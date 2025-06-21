# 
# beta = beta0 []
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.stateless import functional_call
import sys
sys.path.append("../quantization")
# from qlinear import getParamNum
import types
from torch.autograd import Function
# 在forward时不保留中间激活图，到backward再自动重新forward一次来计算梯度
from torch.utils.checkpoint import checkpoint
import math
from datasets import load_dataset
import random
from torch.cuda.amp import autocast
import gc
import time

import logging 
from tqdm import tqdm

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(message)s',
#     handlers=[
#         logging.FileHandler("output.log"),
#         logging.StreamHandler()
#     ]
# )



my_logger = logging.getLogger("my_project")

# 配置自己的 handler
file_handler = logging.FileHandler("../logs/beta_grid.txt")
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler.setFormatter(formatter)
my_logger.addHandler(file_handler)

my_logger.setLevel(logging.INFO)

# 数据构造
def get_epoch_batches(tokenizer, device, dataset, micro_batch_size=4, shuffle=True, drop_last=False, seed=42):
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)    # 每个 epoch 随机打乱
    num = 2048
    for start_idx in range(0, num, micro_batch_size):
    # for start_idx in range(0, len(indices), micro_batch_size):
        batch_indices = indices[start_idx:start_idx + micro_batch_size]
        samples = [dataset[i] for i in batch_indices]

        texts = []
        labels = []

        for sample in samples:
            goal = sample["goal"]
            sol1 = sample["sol1"]
            sol2 = sample["sol2"]
            correct_label = int(sample["label"])  # 0 or 1

            # 构造两个输入，一个对应 sol1，一个对应 sol2
            texts.append(f"{goal}\nOption: {sol1}")
            texts.append(f"{goal}\nOption: {sol2}")

            # 正确答案在哪个文本上就标为 1，其它为 0
            labels.append(1 if correct_label == 0 else 0)
            labels.append(1 if correct_label == 1 else 0)

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(labels).to(device)
        # 返回一个 dict，带有 labels
        # inputs["labels"] = torch.tensor(labels).to(device)
        # inputs = tokenizer("DeepSpeed is great!", return_tensors="pt", padding=True)
        yield inputs


class Round(Function):
    @staticmethod
    def forward(ctx, input):
        sign = torch.sign(input)
        output = (sign * torch.floor(torch.abs(input) + 0.5))
        # if(output<=0):
        #     output = 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class RoundBit(Function):
    @staticmethod
    def forward(ctx, input):
        sign = torch.sign(input)
        output = (sign * torch.floor(torch.abs(input) + 0.5))
        # if(output<=0):
        #     output = 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()




class QuantWrapper(nn.Module):
    def __init__(self, model, init_bit=4.0):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 记录所有 nn.Linear 模块
        seen = set()
        self.linear_modules = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and id(m) not in seen and "lm_head" not in name:
                seen.add(id(m))
                self.linear_modules.append(m)
        # self.linear_modules = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        qlayer_num = len(self.linear_modules)
        torch.manual_seed(42)
        mask = torch.rand(qlayer_num, 1) < 0.5
        # self.bit = nn.Parameter(torch.where(mask, torch.tensor(3.0), torch.tensor(4.0)))
        self.bit = nn.Parameter(torch.load("mix_34_largebeta_bit_init.pth"), requires_grad=True)
        # self.bit = nn.Parameter(init_bit * torch.ones(qlayer_num, 1))
        # self.bit.requires_grad = False
        # self.bit[:50].requires_grad = True

        # print(f"linear modules: {len(self.linear_modules)}")
        self.q_group_size = 0
        # 待调整
        self.beta0 = 50



    def fake_quantize(self, w, i):
        w = w.detach()
        org_w_shape = w.shape
        input_dtype = w.dtype
        # print(f"{i}th time quantize")
        # if self.q_group_size > 0:
        #     assert org_w_shape[-1] % self.q_group_size == 0
        #     w = w.reshape(-1, self.q_group_size)
        assert w.dim() == 2
        # gpt说加上.detach防止计算图生成，更保险一点
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        bit_i = (self.bit[i]).clamp(1.0, 8.0)
        # bit_i = 1.0 + 7.0 * torch.sigmoid(self.bit[i])
        # print("[DEBUG] bit[i].requires_grad:",bit_i.requires_grad)
        # max_int = Round.apply(torch.pow(2.0, bit_i) - 1)
        max_int = torch.pow(2.0, RoundBit.apply(bit_i)) - 1
        # print("[DEBUG] max_int.requires_grad:", max_int.requires_grad)
        # print("[DEBUG] max_int.grad_fn:", max_int.grad_fn)
        max_int = max_int.to(dtype=w.dtype, device=w.device)

        min_int = 0

        
        scales = ((max_val - min_val).clamp(min=1e-5) / max_int)
        # print(f"max_int: {max_int}")
        # print(f"min_int: {min_int}")
        min_int = torch.tensor(min_int).to(w.device)
        zeros = (-Round.apply(min_val / scales)).clamp_(min_int, max_int)
        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = (torch.clamp(Round.apply(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
        

        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)
        w = w.to(input_dtype)

        del min_val, max_val, min_int, max_int, zeros, scales, org_w_shape, input_dtype
        torch.cuda.empty_cache()
        # print(f"Layer {i}: w shape = {w.shape}, scales shape = {scales.shape}, zeros shape = {zeros.shape}")
        # print(f"w's dtype: {type(w)}")
        # w = self.bit[i] * w
        return w

    def forward(self, inputs, expect_acc, orig_acc, quant_acc, step_print=False, bit_used=False, grad_print=True):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        

        with torch.no_grad():
            with autocast():
                orig_out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        # 计算acc
        # for h in hooks:
        #     h.remove()

        torch.cuda.empty_cache()
        # self.layer_loss = torch.zeros(1, device=input_ids.device)

        # Patch 所有 Linear 的 forward 方法
        orig_fwd = {}
        def make_patched_forward(i, layer_name):
            # ref = ref_outs[i]
            def patched_forward(module, input):
                # ref = F.linear(input, module.weight, module.bias)
                w_q = module._quant_ref.fake_quantize(module.weight, i)
                q = F.linear(input, w_q, module.bias)
                torch.cuda.empty_cache()
                # module._quant_ref.layer_loss \
                #     += F.mse_loss(q, ref, reduction='mean')
                
                return q
            return patched_forward

        for i, m in enumerate(self.linear_modules):
            orig_fwd[m] = m.forward
            # module与self绑定
            m._quant_ref = self 
            layer_name = f"linear_{i}"  
            m.forward = types.MethodType(make_patched_forward(i, layer_name), m)
        
    
        # def run_model(*inputs):
        #     output = self.model(input_ids=inputs[0], attention_mask=inputs[1]) 
        #     return output.logits
            
        # # forward with autograd enabled
        # q_out = checkpoint(run_model, input_ids, attention_mask)
        with autocast():
            q_out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            # loss = F.mse_loss(q_out, orig_out)
        if(step_print):
            for i, m in enumerate(self.linear_modules):
                layer_name = f"linear_{i}" 
                tqdm.write(f"[DEBUG] Forwarding module: {layer_name or 'unknown'}, using bit[{i}] = {self.bit[i].item():.4f}")
        
        # 恢复原始 forward
        for m in self.linear_modules:
            m.forward = orig_fwd[m]
        del orig_fwd
        torch.cuda.empty_cache()
        
        
        # 计算acc origin_model和q_model
        if labels is not None:
            with torch.no_grad():
                def get_seq_logprobs(logits, input_ids, attention_mask):
                    logits = logits[:, :-1, :]
                    targets = input_ids[:, 1:]
                    # batch_size, seq_len
                    mask = attention_mask[:, 1:] if attention_mask is not None else torch.ones_like(targets)
                    # logits [batch, seq_len, voc_size]
                    log_prob = F.log_softmax(logits, dim=-1)
                    # 按照id选择对应token的probs -> batch, seq_len
                    token_logprobs = torch.gather(log_prob, 2, targets.unsqueeze(-1)).squeeze(-1)
                    return (token_logprobs * mask).sum(dim=1)

                orig_seq_logprobs = get_seq_logprobs(orig_out, input_ids, attention_mask)
                q_seq_logprobs = get_seq_logprobs(q_out, input_ids, attention_mask)

                batch_size = labels.shape[0] // 2
                gt_labels = torch.zeros(batch_size, dtype=torch.long, device=labels.device)

                for i in range(batch_size):
                    if labels[2 * i] == 1:
                        gt_labels[i] = 0  # 正确答案是第一个输入
                    else:
                        gt_labels[i] = 1  # 正确答案是第二个输入
                with torch.no_grad():
                    orig_pred = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
                    quant_pred = torch.zeros(batch_size, dtype=torch.long, device=labels.device)

                for i in range(batch_size):
                    # sol1 vs sol2
                    if orig_seq_logprobs[2 * i] < orig_seq_logprobs[2 * i + 1]:
                        orig_pred[i] = 1
                    else:
                        orig_pred[i] = 0

                    if q_seq_logprobs[2 * i] < q_seq_logprobs[2 * i + 1]:
                        quant_pred[i] = 1
                    else:
                        quant_pred[i] = 0

                # orig_acc = (orig_pred == gt_labels).float().mean().item()
                # quant_acc = (quant_pred == gt_labels).float().mean().item()
                orig_correct = (orig_pred == gt_labels).sum().item()   # 正确个数
                quant_correct = (quant_pred == gt_labels).sum().item()  # 正确个数
                total_sample = gt_labels.numel()

                # print(f"[ACC] Original model: {orig_acc:.4f} | Quantized model: {quant_acc:.4f}")

        
        q_log_prob = F.log_softmax(q_out, dim=2)
        orig_prob = F.softmax(orig_out, dim=2) + 1e-8
        del q_out, orig_out
        # torch.cuda.empty_cache()
        # gc.collect()
        # kl_loss = F.kl_div(q_log_prob, orig_prob, reduction="none").sum(-1).mean()
        # tqdm.write(f"kl_loss value: {kl_loss.item()}, bit mean: {self.bit.mean().item()}")
        # 根据acc做超参选择
        
        if(expect_acc > orig_acc and orig_acc != 0):
            tqdm.write(f"Error:expect acc {expect_acc} cannot obtained from orig_acc {orig_acc}! Changed to orig_acc. ")
            expect_acc = orig_acc

        # alpha *= 100
        # alpha = 75
        beta = 0.5
 
        q_log_prob = q_log_prob[:, :-1, :]
        targets = input_ids[:, 1:].unsqueeze(-1)
        # (B,S-1,V) -> (B, S-1, 1) -> (B, S-1)
        token_log_probs = q_log_prob.gather(2, targets).squeeze(2)
        token_loss = -token_log_probs                   # (B, S-1)
        token_loss = token_loss * attention_mask[:,1:]
        sft_loss = token_loss.sum() / attention_mask[:,1:].sum()
        # 用小数不round
        bit_vals = self.bit.view(-1).to(torch.float64)
        weights = torch.tensor([m.weight.numel() for m in self.linear_modules], dtype=torch.float64).cuda()
        weighted_bit_mean = (bit_vals * weights).sum() / weights.sum()
        
        # beta = max(0, (quant_acc - expect_acc)) * self.beta0
        beta = 0.0
        

        self.loss = sft_loss + beta * weighted_bit_mean
        

        
        if(grad_print):
                self.bit_grads, = torch.autograd.grad(
                    self.loss, 
                    self.bit, 
                    retain_graph=True,  # 如果后面还要 .backward()
                    create_graph=False  # 不需要二阶导
                )
                self.sft_grads, = torch.autograd.grad(
                    sft_loss, 
                    self.bit, 
                    retain_graph=True,  # 如果后面还要 .backward()
                    create_graph=False  # 不需要二阶导
                )                
        if(step_print):
            my_logger.info(f"sft loss: {sft_loss.item()}, weighted bit mean: {weighted_bit_mean.item()}")
            if(grad_print):    
                my_logger.info(f"bit.grad.mean: {self.bit_grads.mean().item():.6f} | norm: {self.bit_grads.norm().item():.6f}")
                my_logger.info(f"sft_loss.grad.mean: {self.sft_grads.mean().item():.6f} | norm: {self.sft_grads.norm().item():.6f}")
        
        return self.loss, orig_correct, quant_correct, total_sample






def main():
    # model_name = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # qlayer_num = getParamNum(base_model)
    # print(f"q layer num:{qlayer_num}")  
    init_bit = 3.0
    model = QuantWrapper(base_model, init_bit)
    lr = 1e-3
    epsilon = 0.005
    # beta0_search = [25, 50, 75, 100]
    # beta0_search = [100, 125]
    # beta0_used = []
    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 8, 
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True}
        },
        "optimizer": {
            "type": "Adam",
            "params": {"lr": lr}
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": True
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[model.bit],
        config=ds_config
    )
    dataset = load_dataset("piqa", split="train")  
    # 1600

    num_epoches = 16
    micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    seed = 42
    expect_acc = 0.76
    expect_lower_bit = 4.0

    my_logger.info(f"config: lr: {lr}, batch size:{model_engine.train_batch_size()}, num epoches:{num_epoches}")
    my_logger.info(f"loss: kld")
    my_logger.info("origin bit")
    ori_bit = model_engine.module.bit
    # for i in range(len(model_engine.module.bit)):
    #     my_logger.info(f"Forwarding module: linear module{i}, using bit[{i}] = {model_engine.module.bit[i].item():.4f}")
    # for beta0 in beta0_search:
    #     model_engine.module.beta0 = beta0
    orig_acc = [0]
    quant_acc = [0]
    bit_mean = []
    # my_logger.info(f"beta0 used is {beta0}")
    bit_param = torch.load("mix_34_largebeta_bit_init.pth")
    model_engine.module.bit.data.copy_(bit_param)
    start_time = time.time()
    for p in optimizer.state.values():
        p.clear()
    # model_engine.optimizer.state.clear()
    for epoch in range(num_epoches):
        data_size = 0
        orig_corrects = 0
        quant_corrects = 0        
        # input("press enter to continue... ")
        my_logger.info(f"Epoch {epoch+1}")
        if epoch <= 8:
            bit_used = False
        else:
            bit_used = True
        for step, batch in tqdm(enumerate(get_epoch_batches(tokenizer, model_engine.device, dataset, micro_batch_size, seed)), desc="training", leave=False):
            if(step % 100 == 0):
                step_print = True
            else:
                step_print = False
            loss, orig_correct, quant_correct, total_sample  = model_engine(batch, expect_acc, orig_acc[epoch], quant_acc[epoch],  step_print=step_print, bit_used=bit_used)
            model_engine.backward(loss)
            model_engine.step()

        data_size = 0
        orig_corrects = 0
        quant_corrects = 0 

        
        grad_print = False
        for step, batch in tqdm(enumerate(get_epoch_batches(tokenizer, model_engine.device, dataset, micro_batch_size, seed)), desc="testing", leave=False):
        
            if(step % 100 == 0):
                step_print = True
            else:
                step_print = False
            with torch.no_grad():
                loss, orig_correct, quant_correct, total_sample  = model_engine(batch, expect_acc, orig_acc[-1], quant_acc[-1], step_print=step_print, bit_used=bit_used, grad_print=grad_print)
            data_size += total_sample
            orig_corrects += orig_correct
            quant_corrects += quant_correct
        
        orig_acc.append(orig_corrects / data_size)
        quant_acc.append(quant_corrects / data_size)
        tqdm.write(f"forward run time: {time.time() - start_time}")
        bit_vals = Round.apply(model_engine.module.bit.view(-1).to(torch.float64))
        weights = torch.tensor([m.weight.numel() for m in model_engine.module.linear_modules], dtype=torch.float64).cuda()
        weighted_bit_mean = (bit_vals * weights).sum() / weights.sum()
        tqdm.write(f"weighted_bit_mean: {weighted_bit_mean}")
        bit_mean.append(weighted_bit_mean)
        my_logger.info(f"epoch {epoch+1}:ori_model acc:{orig_acc[epoch+1]}, quantized_model acc:{quant_acc[epoch+1]}, weighted mean: {bit_mean[epoch]}")

        
        if abs(expect_acc - quant_corrects / data_size) < epsilon:
            # beta0_used.append(model_engine.module.beta0)
            # my_logger.info(f"beta0 useful is {beta0}")
            my_logger.info(f"epoch{epoch+1}: Quantized bit")
            for i in range(len(model_engine.module.bit)):
                my_logger.info(f"Forwarding module: linear module{i}, using bit[{i}] = {model_engine.module.bit[i].item():.4f}")
            my_logger.info(f"epoch {epoch+1}:ori_model acc: {orig_acc[epoch+1]}, quantized_model acc: {quant_acc[epoch+1]}, bit mean: {bit_mean[epoch]}")
            my_logger.info(f"time used is {time.time()-start_time}")
            break
    



        
        

if __name__ == "__main__":
    main()


