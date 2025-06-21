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
from quant_util import get_epoch_batches, Round, RoundBit
import logging 
from tqdm import tqdm
my_logger = logging.getLogger("my_project1")
# 配置自己的 handler
file_handler = logging.FileHandler("my_log1.txt")
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler.setFormatter(formatter)
my_logger.addHandler(file_handler)
my_logger.setLevel(logging.INFO)
class QuantWrapper(nn.Module):
    def __init__(self, model, init_bit=4.0, quant_idx=0):
        super().__init__()
        self.model = model
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
        self.bit = nn.Parameter(init_bit * torch.ones(qlayer_num, 1))
        self.q_group_size = 0
        self.quant_idx = quant_idx


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
        bit_i = self.bit[i] 
        max_int = torch.pow(2.0, RoundBit.apply(bit_i)) - 1
        max_int = max_int.to(dtype=w.dtype, device=w.device)
        min_int = 0    
        scales = ((max_val - min_val).clamp(min=1e-5) / max_int)
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

    def forward(self, inputs, expect_acc, orig_acc, quant_acc, step_print=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        

        with torch.no_grad():
            
            orig_out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits



        torch.cuda.empty_cache()

        # Patch 所有 Linear 的 forward 方法

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

        # for i, m in enumerate(self.linear_modules):
        #     orig_fwd[m] = m.forward
        #     # module与self绑定
        #     m._quant_ref = self 
        #     layer_name = f"linear_{i}"  
        #     m.forward = types.MethodType(make_patched_forward(i, layer_name), m)
        
        # 只量化单层
        m = self.linear_modules[self.quant_idx]
        orig_forward = m.forward 
        m._quant_ref = self
        layer_name = f"linear_{self.quant_idx}"
        m.forward = types.MethodType(make_patched_forward(self.quant_idx, layer_name), m)
        
    
      
        q_out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

          
        # 恢复原始 forward
        m.forward = orig_forward
            
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

        
        del q_out, orig_out

        return 0.0, orig_correct, quant_correct, total_sample



def test(tokenizer, model_engine, dataset, micro_batch_size, seed):
    data_size = 0
    orig_corrects = 0
    quant_corrects = 0
    expect_acc = 70
    orig_acc = quant_acc = 0
    step_print = False 
    for step, batch in tqdm(enumerate(get_epoch_batches(tokenizer, model_engine.device, dataset, micro_batch_size, seed)), desc="testing", leave=False):
            with torch.no_grad():
                loss, orig_correct, quant_correct, total_sample  = model_engine(batch, expect_acc, orig_acc, quant_acc, step_print)
            data_size += total_sample
            orig_corrects += orig_correct
            quant_corrects += quant_correct
    orig_acc = orig_corrects / data_size
    quant_acc = quant_corrects / data_size
    my_logger.info(f"ori_model acc: {orig_acc}, quantized_model acc: {quant_acc}")



def main():
    # model_name = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
    model_name = "meta-llama/Llama-3.2-3B"        

    # model_name = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # qlayer_num = getParamNum(base_model)
    # print(f"q layer num:{qlayer_num}")  

    # 224层
    # qlayer_num
    qlayer_num = 196
    
    # init_bit = [2.0, 4.0, 8.0]
    init_bit = [3.0]
    quant_idx = [13, 27, 45, 66, 97, 123, 154, 189]

    
    lr = 1e-3
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

    
    dataset = load_dataset("piqa", split="train")  
    # 1600
    
    seed = 42
    
    model = QuantWrapper(base_model, init_bit[0], quant_idx[0])
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[model.bit],
        config=ds_config
    )
    micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    bit_param = torch.load("mix_34_largebeta_bit_init.pth")
    model_engine.module.bit.data.copy_(bit_param)
    test(tokenizer, model_engine, dataset, micro_batch_size, seed)
    # for bit in init_bit:
    #     q_idx = quant_idx[0]
    #     my_logger.info(f"config: layer{q_idx} , quantized bit:{bit}")
    #     model_engine.module.bit = nn.Parameter(bit * torch.ones(qlayer_num, 1))
    #     model_engine.module.quant_idx = q_idx
    #     micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    #     test(tokenizer, model_engine, dataset, micro_batch_size, seed)



    # ori_bit = model_engine.module.bit
    # for i in range(len(model_engine.module.bit)):
    #     my_logger.info(f"Forwarding module: linear module{i}, using bit[{i}] = {model_engine.module.bit[i].item():.4f}")
    
    # input("enter to pause...")

    
if __name__ == "__main__":
    main()   