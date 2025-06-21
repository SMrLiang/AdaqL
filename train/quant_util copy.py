import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../quantization")
# from qlinear import getParamNum
import types
from torch.autograd import Function
# 在forward时不保留中间激活图，到backward再自动重新forward一次来计算梯度
from torch.utils.checkpoint import checkpoint
import math
import random
from torch.cuda.amp import autocast
import gc
import time
from tqdm import tqdm


def get_epoch_batches(tokenizer, device, dataset, micro_batch_size=4, shuffle=True, drop_last=False, seed=42):
    indices = list(range(len(dataset)))
    print(f"dataset length: {len(dataset)}")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)    # 每个 epoch 随机打乱
    
    for start_idx in range(0, len(indices), micro_batch_size):
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
        self.bit = nn.Parameter(torch.where(mask, torch.tensor(3.0), torch.tensor(4.0)))
        # self.bit = nn.Parameter(init_bit * torch.ones(qlayer_num, 1))
        # self.bit.requires_grad = False
        # self.bit[:50].requires_grad = True

        # print(f"linear modules: {len(self.linear_modules)}")
        self.q_group_size = 0



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
        bit_i = (self.bit[i]).clamp(1.0, 18.0)
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

    def forward(self, inputs, expect_acc=0, orig_acc=0, quant_acc=0, step_print=False, grad_print=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        
        # ref_outs = []
        # hooks = []
        # def make_hook():
        #     def _hook(m, inp, out):
        #         ref_outs.append(out.detach())
        #     return _hook
        # for m in self.linear_modules:
        #     hooks.append(m.register_forward_hook(make_hook()))

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
        torch.cuda.empty_cache()
        gc.collect()
        kl_loss = F.kl_div(q_log_prob, orig_prob, reduction="none").sum(-1).mean()
        tqdm.write(f"kl_loss value: {kl_loss.item()}, bit mean: {self.bit.mean().item()}")
        # 根据acc做超参选择
        
        if(expect_acc > orig_acc and orig_acc != 0):
            tqdm.write(f"Error:expect acc {expect_acc} cannot obtained from orig_acc {orig_acc}! Changed to orig_acc. ")
            expect_acc = orig_acc

        if(orig_acc == 0):
            # 0.7 1.5 趋于bit小
            # 0.7 1.2
            # 75 0.5 不动了？
            alpha = expect_acc
            beta = -0.8 
        elif(quant_acc < expect_acc):
            tqdm.write(f"quant_acc {quant_acc} is less than expect_acc{expect_acc}")
            alpha = quant_acc
            beta = -0.8
        elif(quant_acc >= expect_acc):
            tqdm.write(f"quant_acc {quant_acc} is larger than expect_acc {expect_acc}")
            alpha = quant_acc
            beta = 0.8
        alpha *= 100
        alpha = 1
        beta = 0.0
        #  10 0.25-> acc 80 2
        # loss = alpha * kl_loss + beta * self.bit.mean() 
        loss = alpha * kl_loss - beta * self.bit.mean() 
        if(grad_print):
                self.bit_grads, = torch.autograd.grad(
                    loss, 
                    self.bit, 
                    retain_graph=True,  # 如果后面还要 .backward()
                    create_graph=False  # 不需要二阶导
                )
        
        # mse_loss = nn.MSELoss()
        # loss = mse_loss(q_out, orig_out)
        # print(loss.item())
        
        # print("[DEBUG] bit requires_grad:", self.bit.requires_grad)
        # print("[DEBUG] bit.grad_fn is None:", self.bit.grad_fn is None)
        # print("[DEBUG] loss requires_grad:", loss.requires_grad)
        # print("[DEBUG] loss grad_fn:", loss.grad_fn)

        return loss, orig_correct, quant_correct, total_sample
