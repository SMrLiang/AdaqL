# 对于特定输入，只进行一次前向反向，目的是通过梯度查看敏感度
# 聚合梯度来算
# 在forward时不保留中间激活图，到backward再自动重新forward一次来计算梯度
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("../quantization")
from datasets import load_dataset


from quant_util import *
from tqdm import tqdm
import logging

my_logger = logging.getLogger("piqa")
# 配置自己的 handler
file_handler = logging.FileHandler("grad_mix34_bit_kl.txt")
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler.setFormatter(formatter)
my_logger.addHandler(file_handler)
my_logger.setLevel(logging.INFO)
# 纯kl散度

def main():
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # qlayer_num = getParamNum(base_model)
    # print(f"q layer num:{qlayer_num}")  
    init_bit = 4.0
    model = QuantWrapper(base_model, init_bit)
    lr = 1e-3
    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 1, 
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
    num_epoches = 1
    micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    seed = 42
    
    num_layers = model_engine.module.bit.numel()
    grad_sum    = torch.zeros(num_layers, device="cpu")
    step_cnt    = 0  
    
    for step, batch in tqdm(enumerate(get_epoch_batches(tokenizer, model_engine.device, dataset, micro_batch_size, seed))):
        loss, *_ = model_engine(batch, grad_print=True)
        model_engine.backward(loss)
        model_engine.step()
        grads = model_engine.module.bit_grads.detach().cpu().squeeze()
        bits = model_engine.module.bit.detach().cpu().squeeze()
        grad_sum += grads
        step_cnt += 1
        # for idx, (bit, grad) in enumerate(zip(bits, grads)):
        #     my_logger.info(f"Linear module {idx}: bit={bit.item():.4f}, bit_grad={grad.item():.4f}")

        # input("pause.....")
        break

    avg_grad = grad_sum / step_cnt
    for idx, (bit, grad) in enumerate(zip(bits,avg_grad)):
        my_logger.info(f"[AVG] Linear module {idx}: bit={bit.item():.4f}, bit_grad={grad.item():.4f}")







if __name__ == "__main__":
    main()


