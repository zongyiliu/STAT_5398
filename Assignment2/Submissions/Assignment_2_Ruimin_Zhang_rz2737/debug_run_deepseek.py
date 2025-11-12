# debug_run_deepseek.py
import os
import json
import time
import pickle
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from huggingface_hub import hf_hub_download
from peft import PeftModel

"""
要点说明：
1. 训练时用的是原数据集里的 LLaMA-style prompt，所以这里也直接用原 prompt，不做 ChatML 转换。
2. 只加载一次 base，再把 LoRA 套上去。
3. 跑前 80 条，和你 debug_one.py 里保持一致。
"""

# ========== 0. 基础环境 ==========

# 你之前的目录我都保留了
os.environ["HF_HOME"] = r"E:\hf"
os.environ["HF_DATASETS_CACHE"] = r"E:\hf\datasets"
os.environ["HF_HUB_CACHE"] = r"E:\hf\hub"
os.makedirs(r"E:\hf", exist_ok=True)
os.makedirs(r"E:\hf\datasets", exist_ok=True)
os.makedirs(r"E:\hf\hub", exist_ok=True)

# token 从环境里拿，不写死
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

CACHE_DIR = r"E:/Project/FinGPT/fingpt/FinGPT_Forecaster/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ========== 1. 加载官方数据集，直接用原 prompt ==========

dataset_name = "FinGPT/fingpt-forecaster-dow30-202305-202405"
ds = load_dataset(
    dataset_name,
    token=HF_TOKEN,
    cache_dir=r"E:\hf\datasets",
)
test_dataset = ds["test"]  # 一共 300 条


# ========== 2. 加载 deepseek base + 你的 LoRA ==========

deep_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 先拉 config，为了兼容 rope_scaling
cfg_path = hf_hub_download(
    repo_id=deep_name,
    filename="config.json",
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
)
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg_dict = json.load(f)

rs = cfg_dict.get("rope_scaling", None)
if isinstance(rs, dict):
    # 有些版本需要明确指定 type
    cfg_dict["rope_scaling"] = {"type": "dynamic", "factor": rs.get("factor", 1.0)}

deep_cfg = LlamaConfig(**cfg_dict)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    deep_name,
    cache_dir=CACHE_DIR,
)
# deepseek 基本也是 eos=pad 的
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    deep_name,
    trust_remote_code=True,
    config=deep_cfg,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
base_model.to("cuda").eval()

# 这里改成你真正训练出来的 LoRA 路径
# 你之前是这个路径：./finetuned_models/hw2-deepseek_202511090323/checkpoint-50
lora_path = r"./finetuned_models/hw2-deepseek_202511090323/checkpoint-50"

model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
).eval()
model.to("cuda")


# ========== 3. 推理函数 ==========

def generate_one(m, tok, prompt: str, max_new_tokens: int = 256):
    # 注意：这里直接用原 prompt（里面有 [INST] ... [/INST]）
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False,
    )
    inputs = {k: v.to(m.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        out_ids = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 跟训练/官方评估一致，走贪心
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    end = time.time()

    text = tok.decode(out_ids[0], skip_special_tokens=False)
    return text, end - start


def eval_model(ds, m, tok, max_samples=80):
    answers, gts, times = [], [], []
    total = min(len(ds), max_samples)
    for i in tqdm(range(total), desc="DeepSeek (eval)"):
        row = ds[i]
        prompt = row["prompt"]   # 直接拿原 prompt
        gt = row["answer"]
        try:
            out, t = generate_one(m, tok, prompt)
        except Exception as e:
            print(f"error at {i}: {e}")
            continue

        answers.append(out)
        gts.append(gt)
        times.append(t)
    return answers, gts, times


# ========== 4. 主体 ==========

if __name__ == "__main__":
    os.makedirs("./comparison_results", exist_ok=True)

    ans, gts, ts = eval_model(
        test_dataset,
        model,
        tokenizer,
        max_samples=80,   # 和你 debug_one.py 保持一致
    )

    with open("./comparison_results/deepseek_finetuned_answers.pkl", "wb") as f:
        pickle.dump(ans, f)
    with open("./comparison_results/deepseek_gts.pkl", "wb") as f:
        pickle.dump(gts, f)
    with open("./comparison_results/deepseek_finetuned_times.pkl", "wb") as f:
        pickle.dump(ts, f)

    print(f"✅ saved {len(ans)} answers to ./comparison_results/")
