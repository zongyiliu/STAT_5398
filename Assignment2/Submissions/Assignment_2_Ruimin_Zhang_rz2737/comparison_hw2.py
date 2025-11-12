import os
import re
import time
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import load_dataset, calc_metrics  # 用你项目里的
import datasets as hfds
os.environ["HF_HOME"] = r"E:\hf"
os.environ["HF_DATASETS_CACHE"] = r"E:\hf\datasets"
os.environ["HF_HUB_CACHE"] = r"E:\hf\hub"

os.makedirs(r"E:\hf\datasets", exist_ok=True)
os.makedirs(r"E:\hf\hub", exist_ok=True)


def get_hf_token():
    tok = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        # 你项目里原来写死的那一行，也可以放进来
        # or "hf_xxx"
    )
    if not tok:
        raise RuntimeError("远程模式必须有 HF_TOKEN，请设置环境变量 HF_TOKEN")
    return tok


def load_remote_test_dataset(repo_id: str, token: str):
    """
    只走远程，不走本地
    repo_id 例子：FinGPT/fingpt-forecaster-dow30-202305-202405
    """
    # 用你 utils 里那套
    dataset_list = load_dataset(repo_id, from_remote=True)

    # 和训练脚本一样，把所有 test 拼在一起
    test_splits = []
    for d in dataset_list:
        if isinstance(d, hfds.DatasetDict):
            test_splits.append(d["test"])
        else:
            test_splits.append(d)

    if len(test_splits) == 1:
        return test_splits[0]
    else:
        return hfds.concatenate_datasets(test_splits)


def insert_guidance_after_intro(prompt: str) -> str:
    intro_marker = (
        "[INST]<<SYS>>\n"
        "You are a seasoned stock market analyst. Your task is to list the positive developments and "
        "potential concerns for companies based on relevant news and basic financials from the past weeks, "
        "then provide an analysis and prediction for the companies' stock price movement for the upcoming week."
    )
    guidance_start_marker = "Based on all the information before"
    guidance_end_marker = "Following these instructions, please come up with 2-4 most important positive factors"

    intro_pos = prompt.find(intro_marker)
    guidance_start_pos = prompt.find(guidance_start_marker)
    guidance_end_pos = prompt.find(guidance_end_marker)

    if intro_pos == -1 or guidance_start_pos == -1 or guidance_end_pos == -1:
        return prompt

    guidance_section = prompt[guidance_start_pos:guidance_end_pos].strip()

    new_prompt = (
        f"{prompt[:intro_pos + len(intro_marker)]}\n\n"
        f"{guidance_section}\n\n"
        f"{prompt[intro_pos + len(intro_marker):guidance_start_pos]}"
        f"{prompt[guidance_end_pos:]}"
    )

    return new_prompt


def apply_to_all_prompts_in_dataset(test_dataset):
    return test_dataset.map(lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])})


def test_demo(model, tokenizer, prompt, is_deepseek=False):
    if is_deepseek and tokenizer.chat_template is not None:
        # 1) 用 deepseek 的 chat 模板包一层
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # inputs 是个 tensor，我们要手动建 attention_mask
        input_ids = inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids)

        # 2) 这里一定要用关键字参数，不要把 input_ids 当第一个位置参数传
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text, 0.0

    else:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=2048,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        start = time.time()
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=128,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        end = time.time()

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text, end - start

import re

def strip_prompt(generated: str, prompt: str) -> str:
    # 1) 优先用前缀匹配剪掉整个 prompt
    if generated.startswith(prompt):
        return generated[len(prompt):].lstrip()
    # 2) 兼容老的有 [/INST] 的情况
    m = re.search(r"\[/INST\]\s*", generated)
    if m:
        return generated[m.end():].lstrip()
    # 3) 实在剪不了就原样返回点干净的
    return generated.lstrip()

def test_acc(
    test_dataset,
    modelname,
    llama3_base_model, llama3_model, llama3_tokenizer,
    deepseek_base_model, deepseek_model, deepseek_tokenizer
):
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []

    if modelname == "llama3":
        base_model = llama3_base_model
        model = llama3_model
        tokenizer = llama3_tokenizer
    else:
        base_model = deepseek_base_model
        model = deepseek_model
        tokenizer = deepseek_tokenizer

    for i in tqdm(range(len(test_dataset)), desc=f"Processing ({modelname})"):
        row = test_dataset[i]
        prompt = row["prompt"]
        gt = row["answer"]

        out_base, t_base = test_demo(base_model, tokenizer, prompt, is_deepseek=(modelname=="deepseek"))
        ans_base = strip_prompt(out_base, prompt)

        out_ft, t_ft = test_demo(model, tokenizer, prompt, is_deepseek=(modelname=="deepseek"))
        ans_ft = strip_prompt(out_ft, prompt)

        answers_base.append(ans_base)
        answers_fine_tuned.append(ans_ft)
        gts.append(gt)
        times_base.append(t_base)
        times_fine_tuned.append(t_ft)
    
    os.makedirs("./comparison_results", exist_ok=True)
    with open(f"./comparison_results/{modelname}_all.pkl", "wb") as f:
        pickle.dump(
            {
                "answers_base": answers_base,
                "answers_fine_tuned": answers_fine_tuned,
                "gts": gts,
                "times_base": times_base,
                "times_fine_tuned": times_fine_tuned,
            },
            f,
        )

    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="远程数据集名，比如 FinGPT/fingpt-forecaster-dow30-202305-202405"
    )
    parser.add_argument(
        "--llama_ckpt",
        default="./finetuned_models/hw2-llama2_202511090212"
    )
    parser.add_argument(
        "--deepseek_ckpt",
        default="./finetuned_models/hw2-deepseek_202511090323/checkpoint-50"
    )
    parser.add_argument(
        "--cache_dir",
        default=r"E:/Project/FinGPT/fingpt/FinGPT_Forecaster/hf_cache"
    )
    args = parser.parse_args()

    # 1) token
    tok = get_hf_token()
    print(">>> USING HF TOKEN:", tok[:10], "...")

    # 2) 远程加载 test 数据
    test_dataset = load_remote_test_dataset(args.dataset, tok)
    test_dataset = apply_to_all_prompts_in_dataset(test_dataset)

    # 3) 模型 & tokenizer
    #llama3_base_model = AutoModelForCausalLM.from_pretrained(
    #    'meta-llama/Llama-2-7b-chat-hf',
    #    token=tok,
    #    trust_remote_code=True,
    #    device_map="auto",
    #    cache_dir=args.cache_dir,
    #    torch_dtype=torch.float16,
    #)
    deepseek_base_model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        token=tok,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
    )

    #llama3_model = PeftModel.from_pretrained(
    #    llama3_base_model,
    #    args.llama_ckpt,
    #    cache_dir=args.cache_dir,
    #    torch_dtype=torch.float16,
    #    offload_folder=os.path.join(args.cache_dir, "offload_llama3")
    #).eval()

    deepseek_model = PeftModel.from_pretrained(
        deepseek_base_model,
        args.deepseek_ckpt,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        offload_folder=os.path.join(args.cache_dir, "offload_deepseek")
    ).eval()

    #llama3_tokenizer = AutoTokenizer.from_pretrained(
    #    'meta-llama/Llama-2-7b-chat-hf',
    #    token=tok,
    #    cache_dir=args.cache_dir,
    #)
    #llama3_tokenizer.padding_side = "right"
    #llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id

    deepseek_tokenizer = AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        token=tok,
        cache_dir=args.cache_dir,
    )
    deepseek_tokenizer.padding_side = "right"
    deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id

    # 4) 跑两套
    os.makedirs("./comparison_results", exist_ok=True)

    #l_base, l_ft, l_gts, l_bt, l_ftt = test_acc(
    #    test_dataset, "llama3",
    #    llama3_base_model, llama3_model, llama3_tokenizer,
    #    deepseek_base_model, deepseek_model, deepseek_tokenizer
    #)
    #l_base_m = calc_metrics(l_base, l_gts)
    #l_ft_m = calc_metrics(l_ft, l_gts)

    #ith open("./comparison_results/llama3_base_metrics.pkl", "wb") as f:
    #    pickle.dump(l_base_m, f)
    #with open("./comparison_results/llama3_fine_tuned_metrics.pkl", "wb") as f:
    #    pickle.dump(l_ft_m, f)

    d_base, d_ft, d_gts, d_bt, d_ftt = test_acc(
        test_dataset, "deepseek",
        None, None, None,
        #llama3_base_model, llama3_model, llama3_tokenizer,
        deepseek_base_model, deepseek_model, deepseek_tokenizer
    )
    d_base_m = calc_metrics(d_base, d_gts)
    d_ft_m = calc_metrics(d_ft, d_gts)

    with open("./comparison_results/deepseek_base_metrics.pkl", "wb") as f:
        pickle.dump(d_base_m, f)
    with open("./comparison_results/deepseek_fine_tuned_metrics.pkl", "wb") as f:
        pickle.dump(d_ft_m, f)

    l_ft = pd.read_pickle("./comparison_results/llama3_fine_tuned_metrics.pkl")

    # finetune 互相比一下
    comp_m = calc_metrics(l_ft, d_ft)
    with open("./comparison_results/comparison_metrics.pkl", "wb") as f:
        pickle.dump(comp_m, f)

    print("done, saved to ./comparison_results")


if __name__ == "__main__":
    main()
