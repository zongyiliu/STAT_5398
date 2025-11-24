# 后面紧跟上面那整段脚本

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comparison.py

Evaluate base vs fine-tuned models (Llama3-1B & DeepSeek-1.5B) on the
Dow30 forecasting dataset, and compare their performance.

This script is self-contained:
- does NOT depend on utils.py
- loads models in 4-bit to save GPU memory
- evaluates base model and LoRA fine-tuned model separately
- saves all metrics & inference times under ./comparison_results/
"""

import os
import re
import time
import pickle
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import load_dataset, Dataset
from rouge_score import rouge_scorer

# -----------------------
# 1. Global config
# -----------------------

# Hugging Face dataset name (same as training)
DATASET_NAME = "FinGPT/fingpt-forecaster-dow30-202305-202405"

# Paths to your saved LoRA adapters on Google Drive
OUTPUT_ROOT = "/content/drive/MyDrive/STAT5398_A2/outputs"

LLAMA3_BASE_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA3_ADAPTER_DIR = os.path.join(
    OUTPUT_ROOT, "dow30_llama3_1b_lora", "lora_adapter"
)

DEEPSEEK_BASE_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEEPSEEK_ADAPTER_DIR = os.path.join(
    OUTPUT_ROOT, "dow30_deepseek_1p5b_lora", "lora_adapter"
)

RESULT_DIR = "/content/drive/MyDrive/STAT5398_A2/comparison_results"
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_INPUT_LEN = 2048
MAX_NEW_TOKENS = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config (same style as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Rouge scorer
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# -----------------------
# 2. Dataset helpers
# -----------------------

INTRO_MARKER = (
    "[INST]<<SYS>>\n"
    "You are a seasoned stock market analyst. Your task is to list the positive developments and "
    "potential concerns for companies based on relevant news and basic financials from the past weeks, "
    "then provide an analysis and prediction for the companies' stock price movement for the upcoming week."
)
GUIDANCE_START = "Based on all the information before"
GUIDANCE_END = "Following these instructions, please come up with 2-4 most important positive factors"


def insert_guidance_after_intro(prompt: str) -> str:
    """
    Reorder the guidance section so that it appears right after the intro system message.
    This follows the instructor's comparison.py logic.
    """
    intro_pos = prompt.find(INTRO_MARKER)
    g_start = prompt.find(GUIDANCE_START)
    g_end = prompt.find(GUIDANCE_END)

    if intro_pos == -1 or g_start == -1 or g_end == -1:
        # If we cannot find all markers, return original prompt
        return prompt

    guidance_section = prompt[g_start:g_end].strip()

    new_prompt = (
        f"{prompt[:intro_pos + len(INTRO_MARKER)]}\n\n"
        f"{guidance_section}\n\n"
        f"{prompt[intro_pos + len(INTRO_MARKER):g_start]}"
        f"{prompt[g_end:]}"
    )
    return new_prompt


def apply_to_all_prompts(test_dataset: Dataset) -> Dataset:
    """Apply guidance reordering to all prompts in the dataset."""
    return test_dataset.map(
        lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])},
        desc="Rewriting prompts with guidance section",
    )


# -----------------------
# 3. Metrics
# -----------------------

def calc_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Compute text generation metrics between predictions and references.
    - Rouge-1 F1
    - Rouge-L F1
    - Exact match rate (strict string equality after strip)
    """
    rouge1_f, rougeL_f, em = [], [], []

    for p, r in zip(preds, refs):
        p = p.strip()
        r = r.strip()
        scores = rouge.score(r, p)  # ref, hyp
        rouge1_f.append(scores["rouge1"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)
        em.append(float(p == r))

    return {
        "rouge1_f": float(np.mean(rouge1_f)) if rouge1_f else 0.0,
        "rougeL_f": float(np.mean(rougeL_f)) if rougeL_f else 0.0,
        "exact_match": float(np.mean(em)) if em else 0.0,
    }


# -----------------------
# 4. Model loading & generation
# -----------------------

def load_model_and_tokenizer(
    base_model_name: str,
    adapter_dir: str = None,
):
    """
    Load base model in 4-bit and optionally attach a LoRA adapter.
    We use device_map='auto' so that HF dispatches weights to GPU.
    """
    print(f"[INFO] Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if adapter_dir is not None:
        print(f"[INFO] Attaching LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
        )

    model.eval()
    return model, tokenizer


def extract_answer(output_text: str) -> str:
    """
    Post-process the raw generation:
    - keep only content after '[/INST]'
    - strip leading/trailing whitespace
    """
    m = re.search(r"\[/INST\]\s*(.*)", output_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return output_text.strip()


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_input_len: int = MAX_INPUT_LEN,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, float]:
    """Generate one answer and measure latency."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    end = time.time()

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_answer(decoded), (end - start)


def run_eval_loop(
    model,
    tokenizer,
    test_dataset: Dataset,
    desc: str,
) -> Tuple[List[str], List[str], List[float]]:
    """Run generation for every sample in the test set."""
    preds, refs, times = [], [], []

    for row in tqdm(test_dataset, desc=desc):
        prompt = row["prompt"]
        gt = row["answer"]

        ans, t = generate_one(model, tokenizer, prompt)
        preds.append(ans)
        refs.append(gt)
        times.append(t)

    return preds, refs, times


# -----------------------
# 5. High-level evaluation for one model family
# -----------------------

def evaluate_model_family(
    base_model_name: str,
    adapter_dir: str,
    test_dataset: Dataset,
    tag: str,
):
    """
    Evaluate (1) base model and (2) LoRA fine-tuned model for the same family.

    Returns a dict containing:
        - base_metrics, ft_metrics
        - base_times, ft_times
        - base_answers, ft_answers, gts
    """
    # 5.1 Base model
    base_model, tokenizer = load_model_and_tokenizer(base_model_name, adapter_dir=None)
    base_answers, gts, base_times = run_eval_loop(
        base_model,
        tokenizer,
        test_dataset,
        desc=f"{tag} | base model",
    )
    base_metrics = calc_metrics(base_answers, gts)
    print(f"[RESULT] {tag} base metrics:", base_metrics)

    # free memory
    del base_model
    torch.cuda.empty_cache()

    # 5.2 Fine-tuned model
    ft_model, tokenizer = load_model_and_tokenizer(base_model_name, adapter_dir=adapter_dir)
    ft_answers, gts2, ft_times = run_eval_loop(
        ft_model,
        tokenizer,
        test_dataset,
        desc=f"{tag} | fine-tuned",
    )
    assert gts == gts2  # sanity check
    ft_metrics = calc_metrics(ft_answers, gts)
    print(f"[RESULT] {tag} fine-tuned metrics:", ft_metrics)

    del ft_model
    torch.cuda.empty_cache()

    return {
        "base_metrics": base_metrics,
        "ft_metrics": ft_metrics,
        "base_times": base_times,
        "ft_times": ft_times,
        "base_answers": base_answers,
        "ft_answers": ft_answers,
        "gts": gts,
    }


# -----------------------
# 6. Main
# -----------------------

def main():
    # 6.1 Load dataset and test split
    print(f"[INFO] Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    test_ds = raw["test"]
    print(f"[INFO] Raw test size: {len(test_ds)}")

    # 6.2 Apply prompt rewriting (guidance after intro)
    test_ds = apply_to_all_prompts(test_ds)
    print(f"[INFO] Test size after rewriting: {len(test_ds)}")

    # 6.3 Evaluate Llama3 family
    llama3_results = evaluate_model_family(
        LLAMA3_BASE_NAME,
        LLAMA3_ADAPTER_DIR,
        test_ds,
        tag="Llama3-1B",
    )

    # Save Llama3 results
    with open(os.path.join(RESULT_DIR, "llama3_base_metrics.pkl"), "wb") as f:
        pickle.dump(llama3_results["base_metrics"], f)

    with open(os.path.join(RESULT_DIR, "llama3_fine_tuned_metrics.pkl"), "wb") as f:
        pickle.dump(llama3_results["ft_metrics"], f)

    with open(os.path.join(RESULT_DIR, "llama3_base_times.pkl"), "wb") as f:
        pickle.dump(llama3_results["base_times"], f)

    with open(os.path.join(RESULT_DIR, "llama3_fine_tuned_times.pkl"), "wb") as f:
        pickle.dump(llama3_results["ft_times"], f)

    # 6.4 Evaluate DeepSeek family
    deepseek_results = evaluate_model_family(
        DEEPSEEK_BASE_NAME,
        DEEPSEEK_ADAPTER_DIR,
        test_ds,
        tag="DeepSeek-1.5B",
    )

    with open(os.path.join(RESULT_DIR, "deepseek_base_metrics.pkl"), "wb") as f:
        pickle.dump(deepseek_results["base_metrics"], f)

    with open(os.path.join(RESULT_DIR, "deepseek_fine_tuned_metrics.pkl"), "wb") as f:
        pickle.dump(deepseek_results["ft_metrics"], f)

    with open(os.path.join(RESULT_DIR, "deepseek_base_times.pkl"), "wb") as f:
        pickle.dump(deepseek_results["base_times"], f)

    with open(os.path.join(RESULT_DIR, "deepseek_fine_tuned_times.pkl"), "wb") as f:
        pickle.dump(deepseek_results["ft_times"], f)

    # 6.5 Cross-model comparison (fine-tuned Llama3 vs fine-tuned DeepSeek)
    print("\n[INFO] Comparing fine-tuned Llama3 vs fine-tuned DeepSeek...")
    comparison_metrics = calc_metrics(
        llama3_results["ft_answers"],
        deepseek_results["ft_answers"],
    )

    with open(os.path.join(RESULT_DIR, "comparison_metrics.pkl"), "wb") as f:
        pickle.dump(comparison_metrics, f)

    print("\n[RESULT] Cross-model comparison metrics:", comparison_metrics)
    print(f"\nAll evaluation finished. Results saved under {RESULT_DIR}/")


if __name__ == "__main__":
    main()
