import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import *
import time
import json, pickle

cache_dir = "./pretrained-models"

# ========================
# Load Base Models
# ========================
# Llama3
llama3_base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
)

# DeepSeek
deepseek_base_model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
)

# ========================
# Load Fine-tuned Models
# ========================
# Llama3
llama3_model = PeftModel.from_pretrained(
    llama3_base_model, 
    'finetuned_models/llama3-test_202511090744', 
    cache_dir=cache_dir, 
    torch_dtype=torch.float16,
)
llama3_model = llama3_model.eval()

# DeepSeek
deepseek_model = PeftModel.from_pretrained(
    deepseek_base_model, 
    'finetuned_models/deepseek-test_202511090815', 
    cache_dir=cache_dir, 
    torch_dtype=torch.float16,
)
deepseek_model = deepseek_model.eval()

# ========================
# Tokenizers
# ========================
llama3_tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    cache_dir=cache_dir,
)
llama3_tokenizer.padding_side = "right"
llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id

deepseek_tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    cache_dir=cache_dir,
)
deepseek_tokenizer.padding_side = "right"
deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id

# ========================
# Load Dataset
# ========================
test_dataset = load_from_disk(
    "/content/drive/MyDrive/Chuoruo/FinGPT-master/fingpt/FinGPT_Forecaster/FinGPT/fingpt-forecaster-dow30-202305-202405"
)["test"]

# ========================
# Functions
# ========================
def insert_guidance_after_intro(prompt):

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


test_dataset = test_dataset.map(lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])})


def test_demo(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt, return_tensors='pt',
        padding=False, max_length=8000
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    start_time = time.time()
    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    end_time = time.time()
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    return output, end_time - start_time


def test_acc(test_dataset, modelname):
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []
    if modelname == "llama3":
        base_model = llama3_base_model
        model = llama3_model
        tokenizer = llama3_tokenizer
    elif modelname == "deepseek":
        base_model = deepseek_base_model
        model = deepseek_model
        tokenizer = deepseek_tokenizer

    for i in tqdm(range(min(30, len(test_dataset))), desc=f"Processing {modelname} samples"):
        try:
            prompt = test_dataset[i]['prompt']
            gt = test_dataset[i]['answer']

            output_base, time_base = test_demo(base_model, tokenizer, prompt)
            answer_base = re.sub(r'.*\[/INST\]\s*', '', output_base, flags=re.DOTALL)

            output_fine_tuned, time_fine_tuned = test_demo(model, tokenizer, prompt)
            answer_fine_tuned = re.sub(r'.*\[/INST\]\s*', '', output_fine_tuned, flags=re.DOTALL)

            answers_base.append(answer_base)
            answers_fine_tuned.append(answer_fine_tuned)
            gts.append(gt)
            times_base.append(time_base)
            times_fine_tuned.append(time_fine_tuned)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned


# ========================
# Llama3 Result Evaluating
# ========================
print("\n=== Llama3 Result Evaluating ===")
llama3_answers_base, llama3_answers_fine_tuned, llama3_gts, llama3_base_times, llama3_fine_tuned_times = test_acc(test_dataset, "llama3")
llama3_base_metrics = calc_metrics(llama3_answers_base, llama3_gts)
llama3_fine_tuned_metrics = calc_metrics(llama3_answers_fine_tuned, llama3_gts)

os.makedirs("./comparison_results", exist_ok=True)
with open("./comparison_results/llama3_base_metrics.pkl", "wb") as f:
    pickle.dump(llama3_base_metrics, f)
with open("./comparison_results/llama3_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(llama3_fine_tuned_metrics, f)
with open("./comparison_results/llama3_base_times.pkl", "wb") as f:
    pickle.dump(llama3_base_times, f)
with open("./comparison_results/llama3_fine_tuned_times.pkl", "wb") as f:
    pickle.dump(llama3_fine_tuned_times, f)


# ========================
# DeepSeek Result Evaluating
# ========================
print("\n=== DeepSeek Result Evaluating ===")
deepseek_answers_base, deepseek_answers_fine_tuned, deepseek_gts, deepseek_base_times, deepseek_fine_tuned_times = test_acc(test_dataset, "deepseek")
deepseek_base_metrics = calc_metrics(deepseek_answers_base, deepseek_gts)
deepseek_fine_tuned_metrics = calc_metrics(deepseek_answers_fine_tuned, deepseek_gts)

with open("./comparison_results/deepseek_base_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_base_metrics, f)
with open("./comparison_results/deepseek_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_fine_tuned_metrics, f)
with open("./comparison_results/deepseek_base_times.pkl", "wb") as f:
    pickle.dump(deepseek_base_times, f)
with open("./comparison_results/deepseek_fine_tuned_times.pkl", "wb") as f:
    pickle.dump(deepseek_fine_tuned_times, f)


# ========================
# Comparing Llama3 and DeepSeek Results
# ========================
print("\n=== Comparing Llama3 and DeepSeek Results ===")
comparison_matrics = calc_metrics(llama3_answers_fine_tuned, deepseek_answers_fine_tuned)
with open("./comparison_results/comparison_matrics.pkl", "wb") as f:
    pickle.dump(comparison_matrics, f)

print(json.dumps(comparison_matrics, indent=2))
print("\n✅ All results saved under ./comparison_results/")
