import os
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import json
import pickle

# Add FinGPT_Forecaster to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FinGPT_Forecaster'))
from utils import calc_metrics

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use local models and data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fine-tuned model directories - updated to actual trained model paths
llama3_finetuned_dir = os.path.join(SCRIPT_DIR, 'FinGPT_Forecaster', 'finetuned_models', 'llama3_optimized_auto_202511070014')
deepseek_finetuned_dir = os.path.join(SCRIPT_DIR, 'FinGPT_Forecaster', 'finetuned_models', 'deepseek_optimized_auto_202511070407')

# Local base model paths - in current directory
llama3_base_path = os.path.join(SCRIPT_DIR, 'Llama-3.1-8B')
deepseek_base_path = os.path.join(SCRIPT_DIR, 'DeepSeek-R1-Distill-Llama-8B')

# Local dataset path
dataset_path = os.path.join(SCRIPT_DIR, 'FinGPT_Forecaster', 'data', 'data')

# Output directory
output_dir = os.path.join(SCRIPT_DIR, 'comparison_results')
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# LOAD BASE MODELS WITH 8-BIT QUANTIZATION
# ============================================================================

print("="*80)
print("Loading base models with 8-bit quantization...")
print("="*80)

# 8-bit quantization config to save memory
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Llama3 Base Model
print("\nLoading Llama-3.1-8B base model (8-bit)...")
llama3_base_model = AutoModelForCausalLM.from_pretrained(
    llama3_base_path,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,
    local_files_only=True
)
print("✓ Llama-3.1-8B base model loaded (8-bit)")

# DeepSeek Base Model
print("\nLoading DeepSeek-R1-Distill-Llama-8B base model (8-bit)...")
deepseek_base_model = AutoModelForCausalLM.from_pretrained(
    deepseek_base_path,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,
    local_files_only=True
)
print("✓ DeepSeek-R1 base model loaded (8-bit)")

# ============================================================================
# LOAD FINE-TUNED MODELS
# ============================================================================

print("\n" + "="*80)
print("Loading fine-tuned models...")
print("="*80)

# Check if fine-tuned models exist
if os.path.exists(llama3_finetuned_dir):
    print(f"\nLoading Llama3 fine-tuned model from {llama3_finetuned_dir}...")
    llama3_model = PeftModel.from_pretrained(
        llama3_base_model, 
        llama3_finetuned_dir,
    )
    llama3_model = llama3_model.eval()
    print("✓ Llama3 fine-tuned model loaded")
else:
    print(f"\n⚠ WARNING: Llama3 fine-tuned model not found at {llama3_finetuned_dir}")
    print("Using base model for comparison instead.")
    llama3_model = llama3_base_model
    llama3_model = llama3_model.eval()

if os.path.exists(deepseek_finetuned_dir):
    print(f"\nLoading DeepSeek fine-tuned model from {deepseek_finetuned_dir}...")
    deepseek_model = PeftModel.from_pretrained(
        deepseek_base_model, 
        deepseek_finetuned_dir,
    )
    deepseek_model = deepseek_model.eval()
    print("✓ DeepSeek fine-tuned model loaded")
else:
    print(f"\n⚠ WARNING: DeepSeek fine-tuned model not found at {deepseek_finetuned_dir}")
    print("Using base model for comparison instead.")
    deepseek_model = deepseek_base_model
    deepseek_model = deepseek_model.eval()

# ============================================================================
# LOAD TOKENIZERS
# ============================================================================

print("\n" + "="*80)
print("Loading tokenizers...")
print("="*80)

# Llama3 Tokenizer
print("\nLoading Llama3 tokenizer...")
llama3_tokenizer = AutoTokenizer.from_pretrained(
    llama3_base_path,
)
llama3_tokenizer.padding_side = "right"
llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id
print("✓ Llama3 tokenizer loaded")

# DeepSeek Tokenizer
print("\nLoading DeepSeek tokenizer...")
deepseek_tokenizer = AutoTokenizer.from_pretrained(
    deepseek_base_path,
)
deepseek_tokenizer.padding_side = "right"
deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id
print("✓ DeepSeek tokenizer loaded")

# ============================================================================
# LOAD DATASET
# ============================================================================

print("\n" + "="*80)
print("Loading Dow Jones 30 dataset from local files...")
print("="*80)

# Load dataset from local Parquet files
train_file = os.path.join(dataset_path, 'train-00000-of-00001-7c4c80aa07272d4c.parquet')
test_file = os.path.join(dataset_path, 'test-00000-of-00001-28531804b005ddc6.parquet')

print(f"\nLoading test data: {test_file}")
ds = load_dataset('parquet', data_files={
    'train': train_file,
    'test': test_file
})
test_dataset = ds["test"]

# Quick test mode - evaluate only first 50 samples (takes ~30min-1hr)
# For full evaluation, comment out the line below (300 samples need 6-10 hours)
test_dataset = test_dataset.select(range(50))

print(f"✓ Dataset loaded: {len(test_dataset)} test samples")

def filter_by_ticker(test_dataset, ticker_code):

    filtered_data = []

    for row in test_dataset:
        prompt_content = row['prompt']

        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)

        if ticker_symbol and ticker_symbol.group(1) == ticker_code:
            filtered_data.append(row)

    filtered_dataset = Dataset.from_dict({key: [row[key] for row in filtered_data] for key in test_dataset.column_names})

    return filtered_dataset

def get_unique_ticker_symbols(test_dataset):

    ticker_symbols = set()

    for i in range(len(test_dataset)):
        prompt_content = test_dataset[i]['prompt']

        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)

        if ticker_symbol:
            ticker_symbols.add(ticker_symbol.group(1))

    return list(ticker_symbols)

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

def apply_to_all_prompts_in_dataset(test_dataset):

    updated_dataset = test_dataset.map(lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])})

    return updated_dataset

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_by_ticker(test_dataset, ticker_code):
    """Filter dataset by specific ticker symbol"""
    filtered_data = []
    for row in test_dataset:
        prompt_content = row['prompt']
        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)
        if ticker_symbol and ticker_symbol.group(1) == ticker_code:
            filtered_data.append(row)
    filtered_dataset = Dataset.from_dict({key: [row[key] for row in filtered_data] for key in test_dataset.column_names})
    return filtered_dataset

def get_unique_ticker_symbols(test_dataset):
    """Extract unique ticker symbols from the dataset"""
    ticker_symbols = set()
    for i in range(len(test_dataset)):
        prompt_content = test_dataset[i]['prompt']
        ticker_symbol = re.search(r"ticker\s([A-Z]+)", prompt_content)
        if ticker_symbol:
            ticker_symbols.add(ticker_symbol.group(1))
    return list(ticker_symbols)

def insert_guidance_after_intro(prompt):
    """Modify prompt structure for better model performance"""
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
    """Apply prompt modifications to entire dataset"""
    updated_dataset = test_dataset.map(lambda x: {"prompt": insert_guidance_after_intro(x["prompt"])})
    return updated_dataset

print("\nApplying prompt modifications...")
test_dataset = apply_to_all_prompts_in_dataset(test_dataset)
unique_symbols = set(test_dataset['symbol'])
print(f"✓ Prompt modifications applied")
print(f"✓ Found {len(unique_symbols)} unique stock symbols")

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def test_demo(model, tokenizer, prompt):
    """Run inference on a single prompt and measure time"""
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
    """Test both base and fine-tuned models on the dataset"""
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []
    if modelname == "llama3":
        base_model = llama3_base_model
        model = llama3_model
        tokenizer = llama3_tokenizer
    elif modelname == "deepseek":
        base_model = deepseek_base_model
        model = deepseek_model
        tokenizer = deepseek_tokenizer

    for i in tqdm(range(len(test_dataset)), desc=f"Processing {modelname} test samples"):
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
            print(f"\nError processing sample {i}: {e}")
    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

os.makedirs("./comparison_results", exist_ok=True)

# ============================================================================
# EVALUATE LLAMA3 MODELS
# ============================================================================

print("\n" + "="*80)
print("EVALUATING LLAMA3 MODELS")
print("="*80)

llama3_answers_base, llama3_answers_fine_tuned, llama3_gts, llama3_base_times, llama3_fine_tuned_times = test_acc(test_dataset, "llama3")

print("\n--- Llama3 Base Model Results ---")
llama3_base_metrics = calc_metrics(llama3_answers_base, llama3_gts)

print("\n--- Llama3 Fine-tuned Model Results ---")
llama3_fine_tuned_metrics = calc_metrics(llama3_answers_fine_tuned, llama3_gts)

# Save results
with open(os.path.join(output_dir, "llama3_base_metrics.pkl"), "wb") as f:
    pickle.dump(llama3_base_metrics, f)

with open(os.path.join(output_dir, "llama3_fine_tuned_metrics.pkl"), "wb") as f:
    pickle.dump(llama3_fine_tuned_metrics, f)

with open(os.path.join(output_dir, "llama3_base_times.pkl"), "wb") as f:
    pickle.dump(llama3_base_times, f)

with open(os.path.join(output_dir, "llama3_fine_tuned_times.pkl"), "wb") as f:
    pickle.dump(llama3_fine_tuned_times, f)

if llama3_base_times and llama3_fine_tuned_times:
    print(f"\nLlama3 average inference time (base): {sum(llama3_base_times)/len(llama3_base_times):.2f}s")
    print(f"Llama3 average inference time (fine-tuned): {sum(llama3_fine_tuned_times)/len(llama3_fine_tuned_times):.2f}s")

# ============================================================================
# EVALUATE DEEPSEEK MODELS
# ============================================================================

print("\n" + "="*80)
print("EVALUATING DEEPSEEK MODELS")
print("="*80)

deepseek_answers_base, deepseek_answers_fine_tuned, deepseek_gts, deepseek_base_times, deepseek_fine_tuned_times = test_acc(test_dataset, "deepseek")

print("\n--- DeepSeek Base Model Results ---")
deepseek_base_metrics = calc_metrics(deepseek_answers_base, deepseek_gts)

print("\n--- DeepSeek Fine-tuned Model Results ---")
deepseek_fine_tuned_metrics = calc_metrics(deepseek_answers_fine_tuned, deepseek_gts)

# Save results
with open(os.path.join(output_dir, "deepseek_base_metrics.pkl"), "wb") as f:
    pickle.dump(deepseek_base_metrics, f)

with open(os.path.join(output_dir, "deepseek_fine_tuned_metrics.pkl"), "wb") as f:
    pickle.dump(deepseek_fine_tuned_metrics, f)

with open(os.path.join(output_dir, "deepseek_base_times.pkl"), "wb") as f:
    pickle.dump(deepseek_base_times, f)

with open(os.path.join(output_dir, "deepseek_fine_tuned_times.pkl"), "wb") as f:
    pickle.dump(deepseek_fine_tuned_times, f)

if deepseek_base_times and deepseek_fine_tuned_times:
    print(f"\nDeepSeek average inference time (base): {sum(deepseek_base_times)/len(deepseek_base_times):.2f}s")
    print(f"DeepSeek average inference time (fine-tuned): {sum(deepseek_fine_tuned_times)/len(deepseek_fine_tuned_times):.2f}s")

# ============================================================================
# COMPARE LLAMA3 AND DEEPSEEK (FINE-TUNED)
# ============================================================================

print("\n" + "="*80)
print("COMPARING LLAMA3 VS DEEPSEEK (FINE-TUNED MODELS)")
print("="*80)

comparison_matrics = calc_metrics(llama3_answers_fine_tuned, deepseek_answers_fine_tuned)

with open(os.path.join(output_dir, "comparison_matrics.pkl"), "wb") as f:
    pickle.dump(comparison_matrics, f)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nAll results saved to {output_dir}")
print("\nFiles created:")
print("  - llama3_base_metrics.pkl")
print("  - llama3_fine_tuned_metrics.pkl")  
print("  - llama3_base_times.pkl")
print("  - llama3_fine_tuned_times.pkl")
print("  - deepseek_base_metrics.pkl")
print("  - deepseek_fine_tuned_metrics.pkl")
print("  - deepseek_base_times.pkl")
print("  - deepseek_fine_tuned_times.pkl")
print("  - comparison_matrics.pkl")
print("\n" + "="*80)

