"""
FinGPT LoRA Fine-tuning Script
Simplified version for Windows with 16GB VRAM
"""

import os
import json
import argparse
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help='Name of this training run')
    parser.add_argument('--base_model', type=str, required=True, choices=['llama3', 'deepseek'], 
                        help='Base model to fine-tune')
    parser.add_argument('--dataset', type=str, default='data', help='Dataset path')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, 
                        help='Gradient accumulation steps')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', 
                        help='Evaluation strategy')
    return parser.parse_args()

def load_model_and_tokenizer(base_model):
    """Load base model and tokenizer"""
    model_paths = {
        'llama3': '../Llama-3.1-8B',
        'deepseek': '../DeepSeek-R1-Distill-Llama-8B'
    }
    
    model_path = model_paths[base_model]
    print(f"\n=== Loading {base_model} from {model_path} ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Load model with FP16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                       'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== LoRA Configuration ===")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model

def load_and_prepare_dataset(dataset_path, tokenizer, max_length):
    """Load and tokenize dataset"""
    print(f"\n=== Loading dataset from {dataset_path} ===")
    
    # Try loading from disk first
    try:
        if os.path.exists(os.path.join(dataset_path, 'data')):
            dataset = load_from_disk(os.path.join(dataset_path, 'data'))
        else:
            dataset = load_from_disk(dataset_path)
    except:
        # Try loading from HuggingFace
        dataset = load_dataset('FinGPT/fingpt-forecaster-dow30-202305-202405')
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    def tokenize_function(examples):
        # Combine prompt and answer
        texts = []
        for prompt, answer in zip(examples['prompt'], examples['answer']):
            text = f"{prompt}\n\n{answer}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        # Labels are the same as input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing train dataset"
    )
    
    tokenized_eval = dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['test'].column_names,
        desc="Tokenizing eval dataset"
    )
    
    return tokenized_train, tokenized_eval

def main():
    args = parse_args()
    
    # Add timestamp to run name
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    full_run_name = f"{args.run_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Training Run: {full_run_name}")
    print(f"Base Model: {args.base_model}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset, tokenizer, args.max_length
    )
    
    # Training arguments
    output_dir = f'finetuned_models/{full_run_name}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=0.1,
        save_strategy='steps',
        save_steps=0.2,
        save_total_limit=5,
        load_best_model_at_end=True,
        report_to='none',
        remove_unused_columns=True,
        dataloader_num_workers=0,
        optim='adamw_torch',
        warmup_ratio=0.03,
        lr_scheduler_type='constant'
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        'run_name': full_run_name,
        'base_model': args.base_model,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == '__main__':
    main()
