#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinGPT LoRA Fine-tuning Script
Optimized for Windows with 16GB VRAM
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, load_from_disk

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler('training.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--base_model', required=True, choices=['llama3', 'deepseek'])
    p.add_argument('--from_remote', type=bool, default=False)
    p.add_argument('--dataset', default='data')
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--scheduler', default='constant')
    p.add_argument('--evaluation_strategy', default='steps')
    p.add_argument('--eval_steps', type=float, default=0.1)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='finetuned_models')
    p.add_argument('--ds_config', default=None)
    return p.parse_args()


def get_model_path(base_model, from_remote):
    if from_remote:
        return 'meta-llama/Llama-3.1-8B' if base_model == 'llama3' else 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    return '../Llama-3.1-8B' if base_model == 'llama3' else '../DeepSeek-R1-Distill-Llama-8B'


def load_model_and_tokenizer(args):
    path = get_model_path(args.base_model, args.from_remote)
    logger.info(f"Loading: {path}")
    
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=not args.from_remote, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=not args.from_remote, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True, use_cache=False)
    return model, tokenizer


def setup_lora(model, args):
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_dropout=args.lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    return model


def load_dataset_fn(args, tokenizer):
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        if os.path.exists(args.dataset):
            dpath = os.path.join(args.dataset, 'data') if os.path.exists(os.path.join(args.dataset, 'data')) else args.dataset
            ds = load_from_disk(dpath)
        else:
            ds = load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
    
    def tokenize(examples):
        texts = [f"{p}\n\n{a}{tokenizer.eos_token}" for p, a in zip(examples['prompt'], examples['answer'])]
        tok = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
        tok['labels'] = tok['input_ids'].copy()
        return tok
    
    train = ds['train'].map(tokenize, batched=True, remove_columns=ds['train'].column_names)
    test = ds['test'].map(tokenize, batched=True, remove_columns=ds['test'].column_names)
    return train, test


def main():
    args = parse_args()
    set_seed(args.seed)
    
    ts = datetime.now().strftime('%Y%m%d%H%M')
    name = f"{args.run_name}_{ts}"
    outdir = os.path.join(args.output_dir, name)
    os.makedirs(outdir, exist_ok=True)
    
    logger.info(f"Run: {name}")
    
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_lora(model, args)
    train_ds, eval_ds = load_dataset_fn(args, tokenizer)
    
    steps = len(train_ds) // (args.batch_size * args.gradient_accumulation_steps)
    eval_steps = int(steps * args.eval_steps) if args.evaluation_strategy == 'steps' else None
    save_steps = int(steps * 0.2) if args.evaluation_strategy == 'steps' else None
    
    train_args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        fp16=True,
        logging_steps=args.log_interval,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=5,
        load_best_model_at_end=(args.evaluation_strategy != 'no'),
        report_to='none',
        dataloader_num_workers=0,
        deepspeed=args.ds_config,
        seed=args.seed
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator, tokenizer=tokenizer)
    
    logger.info("Training...")
    result = trainer.train()
    logger.info(f"Done! Loss: {result.training_loss:.4f}")
    
    trainer.save_model()
    tokenizer.save_pretrained(outdir)
    
    with open(os.path.join(outdir, 'training_info.json'), 'w') as f:
        json.dump({'run_name': name, 'base_model': args.base_model, 'timestamp': ts, 'final_loss': float(result.training_loss)}, f, indent=2)


if __name__ == '__main__':
    main()
