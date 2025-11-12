from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import re
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils import *
# ==== HF token helper ====
import os
from transformers.models.llama.configuration_llama import LlamaConfig

def _get_hf_token():
    # 优先用环境变量，其次用写死的
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

tok = _get_hf_token()
if not tok:
    raise RuntimeError("请设置 HF_TOKEN 或在脚本里填入 token")
print(">>> USING HF TOKEN:", tok[:10], "...")



# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,   
)

# Replace with your own api_key and project name
os.environ['WANDB_API_KEY'] = ''    # TODO: Replace with your environment variable
os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'


class GenerationEvalCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return
            
        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['tokenizer']
            generated_texts, reference_texts = [], []

            for feature in tqdm(self.eval_dataset):
                prompt = feature['prompt']
                gt = feature['answer']
                inputs = tokenizer(
                    prompt, return_tensors='pt',
                    padding=False, max_length=4096
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                
                res = model.generate(
                    **inputs, 
                    use_cache=True
                )
                output = tokenizer.decode(res[0], skip_special_tokens=True)
                answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

                generated_texts.append(answer)
                reference_texts.append(gt)

                # print("GENERATED: ", answer)
                # print("REFERENCE: ", gt)

            metrics = calc_metrics(reference_texts, generated_texts)
            
            # Ensure wandb is initialized
            if wandb.run is None:
                wandb.init()
                
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()            


def main(args):
        
    model_name = parse_model_name(args.base_model, args.from_remote)

    # load model
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        LlamaConfig._rope_scaling_validation = lambda self: None
    except Exception:
        pass

    

    # 1) load config first
        # 1) load config first
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=tok,
    )

    # 2) force rope_scaling into the old format that transformers==4.32.0 expects
    # old LLaMA impl wants: {"type": "dynamic", "factor": <float>}
    rs = {}
    if getattr(config, "rope_scaling", None) and isinstance(config.rope_scaling, dict):
        rs = dict(config.rope_scaling)
    else:
        rs = {}

    # make sure the two keys exist
    if "type" not in rs:
        rs["type"] = "dynamic"
    if "factor" not in rs:
        rs["factor"] = 8.0

    # drop all llama3-only keys to avoid surprises in old code
    for k in ["low_freq_factor", "high_freq_factor", "original_max_position_embeddings", "rope_type", "name"]:
        rs.pop(k, None)

    config.rope_scaling = rs

    # 3) now actually load the model with this patched config
    import torch  # 顶上有就不用加

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        token=tok,
        torch_dtype=torch.float16,   # 用半精度省显存
        low_cpu_mem_usage=True,
    )

    model.to("cuda")  # 明确放到单卡上




    if args.local_rank == 0:
        print(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=tok,              # ← 分词器也用同一个 token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
        
    # load data
    if args.from_remote:
        dataset_fname = args.dataset
    else:
        dataset_fname = "./data/" + args.dataset
    dataset_list = load_dataset(dataset_fname, args.from_remote)
    
    dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
    
    if args.test_dataset:
        test_dataset_fname = "./data/" + args.test_dataset
        dataset_list = load_dataset(test_dataset_fname, args.from_remote)
            
    dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    
    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(50))
    
    dataset = original_dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}', # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        #deepspeed=args.ds_config,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False
    
    # model = prepare_model_for_int8_training(model)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    
    # Train
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], 
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True,
            return_tensors="pt"
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                ignore_until_epoch=round(0.3 * args.num_epochs)
            )
        ]
        
    )
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str,
                        choices=['llama3.1-8b', 'deepseek-r1-llama-8b', 'llama2'])    
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.1, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)    
    args = parser.parse_args()
    
    wandb.login()
    main(args)