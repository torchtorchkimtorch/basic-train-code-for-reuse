import os
import wandb
import argparse
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
from setproctitle import setproctitle
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from data import SFTDataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise "Boolean value expected."

seperator = f"{datetime.now().month}_{datetime.now().day}_{datetime.now().hour}_{datetime.now().minute}_{datetime.now().microsecond}"

# required
args = argparse.ArgumentParser()
args.add_argument("--base_model", type=str, required=True)
args.add_argument("--data_path", type=str, required=True)
args.add_argument("--data_path_type", type=str, choices=["json", "hf"], required=True)
#To BE Done
# args.add_argument("--max_length", type=int, required=True)

# hyperparameters
args.add_argument("--per_device_train_batch_size", type=int, required=True)
args.add_argument("--per_device_eval_batch_size", type=int, required=True)
args.add_argument("--gradient_accumulation_steps", type=int, required=True)
args.add_argument("--gradient_checkpointing", type=str2bool, default=True)
args.add_argument("--group_by_length", type=str2bool, default=False)
args.add_argument("--logging_steps", type=int, default=1)
args.add_argument("--eval_strategy", type=str, choices=["no", "steps", "epoch"], help='if "no", do not evaluate')
args.add_argument("--eval_steps", type=int)
args.add_argument("--test_size", type=float)
args.add_argument("--metric_for_best_model", type=str, default="eval_loss")
args.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch","best"], required=True)
args.add_argument("--save_steps", type=int)
args.add_argument("--load_best_model_at_end", type=str2bool, default=True)
args.add_argument("--save_total_limit", type=int, required=True)
args.add_argument("--num_train_epochs", type=int, required=True)
args.add_argument("--weight_decay", type=float, required=True)
args.add_argument("--warmup_ratio", type=float, required=True)
args.add_argument("--lr_scheduler_type", type=str, default="cosine")
args.add_argument("--learning_rate", type=float, required=True)
args.add_argument("--bf16", type=str2bool, default=True)
args.add_argument("--tf32", type=str2bool, default=True)

# optional
args.add_argument("--eval_data_path", type=str)
args.add_argument("--reasoning_args", type=str, default="it")
args.add_argument("--reasoning", type=str2bool, default=False)
args.add_argument("--project_name_for_wandb", type=str, default="training")
args.add_argument("--run_name", type=str, default=seperator)
args.add_argument("--output_base_dir", type=str, default="./output")
args.add_argument("--deepspeed", type=str2bool, default=True)
args.add_argument("--deepspeed_config_path", type=str, default="./ds_config.json")
args.add_argument("--seed", type=int, default=42)
args.add_argument("--setproctitle", type=str, default="mhkim")
args = args.parse_args()

setproctitle(args.setproctitle)
os.environ["WANDB_PROJECT"] = args.project_name_for_wandb
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model, 
    dtype="auto", 
    trust_remote_code=True, 
    # device_map="auto"
)
base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
preprocessor = SFTDataset(base_tokenizer, 
                          args.reasoning_args, 
                          args.reasoning, 
                          args.data_path, 
                          args.data_path_type)
preprocessed_data = preprocessor.preprocess()
data_collator = DataCollatorForSeq2Seq(
    tokenizer=base_tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
)

if args.eval_data_path:
    preprocessor = SFTDataset(
        base_tokenizer, 
        args.reasoning_args, 
        args.reasoning,
        args.eval_data_path, 
        args.data_path_type
    )
    train_dataset = preprocessed_data
    eval_dataset = preprocessor.preprocess()
elif args.eval_strategy != "no" and args.eval_data_path is None:
    dataset = preprocessed_data["train"]
    train_test_data = dataset.train_test_split(
    test_size=args.test_size,
    seed=args.seed
    )
    train_dataset = train_test_data["train"]
    eval_dataset = train_test_data["test"]
else:
    train_dataset = preprocessed_data
    eval_dataset = None


args = TrainingArguments(
    run_name=args.run_name,
    report_to="wandb",
    output_dir=os.path.join(args.output_base_dir, f"{seperator}"),
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    group_by_length=args.group_by_length,
    logging_steps=args.logging_steps,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    metric_for_best_model=args.metric_for_best_model,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps,
    load_best_model_at_end=args.load_best_model_at_end,
    save_total_limit=args.save_total_limit,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    learning_rate=args.learning_rate,
    bf16=args.bf16,
    tf32=args.tf32,
    deepspeed=args.deepspeed_config_path if args.deepspeed else None,
    seed=args.seed,
)

trainer = Trainer(
    model=base_model,
    tokenizer=base_tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
