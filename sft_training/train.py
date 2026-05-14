import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sys
import torch
import wandb
from huggingface_hub import login
from patch_llama_chat_template import patch_tokenizer_chat_template


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank_id() -> int:
    return int(os.environ.get("RANK", "0"))


if is_main_process():
    print(torch.cuda.is_available())
# Add Rasingan utils to path
sys.path.insert(0, '../utils')
from hfDataset import MHCoPilot_Dataset


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


SYSTEM_PROMPT = """You are a compassionate, client-centered therapist.

Respond with empathy, warmth, and non-judgmental understanding. Reflect the
client's emotions and perspective using reflective listening (e.g., "It sounds like…", 
"I hear that…", "You're feeling…").

Encourage gentle exploration through open-ended questions and support the
client's autonomy.

Guidelines:
- Focus on the client's feelings and lived experience.
- Be concise, calm, and emotionally attuned.
- Do NOT give advice, instructions, or solutions.
- Do NOT judge, confront, diagnose, or moralize.
- Do NOT assume information not expressed by the client.

Task: Write the next therapist response."""

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Training for mental health language models")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data_dir", type=str, default="../sft_training/respair_mhcopilot_format")
    parser.add_argument("--output_dir", type=str, default="../sft_training/results/llama3.2-1b-sft-respair-new-3")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--context_window", type=int, default=6)
    parser.add_argument("--eval_steps", type=int, default=100)
    return parser.parse_args()

def context_to_chat_messages(context):
    messages = []
    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Patient:"):
            content = line[len("Patient:"):].strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif line.startswith("Therapist:"):
            content = line[len("Therapist:"):].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
    return messages

def format_to_messages(example):
    """Convert dataset example to chat messages format for SFTTrainer."""
    output_text = example['Utterance'].strip()

    history_messages = context_to_chat_messages(example.get('context', ''))
    # Ensure there is always a user turn before generating assistant target.
    if not history_messages or history_messages[-1]["role"] != "user":
        history_messages.append({"role": "user", "content": "Please continue the session."})

    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history_messages + [
            {"role": "assistant", "content": output_text}
        ]
    }

def main():
    args = parse_args()
    main_process = is_main_process()
    rank = rank_id()

    # Avoid side-effectful auth calls at import time and across all ranks.
    if main_process and HF_TOKEN:
        login(token=HF_TOKEN)
    
    if main_process:
        # Login to wandb
        wandb.login(key=WANDB_API_KEY)

        # Initialize wandb
        model_name_short = args.model_name_or_path.split("/")[-1]
        wandb.init(
            project="rasingan",
            name=f"{model_name_short}-sft",
            entity="abdullahm-indraprastha-institute-of-information-technolo",
            config={
                "model": args.model_name_or_path,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "data_dir": args.data_dir,
                "context_window": args.context_window,
            }
        )
    
    print(f"[rank {rank}] Loading tokenizer and model for {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # tokenizer = patch_tokenizer_chat_template(tokenizer)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    # Use dedicated pad token - DO NOT use eos_token as pad_token
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    # elif tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print(f"[rank {rank}] Loading datasets...")
    mhcopilot = MHCoPilot_Dataset(args.data_dir, context_window=args.context_window)
    mhcopilot.get_data()
    
    # Convert to chat messages format
    train_dataset = mhcopilot.train_dataset.map(format_to_messages, remove_columns=mhcopilot.train_dataset.column_names)
    val_dataset = mhcopilot.val_dataset.map(format_to_messages, remove_columns=mhcopilot.val_dataset.column_names)
    test_dataset = mhcopilot.test_dataset.map(format_to_messages, remove_columns=mhcopilot.test_dataset.column_names)
    
    # SFTConfig - optimized for 8x A100 40GB GPUs with DDP
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        # save_total_limit=2,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if main_process else "none",
        bf16=True,
        logging_steps=10,
        warmup_steps=20,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # SFT-specific settings
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=False
    )
        
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    trainer.model.print_trainable_parameters()
    
    print(f"[rank {rank}] Starting training...")
    trainer.train()
    
    # if main_process:
    #     print(f"[rank {rank}] Saving model adapter...")
    #     trainer.model.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #     print(f"[rank {rank}] Training complete!")
    # else:
    #     print(f"[rank {rank}] Training complete (save skipped on non-main rank).")

    # print("Evaluating on test data...")
    # test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    # print("Test Results:", test_results)
    
    # # Log final metrics to wandb
    # wandb.log({"test_results": test_results})
    if main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
