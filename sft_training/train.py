import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sys
import torch
import wandb

# Add Rasingan utils to path
sys.path.insert(0, '/home/umairai/faithfulness_emnlp/Rasingan/utils')
from hfDataset import MHCoPilot_Dataset

WANDB_API_KEY = "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"

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
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/home/umairai/faith_data/dataset")
    parser.add_argument("--output_dir", type=str, default="/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/llama3.2-1b-sft_2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()

def format_to_messages(example):
    """Convert dataset example to chat messages format for SFTTrainer."""
    context_str = f"Context: {example['context']}\nTherapist:"
    output_text = example['Utterance'].strip()
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_str},
            {"role": "assistant", "content": output_text}
        ]
    }

def main():
    args = parse_args()
    
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
        }
    )
    
    print(f"Loading tokenizer and model for {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    # Use dedicated pad token - DO NOT use eos_token as pad_token
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("Loading datasets...")
    mhcopilot = MHCoPilot_Dataset(args.data_dir)
    mhcopilot.get_data()
    
    # Convert to chat messages format
    train_dataset = mhcopilot.train_dataset.map(format_to_messages, remove_columns=mhcopilot.train_dataset.column_names)
    val_dataset = mhcopilot.val_dataset.map(format_to_messages, remove_columns=mhcopilot.val_dataset.column_names)
    test_dataset = mhcopilot.test_dataset.map(format_to_messages, remove_columns=mhcopilot.test_dataset.column_names)
    
    # SFTConfig extends TrainingArguments with SFT-specific options
    training_args = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        bf16=True,
        logging_steps=10,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        # SFT-specific settings
        max_length=512,
        packing=False,
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
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model adapter...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

    # print("Evaluating on test data...")
    # test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    # print("Test Results:", test_results)
    
    # # Log final metrics to wandb
    # wandb.log({"test_results": test_results})
    wandb.finish()

if __name__ == "__main__":
    main()
