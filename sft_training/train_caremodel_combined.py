"""
Train CareModel on combined faith_data + PTSD datasets using curriculum learning.
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sys
import torch
import wandb
import pandas as pd
from datasets import Dataset

# Add Rasingan utils to path
sys.path.insert(0, '/home/umairai/faithfulness_emnlp/Rasingan/utils')

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
    parser = argparse.ArgumentParser(description="Train CareModel on combined datasets with curriculum learning")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/home/umairai/faith_data/dataset")
    parser.add_argument("--output_dir", type=str, default="/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/caremodel_combined")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=3)
    parser.add_argument("--phase2_epochs", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_curriculum", action="store_true", default=True)
    return parser.parse_args()

def format_to_messages(example, source='faith_data'):
    """Convert dataset example to chat messages format."""
    if source == 'faith_data':
        context_str = f"Context: {example['context']}\nTherapist:"
        output_text = example['Utterance'].strip()
    else:  # PTSD
        # Build context from conversation
        context_str = f"Context: {example.get('context', '')}\nTherapist:"
        output_text = example['utterance'].strip()
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_str},
            {"role": "assistant", "content": output_text}
        ]
    }

def load_dataset_phase1(data_dir):
    """Load phase 1 (high-quality) curriculum data."""
    print("Loading Phase 1 (high-quality) data...")
    df = pd.read_csv(os.path.join(data_dir, "curriculum_phase1.csv"))
    
    # Add contexts for examples that need them
    contexts = []
    for _, row in df.iterrows():
        if pd.isna(row.get('context')) or row.get('context') == '':
            # Generate context from previous turns if available
            contexts.append('')
        else:
            contexts.append(row['context'])
    df['context'] = contexts
    
    dataset = Dataset.from_pandas(df)
    
    # Format to messages
    dataset = dataset.map(
        lambda x: format_to_messages(x, source=x.get('dataset_source', 'faith_data')),
        remove_columns=dataset.column_names
    )
    
    return dataset

def load_dataset_phase2(data_dir):
    """Load phase 2 (all data) curriculum data."""
    print("Loading Phase 2 (all data) curriculum...")
    df = pd.read_csv(os.path.join(data_dir, "curriculum_phase2.csv"))
    
    contexts = []
    for _, row in df.iterrows():
        if pd.isna(row.get('context')) or row.get('context') == '':
            contexts.append('')
        else:
            contexts.append(row['context'])
    df['context'] = contexts
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda x: format_to_messages(x, source=x.get('dataset_source', 'faith_data')),
        remove_columns=dataset.column_names
    )
    
    return dataset

def load_val_test(data_dir):
    """Load validation and test sets."""
    print("Loading validation and test sets...")
    val_df = pd.read_csv(os.path.join(data_dir, "combined_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "combined_test.csv"))
    
    val_dataset = Dataset.from_pandas(val_df)
    val_dataset = val_dataset.map(
        lambda x: format_to_messages(x, source=x.get('dataset_source', 'faith_data')),
        remove_columns=val_dataset.column_names
    )
    
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(
        lambda x: format_to_messages(x, source=x.get('dataset_source', 'faith_data')),
        remove_columns=test_dataset.column_names
    )
    
    return val_dataset, test_dataset

def main():
    args = parse_args()
    
    # Login to wandb
    wandb.login(key=WANDB_API_KEY)
    
    # Initialize wandb
    model_name_short = args.model_name_or_path.split("/")[-1]
    wandb.init(
        project="rasingan",
        name=f"{model_name_short}-caremodel-combined",
        entity="abdullahm-indraprastha-institute-of-information-technolo",
        config={
            "model": args.model_name_or_path,
            "batch_size": args.batch_size,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "learning_rate": args.learning_rate,
            "curriculum_learning": args.use_curriculum,
            "data_sources": ["faith_data", "ptsd"],
        }
    )
    
    print(f"Loading tokenizer and model for {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
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
    
    # Load datasets
    val_dataset, test_dataset = load_val_test(args.data_dir)
    
    if args.use_curriculum:
        print("\n" + "=" * 80)
        print("CURRICULUM LEARNING: Phase 1 - High Quality Data")
        print("=" * 80)
        
        # Phase 1: High-quality data
        train_dataset_phase1 = load_dataset_phase1(args.data_dir)
        
        training_args_phase1 = SFTConfig(
            output_dir=os.path.join(args.output_dir, "phase1"),
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.phase1_epochs,
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
            max_length=512,
            packing=False,
        )
        
        trainer_phase1 = SFTTrainer(
            model=model,
            args=training_args_phase1,
            train_dataset=train_dataset_phase1,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        print(f"Phase 1 - Training on {len(train_dataset_phase1)} high-quality examples...")
        trainer_phase1.train()
        trainer_phase1.model.print_trainable_parameters()
        
        # Save phase 1 checkpoint
        trainer_phase1.model.save_pretrained(os.path.join(args.output_dir, "phase1_checkpoint"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "phase1_checkpoint"))
        
        print("\n" + "=" * 80)
        print("CURRICULUM LEARNING: Phase 2 - All Data")
        print("=" * 80)
        
        # Phase 2: All data (continue training from phase 1)
        train_dataset_phase2 = load_dataset_phase2(args.data_dir)
        
        training_args_phase2 = SFTConfig(
            output_dir=os.path.join(args.output_dir, "phase2"),
            eval_strategy="epoch",
            learning_rate=args.learning_rate * 0.5,  # Lower learning rate
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.phase2_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb",
            bf16=True,
            logging_steps=10,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=1,
            max_length=512,
            packing=False,
        )
        
        trainer_phase2 = SFTTrainer(
            model=model,  # Continue from phase 1
            args=training_args_phase2,
            train_dataset=train_dataset_phase2,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        print(f"Phase 2 - Training on {len(train_dataset_phase2)} total examples...")
        trainer_phase2.train()
        trainer_phase2.model.print_trainable_parameters()
        
        # Final model
        final_trainer = trainer_phase2
    else:
        # Simple training (no curriculum)
        train_dataset = load_dataset_phase2(args.data_dir)
        
        training_args = SFTConfig(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.phase1_epochs + args.phase2_epochs,
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
            max_length=512,
            packing=False,
        )
        
        final_trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        
        print(f"Training on {len(train_dataset)} examples (no curriculum)...")
        final_trainer.train()
    
    # Save final model
    print("Saving final model adapter...")
    final_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    test_results = final_trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Test Results:", test_results)
    
    wandb.log({"test_results": test_results})
    wandb.finish()
    
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
