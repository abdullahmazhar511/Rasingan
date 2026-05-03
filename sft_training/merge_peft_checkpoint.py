#!/usr/bin/env python3
"""
Merge PEFT LoRA checkpoint with base model and save as a standalone model.
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHECKPOINT_DIR = "/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/llama3.2-1b-sft/checkpoint-288"
OUTPUT_DIR = "/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/llama3.2-1b-sft-merged"

def get_device():
    """Get the device to use for loading models."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def merge_peft_checkpoint(
    base_model_name: str,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "cuda"
):
    """
    Merge PEFT adapter weights with base model.
    
    Args:
        base_model_name: Name or path of the base model
        checkpoint_dir: Path to the PEFT checkpoint (contains adapter_model.safetensors and adapter_config.json)
        output_dir: Output directory for merged model
        device: Device to use ("cuda" or "cpu")
    """
    
    print(f"Device: {device}")
    print(f"Loading base model: {base_model_name}")
    
    # Load base model in half precision for memory efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    
    print(f"Loading PEFT adapter from: {checkpoint_dir}")
    # Load the PEFT model (which will load the adapter)
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        device_map=device if device == "cuda" else None,
    )
    
    print("Merging PEFT adapter with base model...")
    # Merge and unload the adapter
    merged_model = model.merge_and_unload()
    
    # Load tokenizer
    print(f"Loading tokenizer from: {checkpoint_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from checkpoint ({e}), trying base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving merged model to: {output_dir}")
    # Save the merged model
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Merge complete!")
    print(f"\nMerged model saved to: {output_dir}")
    print("\nYou can now use the merged model with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    device = get_device()
    
    # Verify checkpoint exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory not found: {CHECKPOINT_DIR}")
        sys.exit(1)
    
    if not os.path.exists(os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")):
        print(f"Error: adapter_model.safetensors not found in {CHECKPOINT_DIR}")
        sys.exit(1)
    
    merge_peft_checkpoint(
        base_model_name=BASE_MODEL_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
    )
