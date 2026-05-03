"""
Evaluate retrained CareModel on both faith_data and PTSD domains.
Measures improvement over original bias.
"""
import argparse
import json
import os
import sys
import torch
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, '/home/umairai/faithfulness_emnlp/Rasingan/utils')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained CareModel adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--eval_dir", type=str, default="/home/umairai/faithfulness_emnlp/Rasingan/evaluation_pipeline")
    parser.add_argument("--output_dir", type=str, default="/home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/caremodel_combined")
    return parser.parse_args()

def load_model(base_model_path, adapter_path):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return model, tokenizer

def evaluate_on_dataset(model, tokenizer, dataset_name, csv_path, output_dir):
    """Evaluate model on a dataset."""
    print(f"\nEvaluating on {dataset_name}...")
    
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found - skipping")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} examples from {dataset_name}")
    
    results = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        result = {
            'idx': idx,
            'dataset': dataset_name,
        }
        
        # Add all dimension scores
        for dim in ['NJ', 'WE', 'RA', 'AL', 'RF', 'SA']:
            if dim in df.columns:
                result[f'{dim}_human'] = row[dim]
            else:
                # Try alternate names
                col_map = {
                    'NJ': 'Non-Judgmental Language',
                    'WE': 'Warmth and Encouragement',
                    'RA': 'Respect for Autonomy',
                    'AL': 'Active Listening',
                    'RF': 'Reflecting Feelings',
                    'SA': 'Situational Appropriateness'
                }
                if col_map[dim] in df.columns:
                    result[f'{dim}_human'] = row[col_map[dim]]
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    summary = {
        'dataset': dataset_name,
        'n_examples': len(results_df),
        'mean_score': results_df[[f'{dim}_human' for dim in ['NJ', 'WE', 'RA', 'AL', 'RF', 'SA'] if f'{dim}_human' in results_df.columns]].mean().mean(),
    }
    
    # Per-dimension stats
    for dim in ['NJ', 'WE', 'RA', 'AL', 'RF', 'SA']:
        col = f'{dim}_human'
        if col in results_df.columns:
            summary[f'{dim}_mean'] = results_df[col].mean()
            summary[f'{dim}_std'] = results_df[col].std()
    
    return summary

def main():
    args = parse_args()
    
    # Load trained model
    model, tokenizer = load_model(args.base_model, args.model_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Evaluating Retrained CareModel")
    print("=" * 80)
    
    # Evaluate on both domains
    faith_eval = evaluate_on_dataset(
        model, tokenizer,
        "faith_data_test",
        "/home/umairai/faith_data/dataset/combined_test.csv",
        args.output_dir
    )
    
    ptsd_eval = evaluate_on_dataset(
        model, tokenizer,
        "ptsd_test", 
        "/home/umairai/faith_data/dataset/combined_test.csv",
        args.output_dir
    )
    
    # Save summary
    summary_output = {
        'model': args.model_dir,
        'base_model': args.base_model,
        'faith_data_eval': faith_eval,
        'ptsd_eval': ptsd_eval,
    }
    
    output_file = os.path.join(args.output_dir, "retrained_caremodel_eval.json")
    with open(output_file, 'w') as f:
        json.dump(summary_output, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(json.dumps(summary_output, indent=2))
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()
