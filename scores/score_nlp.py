import os
import pandas as pd
import glob
import json
import evaluate
import numpy as np
from tqdm import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_PATH = os.path.dirname(SCRIPT_DIR)
EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_PATH, "evaluation_pipeline"))
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")

def calculate_metrics(model_name, eval_root):
    response_dir = os.path.join(eval_root, model_name, "responses")
    csv_files = glob.glob(os.path.join(response_dir, "*.csv"))
    
    if not csv_files:
        print(f"[{model_name}] No response files found in {response_dir}")
        return
    
    all_preds = []
    all_refs = []
    
    for f in csv_files:
        df = pd.read_csv(f)
        if 'model_prediction' in df.columns and 'Utterance' in df.columns:
            all_preds.extend(df['model_prediction'].fillna("").astype(str).tolist())
            all_refs.extend(df['Utterance'].fillna("").astype(str).tolist())
            
    if not all_preds:
        print(f"[{model_name}] No valid predictions found.")
        return

    print(f"[{model_name}] Calculating NLP Metrics for {len(all_preds)} samples...")
    
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")
    
    rouge_results = rouge.compute(predictions=all_preds, references=all_refs)
    meteor_results = meteor.compute(predictions=all_preds, references=all_refs)
    bleu_results = bleu.compute(predictions=all_preds, references=[[ref] for ref in all_refs])
    # Using a light model for BERTScore to keep it fast
    bert_results = bertscore.compute(predictions=all_preds, references=all_refs, lang="en", model_type="distilbert-base-uncased", device="cuda")
    
    metrics = {
        "rouge1": round(rouge_results['rouge1'], 4),
        "rougeL": round(rouge_results['rougeL'], 4),
        "meteor": round(meteor_results['meteor'], 4),
        "bleu": round(bleu_results['bleu'], 4),
        "bertscore_precision": round(np.mean(bert_results['precision']), 4),
        "bertscore_recall": round(np.mean(bert_results['recall']), 4),
        "bertscore_f1": round(np.mean(bert_results['f1']), 4)
    }
    
    # Save results to the model's eval folder
    output_path = os.path.join(eval_root, model_name, "nlp_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[{model_name}] Metrics: {metrics}")
    print(f"[{model_name}] Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute NLP metrics for models")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT, help="Evaluation root directory")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to score (optional)")
    args = parser.parse_args()
    
    eval_root = args.eval_root
    models_config = os.path.join(eval_root, "models_config.json")
    
    # If specific model name provided, score only that model
    if args.model_name:
        try:
            calculate_metrics(args.model_name, eval_root)
        except Exception as e:
            print(f"Error computing metrics for {args.model_name}: {e}")
            import traceback
            traceback.print_exc()
    # Otherwise, try to load from config if it exists
    elif os.path.exists(models_config):
        with open(models_config, "r") as f:
            models = json.load(f)
        for m in models:
            model_name = m['name'] if isinstance(m, dict) else m
            try:
                calculate_metrics(model_name, eval_root)
            except Exception as e:
                print(f"Error computing metrics for {model_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"Warning: No models_config.json found at {models_config}")
        print("Please provide --model-name argument or ensure models_config.json exists")
