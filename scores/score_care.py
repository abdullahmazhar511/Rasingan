import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import sys
import argparse

# Load Rasingan path for CareModel
RASINGAN_PATH = "/home/umairai/faithfulness_emnlp/Rasingan"
sys.path.append(RASINGAN_PATH)
from inference import CareModel, CARE_LABELS

EVAL_ROOT = os.environ.get("EVAL_ROOT", "/home/umairai/faith_data/evaluation_pipeline")
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")

def score_model_folder(model_cfg, care_model, eval_root):
    model_name = model_cfg['name']
    input_dir = os.path.join(eval_root, model_name, "responses")
    output_dir = os.path.join(eval_root, model_name, "care_scores")
    
    if not os.path.exists(input_dir):
        print(f"Skipping {model_name}: No 'responses' folder found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        return
        
    print(f"\n>>>> Scoring CARE for {model_name}")
    for file_path in tqdm(csv_files, desc="CSVs"):
        out_file = os.path.join(output_dir, os.path.basename(file_path))
        if os.path.exists(out_file): continue # Skip if already done
        
        df = pd.read_csv(file_path)
        if 'model_prediction' not in df.columns or 'context' not in df.columns:
            continue

        df['model_prediction'] = df['model_prediction'].fillna("")
        mask = df['model_prediction'] != ""
        
        contexts = df.loc[mask, 'context'].tolist()
        utterances = df.loc[mask, 'model_prediction'].tolist()
        
        if not utterances:
            df.to_csv(out_file, index=False)
            continue
            
        results = care_model.batch_predict(contexts, utterances, batch_size=8, include_analysis=True)
        
        # Output directly matching the standard coherence column names for ease of use
        mapping = {
            'Non-Judgmental Language': 'NJ',
            'Warmth and Encouragement': 'WE',
            'Respect for Autonomy': 'RA',
            'Active Listening': 'AL',
            'Reflecting Feelings': 'RF',
            'Situational Appropriateness': 'SA'
        }
        
        for label in CARE_LABELS:
            mapped_col = mapping[label]
            df[mapped_col] = None
            df.loc[mask, mapped_col] = [res[label] for res in results]
            
        df.to_csv(out_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CARE scores for models")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT, help="Evaluation root directory")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to score (optional)")
    args = parser.parse_args()
    
    eval_root = args.eval_root
    models_config = os.path.join(eval_root, "models_config.json")
    
    # We must run this from the Rasingan dir so relative paths inside CareModel work
    os.chdir(RASINGAN_PATH)
    print("Loading CARE Model (Slow)...")
    care_model = CareModel()
    
    # If specific model name provided, score only that model
    if args.model_name:
        model_cfg = {'name': args.model_name}
        try:
            score_model_folder(model_cfg, care_model, eval_root)
        except Exception as e:
            print(f"Error CARE scoring for {args.model_name}: {e}")
    # Otherwise, try to load from config if it exists
    elif os.path.exists(models_config):
        with open(models_config, "r") as f:
            models = json.load(f)
        for m in models:
            try:
                score_model_folder(m, care_model, eval_root)
            except Exception as e:
                print(f"Error CARE scoring for {m['name']}: {e}")
    else:
        print(f"Warning: No models_config.json found at {models_config}")
        print("Please provide --model-name argument or ensure models_config.json exists")
