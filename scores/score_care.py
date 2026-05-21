import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import sys
import argparse

# Load Rasingan path for CareModel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_PATH = os.path.dirname(SCRIPT_DIR)
sys.path.append(RASINGAN_PATH)
from inference import CareModel, CARE_LABELS

EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_PATH, "evaluation_pipeline"))
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")

# Gold CARE scores depend only on (context, Utterance) which are dataset-wide
# constants — identical across every model's responses/ CSVs. Cache them at
# this path so re-scoring N models = 1× CARE pass on gold + N× on model output.
# Delete the file to force a recompute (e.g. if the CARE classifier changed).
GOLD_CACHE_PATH = os.path.join(EVAL_ROOT, "_gold_care_cache.csv")


def _load_gold_cache():
    if os.path.exists(GOLD_CACHE_PATH):
        return pd.read_csv(GOLD_CACHE_PATH).set_index("ID")
    return pd.DataFrame(columns=["ID"] + CARE_LABELS).set_index("ID")


def _save_gold_cache(cache_df):
    os.makedirs(os.path.dirname(GOLD_CACHE_PATH), exist_ok=True)
    cache_df.reset_index().to_csv(GOLD_CACHE_PATH, index=False)


def score_model_folder(model_cfg, care_model, eval_root, gold_cache):
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
        if 'model_prediction' not in df.columns or 'context' not in df.columns \
           or 'Utterance' not in df.columns or 'ID' not in df.columns:
            continue

        df['model_prediction'] = df['model_prediction'].fillna("")
        df['Utterance'] = df['Utterance'].fillna("")
        mask = df['model_prediction'] != ""

        contexts = df.loc[mask, 'context'].tolist()
        model_utts = df.loc[mask, 'model_prediction'].tolist()
        gold_utts  = df.loc[mask, 'Utterance'].tolist()
        ids        = df.loc[mask, 'ID'].tolist()

        if not model_utts:
            df.to_csv(out_file, index=False)
            continue

        # Score the model's response.
        pred_results = care_model.batch_predict(contexts, model_utts, batch_size=8, include_analysis=True)

        # Gold scores: serve from cache when available, score only the misses.
        miss_idx = [i for i, _id in enumerate(ids) if _id not in gold_cache.index]
        if miss_idx:
            new_results = care_model.batch_predict(
                [contexts[i]  for i in miss_idx],
                [gold_utts[i] for i in miss_idx],
                batch_size=8, include_analysis=True,
            )
            new_rows = pd.DataFrame(
                [{label: r[label] for label in CARE_LABELS} for r in new_results],
                index=[ids[i] for i in miss_idx],
            )
            new_rows.index.name = "ID"
            gold_cache = pd.concat([gold_cache, new_rows])
            _save_gold_cache(gold_cache)

        # Abbreviated cols = scores on model_prediction.
        # Full-name cols   = cached gold scores (CARE classifier on gold text).
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
            df[label]      = None
            df.loc[mask, mapped_col] = [res[label] for res in pred_results]
            df.loc[mask, label]      = [gold_cache.at[_id, label] for _id in ids]

        df.to_csv(out_file, index=False)

    return gold_cache

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
    gold_cache = _load_gold_cache()
    print(f"Gold CARE cache: {len(gold_cache)} entries loaded from {GOLD_CACHE_PATH}")

    # If specific model name provided, score only that model
    if args.model_name:
        model_cfg = {'name': args.model_name}
        try:
            gold_cache = score_model_folder(model_cfg, care_model, eval_root, gold_cache)
        except Exception as e:
            print(f"Error CARE scoring for {args.model_name}: {e}")
    # Otherwise, try to load from config if it exists
    elif os.path.exists(models_config):
        with open(models_config, "r") as f:
            models = json.load(f)
        for m in models:
            try:
                gold_cache = score_model_folder(m, care_model, eval_root, gold_cache)
            except Exception as e:
                print(f"Error CARE scoring for {m['name']}: {e}")
    else:
        print(f"Warning: No models_config.json found at {models_config}")
        print("Please provide --model-name argument or ensure models_config.json exists")
