import os
import glob
import json
import pandas as pd
import numpy as np
import warnings
import argparse
from scipy.stats import pearsonr

warnings.filterwarnings('ignore', 'An input array is constant; the correlation coefficient is not defined.')

EVAL_ROOT = os.environ.get("EVAL_ROOT", "/home/umairai/faith_data/evaluation_pipeline")
MODELS_CONFIG = os.path.join(EVAL_ROOT, "models_config.json")

def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0: return 0.0
    if len(x) < 2: return 0.0
    r, _ = pearsonr(x, y)
    if np.isnan(r): return 0.0
    return r

def interpret_coherence(upr, ref):
    if upr >= 3.5 and ref >= 3.5: return "Coherent across both dimensions"
    elif upr < 2.5 and ref < 2.5: return "Incoherent clinical reasoning"
    elif upr >= 2.5 and ref < 2.5: return "Coherent support, incoherent engagement"
    elif upr < 2.5 and ref >= 2.5: return "Incoherent support, coherent engagement"
    else:
        if upr >= 3.5: return "Coherent support, incoherent engagement"
        elif ref >= 3.5: return "Incoherent support, coherent engagement"
        else: return "Mildly coherent overall (scores between 2.5 and 3.5)"

def evaluate_coherence(model_cfg, eval_root):
    model_name = model_cfg['name']
    input_dir = os.path.join(eval_root, model_name, "care_scores")
    
    if not os.path.exists(input_dir): return
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not files: return
    
    all_turns = []
    required = ['NJ', 'WE', 'RA', 'AL', 'RF', 'SA']
    
    for file_path in files:
        conv_id_str = os.path.basename(file_path).replace('.csv', '')
        try:
            df = pd.read_csv(file_path)
            if df.empty or not all(c in df.columns for c in required): continue
                
            for c in required: df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=required)
            
            if df.empty: continue
            df['conversation_id'] = conv_id_str
            df['turn_id'] = range(1, len(df) + 1)
            all_turns.append(df[['conversation_id', 'turn_id'] + required])
        except Exception:
            continue

    if not all_turns: return

    combined_df = pd.concat(all_turns, ignore_index=True)
    combined_df['UPR_raw'] = (combined_df['NJ'] + combined_df['WE'] + combined_df['RA']) / 3.0
    combined_df['REF_raw'] = (combined_df['AL'] + combined_df['RF'] + combined_df['SA']) / 3.0

    results = []
    for conv_id, group in combined_df.groupby('conversation_id'):
        group = group.sort_values('turn_id')
        n = len(group)
        nj, we, ra = group['NJ'].values, group['WE'].values, group['RA'].values
        al, rf, sa = group['AL'].values, group['RF'].values, group['SA'].values
        
        rho_NJ_WE, rho_NJ_RA, rho_WE_RA = safe_pearson(nj, we), safe_pearson(nj, ra), safe_pearson(we, ra)
        rho_AL_RF, rho_AL_SA, rho_RF_SA = safe_pearson(al, rf), safe_pearson(al, sa), safe_pearson(rf, sa)
        
        UPR_C = (rho_NJ_WE + rho_NJ_RA + rho_WE_RA) / 3.0
        REF_C = (rho_AL_RF + rho_AL_SA + rho_RF_SA) / 3.0
        
        UPR_final = 5.0 * (UPR_C + 1.0) / 2.0
        REF_final = 5.0 * (REF_C + 1.0) / 2.0
        
        results.append({
            'model': model_name,
            'conversation_id': conv_id,
            'num_turns': n,
            'UPR_final': round(UPR_final, 4),
            'REF_final': round(REF_final, 4),
            'interpretation': interpret_coherence(UPR_final, REF_final)
        })
        
    res_df = pd.DataFrame(results)
    mean_upr = res_df['UPR_final'].mean()
    mean_ref = res_df['REF_final'].mean()
    print(f"[{model_name}] Mean UPR: {mean_upr:.4f} | Mean REF: {mean_ref:.4f}")
    
    # Save inside model folder
    out_file = os.path.join(eval_root, model_name, "coherence_score.csv")
    res_df.to_csv(out_file, index=False)
    
    # Save mean metrics to JSON
    metrics_file = os.path.join(eval_root, model_name, "coherence_metrics.json")
    metrics = {
        'mean_upr': round(mean_upr, 4),
        'mean_ref': round(mean_ref, 4)
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[{model_name}] Saved mean metrics to {metrics_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute coherence scores for models")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT, help="Evaluation root directory")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to score (optional)")
    args = parser.parse_args()
    
    eval_root = args.eval_root
    models_config = os.path.join(eval_root, "models_config.json")
    
    print("\n>>>> Computing TFS Coherence")
    
    # If specific model name provided, score only that model
    if args.model_name:
        model_cfg = {'name': args.model_name}
        evaluate_coherence(model_cfg, eval_root)
    # Otherwise, try to load from config if it exists
    elif os.path.exists(models_config):
        with open(models_config, "r") as f:
            models = json.load(f)
        for m in models:
            evaluate_coherence(m, eval_root)
    else:
        print(f"Warning: No models_config.json found at {models_config}")
        print("Please provide --model-name argument or ensure models_config.json exists")
