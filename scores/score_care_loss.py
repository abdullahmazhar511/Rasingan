import os
import glob
import json
import pandas as pd
import numpy as np
import argparse

EVAL_ROOT = os.environ.get("EVAL_ROOT", "/home/umairai/faithfulness_emnlp/Rasingan/evaluation_pipeline")

# Ground truth (full name) -> Model prediction (abbreviated) column mapping
GT_TO_PRED = {
    'Non-Judgmental Language': 'NJ',
    'Warmth and Encouragement': 'WE',
    'Respect for Autonomy': 'RA',
    'Active Listening': 'AL',
    'Reflecting Feelings': 'RF',
    'Situational Appropriateness': 'SA'
}

def compute_care_loss(model_name, eval_root):
    care_dir = os.path.join(eval_root, model_name, "care_scores")
    if not os.path.exists(care_dir):
        print(f"[{model_name}] No care_scores directory found.")
        return None

    csv_files = glob.glob(os.path.join(care_dir, "*.csv"))
    if not csv_files:
        print(f"[{model_name}] No care score files found.")
        return None

    all_gt = {col: [] for col in GT_TO_PRED.values()}
    all_pred = {col: [] for col in GT_TO_PRED.values()}

    for f in csv_files:
        df = pd.read_csv(f)

        # Check all required columns exist
        gt_cols = list(GT_TO_PRED.keys())
        pred_cols = list(GT_TO_PRED.values())
        if not all(c in df.columns for c in gt_cols + pred_cols):
            continue

        # Convert to numeric, drop rows with NaN in any required column
        for c in gt_cols + pred_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=gt_cols + pred_cols)

        if df.empty:
            continue

        for gt_col, pred_col in GT_TO_PRED.items():
            all_gt[pred_col].extend(df[gt_col].values.tolist())
            all_pred[pred_col].extend(df[pred_col].values.tolist())

    if not all_gt['NJ']:
        print(f"[{model_name}] No valid data found for loss computation.")
        return None

    # Compute per-dimension and aggregate losses
    results = {}
    all_l1 = []
    all_l2 = []

    for dim in GT_TO_PRED.values():
        gt = np.array(all_gt[dim])
        pred = np.array(all_pred[dim])
        diff = pred - gt

        l1 = np.mean(np.abs(diff))
        l2 = np.mean(diff ** 2)
        mae = l1
        rmse = np.sqrt(l2)

        results[dim] = {
            'l1_mae': round(float(l1), 4),
            'l2_mse': round(float(l2), 4),
            'rmse': round(float(rmse), 4),
            'mean_gt': round(float(gt.mean()), 4),
            'mean_pred': round(float(pred.mean()), 4),
            'mean_diff': round(float(diff.mean()), 4),
            'n_samples': len(gt)
        }

        all_l1.extend(np.abs(diff).tolist())
        all_l2.extend((diff ** 2).tolist())

    # Aggregate across all dimensions
    results['aggregate'] = {
        'l1_mae': round(float(np.mean(all_l1)), 4),
        'l2_mse': round(float(np.mean(all_l2)), 4),
        'rmse': round(float(np.sqrt(np.mean(all_l2))), 4),
        'total_samples': len(all_l1)
    }

    # UPR group (NJ, WE, RA)
    upr_l1 = []
    upr_l2 = []
    for dim in ['NJ', 'WE', 'RA']:
        gt = np.array(all_gt[dim])
        pred = np.array(all_pred[dim])
        diff = pred - gt
        upr_l1.extend(np.abs(diff).tolist())
        upr_l2.extend((diff ** 2).tolist())

    results['UPR_group'] = {
        'l1_mae': round(float(np.mean(upr_l1)), 4),
        'l2_mse': round(float(np.mean(upr_l2)), 4),
        'rmse': round(float(np.sqrt(np.mean(upr_l2))), 4),
    }

    # REF group (AL, RF, SA)
    ref_l1 = []
    ref_l2 = []
    for dim in ['AL', 'RF', 'SA']:
        gt = np.array(all_gt[dim])
        pred = np.array(all_pred[dim])
        diff = pred - gt
        ref_l1.extend(np.abs(diff).tolist())
        ref_l2.extend((diff ** 2).tolist())

    results['REF_group'] = {
        'l1_mae': round(float(np.mean(ref_l1)), 4),
        'l2_mse': round(float(np.mean(ref_l2)), 4),
        'rmse': round(float(np.sqrt(np.mean(ref_l2))), 4),
    }

    # Print summary
    print(f"\n[{model_name}] CARE Loss (Ground Truth vs Model Prediction)")
    print(f"{'Dimension':<12} {'L1 (MAE)':<12} {'L2 (MSE)':<12} {'RMSE':<10} {'Mean GT':<10} {'Mean Pred':<10} {'Bias':<10}")
    print("-" * 76)
    for dim in GT_TO_PRED.values():
        r = results[dim]
        print(f"{dim:<12} {r['l1_mae']:<12} {r['l2_mse']:<12} {r['rmse']:<10} {r['mean_gt']:<10} {r['mean_pred']:<10} {r['mean_diff']:<10}")
    print("-" * 76)
    agg = results['aggregate']
    print(f"{'Aggregate':<12} {agg['l1_mae']:<12} {agg['l2_mse']:<12} {agg['rmse']:<10}")
    upr = results['UPR_group']
    print(f"{'UPR Group':<12} {upr['l1_mae']:<12} {upr['l2_mse']:<12} {upr['rmse']:<10}")
    ref = results['REF_group']
    print(f"{'REF Group':<12} {ref['l1_mae']:<12} {ref['l2_mse']:<12} {ref['rmse']:<10}")

    # Save
    out_file = os.path.join(eval_root, model_name, "care_loss.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n[{model_name}] Saved to {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute L1/L2 loss between ground truth and model CARE scores")
    parser.add_argument("--eval-root", type=str, default=EVAL_ROOT, help="Evaluation root directory")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name (optional)")
    args = parser.parse_args()

    eval_root = args.eval_root

    if args.model_name:
        compute_care_loss(args.model_name, eval_root)
    else:
        # Score all models found in eval_root
        for entry in sorted(os.listdir(eval_root)):
            model_dir = os.path.join(eval_root, entry)
            if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "care_scores")):
                compute_care_loss(entry, eval_root)
