"""
Conversation-level CARE quality metrics (no ground truth needed).

For each conversation, computes:
  - Mean UPR (avg of NJ, WE, RA) and REF (avg of AL, RF, SA) across turns
  - Overall CARE score (mean of all 6 dimensions across turns)
  - Consistency: std-dev across turns (lower = more stable therapeutic quality)
  - Min-turn score: worst turn in the conversation (floor of quality)
  - Trajectory slope: whether quality improves or degrades over the conversation

Aggregates across all conversations for a single summary per model.
"""

import argparse
import os
import glob
import json
import numpy as np
import pandas as pd

PRED_UPR = ["NJ", "WE", "RA"]
PRED_REF = ["AL", "RF", "SA"]
PRED_ALL = PRED_UPR + PRED_REF

GT_UPR = ["Non-Judgmental Language", "Warmth and Encouragement", "Respect for Autonomy"]
GT_REF = ["Active Listening", "Reflecting Feelings", "Situational Appropriateness"]
GT_ALL = GT_UPR + GT_REF


def score_conversation(df, use_gt=False):
    """Compute conversation-level metrics from a single care_scores CSV."""
    all_dims = GT_ALL if use_gt else PRED_ALL
    upr_dims = GT_UPR if use_gt else PRED_UPR
    ref_dims = GT_REF if use_gt else PRED_REF

    pred_cols = [c for c in all_dims if c in df.columns]
    if not pred_cols:
        return None

    df = df.copy()
    for c in pred_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=pred_cols)
    if len(df) == 0:
        return None

    vals = df[pred_cols].values  # (n_turns, 6)
    n_turns = len(vals)

    # Per-turn CARE mean (across dims)
    turn_means = vals.mean(axis=1)  # (n_turns,)

    # UPR / REF per turn
    upr_cols = [pred_cols.index(d) for d in upr_dims if d in pred_cols]
    ref_cols = [pred_cols.index(d) for d in ref_dims if d in pred_cols]
    turn_upr = vals[:, upr_cols].mean(axis=1) if upr_cols else turn_means
    turn_ref = vals[:, ref_cols].mean(axis=1) if ref_cols else turn_means

    # Trajectory slope (linear fit of turn_means over turn index)
    if n_turns >= 2:
        x = np.arange(n_turns, dtype=float)
        slope = np.polyfit(x, turn_means, 1)[0]
    else:
        slope = 0.0

    result = {
        "n_turns": n_turns,
        # Means
        "mean_care": float(turn_means.mean()),
        "mean_upr": float(turn_upr.mean()),
        "mean_ref": float(turn_ref.mean()),
        # Per-dimension means
        **{f"mean_{d}": float(df[d].mean()) for d in pred_cols},
        # Consistency (lower = more stable)
        "std_care": float(turn_means.std()),
        "std_upr": float(turn_upr.std()),
        "std_ref": float(turn_ref.std()),
        # Floor quality
        "min_turn_care": float(turn_means.min()),
        "min_turn_upr": float(turn_upr.min()),
        "min_turn_ref": float(turn_ref.min()),
        # Trajectory (positive = improving)
        "trajectory_slope": float(slope),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--ground-truth", action="store_true",
                        help="Use ground truth columns instead of model predictions")
    args = parser.parse_args()
    use_gt = args.ground_truth

    if args.model_name:
        model_dirs = [os.path.join(args.eval_root, args.model_name)]
    else:
        model_dirs = sorted(glob.glob(os.path.join(args.eval_root, "*")))
        model_dirs = [d for d in model_dirs if os.path.isdir(d)]

    for model_dir in model_dirs:
        care_dir = os.path.join(model_dir, "care_scores")
        if not os.path.isdir(care_dir):
            continue

        model_name = os.path.basename(model_dir)
        csv_files = sorted(glob.glob(os.path.join(care_dir, "*.csv")))
        if not csv_files:
            print(f"[{model_name}] No care_scores CSVs found, skipping.")
            continue

        conv_results = {}
        for csv_path in csv_files:
            conv_id = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            result = score_conversation(df, use_gt=use_gt)
            if result is not None:
                conv_results[conv_id] = result

        if not conv_results:
            print(f"[{model_name}] No valid conversations, skipping.")
            continue

        # Aggregate across conversations
        keys = list(next(iter(conv_results.values())).keys())
        agg = {}
        for k in keys:
            values = [r[k] for r in conv_results.values()]
            agg[f"avg_{k}"] = float(np.mean(values))
            if k != "n_turns":
                agg[f"std_{k}"] = float(np.std(values))

        output = {
            "n_conversations": len(conv_results),
            "aggregate": agg,
            "per_conversation": conv_results,
        }

        out_name = "care_conv_metrics_gt.json" if use_gt else "care_conv_metrics.json"
        out_path = os.path.join(model_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        # Print summary
        label = f"{model_name} [GT]" if use_gt else model_name
        print(f"\n{'='*50}")
        print(f"  {label}  ({len(conv_results)} conversations)")
        print(f"{'='*50}")
        print(f"  Mean CARE Score : {agg['avg_mean_care']:.4f}")
        print(f"  Mean UPR        : {agg['avg_mean_upr']:.4f}")
        print(f"  Mean REF        : {agg['avg_mean_ref']:.4f}")
        print(f"  Consistency (σ) : {agg['avg_std_care']:.4f}  (lower=more stable)")
        print(f"  Min-turn CARE   : {agg['avg_min_turn_care']:.4f}  (floor quality)")
        print(f"  Trajectory      : {agg['avg_trajectory_slope']:+.4f}  (positive=improving)")
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
