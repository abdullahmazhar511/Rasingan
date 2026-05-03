"""
Conversation-level CTRS (Clinical Therapeutic Response System) scoring.

For each conversation, computes CTRS-P score from CARE dimensions:
  - Understanding (U) = avg(AL, RF)
  - Interpersonal Effectiveness (IE) = avg(WE, NJ)
  - Collaboration (C) = RA
  - Technical Appropriateness (TA) = SA

Per construct score: S_k = Q_k - F_k + 0.5 * SP_k
Where:
  - Q_k: mean quality (mean of construct values)
  - F_k: weak/neutral rate (% of values <= 0)
  - SP_k: strong positive rate (% of values == 2)

Final CTRS-P = 0.35*U + 0.25*IE + 0.20*C + 0.20*TA

Aggregates per-conversation scores for single model summary.
"""

import argparse
import os
import glob
import json
import numpy as np
import pandas as pd


# CARE dimension abbrev
PRED_ALL = ["NJ", "WE", "RA", "AL", "RF", "SA"]


def compute_construct_score(values):
    """
    Compute Score_k = Q_k - F_k + 0.5 * SP_k
    
    Args:
        values: array of construct values across turns
    
    Returns:
        dict with Q, F, SP, and score
    """
    values = np.array(values, dtype=float)
    
    # Mean quality
    Q = np.mean(values)
    
    # Weak / neutral rate (<= 0)
    F = np.mean(values <= 0)
    
    # Strong positive rate (= 2)
    SP = np.mean(values == 2)
    
    score = Q - F + 0.5 * SP
    
    return {
        "Q": float(Q),
        "F": float(F),
        "SP": float(SP),
        "score": float(score)
    }


def compute_ctrs_from_df(df):
    """
    Compute CTRS score from a single conversation's CARE scores.
    
    Args:
        df: DataFrame with columns [NJ, WE, RA, AL, RF, SA]
    
    Returns:
        dict with construct scores and final CTRS_P, or None if invalid
    """
    df = df.copy()
    
    # Ensure we have all required dimensions
    dims = ["NJ", "WE", "RA", "AL", "RF", "SA"]
    available_dims = [d for d in dims if d in df.columns]
    if len(available_dims) < 6:
        return None
    
    # Convert to numeric and drop NaN
    for d in dims:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    df = df.dropna(subset=dims)
    
    if len(df) == 0:
        return None
    
    # Extract dimensions
    NJ = df["NJ"].values
    WE = df["WE"].values
    RA = df["RA"].values
    AL = df["AL"].values
    RF = df["RF"].values
    SA = df["SA"].values
    
    # Compute constructs
    U = (AL + RF) / 2.0  # Understanding
    IE = (WE + NJ) / 2.0  # Interpersonal Effectiveness
    C = RA  # Collaboration
    TA = SA  # Technical Appropriateness
    
    # Compute construct scores
    U_stats = compute_construct_score(U)
    IE_stats = compute_construct_score(IE)
    C_stats = compute_construct_score(C)
    TA_stats = compute_construct_score(TA)
    
    # Weighted final CTRS-P score
    ctrs_p = (
        0.35 * U_stats["score"] +
        0.25 * IE_stats["score"] +
        0.20 * C_stats["score"] +
        0.20 * TA_stats["score"]
    )
    
    return {
        "Understanding": U_stats,
        "Interpersonal_Effectiveness": IE_stats,
        "Collaboration": C_stats,
        "Technical_Appropriateness": TA_stats,
        "CTRS_P": float(ctrs_p),
        "n_turns": len(df)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute conversation-level CTRS scores from CARE dimensions"
    )
    parser.add_argument("--eval-root", required=True,
                        help="Root directory containing model evaluation folders")
    parser.add_argument("--model-name", default=None,
                        help="Specific model name (if not provided, processes all models)")
    args = parser.parse_args()
    
    if args.model_name:
        model_dirs = [os.path.join(args.eval_root, args.model_name)]
    else:
        model_dirs = sorted(glob.glob(os.path.join(args.eval_root, "*")))
        model_dirs = [d for d in model_dirs if os.path.isdir(d)]
    
    for model_dir in model_dirs:
        care_scores_dir = os.path.join(model_dir, "care_scores")
        if not os.path.isdir(care_scores_dir):
            continue
        
        model_name = os.path.basename(model_dir)
        csv_files = sorted(glob.glob(os.path.join(care_scores_dir, "*.csv")))
        
        if not csv_files:
            print(f"[{model_name}] No care_scores CSVs found, skipping.")
            continue
        
        # Process each conversation
        conv_results = {}
        for csv_path in csv_files:
            conv_id = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            result = compute_ctrs_from_df(df)
            if result is not None:
                conv_results[conv_id] = result
        
        if not conv_results:
            print(f"[{model_name}] No valid conversations for CTRS scoring, skipping.")
            continue
        
        # Aggregate across conversations
        agg = {}
        
        # Aggregate construct scores
        for construct in ["Understanding", "Interpersonal_Effectiveness", "Collaboration", "Technical_Appropriateness"]:
            for metric in ["Q", "F", "SP", "score"]:
                values = [r[construct][metric] for r in conv_results.values()]
                agg[f"avg_{construct}_{metric}"] = float(np.mean(values))
                agg[f"std_{construct}_{metric}"] = float(np.std(values))
        
        # Aggregate CTRS_P
        ctrs_p_values = [r["CTRS_P"] for r in conv_results.values()]
        agg["avg_CTRS_P"] = float(np.mean(ctrs_p_values))
        agg["std_CTRS_P"] = float(np.std(ctrs_p_values))
        
        # Aggregate n_turns
        n_turns_values = [r["n_turns"] for r in conv_results.values()]
        agg["avg_n_turns"] = float(np.mean(n_turns_values))
        
        output = {
            "n_conversations": len(conv_results),
            "aggregate": agg,
            "per_conversation": conv_results
        }
        
        out_path = os.path.join(model_dir, "ctrs_scores.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        print(f"Conversations scored: {len(conv_results)}")
        print(f"Mean CTRS-P: {agg['avg_CTRS_P']:.4f}")
        print(f"  Understanding: {agg['avg_Understanding_score']:.4f}")
        print(f"  Interpersonal Effectiveness: {agg['avg_Interpersonal_Effectiveness_score']:.4f}")
        print(f"  Collaboration: {agg['avg_Collaboration_score']:.4f}")
        print(f"  Technical Appropriateness: {agg['avg_Technical_Appropriateness_score']:.4f}")
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()