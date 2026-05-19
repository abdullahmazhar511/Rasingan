"""
Rank models by how close their CARE-predicted trait profile is to the GOLD
human-annotated trait scores on the same test rows.

For every <model>.csv in sft_CARE_output/:
    For each row i:
        gold[i]  = human-annotated trait scores from test.csv (6-vector in {-2..2})
        pred[i]  = CARE classifier scores on the model's generated utterance
        per_row_L2[i] = mean over 6 traits of (gold[i,t] - pred[i,t])**2
        per_row_L1[i] = mean over 6 traits of |gold[i,t] - pred[i,t]|
    model_L2 = mean(per_row_L2)
    model_L1 = mean(per_row_L1)

Lower L2 / L1 = predicted trait profile closer to the human gold profile.
GOLD_REFERENCE (CARE run on the gold therapist text itself) is the natural
"error floor" — any model below it has saturated this evaluation.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

CARE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAITS = ["NJ", "WE", "RA", "AL", "RF", "SA"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_dir", default=os.path.join(CARE_DIR, "sft_CARE_output"))
    p.add_argument("--test_csv",
                   default=os.path.normpath(os.path.join(CARE_DIR, "..", "Rasingan/sft_training/respair_mhcopilot_format/test.csv")))
    p.add_argument("--output_csv", default=os.path.join(CARE_DIR, "sft_CARE_output/_comparison.csv"))
    return p.parse_args()


def load_gold(test_csv):
    df = pd.read_csv(test_csv)
    df = df[df["Type"] == "T"].copy()
    df["ID"] = df["ID"].astype(str)
    for t in TRAITS:
        if t not in df.columns:
            raise ValueError(f"test.csv is missing trait column '{t}'")
        df[t] = df[t].fillna(0).astype(int)
    return df.set_index("ID")[TRAITS]


def evaluate_one(model_csv, gold_df):
    df = pd.read_csv(model_csv)
    df["ID"] = df["ID"].astype(str)
    merged = df.merge(gold_df, left_on="ID", right_index=True, suffixes=("_pred", "_gold"), how="inner")
    n = len(merged)
    if n == 0:
        raise ValueError(f"No ID overlap between {model_csv} and gold test.csv")
    per_trait_L2, per_trait_L1, diffs_all = {}, {}, []
    for t in TRAITS:
        diff = merged[f"{t}_gold"].astype(float) - merged[f"{t}_pred"].astype(float)
        per_trait_L2[t] = float((diff ** 2).mean())
        per_trait_L1[t] = float(diff.abs().mean())
        diffs_all.append(diff.values)
    diffs_all = np.stack(diffs_all, axis=1)
    return {
        "n_rows": n,
        "per_trait_L2": per_trait_L2,
        "per_trait_L1": per_trait_L1,
        "L2": float((diffs_all ** 2).mean()),
        "L1": float(np.abs(diffs_all).mean()),
    }


def main():
    args = parse_args()
    print(f"Loading gold scores: {args.test_csv}")
    gold_df = load_gold(args.test_csv)
    print(f"  {len(gold_df)} therapist rows with gold scores\n")

    csv_paths = sorted(glob.glob(os.path.join(args.scores_dir, "*.csv")))
    csv_paths = [p for p in csv_paths if not os.path.basename(p).startswith("_")]
    if not csv_paths:
        print(f"No CSVs found in {args.scores_dir}")
        return

    results = []
    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Evaluating {name}")
        r = evaluate_one(path, gold_df)
        print(f"  rows matched: {r['n_rows']}   overall L2={r['L2']:.4f}   overall L1={r['L1']:.4f}")
        results.append((name, r))
    print()

    table_rows = []
    for name, r in results:
        row = {"model": name, "n_rows": r["n_rows"], "L2": r["L2"], "L1": r["L1"]}
        for t in TRAITS: row[f"L2_{t}"] = r["per_trait_L2"][t]
        for t in TRAITS: row[f"L1_{t}"] = r["per_trait_L1"][t]
        table_rows.append(row)
    table = pd.DataFrame(table_rows).sort_values("L2", ascending=True).reset_index(drop=True)
    table.to_csv(args.output_csv, index=False)

    print("=" * 102)
    print("LEADERBOARD — ranked by overall L2 (lower = closer to gold human profile)")
    print("=" * 102)
    print(f"{'rank':>4}  {'model':<32}  {'L2':>8}  {'L1':>8}   per-trait L2")
    print("-" * 102)
    for i, row in table.iterrows():
        per_trait = "  ".join(f"{t}={row[f'L2_{t}']:.3f}" for t in TRAITS)
        print(f"{i+1:>4}  {row['model']:<32}  {row['L2']:>8.4f}  {row['L1']:>8.4f}   {per_trait}")
    print(f"\nSaved comparison CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
