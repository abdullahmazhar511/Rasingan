"""Build train/val/test splits over the multiturn reddit-post seed data.

Inputs:  anxiety_cleaned_data.csv + depression_cleaned_data.csv (sibling files)
Output:  splits/{train,val,test}.csv with columns [post_id, title, body, category]

The split is stratified by `category` so each split has the same anxiety:depression
ratio. Deterministic via --seed (default 42).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCES = {
    "anxiety": HERE / "anxiety_cleaned_data.csv",
    "depression": HERE / "depression_cleaned_data.csv",
}


def load_data() -> pd.DataFrame:
    frames = []
    for category, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        df = pd.read_csv(path)
        missing = {"post_id", "title", "body"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        df = df[["post_id", "title", "body"]].copy()
        df["category"] = category
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # Drop empty bodies and duplicate post_ids (keep first)
    full["body"] = full["body"].fillna("").astype(str)
    full = full[full["body"].str.strip() != ""].copy()
    before = len(full)
    full = full.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
    dropped = before - len(full)
    if dropped:
        print(f"[info] dropped {dropped} duplicate post_id rows")
    return full


def stratified_split(df: pd.DataFrame, train: float, val: float, test: float,
                     seed: int) -> dict:
    if not abs(train + val + test - 1.0) < 1e-6:
        raise ValueError(f"ratios must sum to 1.0 (got {train+val+test:.4f})")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for category, group in df.groupby("category", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # Whatever is left goes to test so totals always sum to n
        n_test = n - n_train - n_val
        parts["train"].append(group.loc[idx[:n_train]])
        parts["val"].append(group.loc[idx[n_train:n_train + n_val]])
        parts["test"].append(group.loc[idx[n_train + n_val:n_train + n_val + n_test]])
        print(f"[info] {category:11s}  total={n:4d}  train={n_train}  val={n_val}  test={n_test}")
    return {k: pd.concat(v, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
            for k, v in parts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(HERE / "splits"),
                        help="Directory for {train,val,test}.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"[info] {len(df)} posts loaded across {df['category'].nunique()} categories")

    splits = stratified_split(df, args.train, args.val, args.test, args.seed)

    for name, sub in splits.items():
        path = out_dir / f"{name}.csv"
        sub.to_csv(path, index=False)
        print(f"[write] {path}  ({len(sub)} rows)  category counts: {dict(sub['category'].value_counts())}")

    # Sanity: no post_id should appear in more than one split
    all_ids = pd.concat([s["post_id"] for s in splits.values()])
    if all_ids.duplicated().any():
        print("[ERROR] post_id appears in more than one split", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] total={len(all_ids)}  unique={all_ids.nunique()}  → no leakage")


if __name__ == "__main__":
    main()
