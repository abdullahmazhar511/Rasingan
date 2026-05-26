"""Convert multiturn_reddit_data/splits/{train,val,test}.csv into the parquet
schema consumed by the `therapist_multiturn_agent` loop:

  data_source       — provenance string
  prompt            — initial chat messages (system + opening user)
  ability           — "dialogue"
  patient_context   — the Reddit post body (what the patient role-plays from)
  patient_profile   — small dict with category + placeholder demographics
  reward_model      — ground_truth dict (zeros — actual reward comes from the
                      CARE server per turn during rollout)
  extra_info        — split / index / post_id / category

Run:
  python verl/examples/faith/data_preprocess/preprocess_multiturn.py
"""

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]    # .../Rasingan
DEFAULT_SPLITS = REPO_ROOT / "multiturn_reddit_data" / "splits"
DEFAULT_OUT = REPO_ROOT / "verl" / "examples" / "faith" / "data_multiturn"

SYSTEM_PROMPT = (
    "You are a compassionate, client-centered therapist conducting a therapy session.\n"
    "Use person-centered therapy: empathy, unconditional positive regard, and genuineness.\n"
    "Reflect emotions with phrases like \"It sounds like...\" or \"You're feeling...\".\n"
    "Ask open-ended questions, validate feelings, and avoid giving advice or diagnoses.\n"
    "Build rapport and support the client's autonomy."
)

OPENING_USER = (
    "Hello. Thank you for seeing me. I have been struggling with some things "
    "lately and I would like to talk about what is going on."
)

# CARE dimensions the reward model expects — defaults are zero (rule-based reward
# is computed live by the CARE server during training rollouts).
CARE_COLUMNS = (
    "Non-Judgmental Language",
    "Warmth and Encouragement",
    "Respect for Autonomy",
    "Active Listening",
    "Reflecting Feelings",
    "Situational Appropriateness",
)


def synth_profile(row: pd.Series) -> dict:
    """Build a minimal patient profile from the row. The simulator uses
    `patient_context` (the post body) as the actual roleplay source; the
    profile is only for prompt scaffolding."""
    return {
        "name": "Patient",
        "age": "adult",
        "occupation": "not specified",
        "family_structure": "not specified",
        "primary_concern": str(row.get("category", "mental health")),
    }


def build_row(idx: int, split: str, row: pd.Series) -> dict:
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    patient_context = f"{title}\n\n{body}".strip() if title else body

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": OPENING_USER},
    ]

    return {
        "data_source": "multiturn_reddit",
        "prompt": prompt,
        "ability": "dialogue",
        "patient_context": patient_context,
        "patient_profile": synth_profile(row),
        "reward_model": {
            "style": "rule",
            "ground_truth": {col: 0 for col in CARE_COLUMNS},
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "id": str(row.get("post_id", idx)),
            "category": str(row.get("category", "")),
        },
    }


def convert_split(split_csv: Path, split_name: str, out_path: Path, limit: int | None = None) -> int:
    df = pd.read_csv(split_csv)
    if df.empty:
        raise ValueError(f"Empty CSV: {split_csv}")
    if limit and limit > 0 and len(df) > limit and "category" in df.columns:
        # Per-category stratified subsample so anxiety/depression stay balanced.
        cats = sorted(df["category"].unique())
        per_cat = max(1, limit // max(len(cats), 1))
        df = (
            df.groupby("category", group_keys=False)
              .apply(lambda g: g.head(per_cat))
              .reset_index(drop=True)
              .head(limit)
        )
    elif limit and limit > 0 and len(df) > limit:
        df = df.head(limit).reset_index(drop=True)
    rows = [build_row(i, split_name, row) for i, row in df.iterrows()]
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    return len(out_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS),
                        help=f"Directory with train.csv / val.csv / test.csv (default: {DEFAULT_SPLITS})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"Where to write {{train,val,test}}.parquet (default: {DEFAULT_OUT})")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Cap train conversations (stratified by category). Default: no cap.")
    parser.add_argument("--val-limit", type=int, default=None,
                        help="Cap val conversations (stratified by category). Default: no cap.")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Cap test conversations (stratified by category). Default: no cap.")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    for split in ("train", "val", "test"):
        src = splits_dir / f"{split}.csv"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        dst = out_dir / f"{split}.parquet"
        n = convert_split(src, split, dst, limit=limits[split])
        print(f"[write] {dst}  ({n} rows)")

    print(f"\nDone. Use these in run_multiturn.sh:")
    print(f"  data.train_files={out_dir / 'train.parquet'}")
    print(f"  data.val_files={out_dir / 'val.parquet'}")


if __name__ == "__main__":
    main()
