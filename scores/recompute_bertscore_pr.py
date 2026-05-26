"""Recompute BERTScore P/R/F1 over all models in evaluation_pipeline/.

For each model directory under EVAL_ROOT that has a `responses/` folder, this
script reads the saved (gold, prediction) pairs, recomputes BERTScore, and
patches `nlp_metrics.json` with `bertscore_precision`, `bertscore_recall`, and
`bertscore_f1` (the latter is overwritten with the freshly-computed value).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd

EVAL_ROOT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/evaluation_pipeline")


def load_pairs(model_dir: Path) -> tuple[list[str], list[str]]:
    preds, refs = [], []
    for f in glob.glob(str(model_dir / "responses" / "*.csv")):
        df = pd.read_csv(f)
        if "model_prediction" not in df.columns:
            continue
        sub = df[df["Type"] == "T"].dropna(subset=["model_prediction", "Utterance"])
        preds.extend(sub["model_prediction"].astype(str).tolist())
        refs.extend(sub["Utterance"].astype(str).tolist())
    return preds, refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(EVAL_ROOT))
    ap.add_argument("--model", default=None, help="Only this model dir (basename)")
    ap.add_argument("--model-type", default="distilbert-base-uncased",
                    help="Match score_nlp.py default for comparability")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    bertscore = evaluate.load("bertscore")

    targets = (
        [eval_root / args.model] if args.model
        else [p for p in eval_root.iterdir()
              if p.is_dir() and (p / "responses").exists()]
    )

    for model_dir in sorted(targets):
        name = model_dir.name
        preds, refs = load_pairs(model_dir)
        if not preds:
            print(f"[{name}] no responses, skipping")
            continue
        print(f"[{name}] {len(preds)} pairs -> computing BERTScore ...")
        res = bertscore.compute(
            predictions=preds, references=refs,
            lang="en", model_type=args.model_type, device="cuda",
        )
        p = float(np.mean(res["precision"]))
        r = float(np.mean(res["recall"]))
        f1 = float(np.mean(res["f1"]))

        nlp_path = model_dir / "nlp_metrics.json"
        if nlp_path.exists():
            with open(nlp_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        metrics["bertscore_precision"] = round(p, 4)
        metrics["bertscore_recall"] = round(r, 4)
        metrics["bertscore_f1"] = round(f1, 4)
        with open(nlp_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"[{name}] P={p:.4f} R={r:.4f} F1={f1:.4f} -> {nlp_path}")


if __name__ == "__main__":
    main()
