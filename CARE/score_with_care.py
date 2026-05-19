"""
Score model prediction CSVs with the trained CARE classifier, using RAG-retrieved
explanations from the training pool (no per-prediction LLM call required).

Methodology (mirrors CARE training, minus the Qwen3-4B explainer step):
    For each prediction row:
        1. Embed the predicted utterance with all-MiniLM-L6-v2.
        2. For each of the 6 traits:
             - Retrieve top-K most similar utterances from the trait's Pos pool
               (training items with gold trait score > 0).
             - Retrieve top-K most similar utterances from the trait's Neg pool
               (training items with gold trait score < 0).
             - Collect those retrieved utterances' CACHED explanations for THIS trait
               (from rag_cache/train_processed.json — pre-generated during CARE training).
             - Concatenate as the trait's "Analysis" text.
        3. Feed (context, prediction, analysis) into the CARE classifier (Qwen3-4B + LoRA + heads).
        4. Argmax per trait -> integer score in {-2,-1,0,1,2}.

Inputs needed:
    - Predictions CSV (reference, prediction columns), produced by inference_test.py
    - rag_cache/train_processed.json (pre-generated CARE training explanations)
    - care_checkpoint/best_classifier.pt (trained CARE classifier)
    - MHCoPilot test set (for parallel context lookup; uses row order, NOT ID)

Usage:
    python score_with_care.py \\
        --predictions_csv Rasingan/sft_training/<model>_test_predictions.csv \\
        --output_csv sft_CARE_output/<model>.csv
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # CARE/
REPO_ROOT = os.path.dirname(SCRIPT_DIR)                   # EMNLP_FINAL/
sys.path.insert(0, SCRIPT_DIR)
# Importing CARE pulls in QwenHierarchicalClassifier, label maps, and the helper
# `create_ideal_sets` we use to build per-trait Pos/Neg pools.
from EMNLP_FINAL.Rasingan.CARE.CARE import (
    QwenHierarchicalClassifier,
    IDX_TO_LABEL,
    CONFIG as CARE_CONFIG,
    create_ideal_sets,
)

sys.path.insert(0, os.path.join(REPO_ROOT, "Rasingan", "utils"))
from hfDataset import MHCoPilot_Dataset


TRAIT_SHORT_NAMES = ["NJ", "WE", "RA", "AL", "RF", "SA"]  # parallel to CARE_CONFIG["labels"]


def parse_args():
    p = argparse.ArgumentParser(description="Score predictions with CARE classifier (RAG-retrieved explanations)")
    p.add_argument("--predictions_csv", required=True, help="CSV with columns: reference, prediction")
    p.add_argument("--output_csv", required=True, help="Where to save the scored CSV")
    p.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "Rasingan/sft_training/respair_mhcopilot_format"),
                   help="Folder containing test.csv (MHCoPilot format) for parallel context lookup")
    p.add_argument("--context_window", type=int, default=6,
                   help="Must match the context_window used during prediction generation")
    p.add_argument("--train_explanations_json", default=os.path.join(SCRIPT_DIR, "rag_cache/train_processed.json"),
                   help="Pre-generated training-pool explanations (built during CARE training)")
    p.add_argument("--rag_index_cache", default=os.path.join(SCRIPT_DIR, "rag_cache/rag_index.pt"),
                   help="Where to cache the built RAG index (ideal_sets + train embeddings)")
    p.add_argument("--care_checkpoint", default=os.path.join(SCRIPT_DIR, "care_checkpoint/best_classifier.pt"),
                   help="Trained CARE classifier weights")
    p.add_argument("--classifier_model_id", default=CARE_CONFIG["classifier_model_id"])
    p.add_argument("--embedding_model", default=CARE_CONFIG["embedding_model"])
    p.add_argument("--top_k", type=int, default=CARE_CONFIG["top_k"],
                   help="Top-K to retrieve from EACH of Pos and Neg pools per trait")
    p.add_argument("--batch_size", type=int, default=8, help="Classifier inference batch size")
    p.add_argument("--max_len", type=int, default=CARE_CONFIG["max_len"])
    p.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    return p.parse_args()


# ----------------------------- RAG INDEX -----------------------------

def build_or_load_rag_index(train_explanations_path, embed_model, cache_path, device):
    """Build (or load from disk) the per-trait Pos/Neg pools + training-utterance embeddings.

    Returns:
        ideal_sets: dict {trait_full_name: {'Pos': [df_idx,...], 'Neg': [df_idx,...]}}
        train_embeddings: torch.Tensor of shape (N_train, embed_dim), on `device`
        trait_explanations: dict {trait_full_name: list[N_train str]} — cached explanations
            for every training item, indexed parallel to train_embeddings.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached RAG index: {cache_path}")
        cache = torch.load(cache_path, weights_only=False, map_location="cpu")
        train_embeddings = cache["embeddings"].to(device)
        return cache["ideal_sets"], train_embeddings, cache["trait_explanations"]

    print(f"Building RAG index from {train_explanations_path}")
    with open(train_explanations_path) as f:
        train_data = json.load(f)
    print(f"  loaded {len(train_data)} training items with cached explanations")

    df = pd.DataFrame(train_data)
    # `create_ideal_sets` expects each of CARE's 6 trait columns (full names).
    # The cached JSON already has them; this assertion just gives a clear error if not.
    missing = [c for c in CARE_CONFIG["labels"] if c not in df.columns]
    if missing:
        raise ValueError(f"train_processed.json is missing trait columns: {missing}")

    # Reuse CARE's exact ideal-set construction (round-robin disjoint Pos/Neg per trait).
    ideal_sets, train_embeddings = create_ideal_sets(df, embed_model)
    train_embeddings = train_embeddings.to(device)

    # Cache the 6 trait explanations parallel to df rows. Missing -> "" (filtered later).
    trait_explanations = {}
    for trait in CARE_CONFIG["labels"]:
        trait_explanations[trait] = [
            str((item.get("Explanations") or {}).get(trait, "")).strip()
            for item in train_data
        ]

    # Disk cache: keep embeddings on CPU for portability across machines/devices.
    torch.save(
        {
            "ideal_sets": ideal_sets,
            "embeddings": train_embeddings.cpu(),
            "trait_explanations": trait_explanations,
        },
        cache_path,
    )
    print(f"  cached RAG index -> {cache_path} (next run will load instantly)")
    return ideal_sets, train_embeddings, trait_explanations


def build_analyses_via_rag(query_embeddings, ideal_sets, train_embeddings, trait_explanations, top_k):
    """Build the per-trait 'Analysis' text for every query utterance.

    Args:
        query_embeddings: (Q, D) embeddings of the new utterances to score.
        ideal_sets, train_embeddings, trait_explanations: from build_or_load_rag_index.
        top_k: how many to retrieve from EACH of Pos and Neg per trait.

    Returns:
        list of length Q, each element is dict {trait_full_name: analysis_text}.
    """
    analyses = [dict() for _ in range(query_embeddings.shape[0])]

    for trait in CARE_CONFIG["labels"]:
        for polarity in ("Pos", "Neg"):
            pool_idxs = ideal_sets[trait][polarity]
            if not pool_idxs:
                continue
            pool_embs = train_embeddings[pool_idxs]  # (P, D)
            # cos_sim: (Q, P)
            scores = util.cos_sim(query_embeddings, pool_embs)
            k = min(top_k, len(pool_idxs))
            _, topk_inds = torch.topk(scores, k=k, dim=1)  # (Q, k)
            topk_inds = topk_inds.cpu().numpy()
            for q in range(query_embeddings.shape[0]):
                retrieved = []
                for j in topk_inds[q]:
                    pool_row_idx = pool_idxs[int(j)]
                    expl = trait_explanations[trait][pool_row_idx]
                    if expl:
                        retrieved.append(expl)
                # Append to the trait's analysis (Pos first, then Neg)
                prev = analyses[q].get(trait, "")
                joined = "\n".join(retrieved)
                analyses[q][trait] = (prev + ("\n" if prev else "") + joined) if joined else prev

    # Fill any empty trait analyses with the standard CARE fallback.
    for q in range(query_embeddings.shape[0]):
        for trait in CARE_CONFIG["labels"]:
            if not analyses[q].get(trait):
                analyses[q][trait] = "No info."
    return analyses


# --------------------------- CLASSIFIER PATH ---------------------------

class PredictionScoringDataset(Dataset):
    """Builds CARE's text_input format from (context, prediction, retrieved explanations)."""
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        analysis = ""
        for lbl in CARE_CONFIG["labels"]:
            expl = str(r["explanations"].get(lbl, "")).strip() or "No info."
            analysis += f"{lbl}: {expl}\n"
        text_input = (
            f"Context:\n{r['context']}\n"
            f"Therapist: \"{r['utterance']}\"\n"
            f"Analysis:\n{analysis}\n"
            "Classify the clinical traits."
        )
        enc = self.tokenizer(
            text_input,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# -------------------------------- MAIN --------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Predictions
    print(f"Loading predictions: {args.predictions_csv}")
    pred_df = pd.read_csv(args.predictions_csv)
    if "prediction" not in pred_df.columns:
        raise ValueError(f"--predictions_csv must have a 'prediction' column; got {list(pred_df.columns)}")
    n_pred = len(pred_df)
    print(f"  {n_pred} prediction rows")

    # 2) Parallel test data for contexts (row-order aligned with predictions)
    print(f"Loading test data: {args.data_dir}  (context_window={args.context_window})")
    mh = MHCoPilot_Dataset(args.data_dir, context_window=args.context_window)
    mh.get_data()
    test_ds = mh.test_dataset
    n_test = len(test_ds)
    print(f"  {n_test} test rows")
    if n_test != n_pred:
        raise ValueError(
            f"row-count mismatch: {n_test} test vs {n_pred} predictions — "
            f"context_window must match the value used during prediction generation"
        )
    test_ids = [str(test_ds[i]["ID"]) for i in range(n_test)]
    contexts = [str(test_ds[i]["context"]) if test_ds[i]["context"] is not None else "" for i in range(n_test)]
    predictions = [str(pred_df.iloc[i]["prediction"]) if pd.notna(pred_df.iloc[i]["prediction"]) else "" for i in range(n_pred)]

    # 3) RAG index (built from train_processed.json, cached on disk)
    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model, device=device)
    ideal_sets, train_embeddings, trait_explanations = build_or_load_rag_index(
        args.train_explanations_json, embed_model, args.rag_index_cache, device
    )

    # 4) Embed all predictions in one batch and retrieve per-trait Pos/Neg explanations.
    print(f"Embedding {n_pred} predictions and retrieving top-{args.top_k} Pos + top-{args.top_k} Neg per trait...")
    query_embs = embed_model.encode(
        predictions, convert_to_tensor=True, show_progress_bar=True, batch_size=128
    ).to(device)
    analyses = build_analyses_via_rag(query_embs, ideal_sets, train_embeddings, trait_explanations, top_k=args.top_k)

    # Free embedding model — classifier needs the GPU memory.
    del embed_model, query_embs, train_embeddings
    torch.cuda.empty_cache()

    # 5) Build scoring rows.
    rows = [
        {"context": contexts[i], "utterance": predictions[i], "explanations": analyses[i]}
        for i in range(n_pred)
    ]

    # 6) Tokenizer + DataLoader
    print(f"Loading tokenizer: {args.classifier_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(
        PredictionScoringDataset(rows, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=False,
    )

    # 7) Load CARE classifier
    print(f"Loading CARE classifier: {args.care_checkpoint}")
    model = QwenHierarchicalClassifier(args.classifier_model_id)
    state = torch.load(args.care_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 8) Inference
    all_pred_idx = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CARE scoring"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(input_ids, mask)
            preds_idx = torch.argmax(out["logits"], dim=2)  # (B, 6)
            all_pred_idx.append(preds_idx.cpu())
    all_pred_idx = torch.cat(all_pred_idx, dim=0).numpy()
    pred_real = np.vectorize(IDX_TO_LABEL.get)(all_pred_idx)  # values in {-2,-1,0,1,2}

    # 9) Save scored CSV
    out_df = pd.DataFrame({
        "ID": test_ids,
        "reference": pred_df["reference"],
        "prediction": pred_df["prediction"],
    })
    for i, short in enumerate(TRAIT_SHORT_NAMES):
        out_df[short] = pred_real[:, i]
    out_df["AVG"] = pred_real.mean(axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print("\n--- CARE score summary (mean across rows) ---")
    for i, short in enumerate(TRAIT_SHORT_NAMES):
        print(f"  {short}: {pred_real[:, i].mean():+.4f}")
    print(f"  AVG: {pred_real.mean():+.4f}")
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
