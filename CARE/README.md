# CARE — Clinical Trait Reward Model

CARE is a Qwen3-4B + LoRA classifier trained on the RESPAIR dataset to score
therapist utterances on 6 clinical traits (–2 to +2 ordinal):
**NJ** Non-Judgmental, **WE** Warmth/Encouragement, **RA** Respect for Autonomy,
**AL** Active Listening, **RF** Reflecting Feelings, **SA** Situational Appropriateness.

This folder contains everything needed to: (1) train CARE from scratch, (2) score
predictions from any generation model with CARE, (3) compare model outputs against
gold human annotations using L2/L1 loss.

---

## Files

| File | Purpose |
|---|---|
| [`CARE.py`](CARE.py) | Trains the CARE classifier from raw RESPAIR data. Two stages: (1) Qwen3-4B explainer generates per-trait analyses via RAG over training set, (2) hierarchical classifier head trains on (context + utterance + analysis) → 6 trait scores. |
| [`score_with_care.py`](score_with_care.py) | Inference-only entry point. Takes a predictions CSV (any model's outputs) and returns CARE-predicted trait scores per row. Uses RAG retrieval over the cached training-pool explanations — no LLM call required at inference. |
| [`run_care_scoring.sh`](run_care_scoring.sh) | Batch runner: score the 4 latest-checkpoint SFT predictions sequentially. Continues past failures. |
| [`care_checkpoint/best_classifier.pt`](care_checkpoint/) | Trained classifier weights (~7.5 GB: Qwen3-4B base + LoRA + custom heads). |
| [`care_checkpoint/test_metrics.json`](care_checkpoint/test_metrics.json) | CARE's intrinsic test metrics: per-trait QWK, Acc, F1-weighted, F1-macro. Avg F1w ≈ 0.66. |
| [`rag_cache/{train,val,test}_processed.json`](rag_cache/) | Pre-generated per-trait explanations for every RESPAIR utterance (~40 MB total). Keyed by ID. Built once by `CARE.py` Stage 1. |
| [`rag_cache/rag_index.pt`](rag_cache/) | Cached RAG retrieval index (Pos/Neg pools per trait + 13K training-utterance embeddings). Built on first `score_with_care.py` run. |
| [`sft_CARE_output/`](sft_CARE_output/) | Output folder. Contains `<model>.csv` per scored model, plus `_comparison.csv` leaderboard and per-run logs. |

---

## How to run

### 1. Score a single predictions CSV

```bash
cd /home/asbahk/EMNLP_FINAL/CARE
CUDA_VISIBLE_DEVICES=0 /home/asbahk/miniconda3/envs/verl/bin/python score_with_care.py \
    --predictions_csv /path/to/<model>_predictions.csv \
    --output_csv sft_CARE_output/<model_name>.csv \
    --batch_size 8
```

Input CSV must have a `prediction` column. Output CSV has columns:
`ID, reference, prediction, NJ, WE, RA, AL, RF, SA, AVG`. Takes ~2-3 min on H100.

### 2. Score all 4 SFT models at once

```bash
cd /home/asbahk/EMNLP_FINAL/CARE
CUDA_VISIBLE_DEVICES=0 bash run_care_scoring.sh
```

Loops over the latest-checkpoint prediction CSVs for Llama-3.2-1B, Qwen3-4B,
gemma-3-4b, Ministral-8B. Continues past any failure. Logs to
`sft_CARE_output/logs/<model>.log`. Total runtime ~12 min.

### 3. Compute L2/L1 leaderboard vs gold human scores

```bash
cd /home/asbahk/EMNLP_FINAL/CARE
python compare_care_scores.py
```

Joins all CSVs in `sft_CARE_output/` (skipping `_`-prefixed files) with gold
human trait scores from
`/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/respair_mhcopilot_format/test.csv`.
Prints a leaderboard sorted by L2 (lower = closer to gold profile). Saves
`sft_CARE_output/_comparison.csv` with per-trait + overall L2/L1.

### 4. Train CARE from scratch

```bash
cd /home/asbahk/EMNLP_FINAL/CARE
CUDA_VISIBLE_DEVICES=0 python CARE.py 2>&1 | tee run.log
```

Expects raw RESPAIR CSVs at `../train/`, `../val/`, `../test/` (relative to repo
root). Stage 1 (explanation generation) is skipped if `rag_cache/*_processed.json`
exist. Stage 2 (classifier training) writes `outputs/best_classifier.pt`.

Override the classifier base model, hyperparameters, etc. by editing `CONFIG` at
the top of [`CARE.py`](CARE.py).

### 5. Sync rag_cache scores back into raw RESPAIR

```bash
cd /home/asbahk/EMNLP_FINAL/CARE
python sync_scores.py
```

Reads `rag_cache/*_processed.json`, overwrites `/home/asbahk/RESPAIR/{train,val,test}/*.csv`
trait columns with those scores, writes results to `/home/asbahk/RESPAIR_synced/`.
Therapist rows missing from the cache get blanked; patient rows always blanked.

---

## How `score_with_care.py` works (one paragraph)

For each prediction row: embed the predicted utterance with `all-MiniLM-L6-v2`.
For each of the 6 traits, retrieve top-K from the trait's Pos pool (training items
gold-scored > 0 for that trait) and top-K from the Neg pool, then gather those
retrieved items' **cached explanations** from `rag_cache/train_processed.json`.
Feed `(context, prediction, retrieved-explanations)` into the CARE classifier.
Argmax → 6 trait scores in `{-2, -1, 0, 1, 2}`. Default K = 3 per polarity.
No Qwen3-4B explainer call at inference time — retrieval only.

---

## Key paths

- Repo root: `/home/asbahk/EMNLP_FINAL/`
- CARE folder: `/home/asbahk/EMNLP_FINAL/CARE/` (this one)
- Test data (MHCoPilot format): `/home/asbahk/EMNLP_FINAL/Rasingan/sft_training/respair_mhcopilot_format/test.csv`
- Raw RESPAIR (training source): `/home/asbahk/EMNLP_FINAL/{train,val,test}/`
- Python env: `/home/asbahk/miniconda3/envs/verl/bin/python` (has torch, transformers, peft, evaluate, bert_score, sentence_transformers, openai)

---

## Known properties of CARE

From `care_checkpoint/test_metrics.json` (CARE evaluated on the gold RESPAIR test set):

| Trait | QWK | Acc | F1-weighted |
|---|---|---|---|
| NJ | 0.31 | 0.66 | 0.62 |
| WE | 0.30 | 0.62 | 0.62 |
| RA | 0.24 | 0.56 | 0.54 |
| AL | 0.30 | 0.62 | 0.57 |
| RF | 0.37 | 0.71 | 0.66 |
| SA | 0.08 | 0.96 | 0.95 |
| **Avg** | **0.27** | **0.69** | **0.66** |

The high SA F1-weighted with QWK ≈ 0 reflects class imbalance (CARE returns the
majority SA class). RF is the most ordinally reliable trait. RA is hardest.

---

## See also (outside this folder)

- [`../run_gpt_pipeline.sh`](../run_gpt_pipeline.sh) — orchestrator that runs
  GPT zero-shot + few-shot generation **and** calls `score_with_care.py` +
  `compare_care_scores.py` from this folder. Writes outputs to
  `CARE/sft_CARE_output/`.
- [`../Rasingan/sft_training/`](../Rasingan/sft_training/) — SFT training code
  that produced the model prediction CSVs CARE evaluates.
