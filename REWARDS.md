# Reward functions tried

Chronological record of the single-turn RL reward variants for the therapist
model. All implementations live in
[verl/verl/experimental/reward/reward_loop/faith.py](verl/verl/experimental/reward/reward_loop/faith.py)
(method `_compute_six_dim_reward`), with classifier predictions served by
[server/app.py](server/app.py) via the `CareModel` in
[inference.py](inference.py).

The reference signal across all variants is the dataset's **gold therapist
utterance** at the same dialogue position as the model's response. "Reward" =
distance / similarity between the model's response and the gold response.

---

## v1 — CARE argmax L2 (original)

**Formula:**
```
pred_i = argmax_c logits(model_response)[i, c]      ∈ {-2,-1,0,1,2}   for i in 6 CARE dims
ref_i  = argmax_c logits(gold_response)[i, c]       ∈ {-2,-1,0,1,2}
normalized_l2 = sqrt( mean_i (pred_i - ref_i)^2 )
reward = clip( 1 - normalized_l2 / ((CARE_MAX - CARE_MIN) / 1.5), 0, 1 )
       = clip( 1 - normalized_l2 / 2.667, 0, 1 )
```

**Property:** each CARE dim is integer-valued, so reward is **piecewise constant**.
Only ~20 distinct reward values can occur (driven by which integer differences
the 6 dims contribute).

**Observed in `qwen_3_v1_21-09-39` (514 steps):**
- 11.3% of rollouts hit `reward = 1.0` exactly (argmax collisions onto common
  6-tuples, not actual gold-matching quality).
- 0% of rollouts landed in `[0.95, 1.0)` — no gradient between "off by 1
  dim" and "exact match."
- `critic/advantages/mean ≈ 0`, `actor/pg_loss ≈ 0` → no learning signal.
- Training reward stayed flat at ~0.625 throughout.
- Validation `l2_norm` drifted **up** 0.934 → 1.012 across 514 steps.
- Entropy collapsed 1.94 → 0.30; KL grew 0.025 → 0.51; response length
  tripled 37 → 128 and hit the 128-token cap (80% clip ratio by step 400).

**Diagnosis:** reward is a step function over the response space. PPO can't
descend it — most token-level edits give zero gradient. Mode collapse +
length runaway followed.

---

## v2 — CARE expected-value L2 (fix #1: smooth the reward)

**Change:** replace `argmax` with the expected value of the softmaxed class
distribution. Each CARE dim becomes a continuous float in [-2, 2]:
```
pred_i = sum_c softmax(logits)[i, c] * c           ∈ [-2, 2]
ref_i  = sum_c softmax(logits_gold)[i, c] * c      ∈ [-2, 2]
normalized_l2 = sqrt( mean_i (pred_i - ref_i)^2 )
reward = clip( 1 - normalized_l2 / 2.667, 0, 1 )
```

Implemented by adding `return_expected: bool` flag to:
- [inference.py](inference.py) — `CareModel.predict` and `batch_predict`.
- [server/app.py](server/app.py) — `/predict` and `/batch_predict`.
- [faith.py](verl/verl/experimental/reward/reward_loop/faith.py) — RL reward
  sends `return_expected: True`; eval (`score_care.py`) still uses argmax
  default for apples-to-apples comparison with dataset annotations.

**Same forward pass produces both argmax integers and expected floats** —
the server returns argmax in a separate field (`predictions_argmax`) when
`return_expected=True`, so the training log can report both expected-value
L2 and argmax L2 with one CARE call.

**Observed in `qwen_3_v2` (514 steps):**
- Reward histogram: **479 unique values** in [0.45, 1.0], smooth distribution.
- Only **0.2% of rollouts** hit reward = 1.0 exactly.
- `pg_loss` magnitude 0.02-0.18; `critic/advantages/mean` ±0.01-0.26 → real
  gradient signal.
- Entropy stable at ~1.85; KL grew slowly to 0.32; response length capped
  at 95 (clip ratio peak 19%). No collapse, no runaway.
- **But:** training reward moved 0.806 → 0.809 over 500 steps (noise).
- Validation L2 best at step 240 (0.4818), then plateaued; net improvement
  over SFT init: **~4% relative**.

**Diagnosis:** the fix removed the *mechanical* roadblock (zero gradient) but
exposed a second-order problem — the CARE classifier itself can't
discriminate finely enough between "OK therapy" and "good therapy" to drive
further gains. SFT init already sits at the smoothed-reward ceiling.

---

## v3 — CARE expected-value + embedding cosine (hybrid, current)

**Change:** add an auxiliary continuous reward from MiniLM sentence-embedding
similarity. The CARE classifier's discrimination ceiling no longer dominates.

**Formula:**
```
care_reward = same v2 expected-value reward, ∈ [0, 1]
emb_sim     = cosine( MiniLM(model_response), MiniLM(gold_response) )  ∈ [-1, 1]
emb_reward  = clip( (emb_sim + 1) / 2, 0, 1 )                          ∈ [0, 1]
reward      = (1 - W) * care_reward + W * emb_reward
```
where `W = EMB_REWARD_WEIGHT` (env var, default `0.5`).

Why embedding sim:
- **Fully continuous** — no argmax / classifier quantization.
- **Classifier-independent** — the MiniLM encoder doesn't share failure modes
  with the CARE classifier.
- **Cheap** — reuses `CareModel.embedding_model` (already loaded for RAG);
  ~10 ms batched per pair.

**Implementation:**
- New endpoints `POST /embedding_sim` and `POST /batch_embedding_sim` in
  [server/app.py](server/app.py).
- `_get_embedding_sim(text_a, text_b)` helper in
  [faith.py](verl/verl/experimental/reward/reward_loop/faith.py).
- `_compute_six_dim_reward` dispatches CARE-predict and embedding-sim calls
  **concurrently** via `asyncio.create_task` — no added latency in the
  reward path.
- Logged metrics in `reward_extra_info` per rollout:
  - `care_reward`, `emb_reward`, `emb_sim` — components
  - `l2_norm`, `raw_l2_norm` — expected-value CARE L2 (reward-aligned)
  - `l2_norm_argmax`, `raw_l2_norm_argmax` — argmax CARE L2 (eval-aligned,
    apples-to-apples with v1)

**Status:** code complete, **not yet run**. Watch on next training:
- `val-aux/.../emb_reward/mean@1` should climb independently of `care_reward`.
- `val-aux/.../l2_norm_argmax/mean@1` is the metric to compare against v1
  (0.934) and v2 (0.4818) — same metric, different reward.
- If `emb_reward` saturates fast (> 0.95 in < 50 steps), lower `W` to 0.3.

---

## Eval-side reward (for completeness)

Independent of the training reward, the evaluation pipeline computes
**CARE-classifier-on-model vs. CARE-classifier-on-gold L1/L2** in
[scores/score_care.py](scores/score_care.py) → [scores/score_care_loss.py](scores/score_care_loss.py).
This always uses **argmax** (matches dataset annotations).

Two relevant changes during this work:
1. **Regenerate gold scores by the classifier** instead of using dataset
   annotations. Avoids drift between human annotators and the CARE
   classifier (the original eval was punishing the model for matching the
   classifier rather than the humans, even when training optimized for the
   former).
2. **Cache gold scores** in `<eval_root>/_gold_care_cache.csv` keyed by `ID`.
   Same gold dataset across N models → 1 gold CARE pass total, not N passes.

Delete `_gold_care_cache.csv` to force a recompute (e.g. after a CARE
classifier change).

---

## Quick comparison

| Variant | Reward space | Unique values per batch | r=1.0 frequency | pg_loss | Entropy at end | Val L2 (argmax) trend |
|---|---|---|---|---|---|---|
| v1 argmax     | ~20 discrete  | ~4-6     | 11.3%   | ~0      | 0.30 (collapse) | 0.934 → 1.012 (up) |
| v2 expected   | continuous    | ~150-300 | 0.2%    | 0.02-0.18 | 1.85          | best 0.4818 @ step 240 |
| v3 hybrid     | continuous + sim | n/a    | < 0.1%  | TBD     | TBD             | TBD                  |

(v3 numbers will be populated after the next training run.)

---

## How to switch reward variants

The reward function is selected by which code is checked out, not by a CLI
flag. To revert to a previous variant:

- **v1 (argmax)**: in `faith.py`, set `_get_server_predictions` to send
  `"return_expected": False` (or omit the field — default is False).
- **v2 (expected only)**: in `faith.py`, set `EMB_REWARD_WEIGHT = 0.0` (or
  `EMB_REWARD_WEIGHT=0 bash scripts/single_turn_rl.sh`).
- **v3 (hybrid)**: current code, default `EMB_REWARD_WEIGHT=0.5`.

In all cases the CARE server must be restarted to pick up server-side changes
(new endpoints, new fields).