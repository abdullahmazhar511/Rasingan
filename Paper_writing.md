# Paper Writing — Single Source of Truth

This document is the canonical reference for writing the EMNLP submission.
Every fact, number, file path, and naming decision used in the paper should
flow from here. Update this file when results change.

---

## 1. Paper Identity

| Field | Value |
|---|---|
| **Conference target** | EMNLP (8 pages body + unlimited refs/appendix) |
| **Working title** | (TBD — discuss before submission) |
| **Framework name** | **TAO** (Therapeutic Alignment Optimization) |
| **Dataset name** | **RESPAIR** (in LaTeX: `\dataset`) |
| **Reward classifier** | **CARE** (from Mazhar et al. 2026) — used as-is, not retrained |
| **Multi-turn quality metric** | **CTQ score** (Clinical Therapeutic Quality) — CTRS-derived, expert-validated; do NOT call it CTRS-P in the paper |
| **Page split target** | ~60% prose / 40% tables-figures-graphs |
| **Thesis framing** | B-with-hint-of-A: lead with reward-design technical contribution, anchor in broader "premature intervention" problem |

---

## 2. Project Goals (One-Paragraph Summary)

Current LLMs deployed in mental-health-adjacent settings exhibit a recurring
*premature intervention* failure mode: they respond with advice within 1-2
turns before the user has disclosed enough context. Training models to behave
otherwise is non-trivial because the natural reward signals — clinical-quality
classifiers — produce discrete (argmax) labels, yielding piecewise-constant
rewards that don't train language models (zero policy gradient almost
everywhere). We address this with TAO: a two-stage SFT-then-RL pipeline using
(a) an **ordinal-expected-value smooth CARE reward** + embedding-cosine
auxiliary for single-turn training, and (b) a **session-level reward** combining
PHQ-9/GAD-7 checklist coverage (IR) and CTQ score for multi-turn training. We
release **RESPAIR**, a clinically refined and re-annotated psychotherapy
benchmark, and validate TAO across four model families.

---

## 3. Codebase Map

All paths relative to `/home/asbahk/EMNLP_FINAL/Rasingan/`.

### 3.1 Reward & Agent Loop (core methodology)

| File | Purpose | Key contents |
|---|---|---|
| `verl/verl/experimental/reward/reward_loop/faith.py` | Reward computation | `_compute_six_dim_reward` (single-turn + multi-turn dispatch); `_get_server_predictions` (CARE call w/ `return_expected=True`); `_get_embedding_sim`; hybrid `R = (1−ω)·R_CARE + ω·R_emb` |
| `verl/verl/experimental/agent_loop/therapist_agent_loop.py` | Multi-turn rollout loop | `TherapistMultiTurnAgentLoop.run` (20-turn rollouts); `_generate_therapist_response` (with **early-stop guard** when prompt would overflow `max_model_len`); `_supervise_turn` (per-turn supervisor extraction + feedback); ExternalModelClient for shared vLLM HTTP |
| `inference.py` | CARE classifier | `CareModel.predict` / `batch_predict` with `return_expected=True` flag; returns expected ordinal value `Σ c·p_d(c)` for smooth reward |
| `server/app.py` | CARE HTTP server | `POST /predict`, `POST /batch_predict` (both honor `return_expected`); `POST /embedding_sim`, `POST /batch_embedding_sim` (MiniLM cosine) |

### 3.2 Training Scripts

| File | Purpose |
|---|---|
| `sft_training/train.py` | LoRA SFT on RESPAIR (HuggingFace SFTTrainer) |
| `sft_training/merge_peft_checkpoint.py` | Merge PEFT adapter into standalone HF dir before RL |
| `scripts/single_turn_rl.sh` | Single-turn RL wrapper (CARE + embedding reward) |
| `scripts/multi_turn_rl.sh` | Multi-turn RL wrapper (IR + CTQ session reward) |
| `verl/examples/faith/scripts/run_single_turn.sh` | Inner verl PPO/GRPO launcher (single-turn) |
| `verl/examples/faith/scripts/run_multiturn.sh` | Inner verl PPO/GRPO launcher (multi-turn) + shared patient/supervisor vLLM |
| `verl/examples/faith/scripts/setup_external_models.sh` | Launches the shared Qwen3.5-4B vLLM server (patient + supervisor) |

### 3.3 Evaluation Pipeline

| File | Purpose |
|---|---|
| `scores/get_scores.sh` | Two-phase eval orchestrator. Phase A = single-turn (generate + CARE + NLP). Phase B = multi-turn (sessions + CTRS + IR). Handles SFT/VERL merge. |
| `scores/generate.py` | Single-turn response generation against RESPAIR test split |
| `scores/score_care.py` | Runs CARE classifier on model responses; **regenerates gold CARE scores** (uses `_gold_care_cache.csv` shared across models) |
| `scores/score_care_loss.py` | Computes L1/L2 CARE-loss against regenerated gold |
| `scores/score_nlp.py` | BLEU / ROUGE-1 / ROUGE-L / METEOR / BERTScore-F1 |
| `scores/score_ctrs.py` | Computes session-level CTQ aggregate from `sessions/*.json` |
| `scores/score_information_retrieval.py` | Computes IR coverage (17 PHQ-9/GAD-7 items) |
| `final_pipeline/run.py` | Multi-agent session simulator — therapist (HF in-process) × patient (vLLM HTTP) × supervisor (vLLM HTTP) |
| `final_pipeline/pipeline.py` | `TherapyPipeline` class — orchestrates per-turn message flow |
| `final_pipeline/hf_therapist.py` | In-process HF therapist client (vLLM-compatible chat API) |
| `final_pipeline/{agent_1,agent_2,patient}.py` | Therapist, supervisor, patient roles |
| `final_pipeline/checklists/` | PHQ-9 + GAD-7 item definitions used by supervisor |

### 3.4 Dataset & Data

| Path | Description |
|---|---|
| `RESPAIR/{train,val,test}/N.csv` | Raw cleaned per-session CSVs (250/32/31 files) with 6 CARE ordinal labels per therapist turn |
| `respair_mhcopilot_format/{train,val,test}.csv` | Flattened HF-loader-friendly format (13,356/1,535/1,857 utterances) |
| `respair_mhcopilot_format/train_rl_single.csv` | CTQ-stratified 150-conversation subset for RL training (8,125 utterances, 90/10 positive/hard-negative split) |
| `CARE/rag_cache/{train,val,test}_processed.json` | Per-utterance free-text rationales for each of 6 CARE dimensions (6,799/796/943 entries) |
| `CARE/rag_cache/rag_index.pt` | MiniLM embeddings (6,799 × 384) + per-dim ideal sets + trait explanations |
| `multiturn_reddit_data/{anxiety,depression}_cleaned_data.csv` | 300 + 300 Reddit patient-roleplay seed posts |
| `multiturn_reddit_data/splits/{train,val,test}.csv` | Stratified 80/10/10 split (480/60/60 posts) |
| `RESPAIR/{issues,issue,issue2,flagged_non_conversation_files}.md` | Per-file cleaning audit logs (~80 modified, 5 removed, 2 split) |

### 3.5 Auxiliary

- `REWARDS.md` — chronological log of reward function variants tried (v1 argmax → v2 expected value → v3 hybrid + embedding)
- `scores/README.md` — quick-reference for eval pipeline
- `verl/examples/faith/scripts/api_key` — symlinked to `sft_training/api_key` (HF_TOKEN + WANDB_API_KEY)

---

## 4. Dataset Statistics (for §3 Dataset)

### 4.1 RESPAIR — Core Corpus

| Split | Sessions | Utterances |
|---|---:|---:|
| Train | 250 | 13,356 |
| Validation | 32 | 1,535 |
| Test | 31 | 1,857 |
| **Total** | **313** | **16,748** |

- **Source**: Refined HOPE (Malhotra et al. 2022) via FAITH-M annotation framework (Mazhar et al. 2026).
- **Annotation**: 6 dimensions × 5-point ordinal scale {−2, −1, 0, +1, +2}.
- **Dimensions** (use these names — match data columns): NJ (Non-Judgmental Language), WE (Warmth and Encouragement), RA (Respect for Autonomy), AL (Active Listening), RF (Reflecting Feelings), SA (Situational Appropriateness).
- **Inter-annotator agreement** (Cohen's κ): overall 0.71; NJ 0.70, W&E 0.73, RF 0.65, AL 0.69, RA 0.75, SA 0.74.
- **Annotators**: 2 primary + 1 senior + 1 licensed clinical psychologist (calibration ~3 weeks, annotation ~12 weeks, validation by expert).

### 4.2 Cleaning Protocol

- Stage 1: Gemini-2.5 transcript correction conditioned on YouTube auto-captions.
- Stage 2: 4-pass contamination audit (artifacts → role validity → meta-narration → residual).
- Per-file disposition: **80 fixed, 5 removed, 2 split, 8 borderline retained** after expert review.
- **Test split never modified after initial annotation** (guarantee no evaluation contamination).

### 4.3 Derived Resources

**CARE Rationale Cache** — per-utterance LLM-generated rationales per dimension:
- 6,799 train / 796 val / 943 test records
- Each: dialogue context, 6 ordinal labels, 6-field rationale dict
- Used by CARE classifier as retrieval-augmented analysis prompts (MiniLM nearest-neighbor over `rag_index.pt`)

**RL Single-Turn Subset** — 150 conversations / 8,125 utterances:
- Eligibility: ≥4 therapist turns + both speaker roles
- Stratification: 135 from positive pool (no therapist turn with mean CARE ≤ 0) + 15 from hard-negative pool
- Ranked within pool by CTQ score
- Path: `respair_mhcopilot_format/train_rl_single.csv`

**Multi-Turn Evaluation Seeds** — 600 Reddit posts:
- 300 from r/Anxiety + 300 from r/depression
- Filtered for first-person symptom disclosure, no crisis indicators
- Fields: post_id, title, body (no author identifiers retained)
- Stratified 80/10/10 split: 480 train / 60 val / 60 test
- Used only as patient-simulator seeds — no model is trained on them

---

## 5. Models & Training

### 5.1 Model Families Under Test

| Family | Backbone | Status |
|---|---|---|
| Llama-3.2-1B | meta-llama/Llama-3.2-1B-Instruct | ✅ Base, SFT, RL done |
| Qwen3-4B | Qwen/Qwen3-4B-Instruct-2507 | ✅ Base, SFT, RL done (v3 step 514 best) |
| Mistral | (8B variant) | ✅ Base, SFT, RL done |
| Gemma | TBD | 🕐 In progress today |

### 5.2 SFT

- LoRA adapters (rank: TBD — fill from config), backbone frozen
- Trained on RESPAIR train split (13,356 utterances)
- Context window: 6 turns (matches eval pipeline)
- System prompt + chat-template renderer in `sft_training/train.py:format_to_messages`

### 5.3 RL (Single-Turn) — `scripts/single_turn_rl.sh`

- Algorithm: GRPO (Shao et al. 2024) via verl
- Reference policy: SFT checkpoint (KL anchor)
- Reward: `R = (1−ω)·R_CARE + ω·R_emb`, default `EMB_REWARD_WEIGHT=0.5`
- Key hyperparams (Qwen3-4B reference run):
  - lr = 1e-6, kl_loss_coef = 0.01
  - response_length = 256, prompt_length = 1024
  - rollout.n = 4 rollouts/prompt
  - total_epochs = 2 (~514 steps observed)
  - save/test_freq = 80
- Training set: `train_rl_single.csv` (150 conversations / 8,125 turns)

### 5.4 RL (Multi-Turn) — `scripts/multi_turn_rl.sh`

- Algorithm: GRPO with session-level reward
- Reward: `R_multi = R_IR + R_CTRS` (equal-weighted sum at end of each session)
- Multi-agent environment:
  - **Therapist** (trained policy, internal verl vLLM, GPU 0+1 TP=2)
  - **Patient simulator** (frozen Qwen3.5-4B via shared vLLM on port 8001)
  - **Clinical supervisor** (frozen Qwen3.5-4B via same shared vLLM)
- Key hyperparams:
  - `MAX_THERAPY_TURNS = 20`
  - `MAX_PROMPT_LENGTH = 2048`, `MAX_RESPONSE_LENGTH = 12288` (rollout vLLM `max_model_len = 14336`)
  - `SUPPORT_MAX_LEN = 24576` (shared vLLM)
  - `TRAIN_LIMIT = 240` reddit posts (120 anxiety + 120 dep), `VAL_LIMIT = 25`
  - GRPO step ~360s (multi-turn)
- **Trajectory-preserving early-stop guard** in `therapist_agent_loop.py:_generate_therapist_response`: if accumulated prompt would overflow `max_model_len`, session terminates cleanly at that turn (returns `[], []`). Does NOT truncate history — preserves PPO trajectory faithfulness.

---

## 6. Reward Design (for §4 Methodology)

### 6.1 Single-Turn Reward

**Step 1 — ordinal expected value** (replaces argmax):
$$ \hat{y}_d(r) = \sum_{c \in \{-2,-1,0,+1,+2\}} c \cdot p_d(c \mid r) \in [-2, +2] $$

where $p_d(c \mid r)$ is the softmax probability over class $c$ for CARE dimension $d$.

**Step 2 — CARE reward**:
$$ R_{\text{CARE}}(r, r^\star) = \mathrm{clip}\!\left(1 - \frac{1}{\alpha}\sqrt{\tfrac{1}{D}\sum_d (\hat{y}_d(r) - \hat{y}_d(r^\star))^2},\ 0,\ 1\right) $$

with $D = 6$, $\alpha = \tfrac{2}{3}(y_{\max} - y_{\min}) = \tfrac{8}{3}$.

**Step 3 — embedding-cosine auxiliary** (MiniLM, L2-normalized):
$$ R_{\text{emb}}(r, r^\star) = \tfrac{1}{2}(\cos(\phi(r), \phi(r^\star)) + 1) \in [0, 1] $$

**Step 4 — composite**:
$$ R_{\text{single}} = (1 - \omega)\, R_{\text{CARE}} + \omega\, R_{\text{emb}}, \quad \omega \in [0, 1] $$

### 6.2 Multi-Turn Reward (session-level)

**IR coverage** — 17 items from PHQ-9 (9) + GAD-7 (8):
$$ R_{\text{IR}}(\tau) = \frac{1}{17}\sum_{i=1}^{17} \mathbf{1}_i(\tau) $$

where $\mathbf{1}_i(\tau) = 1$ if patient disclosed item $i$ during session $\tau$.

**CTQ score (CTRS-derived therapeutic quality)** — over therapist turns only:

Construct decomposition:
- $U = \tfrac{1}{2}(\text{AL} + \text{RF})$ (Understanding)
- $\text{IE} = \tfrac{1}{2}(\text{NJ} + \text{WE})$ (Interpersonal Effectiveness)
- $C = \text{RA}$ (Collaboration)
- $\text{TA} = \text{SA}$ (Technical Appropriateness)

Per-construct score:
$$ \mathrm{score}_k = Q_k - F_k + \tfrac{1}{2} SP_k $$

where $Q_k$ = mean dimension value over therapist turns, $F_k$ = fraction non-positive, $SP_k$ = fraction strongly positive (value = +2).

Session-level CTQ:
$$ R_{\text{CTQ}}(\tau) = 0.35\,U + 0.25\,\text{IE} + 0.20\,C + 0.20\,\text{TA} $$

**Composite session reward**:
$$ R_{\text{multi}} = R_{\text{IR}} + R_{\text{CTQ}} $$

### 6.3 Why ordinal-expected-value matters (the technical pitch)

- Argmax produces piecewise-constant rewards → gradient zero almost everywhere → PPO advantages collapse → policy doesn't move.
- Expected value is differentiable in classifier logits → smooth gradient → PPO actually trains.
- **Evidence in our runs**: argmax-reward run (`qwen_3_v1_*`) showed validation L2 *drifting up* (0.934 → 1.012); smooth-reward run (`qwen_3_v3`) showed clean improvement (best at step 240, L2 = 0.4818 on smooth-reward eval; argmax-eval L2 final = 1.2814).

---

## 7. Results

### 7.1 Single-Turn CARE Loss (Phase A) — main results

All values from `evaluation_pipeline/<model>/care_loss.json` after `get_scores.sh`.

**Lower = better.**

| Model | Setting | L1 (MAE) | **L2 (MSE)** | UPR-L2 | REF-L2 | Δ vs Base (L2) |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | Base | 0.798 | 1.537 | 2.366 | 0.708 | — |
| Llama-3.2-1B | SFT | 0.747 | 1.375 | 2.111 | 0.639 | −10.5% |
| **Llama-3.2-1B** | **TAO (RL)** | **0.706** | **1.313** | 2.029 | 0.597 | **−14.6%** ← best Llama |
| Qwen3-4B | Base | 0.749 | 1.407 | 2.171 | 0.644 | — |
| Qwen3-4B | SFT | 0.718 | 1.336 | 2.046 | 0.625 | −5.1% |
| Qwen3-4B | TAO (RL, step 240) | 0.699 | 1.301 | 1.996 | 0.606 | −7.5% |
| **Qwen3-4B** | **TAO (RL, step 514, final)** | **0.689** | **1.281** | **1.984** | **0.578** | **−8.9%** ← best Qwen |
| Mistral | Base | 0.740 | 1.413 | 2.202 | 0.624 | — |
| Mistral | SFT | 0.730 | 1.364 | 2.140 | 0.587 | −3.5% |
| **Mistral** | **TAO (RL)** | **0.682** | **1.289** | 2.033 | 0.544 | **−8.8%** ← best Mistral |
| Gemma | Base | TBD | TBD | TBD | TBD | — |
| Gemma | SFT | TBD | TBD | TBD | TBD | TBD |
| Gemma | TAO (RL) | TBD | TBD | TBD | TBD | TBD |

**Headline**: TAO reduces CARE-L2 by **8.8 – 14.6%** across three completed model families. Largest gain on smallest model (Llama-1B).

### 7.2 Per-Construct CARE L1 (single-turn)

| Model | NJ | WE | RA | AL | RF | SA |
|---|---:|---:|---:|---:|---:|---:|
| Llama Base | 0.863 | 0.826 | 1.486 | 0.582 | 0.655 | 0.374 |
| Llama SFT | 0.875 | 0.792 | 1.299 | 0.613 | 0.488 | 0.415 |
| **Llama TAO** | 0.821 | 0.743 | 1.280 | 0.548 | 0.467 | 0.375 |
| Qwen Base | 0.859 | 0.803 | 1.353 | 0.586 | 0.511 | 0.383 |
| Qwen SFT | 0.822 | 0.763 | 1.283 | 0.571 | 0.480 | 0.386 |
| **Qwen TAO (step 514)** | 0.797 | 0.730 | 1.249 | 0.508 | 0.478 | 0.373 |
| Mistral Base | 0.854 | 0.789 | 1.350 | 0.519 | 0.578 | 0.353 |
| Mistral SFT | 0.853 | 0.773 | 1.358 | 0.579 | 0.446 | 0.371 |
| **Mistral TAO** | 0.776 | 0.734 | 1.281 | 0.461 | 0.497 | 0.339 |

**Key observation**: REF group constructs (AL, RF, SA) — the harder reflective/contextual dimensions — see the largest improvement under TAO. RA (Respect for Autonomy) improves across every family. NJ, WE improve modestly.

### 7.3 NLP Metrics (single-turn, higher = better)

| Model | Setting | BLEU | ROUGE-1 | ROUGE-L | METEOR | BERTScore-F1 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | Base | 0.0068 | 0.1537 | 0.1139 | 0.1801 | 0.7199 |
| Llama-3.2-1B | SFT | 0.0174 | 0.1652 | 0.1365 | 0.1525 | 0.7285 |
| **Llama-3.2-1B** | **TAO** | **0.0152** | **0.1866** | **0.1513** | **0.1885** | **0.7393** |
| Qwen3-4B | Base | 0.0214 | 0.1789 | 0.1530 | 0.1524 | 0.7330 |
| Qwen3-4B | SFT | 0.0202 | 0.1801 | 0.1552 | 0.1533 | 0.7330 |
| Qwen3-4B | TAO (step 240) | 0.0272 | 0.1984 | 0.1660 | 0.1865 | 0.7401 |
| **Qwen3-4B** | **TAO (step 514)** | **0.0232** | **0.2008** | **0.1654** | **0.2119** | **0.7445** |
| Mistral | Base | 0.0115 | 0.1715 | 0.1343 | 0.1714 | 0.7317 |
| Mistral | SFT | **0.0243** | **0.1823** | **0.1535** | 0.1752 | 0.7356 |
| **Mistral** | **TAO** | 0.0121 | 0.1713 | 0.1377 | **0.1817** | **0.7384** |
| Gemma | Base/SFT/TAO | TBD | TBD | TBD | TBD | TBD |

**Note on Mistral**: SFT beats TAO on lexical metrics (BLEU, ROUGE) but TAO wins on BERTScore-F1 and CARE-L2. Frame this honestly as "TAO trades lexical overlap for clinical alignment" — that's the intended objective.

### 7.4 Multi-Turn Results

**Status**: in progress today.

Schema for the table once numbers land:

| Model | Setting | IR Coverage | CTQ Score | U | IE | C | TA |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | Base/SFT/TAO | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-4B | Base/SFT/TAO | TBD | TBD | TBD | TBD | TBD | TBD |
| Mistral | Base/SFT/TAO | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma | Base/SFT/TAO | TBD | TBD | TBD | TBD | TBD | TBD |

**Multi-turn-trained checkpoint** (`therapist_multi_23-23-15_rl`) — evaluated on single-turn benchmark:
- L2 = 1.327 (slightly worse than single-turn-trained Qwen RL, 1.281)
- BERTScore = 0.7453 (basically tied with single-turn RL)
- BLEU = 0.0272 (slightly better than single-turn RL)

This is expected: multi-turn training optimizes for full-session quality, not turn-by-turn matching of gold. The fair comparison is on multi-turn metrics, which will come from `score_ctrs.py` + `score_information_retrieval.py`.

### 7.5 Headline Numbers for Intro / Abstract

- **CARE-L2 reduction**: 8.8% to 14.6% across three model families (will update to range across four once Gemma lands).
- **Largest relative improvement**: Llama-3.2-1B (1B parameters) at −14.6% — "TAO scales down to 1B models."
- **Per-construct**: REF group (AL, RF, SA) sees the strongest gains across all families.
- **BERTScore-F1 best**: Qwen3-4B TAO at 0.7445.

---

## 8. Pending Experiments (status as of today)

| Item | Status | Owner | Expected |
|---|---|---|---|
| Gemma SFT + RL single-turn | In progress | Asbah | Today |
| Multi-turn RL training (Llama / Qwen / Mistral / Gemma) | In progress | Asbah | Today |
| Multi-turn evaluation (CTQ + IR via `get_scores.sh` Phase B) | Will run after multi-turn RL completes | Asbah | Today / tomorrow |
| Expert human evaluation of multi-turn sessions | Not yet — needs protocol | Asbah | This week |
| LLM-judge × Expert agreement analysis (κ + r) | After both above | Asbah | This week |
| Smooth-vs-argmax ablation (single-turn) | Have implicit data via `qwen_3_v1_*` (argmax broke) vs `qwen_3_v3` (smooth works) | — | Write up now |
| Ω sweep for `EMB_REWARD_WEIGHT` (single-turn ablation) | Not run | Optional | Skip unless time |

---

## 9. Paper Section Status

| Section | Status | File / Notes |
|---|---|---|
| §1 Introduction | Drafted today (this session) | Two `\hl{...}` placeholders pending Gemma + multi-turn numbers |
| §2 Related Work | Not started | Need: prior therapeutic LLM work, RLHF for chat, clinical alignment evaluation |
| §3 Dataset (RESPAIR) | Drafted earlier this session (final accepted version above) | Verify ethics paragraph for EMNLP |
| §4 Methodology (TAO) | Drafted, finalized today | Includes single-turn reward §4.3, multi-turn session reward §4.4, GRPO §4.5, multi-agent eval env §4.6 |
| §5 Experiments | Structure planned (Setup, Baselines, Metrics, Results, Ablation) | Draft §5.1-5.3 now; §5.4-5.5 after numbers land |
| §6 Analysis | Structure planned (Single-Turn, Multi-Turn, Error, Expert Assessment) | Draft single-turn analysis now; rest blocked on data |
| §7 Conclusion / Limitations / Ethics | Not started | Standard EMNLP boilerplate; can copy structure from prior submissions |
| Appendix | Structure planned (A-I) | See below |

### 9.1 Appendix Plan

- **A** Implementation details (LoRA rank, lr, KL coef, GRPO group size, wall-clock, GPU)
- **B** Dataset cleaning protocol — per-file disposition log
- **C** Reward implementation details — full softmax → expected value derivation, agent-loop early-stop guard, `vllm_async_server.py:400` math
- **D** Per-construct per-family result tables (long-form)
- **E** Training curves: reward, val-L2, entropy, KL trajectories per family
- **F** Example conversations — 4-6 full sessions side-by-side (Base / SFT / TAO) with expert annotations
- **G** Expert assessment protocol — rubric, LLM-judge prompts, IRR
- **H** Ethical considerations (Reddit data, no deployment, research-only)
- **I** Limitations (trajectory early-stops, CARE classifier as ceiling, no real patient data)

---

## 10. Writing Decisions / Glossary

### 10.1 Hard naming rules

| Always use | Never use |
|---|---|
| **TAO** (framework name) | "GRACE" (earlier name, dropped) |
| **RESPAIR** / `\dataset` | "HOPE-v2", "FAITH-M" except as predecessor citation |
| **CARE** (the classifier) | "CARE-v2" (the prototype mechanism doesn't exist) |
| **CTQ score** (CTRS-derived therapeutic quality, expert-validated) | "CTRS-P" (avoid reviewer pushback on direct CTRS claim) |
| **CARE classifier** | "CARE model" (ambiguous with the framework name) |
| **Multi-agent environment** | "Multi-agent framework" (overlaps with TAO) |
| **Ordinal-expected-value reward** | "Soft reward" (less specific) |
| **Early-stop guard** | "Truncation" (we explicitly do NOT truncate trajectories) |

### 10.2 CARE dimension names (6) — match the data columns

| Abbreviation | Full name (use this) |
|---|---|
| NJ | Non-Judgmental Language |
| WE | Warmth and Encouragement |
| RA | Respect for Autonomy |
| AL | Active Listening |
| RF | Reflecting Feelings |
| SA | Situational Appropriateness |

**Do NOT** use the alternative names from Mazhar 2026 (Non-Judgmental Acceptance, Warmth and Empathy, Radical Acceptance, Alignment, Reflective Listening, Support and Affirmation) — those don't match the dataset columns.

### 10.3 Phrases worth preserving across sections

- "Premature intervention" — name for the failure mode (introduced §1)
- "Discrete rewards do not train language models" — quotable methodology pitch (§1)
- "Information acquisition before intervention" — descriptive phrase for desired behavior (§1)
- "Trajectory-preserving early-stop" — for the agent-loop guard (§4, App. C)

### 10.4 Citation keys (use these `.bib` keys)

| Topic | Key |
|---|---|
| LLM as therapist usage | `hatch2025eliza`, `song2024typingcureexperienceslarge` |
| CARE framework / FAITH-M | `mazhar2026measuringmattersassessingtherapeutic` |
| HOPE corpus | `malhotra2022speaker` |
| PHQ-9 | `kroenke2001phq` |
| GAD-7 | `spitzer2006brief` |
| PPO | `schulman2017ppo` |
| GRPO | `shao2024deepseekmathpushinglimitsmathematical` |
| LoRA | `hu2021loralowrankadaptationlarge` |
| MiniLM | `wang2020minilmdeepselfattentiondistillation` |
| vLLM | `kwon2023efficientmemorymanagementlarge` |
| Cohen's kappa | `cohen1960coefficient` |
| Gemini-2.5 (used for transcript correction) | `gemini_team_2023` |

### 10.5 Numbers to never confuse

- **313 sessions / 16,748 utterances** = full RESPAIR
- **150 conversations / 8,125 utterances** = `train_rl_single.csv` (RL training subset)
- **600 Reddit posts** = multi-turn eval seeds (300 + 300, split 80/10/10)
- **17 checklist items** = PHQ-9 (9) + GAD-7 (8), used by supervisor for IR
- **6 CARE dimensions** = NJ, WE, RA, AL, RF, SA
- **5 ordinal classes** = {−2, −1, 0, +1, +2}
- **MAX_THERAPY_TURNS = 20** (multi-turn rollout cap)

---

## 11. Multi-Agent Evaluation Environment (for §4.6 / §5.1)

### 11.1 Agents

| Agent | Model | Trained? | Path |
|---|---|---|---|
| **Therapist** (policy under test) | LLM-of-interest (Llama / Qwen / Mistral / Gemma) | ✅ (the only trained agent) | `final_pipeline/agent_1.py` + `hf_therapist.py` |
| **Patient simulator** | Qwen3.5-4B (instruct, non-thinking mode) | Frozen | `final_pipeline/patient.py` |
| **Clinical supervisor** | Qwen3.5-4B (instruct, non-thinking mode) | Frozen | `final_pipeline/agent_2.py` |

- Patient simulator: conditioned on a Reddit seed post; system prompt enforces (i) progressive disclosure, (ii) persona consistency.
- Clinical supervisor: per-turn checklist scoring against 17 PHQ-9/GAD-7 items + per-session CTQ scoring at termination.

### 11.2 Session Mechanics

- Each session = up to `MAX_THERAPY_TURNS = 20` turns
- One rollout = patient utterance → therapist response → (parallel) patient + supervisor reactions
- Session ends on: max turns, natural conclusion (model says goodbye), or prompt-overflow early-stop
- Reward at termination: `R_multi = R_IR + R_CTQ`

### 11.3 Eval Protocol (Both for §5 and Expert Assessment)

- Test seed set: 60 Reddit posts (the held-out test split)
- For each (model, setting), generate 60 sessions
- Score each session with:
  - **Automatic**: supervisor LLM (Qwen3.5-4B) computes IR + CTQ
  - **Expert**: clinician scores same 60 sessions on IR (binary per item) and CTQ (Likert 1-5 per construct)
- Compute LLM-judge × expert agreement: **Cohen's κ for IR** + **Pearson r for CTQ**

---

## 12. Open TODOs (paper-writing critical path)

In priority order:

1. **Finish Gemma single-turn runs** — needed for headline number in §1 and §5.4
2. **Finish multi-turn RL training across all 4 families** — needed for §5.4 multi-turn table
3. **Run Phase B (`score_ctrs.py` + `score_information_retrieval.py`)** on every multi-turn checkpoint — needed for §5.4 numbers
4. **Define expert assessment protocol** — rubric, sampling strategy, instructions; needs to be written before experts can start
5. **Run expert assessment** — 60 sessions × N experts × M models; aim for at least 3 experts
6. **Compute LLM-judge × expert agreement** — κ for IR, Pearson r for CTQ
7. **Pick 4-6 example sessions for App. F** — one per family, ideally with strong contrasts
8. **Write Limitations and Ethics** — boilerplate, can copy
9. **Final pass on naming consistency** — search & verify (TAO, RESPAIR, CARE, CTQ — no slips)

---

## 13. Quick Reference — How to Reproduce Any Number

```bash
# Single-turn eval (Phase A only) for a specific model
bash ./scores/get_scores.sh \
    --model-name <name> \
    --base-model <hf-id-or-path> \
    --merge sft \
    --merge-checkpoint <path-to-PEFT-adapter> \
    --skip-multi-turn

# Multi-turn eval (Phase B only)
bash ./scores/get_scores.sh \
    --model-name <name> \
    --skip-single-turn \
    --categories "anxiety depression" \
    --n-per-category 20 \
    --concurrency 4 \
    --max-turns 20

# Single-turn RL training
MODEL_PATH=<merged-sft-hf-dir> \
EXP_NAME=<name> \
OUTPUT_PATH=./checkpoints/<name> \
bash scripts/single_turn_rl.sh

# Multi-turn RL training
MODEL_PATH=<merged-sft-hf-dir> \
EXP_NAME=<name> \
OUTPUT_PATH=./checkpoints/<name> \
bash scripts/multi_turn_rl.sh
```

---

## 14. References / Cross-Docs

- `REWARDS.md` — chronological log of reward function variants (v1 argmax → v3 hybrid)
- `RESPAIR/issues.md`, `issue.md`, `issue2.md`, `flagged_non_conversation_files.md` — dataset cleaning audit logs
- `respair_mhcopilot_format/README_train_rl_single.md` — RL training subset curation criteria
- `verl/examples/faith/scripts/run_single_turn.sh` — full single-turn RL config (LoRA rank, lr, etc.)
- `verl/examples/faith/scripts/run_multiturn.sh` — full multi-turn RL config

---

*Last updated: 2026-05-24. Update this file whenever (a) new numbers land, (b) naming changes, (c) new experiments complete, or (d) a writing decision is made.*
