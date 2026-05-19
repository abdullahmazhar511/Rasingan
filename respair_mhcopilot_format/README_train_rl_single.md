# train_rl_single.csv Curation README

This document explains how `train_rl_single.csv` was built from `train.csv`.

## Source and Outputs

- Source: `respair_mhcopilot_format/train.csv`
- Selected train split: `respair_mhcopilot_format/train_rl_single.csv`
- Selection metadata: `respair_mhcopilot_format/train_rl_single_conv_ids.txt`

## Goal

Create a conversation-level RL training subset with:

- Exactly 150 conversations
- Ranking based on CTRS-P (conversation metric)
- 10% of selected conversations containing negative CARE turns

## Conversation Unit

Conversation ID is derived from row ID by removing the final turn suffix:

- Row ID format: `<conv_id>_<turn_idx>`
- Conversation ID extraction regex: `_[0-9]+$` (removed)

## CTRS-P Scoring Rule

Scoring follows `scores/score_ctrs.py` and is computed at conversation level from therapist turns only.

CARE dimensions per therapist turn:

- NJ, WE, RA, AL, RF, SA

Construct definitions:

- Understanding (U) = (AL + RF) / 2
- Interpersonal Effectiveness (IE) = (WE + NJ) / 2
- Collaboration (C) = RA
- Technical Appropriateness (TA) = SA

For each construct `k` over all therapist turns in a conversation:

- Q_k = mean(values)
- F_k = rate(values <= 0)
- SP_k = rate(values == 2)
- Score_k = Q_k - F_k + 0.5 * SP_k

Final conversation score:

- CTRS-P = 0.35 * U + 0.25 * IE + 0.20 * C + 0.20 * TA

## Eligibility Gates

A conversation is eligible if all are true:

- Has both speakers (Therapist and Patient)
- Has at least 4 therapist turns

Negative-CARE flag is computed from therapist-turn CARE mean:

- mean6_per_turn = mean(NJ, WE, RA, AL, RF, SA)
- `has_negative_care = any(mean6_per_turn < 0)`

## Selection Policy

Target size: 150 conversations.

- Positive pool: eligible and `has_negative_care = False`
- Negative pool: eligible and `has_negative_care = True`
- Keep 10% negative-care conversations in final selection:
  - 135 from positive pool
  - 15 from negative pool
- Within each pool, rank descending by:
  1. CTRS-P
  2. mean6_avg
  3. therapist_turns

If a pool is smaller than needed, remainder is backfilled from the other pool by top rank.

## Resulting Dataset Stats

Source `train.csv`:

- Conversations: 250
- Rows: 13,356
- Therapist turns: 6,799
- Patient turns: 6,533

Selected `train_rl_single.csv`:

- Conversations: 150
- Rows: 8,125
- Therapist turns: 4,132
- Patient turns: 3,993
- Negative-CARE conversations: 15 (10.0%)

CTRS-P distribution on selected conversations:

- Mean: 1.0342
- Std: 0.2385
- Min: 0.6415
- Median (p50): 1.0232
- p90: 1.3315
- p95: 1.4908
- Max: 1.6028

## Repro Notes

- Re-run scoring and selection from latest `train.csv` whenever CARE annotations change.
- `train_rl_single_conv_ids.txt` stores per-conversation rank and score fields for traceability.
