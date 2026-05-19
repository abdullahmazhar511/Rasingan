# Remaining RESPAIR Issues

Rescan of `EMNLP_RESPAIR/Rasingan/care_v2/faith_data/RESPAIR` after recent fixes.

Summary:
- Remaining flagged files: 8
- Split counts: `train=8`, `val=0`, `test=0`

Heuristic reasons used:
- `artifact_keywords`: file still contains obvious transcript/meta keywords
- `commentary_turns`: file still contains therapist commentary/teaching rather than pure dialogue
- `multiple_patient_identities`: more than one patient identity addressed in therapist turns
- `missing_role`: file lacks either patient (`P`) or therapist (`T`) turns
- `non_conversation_opening`: first utterance still looks like narration or video intro

## Remaining Files

- `train/1.csv`: artifact_keywords: `act-inconsistent`; commentary_turns (1)
  Example: `So that was an ACT-inconsistent piece. It wasn't terribly ACT-inconsistent, but it did have a few different aspects...` - fixed

- `train/26.csv`: artifact_keywords: `core competency`, `act-consistent`; commentary_turns (1)
  Example: `Yeah. And so I wonder if you and I could kind of recognize our very human experiences... In this session, it was an ACT-con...` - fixed

- `train/324.csv`: artifact_keywords: `act-consistent`, `act-inconsistent`; commentary_turns (2)
  Example: `So this was an ACT-inconsistent vignette. The therapist started out fine with a question that could go either way...` - fixed

- `train/105.csv`: artifact_keywords: `dr. grande` 

- `train/139.csv`: artifact_keywords: `dr. grande`

- `train/256.csv`: artifact_keywords: `dr. grande`

- `train/261.csv`: artifact_keywords: `dr. grande`

- `train/63.csv`: artifact_keywords: `dr. grande`

## Notes

- The remaining `dr. grande` hits may be true residual contamination or may need manual confirmation, depending on the exact row context.
- The strongest remaining issues are the commentary-style files: `train/1.csv`, `train/26.csv`, and `train/324.csv`.
