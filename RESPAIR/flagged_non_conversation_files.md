# Flagged RESPAIR Files

Heuristic scan for files in `EMNLP_RESPAIR/Rasingan/care_v2/faith_data/RESPAIR` that do not look like normal patient-therapist conversations.

Flag reasons used:
- `missing_role`: file lacks either patient (`P`) or therapist (`T`) turns
- `artifact_keywords`: file contains obvious video/transcript/meta phrases
- `non_conversation_opening`: first utterance looks like narration, chapter text, or video intro
- `commentary_turns`: file contains therapist commentary/teaching rather than dialogue

Additional check:
- `multiple_patient_candidates`: heuristic search for more than one distinct patient name addressed by therapist turns within a file
- Stricter result: `train/1.csv` (`Julie`, `Derek`) and `train/37.csv` (`Steph`, `Tom`) were flagged as multi-patient files
- Limitation: this heuristic can still miss files where patient identity changes are introduced indirectly rather than through therapist vocatives/address patterns.

## Train (33 files)

- `train/1.csv`: artifact_keywords: in this vignette, this vignette, chapter , core competency, act-consistent, act-inconsistent; non_conversation_opening; commentary_turns (7); multiple_patient_identities (Julie, Derek) - fixed and split
- `train/105.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/139.csv`: artifact_keywords: dr. grande - no issue
- `train/14.csv`: artifact_keywords: chapter ; non_conversation_opening; commentary_turns (1) - fixed
- `train/142.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/175.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/193.csv`: artifact_keywords: in this third and final section, dr. grande - fixed
- `train/224.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/228.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/235.csv`: artifact_keywords: licensed marriage and family therapist, emma mcadam; missing_role (no P turns) - removed file
- `train/238.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/242.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/245.csv`: artifact_keywords: part one of this video, dr. grande - fixed
- `train/256.csv`: artifact_keywords: dr. grande - no issue
- `train/259.csv`: artifact_keywords: dr. grande - fixed
- `train/26.csv`: artifact_keywords: chapter , core competency, act-consistent; non_conversation_opening; commentary_turns (3) - fixed
- `train/261.csv`: artifact_keywords: dr. grande; missing_role (no P turns) - fixed
- `train/262.csv`: artifact_keywords: chapter , core competency; non_conversation_opening; commentary_turns (2) - fixed
- `train/263.csv`: missing_role (no P turns) - removed
- `train/268.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/273.csv`: artifact_keywords: dr. grande; missing_role (no P turns) - removed
- `train/288.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/296.csv`: missing_role (no P turns) - removed
- `train/317.csv`: missing_role (no P turns) - removed
- `train/37.csv`: multiple_patient_identities (Steph, Tom) - removed
- `train/4.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - removed
- `train/48.csv`: artifact_keywords: chapter , act-consistent; non_conversation_opening; commentary_turns (3) - fixed
- `train/63.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/64.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - removed
- `train/72.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `train/73.csv`: commentary_turns (1) - fixed
- `train/75.csv`: artifact_keywords: subscribe to my channel, video useful, dr. grande; non_conversation_opening - fixed
- `train/87.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed

## Val (5 files)

- `val/116.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `val/136.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `val/151.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `val/174.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `val/3.csv`: artifact_keywords: core competency; commentary_turns (1) - fixed

## Test (4 files)

- `test/161.csv`: missing_role (no P turns) - removed
- `test/260.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `test/51.csv`: artifact_keywords: subscribe to my channel, thanks for watching, video useful, dr. grande; non_conversation_opening - fixed
- `test/7.csv`: missing_role (no P turns) - removed
