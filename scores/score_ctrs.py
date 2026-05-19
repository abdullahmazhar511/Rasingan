"""Conversation-level CTRS (Clinical Therapeutic Response System) scoring,
computed directly from final_pipeline session transcripts.

For every therapist turn in each session, runs CARE inference to get
[NJ, WE, RA, AL, RF, SA], then computes CTRS-P per session:

  Per construct score: S_k = Q_k - F_k + 0.5 * SP_k
    Q_k  = mean of construct values
    F_k  = weak/neutral rate (values <= 0)
    SP_k = strong-positive rate (values == 2)

  Constructs:
    Understanding (U)              = avg(AL, RF)
    Interpersonal Effectiveness    = avg(WE, NJ)
    Collaboration                  = RA
    Technical Appropriateness      = SA

  CTRS-P = 0.35*U + 0.25*IE + 0.20*C + 0.20*TA

Per-session results + aggregate are saved to <model>/ctrs_scores.json.
Per-session CARE scores are also cached to <model>/care_scores/<session_id>.csv
so they can be reused by other scorers without re-running CARE.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_PATH = os.path.dirname(SCRIPT_DIR)
sys.path.append(RASINGAN_PATH)
from inference import CareModel, CARE_LABELS  # noqa: E402

EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_PATH, "evaluation_pipeline"))

CARE_LABEL_TO_ABBREV = {
    "Non-Judgmental Language": "NJ",
    "Warmth and Encouragement": "WE",
    "Respect for Autonomy": "RA",
    "Active Listening": "AL",
    "Reflecting Feelings": "RF",
    "Situational Appropriateness": "SA",
}
ABBREV_ORDER = ["NJ", "WE", "RA", "AL", "RF", "SA"]


def compute_construct_score(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    Q = float(np.mean(values))
    F = float(np.mean(values <= 0))
    SP = float(np.mean(values == 2))
    return {"Q": Q, "F": F, "SP": SP, "score": Q - F + 0.5 * SP}


def compute_ctrs_from_values(values: Dict[str, np.ndarray]) -> Dict:
    """values: {abbrev: np.array of per-turn scores}."""
    if any(len(v) == 0 for v in values.values()):
        return None

    U = (values["AL"] + values["RF"]) / 2.0
    IE = (values["WE"] + values["NJ"]) / 2.0
    C = values["RA"]
    TA = values["SA"]

    U_s = compute_construct_score(U)
    IE_s = compute_construct_score(IE)
    C_s = compute_construct_score(C)
    TA_s = compute_construct_score(TA)

    ctrs_p = 0.35 * U_s["score"] + 0.25 * IE_s["score"] + 0.20 * C_s["score"] + 0.20 * TA_s["score"]
    return {
        "Understanding": U_s,
        "Interpersonal_Effectiveness": IE_s,
        "Collaboration": C_s,
        "Technical_Appropriateness": TA_s,
        "CTRS_P": float(ctrs_p),
        "n_turns": int(len(values["NJ"])),
    }


def _extract_therapist_pairs(transcript: List[Dict]) -> List[Tuple[str, str, int]]:
    """Return [(context, utterance, turn_number), ...] for every therapist message.

    Context is the prior transcript joined as "Role: message\n…".
    """
    pairs = []
    history_lines: List[str] = []
    for entry in transcript:
        role = entry.get("role", "")
        msg = (entry.get("message") or "").strip()
        if not msg:
            continue
        if role == "therapist":
            context = "\n".join(history_lines).strip()
            pairs.append((context, msg, entry.get("turn", len(pairs))))
        history_lines.append(f"{'Therapist' if role == 'therapist' else 'Patient'}: {msg}")
    return pairs


def _care_for_session(care_model: CareModel, transcript: List[Dict],
                      batch_size: int) -> Optional[pd.DataFrame]:
    pairs = _extract_therapist_pairs(transcript)
    if not pairs:
        return None
    contexts = [p[0] for p in pairs]
    utterances = [p[1] for p in pairs]
    turns = [p[2] for p in pairs]

    results = care_model.batch_predict(contexts, utterances,
                                       batch_size=batch_size, include_analysis=False)

    rows = []
    for turn, utt, res in zip(turns, utterances, results):
        row = {"turn": turn, "utterance": utt}
        for label in CARE_LABELS:
            row[CARE_LABEL_TO_ABBREV[label]] = res[label]
        rows.append(row)
    return pd.DataFrame(rows)


def score_model(model_name: str, sessions_dir: str, output_dir: str,
                care_model: CareModel, batch_size: int) -> Dict:
    if not os.path.isdir(sessions_dir):
        print(f"[{model_name}] No sessions dir at {sessions_dir}")
        return {}

    session_files = sorted(glob.glob(os.path.join(sessions_dir, "session_*.json")))
    if not session_files:
        print(f"[{model_name}] No session_*.json files in {sessions_dir}")
        return {}

    care_cache_dir = os.path.join(output_dir, "care_scores")
    os.makedirs(care_cache_dir, exist_ok=True)

    conv_results: Dict[str, Dict] = {}
    print(f"[{model_name}] Scoring {len(session_files)} sessions via CARE → CTRS")

    for path in session_files:
        sid = os.path.splitext(os.path.basename(path))[0].replace("session_", "")
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [{sid}] skipped (read error: {e})")
            continue

        transcript = data.get("transcript", [])
        cache_path = os.path.join(care_cache_dir, f"{sid}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            df = _care_for_session(care_model, transcript, batch_size)
            if df is None or df.empty:
                print(f"  [{sid}] no therapist turns, skipped")
                continue
            df.to_csv(cache_path, index=False)

        values = {abbr: pd.to_numeric(df[abbr], errors="coerce").dropna().values
                  for abbr in ABBREV_ORDER if abbr in df.columns}
        if any(len(v) == 0 for v in values.values()) or len(values) < len(ABBREV_ORDER):
            print(f"  [{sid}] missing CARE dimensions, skipped")
            continue

        res = compute_ctrs_from_values(values)
        if res is not None:
            conv_results[sid] = res
            print(f"  [{sid}] CTRS-P={res['CTRS_P']:.3f}  (n_turns={res['n_turns']})")

    if not conv_results:
        print(f"[{model_name}] No valid sessions for CTRS scoring")
        return {}

    # Aggregate
    agg = {}
    for construct in ["Understanding", "Interpersonal_Effectiveness", "Collaboration", "Technical_Appropriateness"]:
        for metric in ["Q", "F", "SP", "score"]:
            values = [r[construct][metric] for r in conv_results.values()]
            agg[f"avg_{construct}_{metric}"] = float(np.mean(values))
            agg[f"std_{construct}_{metric}"] = float(np.std(values))

    ctrs_p_values = [r["CTRS_P"] for r in conv_results.values()]
    agg["avg_CTRS_P"] = float(np.mean(ctrs_p_values))
    agg["std_CTRS_P"] = float(np.std(ctrs_p_values))
    agg["avg_n_turns"] = float(np.mean([r["n_turns"] for r in conv_results.values()]))

    output = {
        "model_name": model_name,
        "sessions_dir": sessions_dir,
        "n_sessions": len(conv_results),
        "aggregate": agg,
        "per_session": conv_results,
    }

    out_path = os.path.join(output_dir, "ctrs_scores.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}\n  {model_name}\n{'=' * 60}")
    print(f"Sessions scored: {len(conv_results)}")
    print(f"Mean CTRS-P: {agg['avg_CTRS_P']:.4f} ± {agg['std_CTRS_P']:.4f}")
    print(f"  Understanding:                 {agg['avg_Understanding_score']:.4f}")
    print(f"  Interpersonal Effectiveness:   {agg['avg_Interpersonal_Effectiveness_score']:.4f}")
    print(f"  Collaboration:                 {agg['avg_Collaboration_score']:.4f}")
    print(f"  Technical Appropriateness:     {agg['avg_Technical_Appropriateness_score']:.4f}")
    print(f"Saved to: {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compute CTRS from final_pipeline sessions (runs CARE on therapist turns, then CTRS-P per session)."
    )
    parser.add_argument("--eval-root", default=EVAL_ROOT)
    parser.add_argument("--model-name", default=None,
                        help="Model name (folder under eval-root). If omitted, processes all models with a sessions/ dir.")
    parser.add_argument("--sessions-dir", default=None,
                        help="Override the sessions directory (default: <eval-root>/<model>/sessions).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write ctrs_scores.json and care_scores/ (default: <eval-root>/<model>).")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.sessions_dir or args.model_name:
        if args.model_name and not args.sessions_dir:
            args.sessions_dir = os.path.join(args.eval_root, args.model_name, "sessions")
        if not args.output_dir:
            args.output_dir = (os.path.join(args.eval_root, args.model_name)
                               if args.model_name else os.path.dirname(args.sessions_dir.rstrip("/")))
        targets = [(args.model_name or os.path.basename(args.output_dir), args.sessions_dir, args.output_dir)]
    else:
        targets = []
        for d in sorted(glob.glob(os.path.join(args.eval_root, "*"))):
            if os.path.isdir(os.path.join(d, "sessions")):
                targets.append((os.path.basename(d), os.path.join(d, "sessions"), d))
        if not targets:
            print(f"No model directories with sessions/ found under {args.eval_root}")
            return

    os.chdir(RASINGAN_PATH)  # CareModel uses relative paths
    print("Loading CARE Model (slow)…")
    care_model = CareModel()

    for name, sessions_dir, output_dir in targets:
        try:
            score_model(name, sessions_dir, output_dir, care_model, args.batch_size)
        except Exception as e:
            print(f"[{name}] CTRS scoring failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
