"""Information Retrieval score — how much of the clinical screening checklist
(PHQ-9 / GAD-7 / combined) the model's therapist extracted during a session.

The final_pipeline already judges coverage turn-by-turn via the supervisor
(agent_2), so this script just aggregates what's already been written to disk:

  • Per-session: each `sessions/session_<id>.json` carries `extraction_status`
    + `extraction_summary` (added by pipeline.py).
  • Fallback: `sessions/supervisor_progress/session_<id>_checklist.json` written
    directly by agent_2.

Aggregates across all sessions for the model and writes ir_score.json.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_PATH = os.path.dirname(SCRIPT_DIR)
FINAL_PIPELINE_PATH = os.path.join(RASINGAN_PATH, "final_pipeline")
if FINAL_PIPELINE_PATH not in sys.path:
    sys.path.insert(0, FINAL_PIPELINE_PATH)

from checklists import compute_summary  # noqa: E402

EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_PATH, "evaluation_pipeline"))


def _load_session(path: str) -> Optional[Dict]:
    """Read a JSON file once and return the parsed dict (or None on error)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _gather_sessions(sessions_dir: str) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """Return ({session_id: {"status": extraction_status, "category": str|None}},
              {skip_reason: count}).

    Prefer session_<id>.json files; for any session_id missing extraction_status,
    fall back to supervisor_progress/session_<id>_checklist.json. Category is
    read from session_<id>.json's metadata (None if no session file exists).
    Skipped sessions are counted and surfaced — silent drops here hide cases
    where agent_2 crashed mid-session.
    """
    out: Dict[str, Dict] = {}
    skipped: Dict[str, int] = {}

    def _bump(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in sorted(glob.glob(os.path.join(sessions_dir, "session_*.json"))):
        sid = os.path.splitext(os.path.basename(path))[0].replace("session_", "")
        data = _load_session(path)
        if data is None:
            print(f"  [{sid}] session file unreadable, skipped")
            _bump("read_error")
            continue
        status = data.get("extraction_status")
        category = (data.get("metadata") or {}).get("category") or None
        if not status:
            print(f"  [{sid}] no extraction_status in session, skipped")
            _bump("no_extraction_status")
            continue
        out[sid] = {"status": status, "category": category}

    progress_dir = os.path.join(sessions_dir, "supervisor_progress")
    if os.path.isdir(progress_dir):
        for path in sorted(glob.glob(os.path.join(progress_dir, "session_*_checklist.json"))):
            sid = os.path.basename(path).replace("session_", "").replace("_checklist.json", "")
            if sid in out:
                continue  # already captured from the session file
            data = _load_session(path)
            if data is None:
                print(f"  [{sid}] progress file unreadable, skipped")
                _bump("read_error")
                continue
            status = data.get("extraction_status")
            if not status:
                print(f"  [{sid}] no extraction_status in progress, skipped")
                _bump("no_extraction_status")
                continue
            out[sid] = {"status": status, "category": None}

    return out, skipped


def _aggregate_categories(per_session: Dict[str, Dict]) -> Dict[str, Dict]:
    """Mean coverage % per category across sessions."""
    buckets: Dict[str, Dict] = {}
    for summary in per_session.values():
        for cat_key, cat in summary["categories"].items():
            slot = buckets.setdefault(cat_key, {"name": cat["name"], "pcts": [], "total": cat["total"]})
            slot["pcts"].append(cat["pct"])
    return {
        k: {
            "name": v["name"],
            "total_items": v["total"],
            "mean_coverage_pct": round(float(np.mean(v["pcts"])), 2),
            "std_coverage_pct": round(float(np.std(v["pcts"])), 2),
        }
        for k, v in buckets.items()
    }


def _aggregate_by_scenario_category(per_session: Dict[str, Dict],
                                    session_meta: Dict[str, Dict]) -> Dict[str, Dict]:
    """Group sessions by their scenario `category` (e.g. anxiety vs depression)
    and compute mean overall IR + per-checklist-category coverage within each
    group. Sessions without a category land under "(uncategorized)".
    """
    buckets: Dict[str, List[Dict]] = {}
    for sid, summary in per_session.items():
        cat = (session_meta.get(sid) or {}).get("category") or "(uncategorized)"
        buckets.setdefault(cat, []).append(summary)

    out: Dict[str, Dict] = {}
    for cat, summaries in buckets.items():
        overall = [s["overall_pct"] for s in summaries]
        # Per-checklist-category coverage within this scenario category
        cat_breakdown: Dict[str, List[float]] = {}
        cat_names: Dict[str, str] = {}
        for s in summaries:
            for ck, cv in s["categories"].items():
                cat_breakdown.setdefault(ck, []).append(cv["pct"])
                cat_names[ck] = cv["name"]
        per_checklist_cat = {
            ck: {
                "name": cat_names[ck],
                "mean_coverage_pct": round(float(np.mean(vs)), 2),
                "std_coverage_pct": round(float(np.std(vs)), 2),
            }
            for ck, vs in cat_breakdown.items()
        }
        out[cat] = {
            "n_sessions": len(summaries),
            "ir_score": round(float(np.mean(overall)), 2),
            "ir_score_std": round(float(np.std(overall)), 2),
            "checklist_category_coverage": per_checklist_cat,
        }
    return out


def score_sessions_dir(sessions_dir: str, output_path: Optional[str] = None,
                       model_name: Optional[str] = None) -> Dict:
    if not os.path.isdir(sessions_dir):
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    gathered, skipped = _gather_sessions(sessions_dir)
    if not gathered:
        print(f"[IR] No sessions with extraction_status found under {sessions_dir}")
        return {}

    statuses = {sid: g["status"] for sid, g in gathered.items()}
    session_meta = {sid: {"category": g["category"]} for sid, g in gathered.items()}

    per_session: Dict[str, Dict] = {
        sid: compute_summary(st)
        for sid, st in tqdm(statuses.items(), desc=f"[{model_name or 'IR'}] aggregate", unit="session")
    }
    coverage_pcts = [s["overall_pct"] for s in per_session.values()]

    # total_items can legitimately vary by checklist (e.g. PHQ-9 vs GAD-7).
    # Surface the full distribution instead of pretending the first session is canonical.
    item_counts = sorted({int(s["total_items"]) for s in per_session.values()})
    total_items_repr: object = item_counts[0] if len(item_counts) == 1 else item_counts

    # IR fraction in [0, 1] — matches the training reward signal exactly
    # (faith.py: ir_frac = overall_pct / 100.0). overall_pct is already 0-100.
    ir_frac_values = [p / 100.0 for p in coverage_pcts]

    summary = {
        "model_name": model_name,
        "sessions_dir": sessions_dir,
        "n_sessions": len(per_session),
        "n_sessions_skipped": skipped,
        "ir_score": round(float(np.mean(coverage_pcts)), 2),
        "ir_score_std": round(float(np.std(coverage_pcts)), 2),
        # ir_frac mirrors the per-session ir_frac the training reward uses.
        "ir_frac_mean": round(float(np.mean(ir_frac_values)), 4),
        "ir_frac_std": round(float(np.std(ir_frac_values)), 4),
        "total_items": total_items_repr,
        "category_coverage": _aggregate_categories(per_session),
        "by_scenario_category": _aggregate_by_scenario_category(per_session, session_meta),
        "per_session": {
            sid: {
                "overall_pct": per_session[sid]["overall_pct"],
                "ir_frac": per_session[sid]["overall_pct"] / 100.0,
                "total_items": per_session[sid]["total_items"],
                "category": session_meta[sid]["category"],
            }
            for sid in per_session
        },
    }

    if output_path is None:
        output_path = os.path.join(os.path.dirname(sessions_dir.rstrip("/")), "ir_score.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    tag = f"[{model_name}] " if model_name else ""
    if skipped:
        print(f"{tag}Sessions skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    print(f"\n{tag}IR coverage: {summary['ir_score']:.2f}% (±{summary['ir_score_std']:.2f}) "
          f"across {summary['n_sessions']} sessions, {summary['total_items']} checklist items")
    for cat in summary["category_coverage"].values():
        print(f"  • {cat['name']}: {cat['mean_coverage_pct']:.2f}%")

    # Per-scenario-category split (e.g. anxiety vs depression)
    by_scn = summary.get("by_scenario_category", {})
    if by_scn and not (len(by_scn) == 1 and "(uncategorized)" in by_scn):
        print(f"\n{tag}IR by scenario category:")
        for scn_cat, stats in by_scn.items():
            print(f"  [{scn_cat}] n={stats['n_sessions']}  "
                  f"IR={stats['ir_score']:.2f}% (±{stats['ir_score_std']:.2f})")

    print(f"{tag}Saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the checklist coverage already judged by agent_2 during inference."
    )
    parser.add_argument("--sessions-dir", default=None,
                        help="Directory holding session_<id>.json files (and optionally supervisor_progress/). "
                             "If omitted, derives from --eval-root and --model-name.")
    parser.add_argument("--eval-root", default=EVAL_ROOT,
                        help=f"Evaluation root directory (default: {EVAL_ROOT})")
    parser.add_argument("--model-name", default=None,
                        help="Model name (folder under eval-root). Implies sessions-dir=<eval-root>/<model>/sessions.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <model_dir>/ir_score.json or sibling of sessions dir).")
    args = parser.parse_args()

    if args.sessions_dir:
        targets: List[Dict] = [{"name": args.model_name, "dir": args.sessions_dir, "out": args.output}]
    elif args.model_name:
        sessions_dir = os.path.join(args.eval_root, args.model_name, "sessions")
        out = args.output or os.path.join(args.eval_root, args.model_name, "ir_score.json")
        targets = [{"name": args.model_name, "dir": sessions_dir, "out": out}]
    else:
        # all models under eval-root with a sessions/ dir
        targets = []
        for d in sorted(glob.glob(os.path.join(args.eval_root, "*"))):
            if os.path.isdir(os.path.join(d, "sessions")):
                name = os.path.basename(d)
                targets.append({
                    "name": name,
                    "dir": os.path.join(d, "sessions"),
                    "out": os.path.join(d, "ir_score.json"),
                })
        if not targets:
            print(f"No model directories with sessions/ found under {args.eval_root}")
            return

    for t in targets:
        try:
            score_sessions_dir(t["dir"], output_path=t["out"], model_name=t["name"])
        except Exception as e:
            print(f"[{t.get('name')}] IR scoring failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
