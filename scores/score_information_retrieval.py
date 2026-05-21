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
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RASINGAN_PATH = os.path.dirname(SCRIPT_DIR)
FINAL_PIPELINE_PATH = os.path.join(RASINGAN_PATH, "final_pipeline")
if FINAL_PIPELINE_PATH not in sys.path:
    sys.path.insert(0, FINAL_PIPELINE_PATH)

from checklists import compute_summary  # noqa: E402

EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(RASINGAN_PATH, "evaluation_pipeline"))


def _load_status_from_session(path: str) -> Optional[Dict]:
    """Read a session_<id>.json and return its extraction_status dict (or None)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    status = data.get("extraction_status")
    if status:
        return status
    return None


def _load_status_from_progress(path: str) -> Optional[Dict]:
    """Read a supervisor progress JSON and return its extraction_status."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("extraction_status") or None


def _load_category_from_session(path: str) -> Optional[str]:
    """Read session_<id>.json and return the scenario `category` if present."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    meta = data.get("metadata") or {}
    cat = meta.get("category")
    return cat if cat else None


def _gather_sessions(sessions_dir: str) -> Dict[str, Dict]:
    """Return {session_id: {"status": extraction_status, "category": str|None}}.

    Prefer session_<id>.json files; for any session_id missing extraction_status,
    fall back to supervisor_progress/session_<id>_checklist.json. Category is
    read from session_<id>.json's metadata (None if no session file exists).
    """
    out: Dict[str, Dict] = {}

    for path in sorted(glob.glob(os.path.join(sessions_dir, "session_*.json"))):
        sid = os.path.splitext(os.path.basename(path))[0].replace("session_", "")
        status = _load_status_from_session(path)
        category = _load_category_from_session(path)
        if status:
            out[sid] = {"status": status, "category": category}

    progress_dir = os.path.join(sessions_dir, "supervisor_progress")
    if os.path.isdir(progress_dir):
        for path in sorted(glob.glob(os.path.join(progress_dir, "session_*_checklist.json"))):
            sid = os.path.basename(path).replace("session_", "").replace("_checklist.json", "")
            if sid in out:
                continue  # already captured from the session file
            status = _load_status_from_progress(path)
            if status:
                out[sid] = {"status": status, "category": None}

    return out


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

    gathered = _gather_sessions(sessions_dir)
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

    summary = {
        "model_name": model_name,
        "sessions_dir": sessions_dir,
        "n_sessions": len(per_session),
        "ir_score": round(float(np.mean(coverage_pcts)), 2),
        "ir_score_std": round(float(np.std(coverage_pcts)), 2),
        "total_items": per_session[next(iter(per_session))]["total_items"],
        "category_coverage": _aggregate_categories(per_session),
        "by_scenario_category": _aggregate_by_scenario_category(per_session, session_meta),
        "per_session": {
            sid: {
                "overall_pct": per_session[sid]["overall_pct"],
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
