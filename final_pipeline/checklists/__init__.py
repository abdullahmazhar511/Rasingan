"""Clinical screening checklists package.

Each checklist module (phq9, gad7) exposes:
    CHECKLIST       - dict of {category_key: {"name": ..., "items": [...]}}
    PRIORITY_ORDER  - list of category keys, highest priority first
    SCORING         - dict with frequency_options + interpretation bands

This package also provides status-tracking helpers (parameterized over a
checklist dict) so the supervisor can track coverage of any checklist.
"""

from typing import Dict, List, Optional

from . import phq9, gad7


def _merge_checklists(*checklists: Dict) -> Dict:
    merged = {}
    for cl in checklists:
        for key, val in cl.items():
            if key in merged:
                merged[f"{key}_{len(merged)}"] = val
            else:
                merged[key] = val
    return merged


CHECKLISTS: Dict[str, Dict] = {
    "phq9": {
        "checklist": phq9.CHECKLIST,
        "priority_order": phq9.PRIORITY_ORDER,
        "scoring": phq9.SCORING,
        "instrument": "PHQ-9 (depression screening)",
    },
    "gad7": {
        "checklist": gad7.CHECKLIST,
        "priority_order": gad7.PRIORITY_ORDER,
        "scoring": gad7.SCORING,
        "instrument": "GAD-7 (anxiety screening)",
    },
    "combined": {
        "checklist": _merge_checklists(phq9.CHECKLIST, gad7.CHECKLIST),
        "priority_order": [*phq9.PRIORITY_ORDER, *gad7.PRIORITY_ORDER],
        "scoring": None,
        "instrument": "PHQ-9 + GAD-7 (depression and anxiety screening)",
    },
}


def get_checklist(name: str = "combined") -> Dict:
    """Look up a checklist bundle by name. Raises ValueError if unknown."""
    if name not in CHECKLISTS:
        raise ValueError(
            f"Unknown checklist '{name}'. Available: {list(CHECKLISTS)}"
        )
    return CHECKLISTS[name]


def init_status(checklist: Dict) -> Dict:
    """Initialize tracking status for a checklist — all items start uncovered."""
    return {
        category: {
            "name": data["name"],
            "items": {
                item: {"covered": False, "turn_covered": None, "evidence": None}
                for item in data["items"]
            },
        }
        for category, data in checklist.items()
    }


def compute_summary(status: Dict) -> Dict:
    """Compute coverage stats over a status dict."""
    total = covered = 0
    cats = {}
    for category, data in status.items():
        cat_total = len(data["items"])
        cat_cov = sum(1 for v in data["items"].values() if v["covered"])
        total += cat_total
        covered += cat_cov
        cats[category] = {
            "name": data["name"],
            "covered": cat_cov,
            "total": cat_total,
            "pct": round(cat_cov / cat_total * 100, 1) if cat_total else 0,
        }
    return {
        "total_items": total,
        "covered_items": covered,
        "overall_pct": round(covered / total * 100, 1) if total else 0,
        "categories": cats,
        "pending_categories": [
            c["name"] for c in cats.values() if c["covered"] < c["total"]
        ],
    }


def format_status(status: Dict) -> str:
    """Render the checklist as a human-readable progress block."""
    text = ""
    for _, data in status.items():
        cov = sum(1 for v in data["items"].values() if v["covered"])
        total = len(data["items"])
        text += f"\n{data['name']} [{cov}/{total}]:\n"
        for item, s in data["items"].items():
            mark = "✓" if s["covered"] else "☐"
            text += f"  {mark} {item}\n"
    return text


def get_pending_items(status: Dict) -> List[str]:
    """Flat list of still-uncovered items, prefixed with their category name."""
    return [
        f"[{data['name']}] {item}"
        for _, data in status.items()
        for item, s in data["items"].items()
        if not s["covered"]
    ]


def get_next_priority(status: Dict, priority_order: List[str]) -> Optional[str]:
    """Pick the next pending item, respecting the supplied priority order."""
    seen = set()
    for category in priority_order:
        seen.add(category)
        data = status.get(category, {})
        pending = [item for item, s in data.get("items", {}).items() if not s["covered"]]
        if pending:
            return f"[{data['name']}] Focus on: {pending[0]}"
    # any remaining categories not listed in priority_order
    for category, data in status.items():
        if category in seen:
            continue
        pending = [item for item, s in data["items"].items() if not s["covered"]]
        if pending:
            return f"[{data['name']}] Focus on: {pending[0]}"
    return None


__all__ = [
    "CHECKLISTS",
    "get_checklist",
    "init_status",
    "compute_summary",
    "format_status",
    "get_pending_items",
    "get_next_priority",
    "phq9",
    "gad7",
]
