"""Diverging stacked horizontal bar chart of RESPAIR CARE label distributions."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/RESPAIR")
OUT_DIR = Path("/home/asbahk/EMNLP_FINAL/Rasingan/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["NJ", "WE", "RA", "AL", "RF", "SA"]
DIM_LABELS = {
    "NJ": "Non-Judgmental Language",
    "WE": "Warmth & Encouragement",
    "RA": "Respect for Autonomy",
    "AL": "Active Listening",
    "RF": "Reflecting Feelings",
    "SA": "Situational Appropriateness",
}
CLASSES = [-2, -1, 0, 1, 2]
CLASS_LABELS = {-2: "$-2$", -1: "$-1$", 0: "$0$", 1: "$+1$", 2: "$+2$"}
COLORS = {-2: "#d7191c", -1: "#fdae61", 0: "#cccccc", 1: "#a6d96a", 2: "#1a9641"}


def load_all() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        for f in glob.glob(str(ROOT / split / "*.csv")):
            df = pd.read_csv(f)
            df = df[df["Type"] == "T"][DIMS].dropna(how="all")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def class_pct(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for d in DIMS:
        s = df[d].dropna().round().astype(int)
        counts = s.value_counts().reindex(CLASSES, fill_value=0)
        rows[d] = 100.0 * counts / counts.sum()
    return pd.DataFrame(rows).T


def plot(pct: pd.DataFrame, out_stem: Path) -> None:
    # Sort by share of +2 (strongly-aligned) ascending -> strongest at top
    order = pct[2].sort_values(ascending=True).index.tolist()
    pct = pct.loc[order]

    fig, ax = plt.subplots(figsize=(3.45, 2.5), dpi=300)
    y = np.arange(len(order))
    h = 0.62

    left = np.zeros(len(order))
    for c in CLASSES:
        vals = pct[c].values
        ax.barh(y, vals, left=left, height=h,
                color=COLORS[c], edgecolor="white", linewidth=0.5)
        for yi, v, l in zip(y, vals, left):
            if v >= 5.0:
                ax.text(l + v / 2.0, yi, f"{v:.0f}",
                        ha="center", va="center",
                        fontsize=6.4,
                        color="white" if c in (-2, 2) else "black")
        left = left + vals

    ax.set_yticks(y)
    ax.set_yticklabels([DIM_LABELS[d] for d in order], fontsize=7.5)
    ax.set_xlabel("Share of therapist turns (%)", fontsize=7.5, labelpad=2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=6.8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CLASSES]
    labels = [CLASS_LABELS[c] for c in CLASSES]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), ncol=5,
              frameon=False, fontsize=7.2, handlelength=1.1,
              columnspacing=1.4, handletextpad=0.4,
              title="CARE ordinal label", title_fontsize=7.2)

    plt.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_all()
    pct = class_pct(df)
    print("Per-dimension class % (rows sum to 100):")
    print(pct.round(2))
    plot(pct, OUT_DIR / "respair_distribution")
    print(f"\nWrote: {OUT_DIR/'respair_distribution.pdf'}")
    print(f"Wrote: {OUT_DIR/'respair_distribution.png'}")


if __name__ == "__main__":
    main()
