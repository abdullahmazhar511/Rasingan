"""Editorial-style RESPAIR distribution figure.

No axes, no ticks, no colorbar. Each dimension is one row:
    DIMENSION NAME          [====bar====]          mean
with inline % labels on the bar, mean ordinal value as a hero number on
the right, and a single thin legend strip at the bottom.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/RESPAIR")
OUT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/figures")

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
CLASS_LBL = {-2: "$-2$", -1: "$-1$", 0: "$0$", 1: "$+1$", 2: "$+2$"}

# Muted, editorial palette (closer to FT/NYT than ColorBrewer default)
COLORS = {
    -2: "#b2182b",   # deep red
    -1: "#ef8a62",   # muted coral
     0: "#e5e5e5",   # cool grey
     1: "#9ec3a6",   # sage green
     2: "#1b6e3b",   # forest green
}


def load_pct() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        for f in glob.glob(str(ROOT / split / "*.csv")):
            df = pd.read_csv(f)
            df = df[df["Type"] == "T"][DIMS].dropna(how="all")
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    rows = {}
    for d in DIMS:
        s = full[d].dropna().round().astype(int)
        counts = s.value_counts().reindex(CLASSES, fill_value=0)
        rows[d] = 100.0 * counts / counts.sum()
    return pd.DataFrame(rows).T


def plot(pct: pd.DataFrame) -> None:
    means = (pct[CLASSES] * np.array(CLASSES)).sum(axis=1) / 100.0
    order = means.sort_values(ascending=False).index.tolist()
    pct = pct.loc[order]
    means = means.loc[order]

    n = len(order)
    fig_w, fig_h = 6.8, 0.62 * n + 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    # Layout coordinates
    label_x = 0.0          # right-edge of dim name
    bar_x0 = 0.27          # left of bar
    bar_x1 = 0.85          # right of bar
    mean_x = 0.93          # x for hero mean number
    bar_w = bar_x1 - bar_x0

    y_positions = np.linspace(0.92, 0.18, n)
    row_h = (y_positions[0] - y_positions[1]) * 0.45 if n > 1 else 0.08

    for yi, d in zip(y_positions, order):
        # Dimension name (right-aligned, before bar)
        ax.text(label_x + 0.255, yi, DIM_LABELS[d],
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=10.5, color="#222222",
                fontweight="medium")

        # Bar segments
        left = bar_x0
        for c in CLASSES:
            v = pct.loc[d, c] / 100.0
            seg_w = v * bar_w
            if seg_w <= 0:
                continue
            rect = mpatches.FancyBboxPatch(
                (left, yi - row_h / 2), seg_w, row_h,
                boxstyle="round,pad=0,rounding_size=0",
                linewidth=0, facecolor=COLORS[c],
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            if v * 100 >= 5:
                txt_color = "white" if c in (-2, 2) else "#222222"
                ax.text(left + seg_w / 2, yi, f"{v*100:.0f}",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color,
                        fontweight="semibold")
            left += seg_w

        # Hero mean number on the right
        m = means.loc[d]
        # Color: green if positive, grey if near zero, red if negative
        if m >= 1.0:
            mcolor = "#1b6e3b"
        elif m >= 0.4:
            mcolor = "#3a7d49"
        else:
            mcolor = "#6c6c6c"
        ax.text(mean_x, yi, f"{m:+.2f}",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=12.5, color=mcolor,
                fontweight="bold",
                family="DejaVu Sans")

    # Column header for the mean column
    ax.text(mean_x, 0.99, "mean",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=7.8, color="#777777",
            fontweight="medium",
            family="DejaVu Sans",
            style="italic")

    # Legend strip at the bottom
    legend_y = 0.04
    legend_x0 = bar_x0
    seg_w_leg = (bar_x1 - bar_x0) / 5.0
    for i, c in enumerate(CLASSES):
        lx = legend_x0 + i * seg_w_leg
        rect = mpatches.Rectangle(
            (lx + seg_w_leg * 0.30, legend_y - 0.008),
            seg_w_leg * 0.10, 0.018,
            linewidth=0, facecolor=COLORS[c],
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(lx + seg_w_leg * 0.45, legend_y,
                CLASS_LBL[c],
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=8.5, color="#444444")

    # Subtitle / scale note (small grey text under legend)
    ax.text(bar_x0, -0.02, "Share of therapist turns per CARE ordinal label, all splits combined.",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=7.6, color="#888888",
            style="italic")

    # Strip all spines + ticks
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    plt.tight_layout(pad=0.4)
    fig.savefig(OUT / "respair_editorial.pdf", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUT / "respair_editorial.png", bbox_inches="tight", pad_inches=0.10, dpi=300)
    plt.close(fig)


def main() -> None:
    pct = load_pct()
    plot(pct)
    print(f"wrote: {OUT/'respair_editorial.pdf'}")
    print(f"wrote: {OUT/'respair_editorial.png'}")


if __name__ == "__main__":
    main()
