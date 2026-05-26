"""Three alternative figures for RESPAIR CARE label distribution.

Produces:
  respair_heatmap.{pdf,png}
  respair_polar.{pdf,png}
  respair_stacked_v2.{pdf,png}
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/RESPAIR")
OUT = Path("/home/asbahk/EMNLP_FINAL/Rasingan/figures")
OUT.mkdir(parents=True, exist_ok=True)

DIMS = ["NJ", "WE", "RA", "AL", "RF", "SA"]
DIM_LABELS = {
    "NJ": "Non-Judgmental Lang.",
    "WE": "Warmth & Encour.",
    "RA": "Respect for Autonomy",
    "AL": "Active Listening",
    "RF": "Reflecting Feelings",
    "SA": "Situational Approp.",
}
DIM_LABELS_SHORT = {
    "NJ": "Non-Judg.",
    "WE": "Warmth",
    "RA": "Autonomy",
    "AL": "Listening",
    "RF": "Reflecting",
    "SA": "Situational",
}
CLASSES = [-2, -1, 0, 1, 2]
CLASS_LBL = {-2: "$-2$", -1: "$-1$", 0: "$0$", 1: "$+1$", 2: "$+2$"}
COLORS = {-2: "#d7191c", -1: "#fdae61", 0: "#cccccc",
          1: "#a6d96a", 2: "#1a9641"}


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


def fig_heatmap(pct: pd.DataFrame) -> None:
    order = pct[2].sort_values(ascending=False).index.tolist()
    pct = pct.loc[order]
    data = pct.values

    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    cmap = plt.get_cmap("YlGn")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(5))
    ax.set_xticklabels([CLASS_LBL[c] for c in CLASSES], fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DIM_LABELS[d] for d in order], fontsize=8)
    ax.set_xlabel("CARE ordinal label", fontsize=8.5, labelpad=4)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = f"{v:.0f}" if v >= 1 else f"{v:.1f}"
            color = "white" if v > 55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("% of therapist turns", fontsize=7.5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "respair_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "respair_heatmap.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig_polar(pct: pd.DataFrame) -> None:
    """Polar bar: one bar per dimension, length = mean ordinal value (rescaled to [0,1])."""
    means = (pct[CLASSES] * np.array(CLASSES)).sum(axis=1) / 100.0  # in [-2, 2]
    # Also compute % positive (sum of +1, +2) for the secondary ring
    pos = (pct[1] + pct[2]) / 100.0  # in [0, 1]

    order = DIMS  # keep canonical order around the circle
    means = means.loc[order]
    pos = pos.loc[order]

    n = len(order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.85

    fig = plt.figure(figsize=(3.6, 3.6), dpi=300)
    ax = fig.add_subplot(111, projection="polar")

    # Background % positive ring (light)
    ax.bar(angles, pos.values, width=width, bottom=0,
           color="#a6d96a", alpha=0.35, edgecolor="white", linewidth=0.6,
           label="\\% positive ($+1, +2$)")
    # Mean ordinal bar, normalized to [0,1] from [-2,2]
    mean_norm = (means.values + 2) / 4.0
    ax.bar(angles, mean_norm, width=width * 0.55, bottom=0,
           color="#1a9641", edgecolor="white", linewidth=0.8,
           label="Mean ordinal (norm.)")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([DIM_LABELS_SHORT[d] for d in order], fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=6)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", pad=8)
    ax.grid(color="#cccccc", linewidth=0.5, alpha=0.7)
    ax.spines["polar"].set_visible(False)

    # Annotate each mean bar with raw mean value
    for ang, m in zip(angles, means.values):
        r = (m + 2) / 4.0
        ax.text(ang, r + 0.06, f"{m:.2f}",
                ha="center", va="center", fontsize=7,
                color="#1a9641", fontweight="bold")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=7.5,
              handlelength=1.2, handletextpad=0.5, columnspacing=1.5)
    plt.tight_layout()
    fig.savefig(OUT / "respair_polar.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "respair_polar.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig_stacked_v2(pct: pd.DataFrame) -> None:
    """Polished stacked bar: tighter, neutral-class grey, % annotations >=4."""
    order = pct[2].sort_values(ascending=True).index.tolist()
    pct = pct.loc[order]

    fig, ax = plt.subplots(figsize=(3.45, 2.4), dpi=300)
    y = np.arange(len(order))
    h = 0.66

    left = np.zeros(len(order))
    for c in CLASSES:
        vals = pct[c].values
        ax.barh(y, vals, left=left, height=h,
                color=COLORS[c], edgecolor="white", linewidth=0.6)
        for yi, v, l in zip(y, vals, left):
            if v >= 4.0:
                ax.text(l + v / 2.0, yi, f"{v:.0f}",
                        ha="center", va="center", fontsize=6.6,
                        color="white" if c in (-2, 2) else "black")
        left = left + vals

    ax.set_yticks(y)
    ax.set_yticklabels([DIM_LABELS[d] for d in order], fontsize=8)
    ax.set_xlabel("Share of therapist turns (\\%)", fontsize=8, labelpad=2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CLASSES]
    labels = [CLASS_LBL[c] for c in CLASSES]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.36), ncol=5,
              frameon=False, fontsize=7.3, handlelength=1.0,
              columnspacing=1.5, handletextpad=0.4,
              title="CARE ordinal label", title_fontsize=7.5)

    plt.tight_layout()
    fig.savefig(OUT / "respair_stacked_v2.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "respair_stacked_v2.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main() -> None:
    pct = load_pct()
    print(pct.round(2))
    fig_heatmap(pct)
    fig_polar(pct)
    fig_stacked_v2(pct)
    for f in ("respair_heatmap", "respair_polar", "respair_stacked_v2"):
        print(f"wrote: {OUT/f}.pdf  {OUT/f}.png")


if __name__ == "__main__":
    main()
