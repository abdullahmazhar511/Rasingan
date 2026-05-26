"""Publication-quality plot of validation reward on faith_dataset across training steps."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
OUT_PDF = HERE / "faith_reward.pdf"
OUT_PNG = HERE / "faith_reward.png"

RUNS = [
    {
        "csv": HERE / "wandb_export_2026-05-25T17_52_01.698+05_30.csv",
        "label": "Qwen3-4B",
        "color": "#1f4e79",
        "marker": "o",
    },
    {
        "csv": HERE / "wandb_export_2026-05-25T17_59_18.434+05_30.csv",
        "label": "Mistral-8B",
        "color": "#c0392b",
        "marker": "s",
    },
    {
        "csv": HERE / "wandb_export_2026-05-25T18_01_12.884+05_30.csv",
        "label": "Llama-3.2-1B",
        "color": "#2e7d32",
        "marker": "^",
    },
]


def load(path):
    steps, mean, lo, hi = [], [], [], []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            steps.append(int(row[0]))
            mean.append(float(row[1]))
            lo.append(float(row[2]))
            hi.append(float(row[3]))
    return steps, mean, lo, hi


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)

all_lo, all_hi, all_steps = [], [], []
for run in RUNS:
    steps, mean, lo, hi = load(run["csv"])
    ax.plot(steps, mean, marker=run["marker"], markersize=4.5,
            color=run["color"], markerfacecolor="white",
            markeredgewidth=1.2, label=run["label"])
    if any(h > m or l < m for h, m, l in zip(hi, mean, lo)):
        ax.fill_between(steps, lo, hi, color=run["color"],
                        alpha=0.15, linewidth=0)
    all_lo += lo
    all_hi += hi
    all_steps += steps

ax.set_title("Faith Dataset: Validation Reward vs. Training Step", pad=6)
ax.set_xlabel("Training Step")
ax.set_ylabel("Validation Reward")

ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.xaxis.set_major_locator(MultipleLocator(80))
ax.set_xlim(min(all_steps) - 20, max(all_steps) + 20)

ymin, ymax = min(all_lo), max(all_hi)
pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.002
ax.set_ylim(ymin - pad, ymax + pad)

ax.legend(frameon=False, loc="lower right")

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08)
print(f"Wrote {OUT_PDF} and {OUT_PNG}")
