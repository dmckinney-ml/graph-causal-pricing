"""
Plot ablation study results: compare estimated effects across 5 model variants.

Usage:
    cd graph-causal-pricing
    python experiments/ablations/plot_ablations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

ABLATION_LABELS = {
    "full_model":      "Full model",
    "binary_weights":  "Binary weights",
    "no_embeddings":   "No embeddings",
    "sub_only":        "Substitutes only",
    "comp_only":       "Complements only",
}
ABLATION_ORDER = list(ABLATION_LABELS.keys())

EFFECT_TITLES = {
    "direct":       "Direct Effect",
    "sub_spillover": "Substitute Spillover",
    "comp_spillover": "Complement Spillover",
}
EFFECT_ORDER = ["direct", "sub_spillover", "comp_spillover"]

# For sub_only/comp_only, the zeroed-out effects are structural (not estimated).
STRUCTURAL_ZEROS = {
    ("sub_only",  "comp_spillover"),
    ("comp_only", "sub_spillover"),
}


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "tables" / "ablation_results.csv")

    palette = sns.color_palette("muted", n_colors=len(ABLATION_ORDER))
    color_map = dict(zip(ABLATION_ORDER, palette))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, effect in zip(axes, EFFECT_ORDER):
        sub = df[df["effect"] == effect].copy()
        sub["_order"] = sub["ablation"].map({k: i for i, k in enumerate(ABLATION_ORDER)})
        sub = sub.sort_values("_order")

        y_pos = np.arange(len(sub))
        labels = [ABLATION_LABELS[a] for a in sub["ablation"]]

        for i, (_, row) in enumerate(sub.iterrows()):
            ablation = row["ablation"]
            is_zero = (ablation, effect) in STRUCTURAL_ZEROS

            if is_zero:
                # Structural zero — hatched gray bar, no error bars
                ax.barh(
                    i, 0.0001,  # near-zero width just to show the bar slot
                    color="lightgray", edgecolor="gray", linewidth=0.5,
                    hatch="///", height=0.6,
                )
                ax.text(
                    0.0, i, "  (excluded)", va="center", ha="left",
                    fontsize=8.5, color="gray", style="italic",
                )
            else:
                xerr_lo = row["ATE"] - row["CI_lower"]
                xerr_hi = row["CI_upper"] - row["ATE"]
                ax.barh(
                    i, row["ATE"],
                    xerr=[[max(xerr_lo, 0)], [max(xerr_hi, 0)]],
                    color=color_map[ablation],
                    edgecolor="black", linewidth=0.5,
                    capsize=4, height=0.6,
                    error_kw={"elinewidth": 1.2, "ecolor": "black"},
                )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("ATE (log units sold)", fontsize=9.5)
        ax.set_title(EFFECT_TITLES[effect], fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    # Highlight the "no_embeddings" bar with a note on the direct panel
    # no_embeddings is at y_pos index 2 in data coordinates
    axes[0].annotate(
        "+16% vs full model",
        xy=(0.0431, 2),
        xytext=(0.055, 1.3),
        fontsize=8, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
        va="center",
    )

    fig.suptitle(
        "Ablation Study: Effect Estimates Across Model Variants",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = ROOT / "results" / "figures" / "ablation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
