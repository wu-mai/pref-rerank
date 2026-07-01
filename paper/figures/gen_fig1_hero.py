"""Fig 1 (Hero): nDCG@10 across 6 systems with 95% CI error bars.

Visual message:
- Bilinear variants (full / no_user_bias / random_schema / classic) >> baseline
- linear_only = baseline (w^T u contributes nothing)
- All bilinear variants ~ each other (representation does not matter)
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import COLORS, FIG_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_eval(name):
    with open(f"results/r100k_eval_v2_{name}.json") as f:
        return json.load(f)


systems = [
    ("Baseline\n(user-agnostic)", "baseline_only", None),
    ("linear_only\n($w^T u$)", "linear_only", None),
    ("full\n($w^T u + h^T A u$)", "full", None),
    ("no\\_user\\_bias\n($h^T A u$)", "no_user_bias", None),
    ("random\\_schema\n($h^T A u$, rand)", "random_schema", None),
    ("classic\n($h^T A u$, behav.)", "classic", None),
]

# Baseline is the same across all eval files (BM25+queryflow, user-agnostic)
baseline_data = load_eval("no_user_bias")
baseline_ndcg = baseline_data["baseline"]["ndcg10"]

labels = []
means = []
ci_lo = []
ci_hi = []
for label, variant, _ in systems:
    if variant == "baseline_only":
        labels.append(label)
        means.append(baseline_ndcg)
        ci_lo.append(0.0)
        ci_hi.append(0.0)
    else:
        d = load_eval(variant)
        labels.append(label)
        means.append(d["personalized"]["ndcg10"])
        pb = d["paired_bootstrap_ci"]
        ci_lo.append(pb["ci_low"])
        ci_hi.append(pb["ci_high"])

# Convert to arrays; error bars are |mean - ci|, clamped to >= 0
means = np.array(means)
err_lo = np.maximum(means - np.array(ci_lo), 0)
err_hi = np.maximum(np.array(ci_hi) - means, 0)

# Colors: baseline & linear_only grey (no contribution); bilinear variants colored
bar_colors = [COLORS[0], COLORS[0], COLORS[4], COLORS[2], COLORS[1], COLORS[5]]

fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.2))
x = np.arange(len(labels))
bars = ax.bar(x, means, color=bar_colors, edgecolor="black", linewidth=0.6, width=0.7)
ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3, capthick=1, elinewidth=1)

# Annotate bars with values
for i, (bar, m) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width() / 2, m + err_hi[i] + 0.015,
            f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)

# Baseline reference line
ax.axhline(baseline_ndcg, color=COLORS[0], linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(5.4, baseline_ndcg + 0.005, "baseline", fontsize=7.5, color=COLORS[0], ha="right")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Session-level nDCG@10")
ax.set_ylim(0.30, 0.72)

# Legend explaining color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS[0], edgecolor="black", label="no bilinear ($s_{base}$ or $w^T u$ only)"),
    Patch(facecolor=COLORS[4], edgecolor="black", label="bilinear $h^T A u$"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=8)

plt.tight_layout()
out = f"{FIG_DIR}/fig1_hero_ndcg.pdf"
plt.savefig(out)
print(f"Saved: {out}")
