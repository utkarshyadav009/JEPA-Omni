"""figures/make_baseline_1545_figure.py — R@1 grouped bar chart for docs/BASELINE_1545.md.

Reads the same numbers that are in docs/BASELINE_1545.md / docs/BASELINE_1545_PROVENANCE.json
(hardcoded here for a static one-shot figure, values must match the provenance JSON exactly).
Renders figures/baseline_1545.png + .pdf, 300dpi, IEEE-style, colourblind-safe palette
(validated pair: blue #2a78d6 / orange #eb6834, dataviz skill categorical slots 1-2).

Usage:
    conda run -n jepa-omni python figures/make_baseline_1545_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── data (must match docs/BASELINE_1545.md) ─────────────────────────────────
# order: Ours first (highlighted), then baselines by descending max(a->v, v->a)
models = [
    "Ours\n(m2_run2)", "ImageBind", "EquiAV", "CAV-MAE", "LanguageBind",
    "Wav2CLIP", "CAV-MAE\nSync‡", "AVSiam†", "AudioCLIP",
]
a2v = [53.27, 29.70, 24.80, 12.23, 7.57, 5.35, 2.02, 2.48, 0.20]
v2a = [53.72, 29.70, 21.80, 14.24, 10.31, 6.46, 4.90, 3.00, 0.91]
# IN-DISTRIBUTION (VGGSound in pretrain corpus) -> hatched bars
contaminated = [False, False, False, True, False, True, True, False, False]
is_ours = [True, False, False, False, False, False, False, False, False]

CHANCE = 100.0 / 1545  # 0.0647%

BLUE = "#2a78d6"    # categorical slot 1 (a->v)
ORANGE = "#eb6834"  # categorical slot 2 (v->a)
OURS_BG = "#e9f2fd"  # faint highlight band behind the "ours" group
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.edgecolor": BASELINE_AXIS,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "pdf.fonttype": 42,   # embed as real text, not curves (IEEE requirement)
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=300)

n = len(models)
x = np.arange(n)
w = 0.36

# highlight band behind the "ours" group
ax.axvspan(-0.5, 0.5, color=OURS_BG, zorder=0)

bars_a = ax.bar(x - w / 2, a2v, width=w, color=BLUE, label="Audio→Visual R@1",
                 zorder=3, edgecolor="white", linewidth=0.6)
bars_v = ax.bar(x + w / 2, v2a, width=w, color=ORANGE, label="Visual→Audio R@1",
                 zorder=3, edgecolor="white", linewidth=0.6)

# hatch the IN-DISTRIBUTION (contaminated) bars, both series
for i, c in enumerate(contaminated):
    if c:
        bars_a[i].set_hatch("///")
        bars_v[i].set_hatch("///")
        bars_a[i].set_edgecolor("#fcfcfb")
        bars_v[i].set_edgecolor("#fcfcfb")

# chance-level reference line
ax.axhline(CHANCE, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
ax.annotate(
    f"chance level ({CHANCE:.3f}%)",
    xy=(n - 1, CHANCE), xytext=(n - 1.35, 9),
    color=INK_MUTED, fontsize=7.5, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.7,
                     shrinkA=0, shrinkB=2),
)

# value labels on the "ours" bars only (selective direct labels)
for i in range(n):
    if is_ours[i]:
        ax.text(x[i] - w / 2 - 0.03, a2v[i] + 1.0, f"{a2v[i]:.1f}", ha="right", va="bottom",
                 fontsize=7.5, color=INK, fontweight="bold")
        ax.text(x[i] + w / 2 + 0.03, v2a[i] + 1.0, f"{v2a[i]:.1f}", ha="left", va="bottom",
                 fontsize=7.5, color=INK, fontweight="bold")

ax.set_xticks(x)
tick_labels = ax.set_xticklabels(models, fontsize=7.8)
tick_labels[0].set_fontweight("bold")

ax.set_ylabel("R@1 (%)", fontsize=9.5)
ax.set_ylim(0, 62)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="both", length=0)

ax.set_title(
    "Audio-Visual Retrieval R@1 on the Fixed 1,545-clip VGGSound Gallery\n"
    "(directly measured, same protocol, our code — not quoted from papers)",
    fontsize=9.5, color=INK, pad=12,
)

# legend: direction (color) + contamination (hatch)
from matplotlib.patches import Patch
hatch_patch = Patch(facecolor="white", edgecolor=INK_SECONDARY, hatch="///",
                     label="IN-DISTRIBUTION (VGGSound in pretrain corpus)")
ax.legend(
    handles=[bars_a, bars_v, hatch_patch], loc="upper right", frameon=False,
    fontsize=7.5, handlelength=1.6, handleheight=1.2, borderaxespad=0.3,
)

ax.text(0.002, 0.98,
        "† AVSiam: checkpoint loaded 670/963 keys (293 missing)\n"
        "‡ CAV-MAE Sync: released checkpoint is sha256-identical to CAV-MAE (see caveat)",
        transform=ax.transAxes, fontsize=6.3, color=INK_MUTED, va="top", ha="left")

fig.tight_layout()
fig.savefig("figures/baseline_1545.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("figures/baseline_1545.pdf", bbox_inches="tight", facecolor="white")
print("Wrote figures/baseline_1545.png + .pdf")
