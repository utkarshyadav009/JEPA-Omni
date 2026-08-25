#!/usr/bin/env python3
"""
Dissertation figure generator for BMO/JEPA-Omni.

RULE: every numeric value plotted here is either (a) copy-typed from a real file in this
repo (path given in the SOURCE comment above each data block) or (b) a trivial arithmetic
derivation of such values (e.g. an average of two directions, or 1/latency for a rate),
which is noted explicitly where it occurs. Nothing is interpolated, smoothed, or guessed.
Where the brief's requested number could not be found anywhere in this repo, the figure
uses the real number found instead and the substitution is logged to
figures/FIGURE_MANIFEST.md, or the figure is skipped outright (search this file for SKIPPED).

Run: conda activate jepa-omni && python figures/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.ticker as mticker
import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------------------
# IEEE-ish style: serif fonts, small readable sizes, colourblind + greyscale safe palette
# (Okabe-Ito), consistent hatches so series are distinguishable without colour.
# ----------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colourblind-safe palette
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_VERM   = "#D55E00"
C_PINK   = "#CC79A7"
C_YELLOW = "#F0E442"
C_SKY    = "#56B4E9"
C_GREY   = "#999999"
C_BLACK  = "#000000"

HATCHES = ["", "///", "xxx", "...", "\\\\\\", "ooo", "+++", "---"]

SINGLE_W = 3.5   # in
DOUBLE_W = 7.16  # in


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png / {name}.pdf")


def chance_line(ax, y, label, x0=None, x1=None, **kw):
    kw.setdefault("color", C_BLACK)
    kw.setdefault("linestyle", "--")
    kw.setdefault("linewidth", 0.8)
    ax.axhline(y, **kw)
    ax.text(ax.get_xlim()[1] if x1 is None else x1, y, f" {label}",
            va="center", ha="left", fontsize=6.5)


# ==========================================================================================
# FIGURE 1 — System architecture block diagram
# SOURCE: docs/CURRENT_ARCHITECTURE.md (live production path, 2026-08-16 Jetson sync);
#         cross-checked against ARCHITECTURE.md secs 1b/2/6 (superseded proposal doc, used
#         only to confirm no contradiction on frozen/trained status).
# Param counts: MEASURED LIVE by running each module's own __main__ smoke test in the
#         jepa-omni conda env on 2026-08-18 (models/av_jepa_predictor.py -> 104,974,080;
#         models/jepa_identity_head.py -> 8,147,200; models/m4_decision_head.py -> 1,316,353;
#         models/query_predictor.py, sources=[m2,vision,ambient] -> 32.3M). SigLIP2 projection
#         trainable_text_params=590,592 for the deployed sig_runD_proj768 run is from
#         docs/METHODOLOGY_FORENSICS.md line 307 (script's own logged
#         "[ddp] trainable_text_params=" line). Excludes M3 connector and WavJEPA-nat per
#         CURRENT_ARCHITECTURE.md (both dropped from the live path 2026-08-16) and excludes
#         Qwen2.5-1.5B (no longer in the architecture, same source).
# Clock domains: tick_interval_sec=0.25 is a literal constant at
#         models/m5_streaming_loop.py:84 (StreamingConfig). stride_vision_sec=10.0 at
#         m5_streaming_loop.py:78 is the code's own perception-refresh period (0.1 Hz);
#         the measured full round latency (perception->query->thinker->speaker, no-nat) is
#         2,886 ms in ARCHITECTURE.md sec 8.2 (Jetson, MAXN_SUPER), i.e. 1/2.886s = 0.35 Hz --
#         shown as a secondary, explicitly-derived annotation since it's the closer real match
#         to a "~0.3 Hz" figure than the 0.1 Hz stride is.
# ==========================================================================================

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, text, frozen, sub="", fc=None, fontsize=6.3):
        ec = C_GREY if frozen else C_BLACK
        ls = "--" if frozen else "-"
        fc = fc or ("#f2f2f2" if frozen else "#dbe9f6")
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.6",
                            linewidth=1.1, edgecolor=ec, linestyle=ls, facecolor=fc, zorder=3)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + (1.1 if sub else 0), text, ha="center", va="center",
                fontsize=fontsize, zorder=4, wrap=True)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.3, sub, ha="center", va="center",
                     fontsize=5.6, style="italic", color="#333333", zorder=4)
        return (x, y, w, h)

    def arrow(b1, b2, side1="right", side2="left", **kw):
        def pt(b, side):
            x, y, w, h = b
            return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
                    "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]
        p1, p2 = pt(b1, side1), pt(b2, side2)
        kw.setdefault("arrowstyle", "-|>")
        kw.setdefault("mutation_scale", 8)
        kw.setdefault("linewidth", 0.9)
        kw.setdefault("color", "#333333")
        a = FancyArrowPatch(p1, p2, zorder=2, **kw)
        ax.add_patch(a)

    # --- Row 1: frozen perception encoders ---
    y_enc = 46
    b_vit = box(2, y_enc, 16, 7, "V-JEPA2 ViT-L", True, "frozen, 16 f/win")
    b_wav = box(20, y_enc, 16, 7, "WavJEPA-base", True, "frozen, ambient")
    b_sig = box(38, y_enc, 16, 7, "SigLIP2 base", True, "frozen vision tower\n92.9M / 177MiB")
    b_moo = box(56, y_enc, 16, 7, "Moonshine-base", True, "frozen, STT encoder\n(decision head only)")

    # --- Row 2: fusion / M2 ---
    y_fus = 34
    b_m2 = box(20, y_fus, 22, 7, "M2 fusion predictor\n(AVJepaPredictor)", False,
               "TRAINED  104.97M params\nRUN-2 step19000.pt", fc="#ffe3c9")
    b_ws = box(46, y_fus, 12, 7, "World-State\n1024-d", False, "not L2-normalised", fc="#eeeeee")

    arrow(b_vit, b_m2, "bottom", "top")
    arrow(b_wav, b_m2, "bottom", "top")
    arrow(b_m2, b_ws)

    # --- Row 3: specialists ---
    y_spec = 22
    b_qp = box(4, y_spec, 22, 7, "Query predictor\n(Perceiver)", False,
               "TRAINED  32.3M + 0.59M\nsig_runD_proj768", fc="#ffe3c9")
    b_id = box(28, y_spec, 20, 7, "Identity head", False,
               "TRAINED  8.15M params\nidentity_head_joint.pt", fc="#ffe3c9")
    b_dh = box(50, y_spec, 20, 7, "Decision head\n(turn-taking)", False,
               "TRAINED  1.32M params\nspeechonly_moonshine", fc="#ffe3c9")

    arrow(b_ws, b_qp, "bottom", "top")
    arrow(b_ws, b_id, "bottom", "top")
    arrow(b_sig, b_qp, "bottom", "top")
    arrow(b_moo, b_dh, "bottom", "top")

    # --- Row 4: text-tag interface ---
    y_tag = 12
    b_tag = box(4, y_tag, 44, 6, 'RETRIEVED TEXT TAGS  (nearest-neighbour vs 1,482 tags)', False,
                "not a learned embedding projection -- M3 dropped 2026-08-16", fc="#fff6cf", fontsize=6.0)
    arrow(b_qp, b_tag, "bottom", "top")

    # --- Row 5: generation ---
    y_gen = 2
    b_fast = box(4, y_gen, 20, 7, "Fast tier\nLFM2.5-350M", False,
                 "TRAINED (LoRA)\nenable_thinking=False", fc="#cfe8cf")
    b_think = box(26, y_gen, 20, 7, "Thinker\nQwen3-0.6B", False,
                  "TRAINED (LoRA)\nenable_thinking=True", fc="#cfe8cf")
    b_tts = box(48, y_gen, 20, 7, "StreamingVoice\n(NeuTTS-Air GGUF)", False,
                "TRAINED (fine-tune)\n+ NeuCodec INT8 decode", fc="#cfe8cf")

    arrow(b_tag, b_fast, "bottom", "top")
    arrow(b_fast, b_think, "right", "left")
    arrow(b_think, b_tts, "right", "left")
    arrow(b_dh, b_fast, "bottom", "top")
    arrow(b_id, b_think, "bottom", "top")

    # legend
    leg_frozen = mpatches.Patch(facecolor="#f2f2f2", edgecolor=C_GREY, linestyle="--", label="frozen")
    leg_trained = mpatches.Patch(facecolor="#ffe3c9", edgecolor=C_BLACK, label="trained (perception-side)")
    leg_gen = mpatches.Patch(facecolor="#cfe8cf", edgecolor=C_BLACK, label="trained (generation-side)")
    ax.legend(handles=[leg_frozen, leg_trained, leg_gen], loc="upper right",
              bbox_to_anchor=(1.0, 1.02), frameon=False, ncol=1)

    ax.text(78, 34 + 3.5, "clock domains:\ndecision tick: 0.25 s\n(tick_interval_sec,\nm5_streaming_loop.py:84)\n\nperception stride: 10 s\n(0.1 Hz, m5_streaming_loop.py:78)\n\nmeasured full round: 2.886 s\n(~0.35 Hz, ARCHITECTURE.md §8.2,\nno-nat, MAXN_SUPER)",
            fontsize=5.6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor=C_GREY, linewidth=0.6))

    ax.set_title("BMO system architecture (live production path, 2026-08-16 Jetson sync)", fontsize=9)
    save(fig, "fig01_architecture")


# ==========================================================================================
# FIGURE 2 — World-State construction detail
# SOURCE: CLAUDE.md "Spatial pooling for M2's vision cache..." paragraph (8192->512 via
#         4x4 avg-pool over a 16x16 grid, scripts/extract_features_av.py::_spatial_pool);
#         scripts/extract_features_av.py:67 (VISION_TEMP=32); data/av_cached_dataset.py:78
#         (_ts_to_tdm_bins, staircase, called with n_temp*16 bins); models/world_state_builder.py
#         lines 15-90 (docstring: staircase vs the retired linspace-ramp bug; SAME function
#         reused, not reimplemented, for both vision and ambient -- see line 175 comment
#         "SAME function, no linspace"); models/m3_connector.py:28 (d_model=1024, world-state
#         dimensionality). This is a schematic of code structure, not a data plot -- no
#         invented numbers, only structural facts named above.
# ==========================================================================================

def fig2_worldstate():
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", fontsize=6.4, ec=C_BLACK, ls="-"):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.5",
                            linewidth=1.0, edgecolor=ec, linestyle=ls, facecolor=fc, zorder=3)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)
        return (x, y, w, h)

    def arrow(p1, p2, **kw):
        kw.setdefault("arrowstyle", "-|>")
        kw.setdefault("mutation_scale", 8)
        kw.setdefault("linewidth", 0.9)
        kw.setdefault("color", "#333333")
        ax.add_patch(FancyArrowPatch(p1, p2, zorder=2, **kw))

    # Vision path
    b1 = box(2, 28, 20, 8, "8,192 raw vision tokens\n(32 temporal x 256 spatial\n16x16 grid, 1024-d)")
    b2 = box(28, 28, 20, 8, "4x4 avg-pool\n(F.avg_pool2d, k=4,s=4)\nextract_features_av.py\n::_spatial_pool")
    b3 = box(54, 28, 20, 8, "512 pooled tokens\n(32 temporal x 16 spatial,\n1024-d) -- cached as-is")
    arrow((22, 32), (28, 32))
    arrow((48, 32), (54, 32))

    # Temporal binning: staircase vs ramp
    b4 = box(2, 14, 34, 8,
             "STAIRCASE temporal binning\n_ts_to_tdm_bins(vis_ts, dur, n_temp*16)\n32 temporal values, each held constant\nacross its 16 spatial-pooled tokens\n(replaces a retired linspace-RAMP bug)",
             fc="#ffe9d6")
    b5 = box(40, 14, 34, 8,
             "Ambient (WavJEPA) uses the\nSAME _ts_to_tdm_bins function\n(no separate ramp/linspace path --\nworld_state_builder.py:175 comment:\n\"SAME function, no linspace\")",
             fc="#e3f2e3")
    arrow((54, 28), (19, 22))
    arrow((60, 28), (57, 22))

    # Fusion bridge + attentive pooling
    b6 = box(20, 2, 24, 8, "Cross-attention\nfusion bridge\n(av_jepa_predictor.py)", fc="#ffe3c9")
    b7 = box(50, 2, 24, 8, "Attentive pooling\n-> World-State (1024-d)\nnot L2-normalised", fc="#eeeeee")
    arrow((19, 14), (30, 10))
    arrow((57, 14), (60, 10))
    arrow((44, 6), (50, 6))

    ax.set_title("World-State construction (train == inference; both call the SAME functions)", fontsize=9)
    save(fig, "fig02_worldstate")


# ==========================================================================================
# FIGURE 3 — M2 experimental chronology (8 decision points, single-axis)
# SOURCE: docs/METHODOLOGY_FORENSICS.md Part 1.1 chronological table (rows 1-20), which is
#         itself sourced to checkpoints/falsifier_tracking.md and the per-run PROVENANCE/
#         results files. Metric plotted: VGGSound v->a R@1 (first of each pair) where the
#         run reports it; for the two hyper-parameter sweeps (temp, negatives) that never
#         reported a VGGSound-scale R@1, the sweep's own reported R@1 is used instead and
#         labelled. Points, in run order:
#   1. temp sweep 0.03/0.05/0.07 -> 39.84/39.58/38.80 (temp=0.05 shown, adopted)
#            [EVIDENCE_LEDGER_V2.md Table1 1.A, "M2 | temp sweep 0.03/0.05/0.07"]
#   2. 192 vs 200-neg diagnostic (51k corpus) -> 45.31/46.88 shown (200-neg), "a wash"
#            [same table, "M2 | 200-neg diagnostic"]
#   3. Matched-step scale check: 199,007 clips -> 44.27 (v->a)
#            [same table, "M2 | matched-step check, 199,007 clips"]
#   4. VGGSound-60k+Ego4D-17.1k (22.2% share) -> 42.27 (FAIL <52%)
#            [METHODOLOGY_FORENSICS.md Part1.1 row 14]
#   5. RUN-1 (8.0% share) -> 55.15 (PASS, but Ego4D held-out FAILS)
#            [row 15]
#   6. RUN-2 / LOCKED (40.5% share, neg 200) -> 53.27 (PASS; adopted, current production)
#            [row 16]
#   7. RUN-3 +AudioSet (confounded) -> 23.04 (FAIL, rejected)
#            [row 17]
#   8. GradCache 1536-neg scaled SIGReg -> 24.76 (step1000, then collapsed to 19.19 at
#            step2000, killed) [EVIDENCE_LEDGER_V2.md Table1 1.A, "GradCache 1536-neg..."]
# ==========================================================================================

def fig3_m2_chronology():
    points = [
        ("temp sweep\n(temp=0.05\nadopted)", 39.58, "adopted"),
        ("192->200\nnegatives\n(51k, “wash”)", 46.88, "no-op"),
        ("corpus scale\n51.5k->199k\nclips (step6000)", 44.27, "informative"),
        ("+Ego4D 17.1k\n(22.2% share)", 42.27, "rejected"),
        ("RUN-1\n(8.0% share)", 55.15, "partial"),
        ("RUN-2 / LOCKED\n(40.5% share,\nneg=200)", 53.27, "adopted"),
        ("RUN-3\n+AudioSet\n(confounded)", 23.04, "rejected"),
        ("GradCache 1536-neg\nscaled SIGReg\n(collapsed)", 24.76, "rejected"),
    ]
    labels = [p[0] for p in points]
    vals = [p[1] for p in points]
    status = [p[2] for p in points]
    color_map = {"adopted": C_GREEN, "rejected": C_VERM, "partial": C_ORANGE,
                 "no-op": C_GREY, "informative": C_SKY}

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.9))
    x = np.arange(len(points))
    colors = [color_map[s] for s in status]
    ax.plot(x, vals, color="#888888", linewidth=1.0, zorder=1)
    ax.scatter(x, vals, c=colors, s=55, zorder=3, edgecolor="black", linewidth=0.6)
    for xi, v, s in zip(x, vals, status):
        ax.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.3)
    ax.axhline(52, color=C_BLACK, linestyle=":", linewidth=0.8)
    ax.text(len(points) - 1 + 0.15, 52, " 52% VGGSound gate", va="center", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.9)
    ax.set_ylabel("VGGSound R@1 (%)\n(v→a, or run's own reported metric)")
    ax.set_ylim(0, 65)
    handles = [mpatches.Patch(color=c, label=s) for s, c in color_map.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=3, fontsize=6.0)
    ax.set_title("M2 experimental chronology — 8 decision points (of 36 total runs; see METHODOLOGY_FORENSICS.md §1.1 for the full run table)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig03_m2_chronology")


# ==========================================================================================
# FIGURE 4 — Corpus scale at matched steps (both directions)
# SOURCE: docs/EVIDENCE_LEDGER_V2.md Table 1.A, "M2 | matched-step check" rows:
#   51,508 clips, step6000: v->a 33.46%, a->v 34.24%  [logs/m2_diag192.log]
#   199,007 clips, step6000: v->a 44.27%, a->v 43.95%  [logs/m2_fusion_fullscale.log]
# ==========================================================================================

def fig4_corpus_scale():
    corpora = ["51,508 clips", "199,007 clips"]
    va = [33.46, 44.27]
    av = [34.24, 43.95]
    x = np.arange(len(corpora))
    w = 0.32

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    b1 = ax.bar(x - w / 2, va, width=w, label="v→a", color=C_BLUE, hatch=HATCHES[0], edgecolor="black")
    b2 = ax.bar(x + w / 2, av, width=w, label="a→v", color=C_ORANGE, hatch=HATCHES[1], edgecolor="black")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylabel("R@1 (%), step 6000, 1545-clip gallery")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False)
    ax.set_title("Corpus scale at matched training step (step 6000)", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig04_corpus_scale")


# ==========================================================================================
# FIGURE 5 — Batch-share isolation (Ego4D 22.2% / 8.0% / 40.5%)
# SOURCE: docs/EVIDENCE_LEDGER_V2.md Table 1.A rows:
#   22.2% share (VGGSound-60k+Ego4D-17.1k): VGGSound v->a/a->v 42.27/41.68; Ego4D
#       sibling-excl 18.40/18.40
#   8.0% share (RUN-1): VGGSound 55.15/55.53; Ego4D sibling-excl 11.57/10.68
#   40.5% share (RUN-2/LOCKED): VGGSound 53.27/53.72; Ego4D sibling-excl 27.60/27.00
# Gate thresholds: VGGSound >=52% ("PASS (>=52%)", multiple citations in Table 1.A);
#   Ego4D >=10% ("PASS vs later pre-registered >=10% threshold", same table, 22.2%-share row).
# ==========================================================================================

def fig5_batch_share():
    shares = ["22.2%", "8.0%", "40.5%"]
    vgg_va = [42.27, 55.15, 53.27]
    vgg_av = [41.68, 55.53, 53.72]
    ego_va = [18.40, 11.57, 27.60]
    ego_av = [18.40, 10.68, 27.00]

    x = np.arange(len(shares))
    w = 0.19
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.9))
    ax.bar(x - 1.5 * w, vgg_va, width=w, color=C_BLUE, hatch=HATCHES[0], edgecolor="black", label="VGGSound v→a")
    ax.bar(x - 0.5 * w, vgg_av, width=w, color=C_SKY, hatch=HATCHES[1], edgecolor="black", label="VGGSound a→v")
    ax.bar(x + 0.5 * w, ego_va, width=w, color=C_ORANGE, hatch=HATCHES[2], edgecolor="black", label="Ego4D v→a (sibling-excl)")
    ax.bar(x + 1.5 * w, ego_av, width=w, color=C_VERM, hatch=HATCHES[3], edgecolor="black", label="Ego4D a→v (sibling-excl)")

    ax.axhline(52, color=C_BLUE, linestyle=":", linewidth=0.9)
    ax.text(0.02, 52 / 62 + 0.02, "VGGSound gate 52%", fontsize=6.0, color=C_BLUE,
            ha="left", transform=ax.transAxes)
    ax.axhline(10, color=C_VERM, linestyle=":", linewidth=0.9)
    ax.text(0.98, 10 / 62 + 0.025, "Ego4D gate 10%", fontsize=6.0, color=C_VERM,
            ha="right", transform=ax.transAxes)

    ax.set_xticks(x)
    ax.set_xticklabels([f"share {s}" for s in shares], fontsize=7.2)
    ax.set_xlabel("Ego4D batch share")
    ax.set_ylabel("R@1 (%)")
    ax.set_ylim(0, 62)
    ax.set_xlim(-0.6, 2.6)
    ax.legend(frameon=False, fontsize=6.0, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.32))
    ax.set_title("Batch-share isolation: VGGSound and Ego4D R@1 move in opposition", fontsize=8.3, pad=2)
    fig.tight_layout()
    save(fig, "fig05_batch_share")


# ==========================================================================================
# FIGURE 6 — Interface comparison (three attempts)
# SOURCE latency: checkpoints/falsifier_tracking.md:2296-2298 (quoted in
#   docs/METHODOLOGY_FORENSICS.md line 234): "predictor + nearest-neighbor lookup = 8ms
#   (5ms predictor + 3ms NN), vs the old M3-connector-to-Qwen autoregressive path's 1-6s
#   generation." Perception-prefix latency is NOT independently measured anywhere in this
#   repo; METHODOLOGY_FORENSICS.md line 239 characterises the combined soft-prompt-family
#   cost as "1-12+ seconds", used here ONLY as an explicitly-labelled upper-bound range for
#   perception-prefix, not a point measurement.
# SOURCE quality: ARCHITECTURE.md sec 1b / 6 table:
#   M3 connector: F1 0.317 (VGGSound rich captions)
#   perception prefix: F1 0.269
#   query predictor (retrieval): R@1 0.737 (sig_runD_proj768, SigLIP2 space) vs the
#     protocol-matched EmbeddingGemma reference R@1 0.681 (both same 518k pool, same
#     protocol -- docs/EVIDENCE_LEDGER_V2.md Part C §C3).
# Deployed retrieval latency in the CURRENT (SigLIP2, 4-stream) stack is 25ms/33ms
#   (no-nat)/23ms(with-nat) per ARCHITECTURE.md §8.2 -- shown as a second, smaller marker
#   distinct from the historical 8ms figure that originally motivated the pivot.
# ==========================================================================================

def fig6_interface_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7))

    # Latency, log scale
    names = ["M3 soft-prompt\n(generate)", "Perception-prefix\n(generate)", "Query predictor\n(retrieve)"]
    lo = [1000, 1000, 8]
    hi = [6000, 12000, 8]
    y = np.arange(len(names))
    for yi, (l, h) in enumerate(zip(lo, hi)):
        if l == h:
            ax1.plot([l, l], [yi, yi], marker="o", color=C_GREEN, markersize=6, zorder=3)
            ax1.annotate(f"{l} ms", (l, yi), textcoords="offset points", xytext=(6, 0), fontsize=6.3, va="center")
        else:
            ax1.plot([l, h], [yi, yi], color=C_VERM, linewidth=4, solid_capstyle="butt", zorder=3)
            ax1.annotate(f"{l/1000:.0f}–{h/1000:.0f} s", (h, yi), textcoords="offset points",
                        xytext=(6, 0), fontsize=6.3, va="center")
    ax1.scatter([25], [2], marker="D", color=C_SKY, s=32, zorder=4)
    ax1.annotate("current deployed\n(SigLIP2, 25 ms)", (25, 2), textcoords="offset points",
                xytext=(8, 10), fontsize=5.8, va="bottom", ha="left", color=C_SKY)
    ax1.set_xscale("log")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=6.6)
    ax1.set_xlabel("latency, ms (log scale)")
    ax1.set_xlim(5, 30000)
    ax1.set_ylim(-0.6, 2.7)
    ax1.set_title("(a) Latency", fontsize=8)

    # Quality, protocol-matched only for the retrieval pair
    bars = ["M3\n(F1)", "Perception-\nprefix (F1)", "EmbeddingGemma\nretrieval (R@1)", "SigLIP2\nretrieval (R@1)"]
    vals = [0.317, 0.269, 0.681, 0.737]
    colors = [C_GREY, C_GREY, C_SKY, C_BLUE]
    hatches = ["///", "///", "", ""]
    xb = np.arange(len(bars))
    barsobj = ax2.bar(xb, vals, color=colors, hatch=hatches, edgecolor="black", width=0.62)
    for r, v in zip(barsobj, vals):
        ax2.annotate(f"{v:.3f}", (r.get_x() + r.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(bars, fontsize=5.9)
    ax2.set_ylabel("metric value", fontsize=7.5)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("(b) Quality — hatched bars are F1 (generation),\nsolid bars are protocol-matched R@1 (retrieval);\nF1 and R@1 are NOT comparable to each other", fontsize=7.0)
    ax2.axvspan(1.5, 3.5, color=C_SKY, alpha=0.06)

    fig.suptitle("Three attempts at the perception→language interface", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, "fig06_interface_comparison")


# ==========================================================================================
# FIGURE 7 — Stream ablation (4-way, cross-clip R@1)
# SOURCE: docs/EVIDENCE_LEDGER_V2.md Table 1.B, row "4-way stream ablation (EmbeddingGemma
#   geometry)": cross-clip R@1 A/B/C/D = 0.441/0.564/0.566/0.546
#   [checkpoints/abl_A_m2_vision_ambient/, abl_B_plus_scene/, abl_C_scene_baseonly/,
#    abl_D_scene_vision_only/ -- file names give the exact stream composition per arm].
# NOTE: a second, non-identical 4-arm ablation exists in v1's Table 1 ("unified architecture
#   ablation", 2026-08-11, R@1 0.385/0.447/0.478/0.458, arms = m2/vision/m2+vision/unified --
#   a stream-SUBSET study, not a scene-addition study). The two are flagged as an unreconciled
#   conflict in EVIDENCE_LEDGER_V2.md Part E, E3 row 4. This figure uses ONLY the abl_A-D set
#   (2026-08-14), because its arm names map 1:1 onto real file names and it is the ablation
#   that directly motivated wiring SigLIP2 into production.
# ==========================================================================================

def fig7_stream_ablation():
    arms = ["A: m2+vision\n+ambient", "B: A + scene\n(SigLIP2)", "C: scene +\nambient-base only", "D: scene +\nvision only"]
    vals = [0.441, 0.564, 0.566, 0.546]
    colors = [C_GREY, C_BLUE, C_SKY, C_ORANGE]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    bars = ax.bar(arms, vals, color=colors, edgecolor="black",
                   hatch=[HATCHES[i] for i in range(4)])
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (r.get_x() + r.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.8)
    ax.set_xticklabels(arms, fontsize=6.2)
    ax.set_ylabel("cross-clip R@1")
    ax.set_ylim(0, 0.68)
    ax.set_title("4-way stream ablation (2026-08-14, abl_A–D)", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig07_stream_ablation")


# ==========================================================================================
# FIGURE 8 — A1 seven-condition falsifier
# SOURCE (six of seven conditions, all n=300, 100/class): direct read of
#   checkpoints/m4_decision_head_3class_bothpresent/A1_FALSIFIER_RESULTS.json and its
#   PROVENANCE.txt (bootstrap n_boot=10,000):
#     a_real_fresh                = 93.67%
#     b_ws_zeroed                 = 80.67%   (a-b: +13.00pp, CI[+8.67,+17.33], EXCLUDES ZERO)
#     c_ws_swapped_within_session = 92.33%   (a-c_within: +1.33pp, CI[-0.33,+3.33], NOT sig.)
#     c_ws_swapped_cross_session  = 93.33%   (a-c_cross: +0.33pp, CI[-1.33,+2.00], NOT sig.)
#     e_random_matched_stats      = 94.33%   (a-e: -0.67pp, CI[-2.33,+1.00], NOT sig.)
#     f_dataset_mean              = 94.67%   (a-f: -1.00pp, CI[-2.67,+0.33], NOT sig.)
#   Seventh condition (g, speech-only -- a DIFFERENT checkpoint that removes the World-State
#   input branch entirely, not a perturbation of the same model):
#     checkpoints/m4_decision_head_3class_speechonly/gate_results.json -> accuracy 95.00%
#   No bootstrap CI exists for condition g against condition a (different checkpoints); the
#   bracket for it is therefore omitted, not fabricated. Per-condition CIs (as opposed to the
#   six pairwise-vs-a CIs above) are not present in any file found -- bars show point
#   estimates only, with the six real pairwise CIs shown as bracket annotations instead of
#   invented symmetric whiskers.
# Chance line: 33.3% (balanced 3-class, n=100/class).
# ==========================================================================================

def fig8_a1_falsifier():
    conds = ["(a) real, fresh WS", "(b) WS zeroed", "(c-within) swapped,\nsame session",
             "(c-cross) swapped,\ncross-session", "(e) random,\nmatched stats",
             "(f) dataset mean", "(g) speech-only*"]
    vals = [93.67, 80.67, 92.33, 93.33, 94.33, 94.67, 95.00]
    colors = [C_GREEN, C_VERM, C_ORANGE, C_ORANGE, C_SKY, C_SKY, C_GREY]
    # pairwise bootstrap delta vs (a), n_boot=10,000 -- None where no CI exists (condition g,
    # a different checkpoint) -- see PROVENANCE.txt quoted in the module docstring above.
    ci_text = [
        None,
        "Δ=+13.00pp  CI[+8.67,+17.33]  excludes zero",
        "Δ=+1.33pp  CI[-0.33,+3.33]  n.s.",
        "Δ=+0.33pp  CI[-1.33,+2.00]  n.s.",
        "Δ=-0.67pp  CI[-2.33,+1.00]  n.s.",
        "Δ=-1.00pp  CI[-2.67,+0.33]  n.s.",
        None,
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.3))
    y = np.arange(len(conds))
    bars = ax.barh(y, vals, color=colors, edgecolor="black", height=0.62)
    for r, v, ci in zip(bars, vals, ci_text):
        ax.annotate(f"{v:.2f}%", (v, r.get_y() + r.get_height() / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", ha="left", fontsize=6.8)
        if ci:
            ax.annotate(f"vs (a): {ci}", (0, r.get_y() + r.get_height() / 2),
                        textcoords="offset points", xytext=(6, 0), va="center", ha="left",
                        fontsize=5.7, color="white", fontweight="bold")
    ax.axvline(33.3, color=C_BLACK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy (%), n=300 (100/class), real EasyCom test sessions")
    ax.set_xlim(0, 100)
    ax.annotate("chance = 33.3%", (33.3, 1.0), xycoords=("data", "axes fraction"),
                fontsize=6.2, ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")

    ax.set_title("A1 seven-condition falsifier: World-State content is not the task-specific\n"
                  "signal (five of six perturbations vs. real WS are statistically indistinguishable)",
                  fontsize=8.0, pad=8)
    fig.text(0.02, -0.02,
             "*condition (g) is a separate checkpoint (World-State input branch removed entirely) — no pairwise bootstrap CI vs (a) exists across different checkpoints.\n"
             "CI text inside each bar is the paired bootstrap delta of that condition's accuracy vs. condition (a), n_boot=10,000 (A1_PROVENANCE.txt).",
             fontsize=5.6, ha="left", va="top")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig08_a1_falsifier")


# ==========================================================================================
# FIGURE 9 — Identity head, frozen vs trained TAR@FAR1%
# SOURCE: direct read of checkpoints/jepa_identity_head_voice_full/results.json
#   (frozen_mean.tar_at_far1pct=0.302118989405053, trained_head.tar_at_far1pct=0.7047269763651182,
#    n_test_speakers=800) and checkpoints/jepa_identity_head_av_full/results.json
#   (results.joint.frozen.tar_at_far1pct=0.35139941963867827,
#    results.joint.trained.tar_at_far1pct=0.7650472713657213, n_test_usable=884).
#   Both are the FULL-CORPUS runs (voice: 122,235 clips/4,000 speakers; joint AV: 106,736
#   clips/4,420 speakers) -- the current-best pair per EVIDENCE_LEDGER_V2.md Table 7.
# ==========================================================================================

def fig9_identity_head():
    groups = ["Voice-only\n(n=800 test spk)", "Joint AV\n(n=884 test id)"]
    frozen = [0.3021, 0.3514]
    trained = [0.7047, 0.7650]
    x = np.arange(len(groups))
    w = 0.32

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
    b1 = ax.bar(x - w / 2, frozen, width=w, color=C_GREY, hatch="///", edgecolor="black", label="frozen encoder")
    b2 = ax.bar(x + w / 2, trained, width=w, color=C_GREEN, edgecolor="black", label="trained head")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("TAR@FAR1%")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Identity head: frozen encoder vs. trained head\n(full-corpus, current-best checkpoints)", fontsize=8.2)
    fig.tight_layout()
    save(fig, "fig09_identity_head")


# ==========================================================================================
# FIGURE 10 — Encoder latency, log scale (Mercury vs Jetson)
# SOURCE: docs/EVIDENCE_LEDGER_V2.md Table 6 (carried from v1 verbatim):
#   V-JEPA2 ViT-L, Jetson (production, int8): 2.43-2.45 s (checkpoints/m5_jetson/
#       PHASE0_PROVENANCE.txt)
#   V-JEPA2 ViT-L, Mercury/Blackwell (bf16): 47.9-48.0 ms (checkpoints/vjepa21_shelved/
#       HEAD_TO_HEAD_LATENCY.json)
#   V-JEPA 2.1 ViT-B 384px, Mercury: 64.1 ms (same file)
#   V-JEPA 2.1 ViT-L 384px, Mercury: 174.6 ms (same file)
#   V-JEPA 2.1 ViT-L 384px, Jetson: 37.98 s (checkpoints/vjepa21_shelved/
#       B_2B_JETSON_VITL_DECISIVE.txt) -- ruled out for Jetson permanently.
# ==========================================================================================

def fig10_encoder_latency():
    rows = [
        ("V-JEPA2 ViT-L\n(production)", 48.0, 2440),
        ("V-JEPA2.1 ViT-B\n384px", 64.1, None),
        ("V-JEPA2.1 ViT-L\n384px", 174.6, 37980),
    ]
    labels = [r[0] for r in rows]
    mercury = [r[1] for r in rows]
    jetson = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.1))
    y = np.arange(len(labels))
    ax.scatter(mercury, y - 0.12, color=C_BLUE, marker="o", s=45, zorder=3, label="Mercury / Blackwell (bf16)")
    for yi, v in zip(y, mercury):
        ax.annotate(f"{v:.1f} ms", (v, yi - 0.12), textcoords="offset points", xytext=(6, 4), fontsize=6.3)
    jy = [yi + 0.12 for yi, v in zip(y, jetson) if v is not None]
    jv = [v for v in jetson if v is not None]
    ax.scatter(jv, jy, color=C_VERM, marker="s", s=45, zorder=3, label="Jetson Orin (int8)")
    for yi, v in zip(jy, jv):
        lab = f"{v:.0f} ms" if v < 1000 else f"{v/1000:.2f} s"
        ax.annotate(lab, (v, yi), textcoords="offset points", xytext=(6, -8), fontsize=6.3)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("isolated forward latency, ms (log scale)")
    ax.set_xlim(30, 60000)
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    ax.set_title("Encoder latency: production ViT-L vs.\nshelved V-JEPA 2.1 variants", fontsize=8.2)
    fig.tight_layout()
    save(fig, "fig10_encoder_latency")


# ==========================================================================================
# FIGURE 11 — Jetson memory: waterfall of components against the 7,620 MiB ceiling
# SOURCE: ARCHITECTURE.md sec 8.1 "IT FITS" table, no-nat column (the CURRENT production
#   configuration -- WavJEPA-nat is dropped, EVIDENCE_LEDGER_V2.md Part F). Deltas (MiB):
#   speaker +253, thinker +780, ViT-L int8 +629, WavJEPA base +618, M2 -53 (net memory freed
#   at this step, shown as a release, not a cost), SigLIP2 vision tower +1170, pre-encoded
#   queries -1, query predictor + 1,372 tags +134, camera +18 (avail after camera: 1,392 MiB,
#   matches this waterfall's running total exactly). The table's own 10th row, "after 3
#   rounds", gives avail=429 MiB directly (no delta stated) -- this figure's "3 rounds of
#   inference" bar uses delta = 1,392 - 429 = 963 MiB, i.e. simple subtraction of two real
#   avail numbers from the same table row, not an independent measurement or an invented
#   number. Starting avail before "speaker" (4,940 MiB) is back-derived the same way:
#   4,687 (avail after speaker) + 253 (speaker's own delta) = 4,940.
#   Ceiling 7,620 MiB: docs/EVIDENCE_LEDGER.md row "Full stack, malloc_trim+int8 fix...
#   6647MiB / 7620MiB" (checkpoints/m5_jetson/PHASE0_CLARIFICATION_PROVENANCE.txt) --
#   "already used before this test" (2,680 MiB) is the ceiling minus the back-derived 4,940
#   starting-avail figure, i.e. OS + kiosk/face-engine + Xorg + whatever was resident before
#   this specific isolated fit-test began.
# ==========================================================================================

def fig11_jetson_memory():
    ceiling = 7620
    start_avail = 4940  # back-derived: avail-after-speaker (4687) + speaker's own delta (253)
    already_used = ceiling - start_avail
    # "after 3 rounds" avail=429 is the table's own 10th row (ARCHITECTURE.md sec 8.1); its
    # delta here (963) is derived by simple subtraction from two real avail numbers in that
    # same row (camera-step avail 1392 minus the row's own avail 429), not invented.
    steps = [
        ("already used\n(OS/kiosk/Xorg)", already_used, "base"),
        ("speaker\n(LFM2.5-350M)", 253, "cost"),
        ("thinker\n(Qwen3-0.6B)", 780, "cost"),
        ("ViT-L int8", 629, "cost"),
        ("WavJEPA base", 618, "cost"),
        ("M2 predictor", -53, "release"),
        ("SigLIP2 vision\ntower", 1170, "cost"),
        ("pre-encoded\nqueries", -1, "release"),
        ("query predictor\n+ 1,372 tags", 134, "cost"),
        ("camera", 18, "cost"),
        ("3 rounds of\ninference", 963, "cost"),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.3))
    cum = 0
    xs = []
    for i, (name, delta, kind) in enumerate(steps):
        color = C_GREY if kind == "base" else (C_VERM if kind == "cost" else C_GREEN)
        bottom = cum
        ax.bar(i, delta, bottom=bottom, color=color, edgecolor="black", width=0.65)
        cum += delta
        xs.append(name)
        label = f"{delta:+d}" if kind != "base" else f"{delta}"
        va = "center" if abs(delta) > 250 else "bottom"
        yoff = 0 if abs(delta) > 250 else 6
        ax.annotate(label, (i, bottom + delta / 2 if abs(delta) > 250 else bottom + max(delta, 0)),
                    ha="center", va=va, fontsize=6.0,
                    xytext=(0, yoff), textcoords="offset points",
                    color=("white" if abs(delta) > 250 else "black"))
    ax.axhline(ceiling, color=C_BLACK, linestyle="--", linewidth=0.9)
    ax.text(0.15, ceiling, f"ceiling {ceiling} MiB", fontsize=6.3, va="bottom", ha="left")
    final_avail = ceiling - cum
    ax.axhline(cum, color=C_BLUE, linestyle=":", linewidth=1.0)
    ax.text(0.15, cum - 60, f"final: {cum} MiB used → {final_avail} MiB avail",
            fontsize=6.3, va="top", ha="left", color=C_BLUE)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(xs, fontsize=5.7, rotation=0)
    ax.set_ylabel("cumulative resident memory (MiB)")
    ax.set_ylim(0, ceiling * 1.08)
    handles = [mpatches.Patch(color=C_GREY, label="pre-existing"),
               mpatches.Patch(color=C_VERM, label="component cost"),
               mpatches.Patch(color=C_GREEN, label="memory released")]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("Jetson memory waterfall, current (no-nat) production stack, MAXN_SUPER\n(ARCHITECTURE.md §8.1, 2026-08-15)", fontsize=8.0)
    fig.tight_layout()
    save(fig, "fig11_jetson_memory")


# ==========================================================================================
# FIGURE 12 — Jetson latency: (a) tick distribution by vision-overlap, (b) power-mode effect
# SOURCE (a): direct read of checkpoints/vjepa21_shelved/JETSON_PHASE4_4_RESULTS.json
#   strided.overlapped_vision_forward_ticks   n=54  mean=155.79 p50=5.73  p95=845.23 max=856.89
#   strided.not_overlapped_ticks              n=186 mean=49.70  p50=0.18  p95=284.70 max=806.53
#   opportunistic.overlapped_vision_forward_ticks n=196 mean=63.64 p50=1.13 p95=348.02 max=828.45
#   opportunistic.not_overlapped_ticks        n=44  mean=122.34 p50=0.25  p95=340.39 max=341.59
# SOURCE (b): docs/EVIDENCE_LEDGER_V2.md Table 1.B, "describe-demo, 7W vs MAXN_SUPER power
#   mode" row: perception 3828-12368ms(7W)->1089-1732ms(MAXN); total-round
#   14971-18401ms(7W)->3892-4994ms(MAXN)
#   [jetson_artifacts/benchmarks/home/jetson_describe_demo_results{,_MAXN}.json]
# No sourced "~250ms conversational budget" line was found anywhere in EVIDENCE_LEDGER_V2.md
# or METHODOLOGY_FORENSICS.md; omitted rather than invented (see FIGURE_MANIFEST.md).
# ==========================================================================================

def fig12_jetson_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.8))

    conds = [
        ("strided\noverlapped\n(n=54)", 155.79, 5.73, 845.23, 856.89),
        ("strided\nnot-overlapped\n(n=186)", 49.70, 0.18, 284.70, 806.53),
        ("opportunistic\noverlapped\n(n=196)", 63.64, 1.13, 348.02, 828.45),
        ("opportunistic\nnot-overlapped\n(n=44)", 122.34, 0.25, 340.39, 341.59),
    ]
    x = np.arange(len(conds))
    means = [c[1] for c in conds]
    p50s = [c[2] for c in conds]
    p95s = [c[3] for c in conds]
    maxs = [c[4] for c in conds]
    ax1.vlines(x, p50s, maxs, color="#bbbbbb", linewidth=6, zorder=1)
    ax1.vlines(x, p50s, p95s, color=C_SKY, linewidth=6, zorder=2)
    ax1.scatter(x, means, marker="D", color=C_VERM, s=30, zorder=3, label="mean")
    ax1.scatter(x, p50s, marker="_", color="black", s=200, zorder=3, label="p50")
    ax1.scatter(x, p95s, marker="_", color=C_BLUE, s=200, zorder=3, label="p95")
    ax1.scatter(x, maxs, marker="x", color="black", s=25, zorder=3, label="max")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in conds], fontsize=6.0)
    ax1.set_ylabel("tick latency (ms)")
    ax1.legend(frameon=False, fontsize=5.6, loc="upper left")
    ax1.set_title("(a) Tick latency by vision-overlap\n(JETSON_PHASE4_4_RESULTS.json)", fontsize=7.6)

    legs = ["perception", "total round"]
    lo7w = [3828, 14971]
    hi7w = [12368, 18401]
    loM = [1089, 3892]
    hiM = [1732, 4994]
    y = np.arange(len(legs))
    h = 0.32
    ax2.barh(y - h / 2, [hi - lo for hi, lo in zip(hi7w, lo7w)], left=lo7w, height=h,
             color=C_VERM, edgecolor="black", label="7W mode")
    ax2.barh(y + h / 2, [hi - lo for hi, lo in zip(hiM, loM)], left=loM, height=h,
             color=C_GREEN, edgecolor="black", label="MAXN_SUPER")
    for yi, (lo, hi) in zip(y - h / 2, zip(lo7w, hi7w)):
        ax2.annotate(f"{lo}–{hi} ms", (hi, yi), textcoords="offset points", xytext=(4, 0), va="center", fontsize=6.2)
    for yi, (lo, hi) in zip(y + h / 2, zip(loM, hiM)):
        ax2.annotate(f"{lo}–{hi} ms", (hi, yi), textcoords="offset points", xytext=(4, 0), va="center", fontsize=6.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(legs)
    ax2.set_xlabel("latency (ms)")
    ax2.set_xlim(0, 20000)
    ax2.legend(frameon=False, fontsize=6.2)
    ax2.set_title("(b) Power-mode effect\n(describe-demo, min–max range)", fontsize=7.6)

    fig.tight_layout()
    save(fig, "fig12_jetson_latency")


# ==========================================================================================
# FIGURE 13 — Ego4D filter funnel (REAL numbers substituted for the brief's un-sourced
# 218,096 figure -- see FIGURE_MANIFEST.md)
# SOURCE: checkpoints/vjepa21_shelved/ego4d_av_filter_scores.json (n_candidates=57822,
#   top_n=42000 kept AV-relevance, n_dropped=15822); checkpoints/vjepa21_shelved/
#   EGO4D_RECUT_V5_SUMMARY.json (floor=0.1, vad_speech_exclude_threshold=0.84,
#   n_after_floor_and_exclusions=28889, per-tag caps music/conv/narr=722/577/577,
#   final_kept_count=23303, file_coverage=1296); checkpoints/vjepa21_shelved/
#   EGO4D_HELDOUT_SPLIT_SUMMARY_V2.json (n_heldout_windows=674, cap_per_file=2,
#   n_heldout_files=350, n_train_windows=17140, n_train_files=946).
# NOTE: this funnel produced the 674-window Ego4D held-out GALLERY still used by the
#   LOCKED RUN-2 checkpoint's evaluation gate, and the 17,140-clip Ego4D TRAINING set used
#   by the early scaling runs (VGGSound-60k+Ego4D-17.1k, RUN-1). It is NOT the funnel for
#   RUN-2's full 134,491-clip Ego4D training corpus -- that later "expand" pipeline's
#   intermediate stage-by-stage counts are not documented with a comparable funnel anywhere
#   in EVIDENCE_LEDGER_V2.md or METHODOLOGY_FORENSICS.md (partial shard files exist under
#   checkpoints/vjepa21_shelved/ego4d_expand_*.json but do not sum to a single documented
#   total); not fabricated here.
# ==========================================================================================

def fig13_ego4d_funnel():
    stages = [
        ("raw sampled\nwindows", 57822),
        ("AV-relevance\ntop_n kept", 42000),
        ("floor 0.1 +\nVAD-speech excl.\n(≤84%)", 28889),
        ("per-tag caps\n(music/conv/narr\n≤722/577/577)", 23303),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.8), gridspec_kw={"width_ratios": [2, 1]})

    names = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    y = np.arange(len(stages))
    bars = ax1.barh(y, counts, color=C_BLUE, edgecolor="black", height=0.55)
    for r, c in zip(bars, counts):
        ax1.annotate(f"{c:,}", (c, r.get_y() + r.get_height() / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=6.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=6.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("windows remaining")
    ax1.set_xlim(0, 65000)
    ax1.set_title("(a) Ego4D pre-processing funnel\n(V5 recut, 1,296 files)", fontsize=7.6)

    split_names = ["held-out gallery\n(cap ≤2/file,\n350 files)", "train split\n(946 files)"]
    split_vals = [674, 17140]
    bars2 = ax2.bar(split_names, split_vals, color=[C_VERM, C_GREEN], edgecolor="black")
    for r, v in zip(bars2, split_vals):
        ax2.annotate(f"{v:,}", (r.get_x() + r.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
    ax2.set_ylabel("windows")
    ax2.set_title("(b) Final 23,303-window\nfile-disjoint split", fontsize=7.6)
    ax2.set_xticklabels(split_names, fontsize=6.0)

    fig.suptitle("Ego4D filter funnel — the ORIGINAL corpus (674-window eval gallery still in use;\n17,140-clip train set used by early scaling runs, NOT RUN-2's full 134,491-clip corpus)",
                  fontsize=7.3, y=1.08)
    fig.tight_layout()
    save(fig, "fig13_ego4d_funnel")


# ==========================================================================================
# FIGURE 14 — Query predictor swapped-query control
# SOURCE: docs/EVIDENCE_LEDGER.md (v1, carried forward verbatim into EVIDENCE_LEDGER_V2.md
#   Table 1.A / Table 4): "Query-predictor swapped-query control | correct query vs swapped
#   (wrong) query | within-clip acc 0.897 (correct) vs 0.002-0.006 (swapped) | n=varies |
#   far-below-chance comparison (chance=0.167 for 6-way) | PASS".
# ==========================================================================================

def fig14_swapped_query():
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.5))
    cats = ["correct query", "swapped query\n(range)"]
    correct = 0.897
    swap_lo, swap_hi = 0.002, 0.006
    ax.bar([0], [correct], width=0.5, color=C_GREEN, edgecolor="black")
    ax.annotate(f"{correct:.3f}", (0, correct), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.bar([1], [swap_hi], width=0.5, color=C_VERM, edgecolor="black")
    ax.annotate(f"{swap_lo:.3f}–{swap_hi:.3f}", (1, swap_hi), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=6.6)
    ax.axhline(0.167, color=C_BLACK, linestyle="--", linewidth=0.9)
    ax.text(1.35, 0.167, " chance (1/6 = 0.167)", va="center", fontsize=6.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cats)
    ax.set_ylabel("within-clip accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 2.1)
    ax.set_title("Query predictor swapped-query control", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig14_swapped_query")


if __name__ == "__main__":
    fig1_architecture()
    fig2_worldstate()
    fig3_m2_chronology()
    fig4_corpus_scale()
    fig5_batch_share()
    fig6_interface_comparison()
    fig7_stream_ablation()
    fig8_a1_falsifier()
    fig9_identity_head()
    fig10_encoder_latency()
    fig11_jetson_memory()
    fig12_jetson_latency()
    fig13_ego4d_funnel()
    fig14_swapped_query()
    print("\nAll figures written to", OUTDIR)
