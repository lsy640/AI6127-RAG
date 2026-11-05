"""Report figures: retrieval precision@5 and ablation results.
Usage (from the repository root): python eval/figures/charts.py"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
HERE = Path(__file__).resolve().parent
U = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent)
for f in font_manager.findSystemFonts():
    if "Inter" in f: font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 7.5, "axes.edgecolor": "#d9d8d3",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
SURF, GRID, INK2 = "#fcfcfb", "#e7e6e1", "#52514e"
C1, C2 = "#2a78d6", "#eb6834"

def hbars(ax, labels, series, names, colors, xmax=1.0, fmt="{:.2f}"):
    n = len(series); h = 0.36; gap = 0.04
    ys = list(range(len(labels)))[::-1]
    for j, (vals, name, col) in enumerate(zip(series, names, colors)):
        off = -(j - (n - 1) / 2) * (h + gap)
        for y, v in zip(ys, vals):
            ax.barh(y + off, v, height=h, color=col, label=name if y == ys[0] else None, zorder=3)
            ax.text(v + xmax * 0.01, y + off, fmt.format(v), va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_xlim(0, xmax); ax.set_facecolor(SURF)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

# ---- Figure 2: retrieval
S = json.load(open(f"{U}/q3_retrieval/summary.json")); C = json.load(open(f"{U}/q3_retrieval/condensed_p5.json"))
cfg = [("BM25", "chunk:bm25"), ("Dense (bge-m3)", "chunk:dense"), ("RRF fusion", "chunk:rrf"),
       ("RRF + re-ranking (RR)", "chunk:rrf+rerank"), ("RRF + filters", "chunk:rrf+filters"),
       ("RRF + RR + filters", "chunk:rrf+rerank+filters"),
       ("Chinese, dense", "chunk:zh:dense"), ("Chinese → English + RR", "chunk:zh→en:rrf+rerank")]
prog = [S[k]["prog"]["P@5"] for _, k in cfg]; hum = [C[k.replace(":", "_", 1)] for _, k in cfg]
fig, ax = plt.subplots(figsize=(3.45, 3.0), dpi=400); fig.patch.set_facecolor(SURF)
hbars(ax, [c for c, _ in cfg], [prog, hum], ["Programmatic qrels", "Human qrels (judged only)"], [C1, C2])
ax.set_xlabel("Precision@5", fontsize=7)
fig.legend(loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, frameon=False, fontsize=6.8, handlelength=1.0, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(HERE / "fig_retrieval.png", facecolor=SURF); fig.savefig(HERE / "fig_retrieval.pdf", facecolor=SURF)

# ---- Figure 3: ablation (two panels, separate scales)
Q = json.load(open(f"{U}/q5_runs/summary.json"))
ab = [("B0 single-turn RAG", "B0"), ("B1 + rewriting", "B1"), ("B2 + LLM clarification", "B2"),
      ("B3 + entropy clarification", "B3"), ("B4 + metadata filters", "B4"), ("Full + evidence & self-check", "full"),
      ("Full − I1 (no clarification)", "full-I1"), ("Full − I2", "full-I2"), ("Full − I3", "full-I3")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=400, gridspec_kw={"width_ratios": [1.5, 1]}); fig.patch.set_facecolor(SURF)
hbars(a1, [n for n, _ in ab], [[Q[k]["success_rate"] for _, k in ab], [Q[k]["first_recommendation_success"] for _, k in ab]],
      ["Success@3 within 5 turns", "First recommendation already correct"], [C1, C2])
a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7, handlelength=1.0); a1.set_xlabel("Share of dialogues")
a1.set_title("Dialogue success (user simulator)", loc="left", fontsize=8.5, fontweight="semibold", pad=16)
hbars(a2, [n for n, _ in ab], [[Q[k]["constraint_violation_rate"] for _, k in ab]], ["Violation rate"], [C1], xmax=0.85)
a2.set_yticklabels([]); a2.set_xlabel("Share of recommended products")
a2.set_title("Constraint violations", loc="left", fontsize=8.5, fontweight="semibold", pad=16)
fig.tight_layout(); fig.savefig(HERE / "fig_ablation.png", facecolor=SURF); fig.savefig(HERE / "fig_ablation.pdf", facecolor=SURF)
