"""Yin (1969) study/test orientation matching: humans vs. LP-Net vs. Plain.

Data source: paper/results_summary.md (Sections 1, 2, 4).
Outputs: yin_human_vs_models.{png,pdf} and yin_human_vs_models.csv (table view).
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

OUT = Path(__file__).parent
CONDITIONS = ["UU", "II", "UI", "IU"]
CATEGORIES = ["Faces", "Objects", "Houses"]

# --- data (accuracy %, 2AFC matching; chance = 50) ---------------------------
# Human: Yin (1969) Tables 1 & 2, objects = airplanes.
HUMAN = {
    "Faces":   [96.29, 81.88, 84.13, 78.58],
    "Objects": [84.79, 83.96, 86.71, 82.75],
    "Houses":  [90.71, 85.75, 88.08, 85.71],
}
# Models: per-backbone runs, r5b_houses_zubud_30ep (LP-Net) / r6_plain_30ep (Plain).
LPNET = {
    "Faces":   {"r18": [95.83, 70.83, 83.33, 79.17], "r34": [95.83, 79.17, 70.83, 79.17]},
    "Objects": {"r18": [83.33, 79.17, 75.00, 79.17], "r34": [83.33, 79.17, 75.00, 83.33]},
    "Houses":  {"r18": [87.50, 83.33, 87.50, 83.33], "r34": [87.50, 75.00, 83.33, 70.83]},
}
PLAIN = {
    "Faces":   {"r18": [95.83, 83.33, 54.17, 62.50], "r34": [95.83, 70.83, 50.00, 70.83]},
    "Objects": {"r18": [83.33, 83.33, 70.83, 83.33], "r34": [83.33, 62.50, 70.83, 79.17]},
    "Houses":  {"r18": [87.50, 62.50, 58.33, 54.17], "r34": [87.50, 70.83, 62.50, 66.67]},
}

# --- palette (validated categorical slots 1-3, light surface) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"Human": "#2a78d6", "LP-Net": "#eb6834", "Plain": "#1baf7a"}
ORDER = ["Human", "LP-Net", "Plain"]


def rounded_bar(ax, x, width, top, color, radius_px=4.0):
    """Bar anchored at the chance baseline with a 4px-rounded data end."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = radius_px * (x1 - x0) / bbox.width
    ry = radius_px * (y1 - y0) / bbox.height
    base = y0
    top = max(top, base)
    rx, ry = min(rx, width / 2), min(ry, max(top - base, 1e-9))
    k = 0.4477  # circle-ish Bezier handle
    left, right = x - width / 2, x + width / 2
    verts = [
        (left, base), (left, top - ry),
        (left, top - ry + k * ry), (left + rx - k * rx, top), (left + rx, top),
        (right - rx, top),
        (right - rx + k * rx, top), (right, top - ry + k * ry), (right, top - ry),
        (right, base), (left, base),
    ]
    codes = [MPath.MOVETO, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True, dpi=200)
    fig.subplots_adjust(left=0.068, right=0.995, top=0.745, bottom=0.185, wspace=0.075)

    xs = np.arange(len(CONDITIONS))
    bar_w = 0.25
    gap = 0.02  # ~2px surface gap between adjacent bars
    offsets = {"Human": -(bar_w + gap), "LP-Net": 0.0, "Plain": bar_w + gap}

    for ax in axes:
        ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
        ax.set_ylim(50, 100)  # baseline = chance for the 2AFC matching task
    fig.canvas.draw()  # fix axes geometry so px-based corner radii are correct

    for ax, cat in zip(axes, CATEGORIES):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.xaxis.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        ax.set_xticks(xs)
        ax.set_xticklabels(CONDITIONS, color=INK_2, fontsize=9.5)

        for name in ORDER:
            color = SERIES[name]
            for i, cond in enumerate(CONDITIONS):
                x = xs[i] + offsets[name]
                if name == "Human":
                    rounded_bar(ax, x, bar_w, HUMAN[cat][i], color)
                    continue
                runs = (LPNET if name == "LP-Net" else PLAIN)[cat]
                vals = [runs["r18"][i], runs["r34"][i]]
                rounded_bar(ax, x, bar_w, float(np.mean(vals)), color)
                ax.scatter([x - 0.055, x + 0.055], vals, s=22, facecolor=INK_2,
                           edgecolor=SURFACE, linewidth=1.4, zorder=5)

        ax.set_title(cat, color=INK, fontsize=11, fontweight="bold", pad=8, loc="left")

    axes[0].set_ylabel("Matching accuracy (%)", color=INK_2, fontsize=9.5)
    axes[1].set_xlabel("study → test orientation  (U = upright, I = inverted)",
                       color=INK_2, fontsize=9.5, labelpad=6)
    fig.text(0.068, 0.028, "Bars are anchored at chance (50%, 2AFC matching); "
                           "24 test pairs per condition, so each step is 4.2 pts.",
             color=MUTED, fontsize=8, ha="left")

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                          markerfacecolor=SERIES[n], markeredgecolor="none", label=n)
               for n in ORDER]
    handles.append(plt.Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                              markerfacecolor=INK_2, markeredgecolor=SURFACE,
                              markeredgewidth=1.2, label="ResNet-18 / ResNet-34 (bar = mean)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.905),
               frameon=False, ncol=4, handletextpad=0.5, columnspacing=1.6,
               labelcolor=INK_2, fontsize=9)

    fig.suptitle("Yin (1969) orientation-matching: humans vs. LP-Net vs. Plain",
                 x=0.062, y=0.965, ha="left", color=INK, fontsize=13, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"yin_human_vs_models.{ext}", facecolor=SURFACE, dpi=200)

    # table view (the WCAG-clean twin of the chart)
    with open(OUT / "yin_human_vs_models.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "source", "backbone"] + CONDITIONS)
        for cat in CATEGORIES:
            w.writerow([cat, "Human (Yin 1969)", ""] + HUMAN[cat])
            for name, data in (("LP-Net", LPNET), ("Plain", PLAIN)):
                for bb in ("r18", "r34"):
                    w.writerow([cat, name, bb] + data[cat][bb])
    print("wrote", OUT / "yin_human_vs_models.png")


if __name__ == "__main__":
    main()
