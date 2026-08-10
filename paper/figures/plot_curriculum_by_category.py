"""Curriculum training, per category: upright vs. inverted validation accuracy.

Data source: the r7 curriculum run's training_history_*.csv.
Outputs: curriculum_by_category.{png,pdf} and curriculum_by_category.csv.

The gap between the two lines is the inversion cost. Faces and houses are
identity-level tasks and carry a large cost; objects are basic-level and carry a
small one. Accuracy is not comparable across stages (each stage scores only its
own active class subset), but the *gap* within an epoch is, since both curves see
the same class set.
"""

import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent
RUN = (Path(__file__).parent.parent.parent
       / "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r7_curriculum")

# --- palette (validated categorical slots 2-3, light surface) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
UPRIGHT = "#eb6834"
INVERTED = "#1baf7a"

PANELS = [("Faces", "faces"), ("Objects", "objects"), ("Houses", "houses_zubud")]


def main():
    hist_csv = sorted(glob.glob(str(RUN / "training_history_*.csv")))[-1]
    h = pd.read_csv(hist_csv)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3), sharey=True, dpi=200)
    fig.subplots_adjust(left=0.058, right=0.995, top=0.695, bottom=0.245, wspace=0.085)

    starts = h.groupby("stage").epoch.min().tolist()

    for ax, (title, cat) in zip(axes, PANELS):
        ax.set_xlim(0.5, h.epoch.max() + 0.5)
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.xaxis.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        for start in starts:
            if start > 1:
                ax.axvline(start - 0.5, color=AXIS, linewidth=0.8,
                           linestyle=(0, (3, 3)), zorder=1)

        up = h[f"valid_acc_{cat}"] * 100
        inv = h[f"test_acc_{cat}"] * 100
        # shade the inversion cost so the gap itself is the readable quantity
        ax.fill_between(h.epoch, inv, up, where=(up >= inv), color=UPRIGHT,
                        alpha=0.11, linewidth=0, zorder=2)
        ax.plot(h.epoch, up, color=UPRIGHT, linewidth=2.0,
                solid_capstyle="round", zorder=4)
        ax.plot(h.epoch, inv, color=INVERTED, linewidth=2.0,
                solid_capstyle="round", zorder=4)

        gap = up.iloc[-1] - inv.iloc[-1]
        ax.set_title(f"{title}   ", color=INK, fontsize=11, fontweight="bold",
                     pad=8, loc="left")
        ax.text(0.985, 0.055, f"final gap  {gap:+.1f} pts", transform=ax.transAxes,
                color=INK_2, fontsize=9, ha="right", va="bottom")

    axes[0].set_ylabel("Validation accuracy (%)", color=INK_2, fontsize=9.5)
    axes[1].set_xlabel("Epoch", color=INK_2, fontsize=9.5, labelpad=6)

    handles = [plt.Line2D([], [], color=UPRIGHT, linewidth=2.4, label="Upright"),
               plt.Line2D([], [], color=INVERTED, linewidth=2.4, label="Inverted")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.885),
               frameon=False, ncol=2, handletextpad=0.6, columnspacing=1.8,
               labelcolor=INK_2, fontsize=9.5)

    fig.suptitle("Inversion cost emerges over curriculum training, per category",
                 x=0.052, y=0.965, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.052, 0.028,
             "Dashed rules mark each class introduction. Shaded band = inversion cost. "
             "Accuracy is not comparable across stages (each stage scores its own class "
             "subset), but the\nupright–inverted gap within an epoch is, since both "
             "orientations see the same classes. Houses have 1 validation image per class, "
             "so that panel is the noisiest.",
             color=MUTED, fontsize=8, ha="left", va="bottom", linespacing=1.5)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"curriculum_by_category.{ext}", facecolor=SURFACE, dpi=200)

    # table view (the WCAG-clean twin of the chart)
    with open(OUT / "curriculum_by_category.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "stage", "active_classes", "category",
                    "upright_pct", "inverted_pct", "gap_pts"])
        for _, r in h.iterrows():
            for title, cat in PANELS:
                u, i = r[f"valid_acc_{cat}"] * 100, r[f"test_acc_{cat}"] * 100
                w.writerow([int(r.epoch), int(r.stage), int(r.active_classes), title,
                            round(u, 2), round(i, 2), round(u - i, 2)])
    print("wrote", OUT / "curriculum_by_category.png")


if __name__ == "__main__":
    main()
