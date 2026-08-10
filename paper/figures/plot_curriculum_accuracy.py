"""Curriculum (sequential) training: train vs. validation accuracy across epochs.

Data source: the r7 curriculum run's training_history_*.csv.
Outputs: curriculum_accuracy.{png,pdf} and curriculum_accuracy.csv (table view).

Note on comparability: within a stage both curves are scored on that stage's
active class subset, so accuracy is NOT comparable across stages -- the model is
solving a 12-way problem in stage 1 and a 393-way problem in stage 7. The dashed
rules mark each class introduction; the numbers along the top are the class count
the curves are being scored against.
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

# --- palette (validated categorical slots 1-2, light surface) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"Train (upright)": "#2a78d6", "Validation (upright)": "#eb6834",
          "Validation (inverted)": "#1baf7a"}


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

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    fig.subplots_adjust(left=0.068, right=0.815, top=0.745, bottom=0.20)

    ax.set_xlim(0.5, h.epoch.max() + 0.5)
    ax.set_ylim(40, 100)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])

    # stage rules + the class count each segment is scored against
    starts = h.groupby("stage").epoch.min().tolist()
    for st, start in zip(sorted(h.stage.unique()), starts):
        seg = h[h.stage == st]
        if start > 1:
            ax.axvline(start - 0.5, color=AXIS, linewidth=0.8,
                       linestyle=(0, (3, 3)), zorder=1)
        mid = (seg.epoch.min() + seg.epoch.max()) / 2
        ax.text(mid, 97.4, f"{int(seg.active_classes.iloc[0])}", color=MUTED,
                fontsize=8, ha="center", va="center")

    for name, col in (("Train (upright)", "train_acc"),
                      ("Validation (upright)", "valid_acc"),
                      ("Validation (inverted)", "test_acc")):
        y = h[col] * 100
        ax.plot(h.epoch, y, color=SERIES[name], linewidth=2.0,
                solid_capstyle="round", zorder=4)
        # direct label at the line end, so identity is never color-alone
        ax.text(h.epoch.iloc[-1] + 0.9, y.iloc[-1], f"{name}\n{y.iloc[-1]:.1f}%",
                color=SERIES[name], fontsize=9, va="center", ha="left",
                linespacing=1.45)

    ax.set_xlabel("Epoch", color=INK_2, fontsize=9.5, labelpad=6)
    ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=9.5)

    fig.suptitle("Curriculum training: accuracy across epochs (LP-Net, ResNet-18)",
                 x=0.072, y=0.955, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.072, 0.895,
             "Seven stages of 4 → 8 → 16 → 32 → 64 → 128 → all classes per category "
             "(faces, objects, houses);\nthe top row gives the classes active in each stage.",
             color=INK_2, fontsize=9.5, ha="left", va="top", linespacing=1.5)
    fig.text(0.068, 0.038,
             "Each stage is scored on its own active class subset, so accuracy is not comparable "
             "across stages: the drop at every dashed\nrule is the class count rising, not the "
             "model degrading. Final stage = all 393 classes.",
             color=MUTED, fontsize=8, ha="left", va="bottom", linespacing=1.5)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"curriculum_accuracy.{ext}", facecolor=SURFACE, dpi=200)

    # table view (the WCAG-clean twin of the chart)
    with open(OUT / "curriculum_accuracy.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "stage", "active_classes", "train_acc_pct",
                    "valid_upright_pct", "valid_inverted_pct"])
        for _, r in h.iterrows():
            w.writerow([int(r.epoch), int(r.stage), int(r.active_classes),
                        round(r.train_acc * 100, 2), round(r.valid_acc * 100, 2),
                        round(r.test_acc * 100, 2)])
    print("wrote", OUT / "curriculum_accuracy.png")


if __name__ == "__main__":
    main()
