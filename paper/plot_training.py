"""
plot_training.py

Training curves for a run, from the training_history_*.csv train.py already
writes. Three figures:

  <run>/fig_training_overall.png   train / upright / inverted accuracy
  <run>/fig_training_bycat.png     upright vs inverted, one panel per category
  <run>/fig_inversion_effect.png   upright - inverted over epochs, all categories

The `valid` split is upright presentation and `test` is the SAME images rotated
180 degrees (salience_trans.py:45), so `valid_acc - test_acc` is an inversion
effect measured every epoch, per category, for free. That third figure is the
developmental one: it shows WHEN the face inversion effect separates from the
object and house baselines over the course of training.

Usage:
    python paper/plot_training.py runs/r10_dev_h80 [runs/r11_... ...]
    python paper/plot_training.py --compare runs/r10_dev_h80 runs/r12_...
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Faces are the category the whole project is about, so they get the strong
# colour; houses and objects are the controls.
CAT_COLORS = {"faces": "#c0392b", "houses_zubud": "#2874a6",
              "objects": "#7f8c8d", "houses": "#48a068"}
CAT_LABELS = {"faces": "faces", "houses_zubud": "houses (ZuBuD identities)",
              "objects": "objects", "houses": "houses (generic category)"}


def load_history(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "training_history_*.csv")))
    if not files:
        raise SystemExit(f"no training_history_*.csv in {run_dir}")
    return pd.read_csv(files[-1])


def categories_in(df):
    return [c[len("valid_acc_"):] for c in df.columns if c.startswith("valid_acc_")]


def stage_boundaries(df):
    """Epochs where a new curriculum stage starts, with their class counts."""
    if "stage" not in df.columns:
        return []
    marks = []
    for prev, cur in zip(df.itertuples(), df.iloc[1:].itertuples()):
        if cur.stage != prev.stage:
            marks.append((cur.epoch, getattr(cur, "active_classes", None)))
    return marks


def _draw_stages(ax, df, label=True):
    for epoch, ncls in stage_boundaries(df):
        ax.axvline(epoch, color="0.75", ls="--", lw=0.8, zorder=0)
        if label and ncls is not None:
            ax.annotate(f"{ncls} cls", xy=(epoch, ax.get_ylim()[1]),
                        xytext=(2, -10), textcoords="offset points",
                        fontsize=6, color="0.45", rotation=90, va="top")


def plot_overall(df, run_dir, name):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.epoch, df.train_acc * 100, label="train", color="0.35", lw=1.4)
    ax.plot(df.epoch, df.valid_acc * 100, label="upright (valid)",
            color="#1a5276", lw=2)
    ax.plot(df.epoch, df.test_acc * 100, label="inverted (test, 180$\\degree$)",
            color="#b03a2e", lw=2)
    ax.fill_between(df.epoch, df.test_acc * 100, df.valid_acc * 100,
                    color="#b03a2e", alpha=0.10, lw=0,
                    label="inversion effect")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{name} — overall training")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    _draw_stages(ax, df)
    out = os.path.join(run_dir, "fig_training_overall.png")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_by_category(df, run_dir, name):
    cats = categories_in(df)
    fig, axes = plt.subplots(1, len(cats), figsize=(4.6 * len(cats), 4.4),
                             sharey=True, squeeze=False)
    for ax, cat in zip(axes[0], cats):
        up, inv = df[f"valid_acc_{cat}"] * 100, df[f"test_acc_{cat}"] * 100
        col = CAT_COLORS.get(cat, "#8e44ad")
        ax.plot(df.epoch, up, color=col, lw=2, label="upright")
        ax.plot(df.epoch, inv, color=col, lw=1.6, ls="--", label="inverted")
        ax.fill_between(df.epoch, inv, up, color=col, alpha=0.15, lw=0)
        ax.set_title(CAT_LABELS.get(cat, cat), fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, lw=0.5)
        _draw_stages(ax, df, label=False)
        ax.legend(loc="lower right", fontsize=8)
    axes[0][0].set_ylabel("accuracy (%)")
    fig.suptitle(f"{name} — upright vs inverted by category", fontsize=11)
    out = os.path.join(run_dir, "fig_training_bycat.png")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_inversion_effect(df, run_dir, name):
    """upright - inverted, per category, over training.

    The human pattern is a large and growing gap for faces and a flat near-zero
    one for objects and houses. This is the figure that says whether a model
    develops that dissociation, and when.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat in categories_in(df):
        gap = (df[f"valid_acc_{cat}"] - df[f"test_acc_{cat}"]) * 100
        ax.plot(df.epoch, gap, lw=2, color=CAT_COLORS.get(cat, "#8e44ad"),
                label=CAT_LABELS.get(cat, cat))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("inversion effect (upright − inverted, % points)")
    ax.set_title(f"{name} — inversion effect over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    _draw_stages(ax, df)
    out = os.path.join(run_dir, "fig_inversion_effect.png")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_compare(run_dirs, out_path):
    """One inversion-effect panel per model, shared axes, for the r10/r11/r12
    contrast. Only the categories a model actually has are drawn."""
    fig, axes = plt.subplots(1, len(run_dirs), figsize=(4.8 * len(run_dirs), 4.4),
                            sharey=True, squeeze=False)
    for ax, rd in zip(axes[0], run_dirs):
        df = load_history(rd)
        for cat in categories_in(df):
            gap = (df[f"valid_acc_{cat}"] - df[f"test_acc_{cat}"]) * 100
            ax.plot(df.epoch, gap, lw=2, color=CAT_COLORS.get(cat, "#8e44ad"),
                    label=CAT_LABELS.get(cat, cat))
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(os.path.basename(rd.rstrip("/")), fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, lw=0.5)
        _draw_stages(ax, df, label=False)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("inversion effect (% points)")
    fig.suptitle("Inversion effect over training", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--compare", default=None,
                    help="also write a side-by-side inversion-effect figure here")
    args = ap.parse_args()

    for rd in args.run_dirs:
        df = load_history(rd)
        name = os.path.basename(rd.rstrip("/"))
        for out in (plot_overall(df, rd, name),
                    plot_by_category(df, rd, name),
                    plot_inversion_effect(df, rd, name)):
            print(f"wrote {out}")
        last = df.iloc[-1]
        cats = categories_in(df)
        print(f"  {name}: epoch {int(last.epoch)}  upright {last.valid_acc*100:.2f}%  "
              f"inverted {last.test_acc*100:.2f}%")
        for c in cats:
            gap = (last[f"valid_acc_{c}"] - last[f"test_acc_{c}"]) * 100
            print(f"    {c:14s} upright {last[f'valid_acc_{c}']*100:6.2f}%  "
                  f"inverted {last[f'test_acc_{c}']*100:6.2f}%  effect {gap:+6.2f}")

    if args.compare:
        print(f"wrote {plot_compare(args.run_dirs, args.compare)}")


if __name__ == "__main__":
    main()
