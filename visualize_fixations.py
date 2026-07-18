"""
visualize_fixations.py

Diagnostic: for a handful of (failing) classes, render each sampled valid image
as a 3-band figure:
  band 1  the full 224x224 image with its Gabor-saliency fixation points
  band 2  per-fixation crops after the cnn-variant eval transform (foveate only)
  band 3  the same crops after the lp-variant eval transform (foveate + log-polar)
Both bands use OnTheFlyTransform('valid', ...), i.e. exactly what the model
sees at eval time minus/plus the log-polar step, so coverage failures
(fixations missing the object) and representation failures (log-polar erasing
thin structure) can be told apart by eye.

Example:
    python visualize_fixations.py                          # 10 worst tool classes
    python visualize_fixations.py --classes n04270147 n03481172 --num-images 4
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import _PackedSplit, _crop_at
from salience_trans import OnTheFlyTransform

# worst valid-accuracy object classes from the mid-train per-class eval
# (per_class_accuracy_midtrain.csv, resnet18-pt + resnet34, 2026-07-16)
WORST_CLASSES = {
    "n04270147": "spatula",
    "n03498962": "hatchet",
    "n03633091": "ladle",
    "n03481172": "hammer",
    "n04376876": "syringe",
    "n03958227": "plastic bag",
    "n04154565": "screwdriver",
    "n04008634": "projectile",
    "n03970156": "plunger",
    "n03532672": "hook",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(WORST_CLASSES),
                    help="class names within --category (default: 10 worst object classes)")
    ap.add_argument("--category", default="objects")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--fixation-root", default="fixation_data")
    ap.add_argument("--num-images", type=int, default=2, help="images per class")
    ap.add_argument("--num-fixations", type=int, default=16,
                    help="fixations to mark on the full image (training order)")
    ap.add_argument("--show-fixations", type=int, default=8,
                    help="how many of those fixations to render as crop columns")
    ap.add_argument("--seed", type=int, default=0, help="image sampling seed")
    ap.add_argument("--out-dir", default="runs/fixation_viz")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sp = _PackedSplit(args.fixation_root, args.category, args.split)
    class_to_imgs = {}
    for i, ci in enumerate(sp.labels):
        class_to_imgs.setdefault(sp.classes[ci], []).append(i)

    # eval-time transforms: no rotation/flip/aug on 'valid', only foveation (+ log-polar)
    tf_cnn = OnTheFlyTransform("valid", "cnn")
    tf_lp = OnTheFlyTransform("valid", "lp")

    rng = np.random.default_rng(args.seed)
    n_show = args.show_fixations
    for cls in args.classes:
        if cls not in class_to_imgs:
            print(f"[skip] {cls}: not found in {args.category}/{args.split}")
            continue
        pretty = WORST_CLASSES.get(cls, "")
        picks = rng.choice(class_to_imgs[cls], size=min(args.num_images, len(class_to_imgs[cls])),
                           replace=False)
        for k, i in enumerate(picks):
            img = np.array(sp.images[i])                      # (H, W, 3) uint8
            coords = np.array(sp.coords[i, :args.num_fixations])  # (N, 2) as (x, y)
            crops = torch.stack([_crop_at(sp.images[i], x, y, 180)
                                 for x, y in coords[:n_show]])     # (n, 3, 180, 180)
            with torch.no_grad():
                cnn = tf_cnn(crops.clone()).permute(0, 2, 3, 1).numpy()
                lp = tf_lp(crops.clone()).permute(0, 2, 3, 1).numpy()

            fig = plt.figure(figsize=(2.0 * n_show, 6.8))
            gs = fig.add_gridspec(3, n_show, height_ratios=[1.3, 1, 1], hspace=0.15, wspace=0.05)

            ax = fig.add_subplot(gs[0, :])
            ax.imshow(img)
            ax.scatter(coords[:, 0], coords[:, 1], s=90, facecolors="none",
                       edgecolors="red", linewidths=1.5)
            for j, (x, y) in enumerate(coords):
                ax.annotate(str(j), (x, y), color="yellow", fontsize=8,
                            ha="center", va="center")
            ax.set_title(f"{args.category}/{cls} {pretty}  ({sp.stems[i]}) — "
                         f"first {args.num_fixations} fixations", fontsize=11)
            ax.axis("off")

            for j in range(n_show):
                for row, (band, name) in enumerate([(cnn, "foveated (cnn)"),
                                                    (lp, "log-polar (lp)")], start=1):
                    ax = fig.add_subplot(gs[row, j])
                    ax.imshow(band[j].clip(0, 1))
                    ax.axis("off")
                    if j == 0:
                        ax.set_ylabel(name)
                        ax.axis("on")
                        ax.set_xticks([]); ax.set_yticks([])
                    if row == 1:
                        ax.set_title(f"fix {j}", fontsize=9)

            out = os.path.join(args.out_dir, f"{args.category}_{cls}_{k}_{sp.stems[i]}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
