"""
make_deck_figures.py

Render the real-image assets the methods deck needs (slides 1, 2, 3, 6) straight
from the packed fixation stores, using the SAME transform classes the model is
trained and evaluated with -- so the pipeline panels are what the network
actually sees, not a redrawing of it.

    python scripts/make_deck_figures.py --out paper/figures

Everything runs on CPU: it is a handful of images and the GPUs are busy with the
simulation battery.

NOTE on the saliency heat map (slide 2). The stored coords in coords.npy cannot
be reproduced by re-running the sampler -- torch.multinomial is not seeded there,
and preprocess_fixations.py deliberately never re-packs for that reason. So the
heat map here is RECOMPUTED for display while the dots are the REAL stored
coordinates the model trained on. They come from the same distribution, and the
dots are the ground truth; the heat map is illustrative. Say so in the caption.

The saliency operator runs in log-polar space, so the map is scattered back to
image coordinates through the same xMap/yMap the sampler uses, mean-pooled per
image pixel. That back-projection is for display only and exists nowhere in the
training path.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fnn

from datasets import _PackedSplit, _crop_at
from salience_trans import SaliencePipeline, OnTheFlyTransform, ImageNetAugment
from trans import Rotate, Foveate, LogPolar

DEV = "cpu"
CROP = 180


def img_t(sp, i):
    """Packed uint8 image -> float (1,3,224,224) in [0,1]."""
    return torch.from_numpy(np.array(sp.images[i])).permute(2, 0, 1).unsqueeze(0).float() / 255.


def show(ax, t, title=None, ts=8):
    """t: (C,H,W) or (1,C,H,W) float in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    ax.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=ts)


def saliency_map(pipe, im):
    """Recompute the Gabor-variance saliency map and scatter it back to image
    space for display. Mirrors SaliencePipeline.sample_salience_points up to the
    variance, then mean-pools into a 224x224 grid via xMap/yMap."""
    import torchvision.transforms.functional as TF
    B, _, H, W = im.shape
    g = TF.rgb_to_grayscale(im, num_output_channels=1)
    x = torch.arange(0, W); y = torch.arange(0, H)
    yg, xg = torch.meshgrid(y, x, indexing="ij")
    sigma, alpha = W / 4, 2
    mask = torch.exp(-((xg - (W/2-.5))**2 + (yg - (H/2-.5))**2) / (2*sigma**2)) ** alpha
    wimg = mask.repeat(B, 1, 1).unsqueeze(1) * g
    wimg, xMap, yMap = pipe.gaborlogpolar.forwardReturnMapping(wimg)
    wimg = (wimg - wimg.mean((2, 3), keepdim=True)) / (wimg.std((2, 3), keepdim=True) + 1e-9)
    with torch.no_grad():
        f = pipe.kernels(wimg)
    f[..., :10, :] = 0; f[..., -10:, :] = 0; f[..., :, :10] = 0; f[..., :, -10:] = 0
    n = f.shape[1] // 2
    mag = torch.sqrt(f[:, :n]**2 + f[:, n:]**2 + 1e-9)
    var = torch.var(mag, dim=1)[0]                       # (h, w) in log-polar space
    acc = torch.zeros(H, W); cnt = torch.zeros(H, W)
    xi = xMap.long().clamp(0, W-1); yi = yMap.long().clamp(0, H-1)
    acc.index_put_((yi.flatten(), xi.flatten()), var.flatten(), accumulate=True)
    cnt.index_put_((yi.flatten(), xi.flatten()), torch.ones_like(var).flatten(), accumulate=True)
    m = acc / cnt.clamp(min=1)
    k = torch.tensor([1., 4., 6., 4., 1.]); k = (k / k.sum())
    m = m[None, None]
    m = Fnn.conv2d(Fnn.pad(m, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = Fnn.conv2d(Fnn.pad(m, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, -1, 1))
    return m[0, 0]


# ---------------------------------------------------------------- slide 1 + 6
FACE_SRC = [("faces", "CelebA"), ("faces_vgg", "VGGFace2"),
            ("faces_rfwW", "RFW — Caucasian"), ("faces_rfwO", "RFW — other")]
HELD = [("faces_cfdWM64", "faces_cfdWM64\nCFD, studio"),
        ("faces_rfwWM64", "faces_rfwWM64\nRFW, in-the-wild"),
        ("houses_yin64",  "houses_yin64\nZuBuD 41–104"),
        ("objects",       "objects (valid)\ncategories ARE trained")]


def slide1(out, pick):
    fig, axes = plt.subplots(3, 4, figsize=(9, 7.2))
    rows = [("Faces — identity", FACE_SRC),
            ("Objects — basic-level category", [("objects", "ImageNet")] * 4),
            ("Houses", [("houses_zubud137", "ZuBuD identity")] * 3 + [("houses", "generic 'house'")])]
    for r, (label, srcs) in enumerate(rows):
        for c, (cat, name) in enumerate(srcs):
            sp = _PackedSplit("fixation_data", cat, "train")
            i = pick(sp, r * 4 + c)
            first_of_source = (c == 0) or (srcs[c][1] != srcs[c-1][1])
            show(axes[r, c], img_t(sp, i)[0], name if first_of_source else None)
        axes[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Training sources — solid = trained on", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/slide1_training_sources.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.9))
    for c, (cat, name) in enumerate(HELD):
        sp = _PackedSplit("fixation_data", cat, "valid")
        show(axes[c], img_t(sp, pick(sp, c))[0], name, ts=8)
        for s in axes[c].spines.values():
            s.set_linestyle((0, (3, 3))); s.set_linewidth(1.6)
    fig.suptitle("Held-out test stores — 64 items each", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/slide1_heldout_stores.png", dpi=220); plt.close(fig)


# -------------------------------------------------------------------- slide 2
def slide2(out, sp, i):
    pipe = SaliencePipeline(type="valid", device=DEV, num_salient_points=32)
    im = img_t(sp, i)
    sal = saliency_map(pipe, im).numpy()
    lo, hi = np.percentile(sal, 40), np.percentile(sal, 99.5)   # stretch: the map is very long-tailed
    sal = np.clip((sal - lo) / (hi - lo + 1e-9), 0, 1)
    xy = np.array(sp.coords[i])

    fig = plt.figure(figsize=(12.6, 3.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .72, 2.0], wspace=.22)

    show(fig.add_subplot(gs[0]), im[0], "224×224 source", ts=9)

    ax = fig.add_subplot(gs[1]); ax.imshow(sal, cmap="inferno")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Gabor orientation-energy\nsaliency (recomputed)", fontsize=9)

    ax = fig.add_subplot(gs[2]); show(ax, im[0])
    ax.imshow(sal, alpha=.35, cmap="inferno")
    ax.scatter(xy[:16, 0], xy[:16, 1], s=30, facecolors="#18f0c8", edgecolors="k", linewidths=.6, zorder=3)
    ax.scatter(xy[16:, 0], xy[16:, 1], s=30, facecolors="none", edgecolors="w", linewidths=1.2, zorder=3)
    ax.set_title("32 sampled points\nfilled = the 16 used in training", fontsize=9)

    # use a visibly off-centre fixation so the crop is not a copy of the source
    ecc = np.sqrt((xy[:16, 0] - 112.) ** 2 + (xy[:16, 1] - 112.) ** 2)
    j = int(np.argmin(np.abs(ecc - 30)))     # off-centre, but the crop stays mostly on-image
    base = _crop_at(sp.images[i], xy[j, 0], xy[j, 1], CROP).float().unsqueeze(0) / 255.
    show(fig.add_subplot(gs[3]), base[0], f"180×180 crop\nat fixation {j}", ts=9)

    sub = gs[4].subgridspec(1, 4, wspace=.08)
    aug = ImageNetAugment(); rot = Rotate(deg=15.)
    torch.manual_seed(3)
    for k in range(4):
        ax = fig.add_subplot(sub[k]); show(ax, rot(aug(base))[0])
        if k == 0:
            ax.annotate("the SAME crop, re-augmented each epoch — scale jitter + ±15°",
                        xy=(0, 1.06), xycoords="axes fraction", fontsize=9, ha="left")
    fig.savefig(f"{out}/slide2_fixations_and_augmentation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- slide 3
def slide3(out, sp, i):
    xy = np.array(sp.coords[i])
    im = img_t(sp, i)
    ecc = np.sqrt((xy[:16, 0] - 112.) ** 2 + (xy[:16, 1] - 112.) ** 2)
    j = int(np.argmin(np.abs(ecc - 30)))     # off-centre, but the crop stays mostly on-image
    crop = _crop_at(sp.images[i], xy[j, 0], xy[j, 1], CROP).float().unsqueeze(0) / 255.
    rot = Rotate(deg=15.); fov = Foveate(crop_size=CROP)
    lp = LogPolar(input_shape=(CROP, CROP), output_shape=(CROP, CROP), device=DEV)
    torch.manual_seed(1)
    r = rot(crop); f = fov(r); l = lp(f)
    stages = [(im, "224×224 source"), (crop, f"180×180 crop\n@ fixation {j}"),
              (r, "rotation\nU[−15°,15°]"), (f, "foveation\nJiang 2015"), (l, "log-polar\n→ ResNet-18")]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    for ax, (t, name) in zip(axes, stages):
        show(ax, t, name, ts=9)
    fig.suptitle("Transformation pipeline — every crop, on GPU, re-randomised each epoch", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/slide3_pipeline.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.7))
    for ax, v in zip(axes, ["lp", "cnn", "plain"]):
        t = OnTheFlyTransform("valid", v, DEV, crop_size=CROP)(crop)
        show(ax, t, {"lp": "lp  ← used\nfoveation + log-polar", "cnn": "cnn\nfoveation only",
                     "plain": "plain\nneither"}[v], ts=9)
    axes[0].patch.set_edgecolor("k")
    fig.suptitle("Front-end variants (the project's core ablation)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/slide3_variants.png", dpi=220); plt.close(fig)


# -------------------------------------------------------------------- slide 6
def slide6(out, sp, i):
    """The Yin 2x2 as STIMULUS orientation. The log-polar render is slide 3's job;
    mixing the two in one panel made it unreadable."""
    xy = np.array(sp.coords[i])
    ecc = np.sqrt((xy[:16, 0] - 112.) ** 2 + (xy[:16, 1] - 112.) ** 2)
    j = int(np.argmin(np.abs(ecc - 30)))     # off-centre, but the crop stays mostly on-image
    crop = _crop_at(sp.images[i], xy[j, 0], xy[j, 1], CROP).float().unsqueeze(0) / 255.
    # upright at simulation time is the IDENTITY, not the +-15 deg train jitter:
    # OnTheFlyTransform('valid') sets self.rotate = torch.nn.Identity().
    up, inv = torch.nn.Identity(), Rotate(invert=True)
    cells = [("UU", up, up), ("UI", up, inv), ("IU", inv, up), ("II", inv, inv)]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for ax, (name, srot, trot) in zip(axes.ravel(), cells):
        pair = torch.cat([srot(crop)[0], torch.ones(3, CROP, 10), trot(crop)[0]], dim=2)
        ax.imshow(pair.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_xticks([]); ax.set_yticks([])
        anchored = name in ("UU", "II")
        ax.set_title(f"{name}" + ("   ← anchored to human" if anchored else ""),
                     fontsize=11, fontweight="bold" if anchored else "normal")
        ax.set_xlabel("study            test", fontsize=9)
        for sp_ in ax.spines.values():
            sp_.set_linewidth(2.2 if anchored else .8)
    fig.suptitle("Yin 2×2 — orientation crossed between study and test\n"
                 "same images throughout; inverted = exactly 180°", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/slide6_yin_2x2.png", dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--face-index", type=int, default=7, help="index into faces/train")
    ap.add_argument("--seed", type=int, default=0,
                    help="reshuffles which example images slide 1 picks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pick = lambda sp, k: int(rng.integers(0, len(sp.labels)))
    faces = _PackedSplit("fixation_data", "faces", "train")
    slide1(a.out, pick)
    slide2(a.out, faces, a.face_index)
    slide3(a.out, faces, a.face_index)
    slide6(a.out, faces, a.face_index)
    print("wrote:", *sorted(os.listdir(a.out)), sep="\n  ")


if __name__ == "__main__":
    main()
