"""
plot_fixation_pca.py

PCA scatter of the model's 256-d code, one marker per fixation:

    SQUARE  = a study fixation  (10 per item, the Yin study budget)
    CIRCLE  = a test fixation   (32 per item, the Yin test budget)
    COLOR   = identity          (one color per identity, shared by both phases)

Each phase carries its own ORIENTATION (--study-orientation /
--test-orientation, default upright for both). Setting them apart turns the
square/circle contrast into the Yin manipulation itself -- e.g.

    --study-orientation upright --test-orientation inverted

draws each identity's 10 upright fixations as squares and its 32 inverted
fixations as circles, in one PCA fitted over both, so the question "does
inverting the face move the code off its upright cluster, and does it move
every identity the same way?" is read directly off the plot. Inverted here is
exactly what simulate_yin1969.py uses: OnTheFlyTransform("test"), a 180-degree
rotation applied to the crop before foveation and the log-polar transform.

fixation_data is packed with 32 coords, so both budgets are drawn from the same
fixation sequence at offset 0 -- the study squares are the first 10 of the test
circles' 32, which is exactly what simulate_yin1969.py feeds the memory bank vs.
the 2AFC probe.

WHICH representation (--rep) is the thing to be careful about, because the Yin
simulation does not score the clean code:

    probs   deterministic sigmoid, no sampling, no noise          (default here)
    h       Bernoulli sample of probs   (model.stochastic = True)
    noisy   Bernoulli sample + binomial retrieval noise at --noise p

`noisy` is what simulate_yin1969.encode() actually builds -- it sets
stochastic=True and then calls apply_binomial_noise(h, p_noise). `probs` is the
clean upper bound: useful for seeing whether identity structure exists at all,
but it is NOT what the familiarity model reads. Compare the two when asking why
a set that looks separable still scores near chance under Yin.

Four plots by default, matching the faces-familiarity design already used by
simulate_yin1969.py / make_faces_mix64.py ('faces' = the 128 identities every
model trained on; 'faces_setA' = 40 identities novel to every model):

    setA40   all 40 novel Set_A identities
    known64  64 random trained identities   (of 128 in 'faces')
    setA4    4 random novel Set_A identities
    known4   4 random trained identities

Each plot fits its own PCA (2 components) on just its own points, so it shows
the structure of that specific stimulus set rather than a shared projection.

One representative photo per identity is used (same convention as
simulate_yin1969.py's build_items), since these plots are about fixation
structure within/across identities, not photo-to-photo variation.

Plots are written to fixation_pca/<run-dir basename>/ (override with --out-dir),
one file per preset per representation:

    fixation_pca/r10_dev_h80/fixation_pca_known4_probs.png
    fixation_pca/r10_dev_h80/fixation_pca_known4_noisy024.png

Example (point --run-dir at a training run):
    python plot_fixation_pca.py --run-dir runs/r10_dev_h80
    python plot_fixation_pca.py --run-dir runs/r10_dev_h80 --noise 0.24
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import _PackedSplit
from model import Model
from salience_trans import OnTheFlyTransform
from simulate_yin1969 import (apply_binomial_noise, build_items,
                              load_item_fixations, resolve_from_run_dir, set_seed)

# name -> (category, num_items or None for "all"). The random draw uses --seed,
# shared across presets so a run is reproducible.
PRESETS = {
    "setA40":  ("faces_setA", None),
    "known64": ("faces", 64),
    "setA4":   ("faces_setA", 4),
    "known4":  ("faces", 4),
    # Chicago Face Database white males: 93 identities, one neutral photo each,
    # novel to every model. One image per identity is all these plots use
    # anyway (build_items convention), so the single-photo store is not a
    # limitation here the way it would be for the Kanwisher matching task.
    "cfdWM4":  ("faces_cfdWM", 4),
    "cfdWMall": ("faces_cfdWM", None),
    # For r13 / r14, which trained CFD faces in from scratch. Four groups that
    # cross stimulus domain (CFD studio vs CelebA in-the-wild) with familiarity
    # (identity trained-on vs never seen), so the plots isolate which of the two
    # the 256-d code separates on.
    "cfdKnown4":   ("faces_cfd", 4),        # trained CFD identity, held-out photo
    "celebKnown4": ("faces", 4),            # trained CelebA identity, held-out photo
    "cfdNovel4":   ("faces_cfdWM64", 4),    # CFD domain, identity never seen
    "setANovel4":  ("faces_setA", 4),       # novel domain AND novel identity
    # For r15_vgg, which trained 480 VGGFace2 identities. faces_vggHO is the
    # mirror's val/ folder -- 60 people absent from faces_vgg -- so vggNovel4 is
    # the same DOMAIN the model was trained on with identities it has never seen,
    # the cleanest available split of "knows this kind of photograph" from
    # "knows this person". vggKnown4 is its trained-identity counterpart.
    "vggKnown4": ("faces_vgg", 4),          # trained VGGFace2 identity, held-out photo
    "vggNovel4": ("faces_vggHO", 4),        # trained domain, identity never seen
    "vggNovelAll": ("faces_vggHO", None),   # all 60 held-out identities
}
TITLES = {
    "setA40":  "Set A faces -- all 40 identities (novel)",
    "known64": "Known faces -- 64 random identities (trained)",
    "setA4":   "Set A faces -- 4 random identities (novel)",
    "known4":  "Known faces -- 4 random identities (trained)",
    "cfdWM4":  "CFD white male faces -- 4 random identities (novel)",
    "cfdWMall": "CFD white male faces -- all 93 identities (novel)",
    "cfdKnown4":   "CFD faces -- 4 TRAINED identities (held-out photo)",
    "celebKnown4": "CelebA faces -- 4 TRAINED identities (held-out photo)",
    "cfdNovel4":   "CFD faces -- 4 NOVEL identities (held-out 64)",
    "setANovel4":  "Set_A faces -- 4 NOVEL identities (novel domain)",
    "vggKnown4": "VGGFace2 -- 4 TRAINED identities (held-out photo)",
    "vggNovel4": "VGGFace2 -- 4 NOVEL identities (trained domain)",
    "vggNovelAll": "VGGFace2 -- all 60 NOVEL identities (trained domain)",
}

# Validated categorical slots (light surface, CVD-checked) for the small plots,
# where identity is a lookup key and gets a legend. The large plots need 40-64
# distinct colors, which no validated categorical set provides -- there color is
# a grouping cue for reading cluster structure, not a key, so the tab20 family
# is sampled instead and no legend is drawn.
SMALL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59d0",
                 "#d64a8a", "#c9a227", "#4aa8c9", "#6b6b6b"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="training run dir; auto-loads config.json (variant, "
                         "backbone, temperature), best_model.pth and "
                         "label_map.json unless the matching flag is given")
    ap.add_argument("--variant", choices=["lp", "cnn", "plain"], default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--packed-root", default="fixation_data")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--study-fixations", type=int, default=10)
    ap.add_argument("--test-fixations", type=int, default=32)
    ap.add_argument("--study-orientation", choices=["upright", "inverted"],
                    default="upright",
                    help="orientation of the square (study) fixations")
    ap.add_argument("--test-orientation", choices=["upright", "inverted"],
                    default="upright",
                    help="orientation of the circle (test) fixations; set it to "
                         "'inverted' against an upright study to plot the Yin "
                         "upright-vs-inverted contrast in one projection")
    ap.add_argument("--rep", choices=["probs", "h", "noisy"], default=None,
                    help="which 256-d code to plot: 'probs' = clean deterministic "
                         "sigmoid (default), 'h' = Bernoulli sample, 'noisy' = "
                         "Bernoulli sample + binomial retrieval noise at --noise "
                         "(what simulate_yin1969.encode actually scores). Passing "
                         "--noise on its own implies --rep noisy.")
    ap.add_argument("--noise", type=float, default=None,
                    help="binomial retrieval-noise p. Giving it selects --rep noisy "
                         "unless --rep says otherwise; defaults to 0.24 if --rep "
                         "noisy is asked for without a p.")
    ap.add_argument("--presets", nargs="+", default=list(PRESETS),
                    choices=list(PRESETS), help="which plots to produce")
    # ---- naming identities explicitly instead of drawing them ----
    ap.add_argument("--identities", nargs="+", default=None,
                    help="plot exactly these class names (e.g. EmmaStone "
                         "MargotRobbie) as a single figure, instead of the "
                         "--presets random draws. Names must match the packed "
                         "store's classes exactly.")
    ap.add_argument("--identity-category", default="faces",
                    help="packed category --identities are drawn from")
    ap.add_argument("--title", default=None, help="title override for --identities")
    ap.add_argument("--out-name", default="named",
                    help="filename stem for the --identities figure")
    ap.add_argument("--seed", type=int, default=0, help="item-sampling / noise seed")
    ap.add_argument("--out-dir", default=None,
                    help="default: fixation_pca/<run-dir basename>")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # --noise on its own means "show me what Yin actually scores", so it selects
    # the noisy code rather than being silently ignored under the default rep.
    if args.rep is None:
        args.rep = "noisy" if args.noise is not None else "probs"
    if args.rep == "noisy" and args.noise is None:
        args.noise = 0.24
    if args.rep != "noisy" and args.noise is not None:
        print(f"[!] --noise {args.noise} ignored: --rep {args.rep} is a clean code.")

    # Outputs go beside the repo, not into runs/ (which holds checkpoints and
    # training artefacts); one subfolder per run so several runs can coexist.
    if args.out_dir is None:
        tag = os.path.basename(args.run_dir.rstrip("/")) if args.run_dir else "adhoc"
        args.out_dir = os.path.join("fixation_pca", tag)
    return args


def load_model(args, device):
    if args.label_map:
        with open(args.label_map) as f:
            num_classes = len(json.load(f))
    elif args.num_classes:
        num_classes = args.num_classes
    else:
        raise SystemExit("Provide --label-map (or --run-dir) or --num-classes to size the model head.")
    model = Model(size=180, num_classes=num_classes, pretrained=False,
                  T=args.temperature, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()
    return model


def encode_crops(model, transform, crops, device, rep, p_noise):
    """Raw uint8 fixation crops (F, 3, H, W) -> (F, 256) numpy code.

    Mirrors simulate_yin1969.encode() when rep == 'noisy'."""
    x = transform(crops.to(device))
    model.stochastic = rep in ("h", "noisy")
    with torch.no_grad():
        _, h, probs = model(x, return_rep=True)
    if rep == "probs":
        return probs.cpu().numpy()
    if rep == "h":
        return h.cpu().numpy()
    return apply_binomial_noise(h.cpu(), p_noise).numpy()


def collect_embeddings(model, transforms, sp, items, args, device):
    """Return X (N, 256), phase (N,) in {'study','test'}, ident (N,) identity
    name, for every item with enough packed coords for both budgets.

    `transforms` maps phase -> OnTheFlyTransform, so the two phases can be shown
    in different orientations (the crops are identical; only the rotation the
    transform applies differs)."""
    X, phase, ident = [], [], []
    for img_idx, iid in items:
        study_crops = load_item_fixations(sp, img_idx, args.study_fixations, offset=0)
        test_crops = load_item_fixations(sp, img_idx, args.test_fixations, offset=0)
        if study_crops is None or test_crops is None:
            continue
        for crops, tag, n in ((study_crops, "study", args.study_fixations),
                              (test_crops, "test", args.test_fixations)):
            X.append(encode_crops(model, transforms[tag], crops, device,
                                  args.rep, args.noise))
            phase += [tag] * n
            ident += [iid] * n
    if not X:
        return None
    return np.concatenate(X, axis=0), np.array(phase), np.array(ident)


def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S[:2] ** 2) / (S ** 2).sum()


def identity_colors(names):
    """One color per identity, in sorted order so it is stable across reps."""
    n = len(names)
    if n <= len(SMALL_PALETTE):
        cols = SMALL_PALETTE[:n]
    else:
        pool = []
        for cm in ("tab20", "tab20b", "tab20c"):
            pool += [matplotlib.colors.to_hex(c) for c in plt.get_cmap(cm).colors]
        if n > len(pool):  # >60 identities: fall back to an even hue sweep
            pool = [matplotlib.colors.to_hex(plt.get_cmap("hsv")(i / n)) for i in range(n)]
        cols = pool[:n]
    return dict(zip(names, cols))


def plot_pca(coords, phase, ident, var_ratio, title, subtitle, out_path,
             study_label="study fixation", test_label="test fixation"):
    names = sorted(set(ident))
    cmap = identity_colors(names)
    point_colors = np.array([cmap[i] for i in ident])

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # test circles first (more numerous, sit underneath), study squares on top
    test_mask, study_mask = phase == "test", phase == "study"
    ax.scatter(coords[test_mask, 0], coords[test_mask, 1], marker="o", s=20,
               c=point_colors[test_mask], edgecolors="none", alpha=0.55, zorder=2)
    ax.scatter(coords[study_mask, 0], coords[study_mask, 1], marker="s", s=44,
               c=point_colors[study_mask], edgecolors=SURFACE, linewidths=0.7, zorder=3)

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)", color=INK)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)", color=INK)
    ax.set_title(f"{title}\n{subtitle}", color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)

    # Legends sit OUTSIDE the axes: inside, a box large enough to hold 4 identity
    # names reliably covered part of a cluster, which is the one thing this plot
    # exists to show.
    shape_handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                   markerfacecolor=MUTED, markeredgecolor=SURFACE, label=study_label),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                   markerfacecolor=MUTED, markeredgecolor="none", label=test_label),
    ]
    leg = ax.legend(handles=shape_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    leg.get_frame().set_linewidth(0.6)
    legends = [leg]
    # identity legend only when it is readable (<= 8); beyond that color is a
    # grouping cue for cluster structure, not a per-name lookup key
    if len(names) <= len(SMALL_PALETTE):
        id_handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                                 markerfacecolor=cmap[n], markeredgecolor=SURFACE, label=n)
                      for n in names]
        ax.add_artist(leg)
        leg2 = ax.legend(handles=id_handles, loc="upper left", bbox_to_anchor=(1.02, 0.84),
                         frameon=True, title="identity", facecolor=SURFACE,
                         edgecolor=GRID, fontsize=9)
        leg2.get_frame().set_linewidth(0.6)
        legends.append(leg2)

    # Reserve the right margin for the outside legends, then grow the saved bbox
    # to whatever they actually needed. bbox_extra_artists is required: a legend
    # re-attached with add_artist is not picked up by bbox_inches="tight" alone,
    # which silently clipped the wider (shape) legend.
    fig.subplots_adjust(left=0.08, right=0.76, top=0.90, bottom=0.10)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight",
                bbox_extra_artists=legends)
    plt.close(fig)


def main():
    args = parse_args()
    resolve_from_run_dir(args)
    if args.variant is None:
        args.variant = "lp"
    if args.backbone is None:
        args.backbone = "resnet18"
    if args.temperature is None:
        args.temperature = 2.0
    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint or --run-dir.")

    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args, device)
    # "test" is OnTheFlyTransform's name for the 180-degree rotation, the same
    # one simulate_yin1969.py inverts with; "valid" is the identity rotation.
    tf_by_orientation = {
        "upright": OnTheFlyTransform("valid", args.variant, device).to(device),
        "inverted": OnTheFlyTransform("test", args.variant, device).to(device),
    }
    transforms = {"study": tf_by_orientation[args.study_orientation],
                  "test": tf_by_orientation[args.test_orientation]}
    study_label = f"{args.study_fixations} {args.study_orientation} fixations"
    test_label = f"{args.test_fixations} {args.test_orientation} fixations"
    # Only tag the filename when an orientation is actually manipulated, so
    # existing upright-upright outputs keep their current names.
    orient_tag = "" if args.study_orientation == args.test_orientation == "upright" \
        else f"_{args.study_orientation[:3]}{args.test_orientation[:3]}"
    if args.study_orientation != args.test_orientation:
        print(f"orientation: study {args.study_orientation}, "
              f"test {args.test_orientation} (one PCA fitted over both)")

    rep_label = {"probs": "deterministic sigmoid, no noise",
                 "h": "Bernoulli sample of h, no retrieval noise",
                 "noisy": f"Bernoulli h + binomial noise p={args.noise:.2f}"
                          if args.noise is not None else "noisy"}[args.rep]
    print(f"representation: {args.rep} ({rep_label})")

    os.makedirs(args.out_dir, exist_ok=True)
    packed_cache = {}

    # A "job" is (stem, category, selector, title); selector is a count to draw
    # (or None for all) from a preset, or an explicit list of class names.
    if args.identities:
        n = len(args.identities)
        jobs = [(args.out_name, args.identity_category, list(args.identities),
                 args.title or f"{n} named identities -- {args.identity_category}")]
    else:
        jobs = [(k, PRESETS[k][0], PRESETS[k][1], TITLES[k]) for k in args.presets]

    for name, category, selector, title in jobs:
        if category not in packed_cache:
            packed_cache[category] = _PackedSplit(args.packed_root, category, args.split)
        sp = packed_cache[category]

        need = max(args.study_fixations, args.test_fixations)
        if need > sp.num_coords:
            print(f"[skip] {name}: {category}/{args.split} packed with "
                  f"{sp.num_coords} coords, need {need}.")
            continue

        items = build_items(sp)
        if isinstance(selector, list):
            wanted = set(selector)
            items = [(i, iid) for i, iid in items if iid in wanted]
            missing = wanted - {iid for _, iid in items}
            if missing:
                raise SystemExit(
                    f"--identities not found in {category}/{args.split}: "
                    f"{sorted(missing)}. The store has {len(sp.classes)} classes; "
                    f"names must match exactly (e.g. 'EmmaStone', not 'Emma Stone').")
        elif selector is not None and len(items) > selector:
            # Seeded PER PRESET, not from one shared stream: a shared RNG makes
            # the draw depend on which other presets are being plotted, so
            # `--presets known4` picked different identities than the same
            # preset inside a full run and the two plots were not comparable.
            items = random.Random(f"{args.seed}:{name}").sample(items, selector)
        items = sorted(items, key=lambda t: t[1])

        result = collect_embeddings(model, transforms, sp, items, args, device)
        if result is None:
            print(f"[skip] {name}: no items had enough packed coords.")
            continue
        X, phase, ident = result
        n_items = len(set(ident))
        print(f"[{name}] {n_items} identities, {X.shape[0]} fixation points, "
              f"embedding dim {X.shape[1]}")

        coords, var_ratio = pca_2d(X)
        suffix = args.rep if args.rep != "noisy" else f"noisy{args.noise:.2f}".replace(".", "")
        out_path = os.path.join(args.out_dir,
                                f"fixation_pca_{name}{orient_tag}_{suffix}.png")
        subtitle = (f"{n_items} identities x ({args.study_fixations} "
                    f"{args.study_orientation} + {args.test_fixations} "
                    f"{args.test_orientation}) fixations | 256-d code, {rep_label}")
        plot_pca(coords, phase, ident, var_ratio, title, subtitle, out_path,
                 study_label=study_label, test_label=test_label)
        print(f"  -> {out_path}  (PC1+PC2 = {var_ratio.sum()*100:.1f}% var)")


if __name__ == "__main__":
    main()
