"""
simulate_yin1969.py

Replicate Yin (1969) for a single category (faces / houses / objects) using the
trained network and the Barrington-NIMBLE KDE familiarity model on the shared
256-d binary code `h`.

Design (per the report):
  - study : 40 items, first 10 fixations each -> noisy memory bank
  - test  : 24 OLD (subset of studied) vs 24 NEW (never-studied distractors),
            32 fixations each, retrieval noise applied, 2AFC familiarity decision
  - orientation conditions are now applied ON THE FLY (packed pipeline):
        upright  -> OnTheFlyTransform("valid")  (identity rotation)
        inverted -> OnTheFlyTransform("test")   (180-degree rotation)
    There is no longer a separate "valid_inverted" split; both orientations are
    rendered from the same packed 'valid' images.

"Items" are drawn from the packed 'valid' split of the category. For multi-class
categories (faces identities, object categories) items are classes — one
representative image per class. For single-class categories (houses: one generic
"house" class, many photos) items are individual photos, since each photo is its
own memory item to recognize.

Data now comes from the packed fixation store produced by
preprocess_fixations.py:

    fixation_data/<category>/<split>/{images.npy, coords.npy, meta.json}

rather than the retired PNG pipeline (processed_data/.../*_proc<n>.png). All
rotate/foveate/log-polar rendering is done on the GPU on the fly, exactly as at
train time (salience_trans.OnTheFlyTransform).

Example (point --run-dir at a training run to auto-load its config, checkpoint
and label map):
    python simulate_yin1969.py --category faces \
        --run-dir runs/faces_objects_houses_lp_16fix_lr0.001_resnet34_r3_pruned_60ep

or spell the pieces out explicitly:
    python simulate_yin1969.py --category faces --variant lp --backbone resnet34 \
        --checkpoint runs/<run>/best_model.pth \
        --label-map  runs/<run>/label_map.json
"""

import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch

from datasets import _PackedSplit, _crop_at
from model import Model
from salience_trans import OnTheFlyTransform


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- data -----------------------------
def build_items(sp):
    """Return the study/test "items" for a packed split as a list of
    (img_idx, item_id).

    Multi-class categories (faces identities, object categories): one item per
    class, using that class's first packed image -- matches the original
    per-class behaviour.

    Single-class categories (houses: one generic "house" class with many
    photos): one item per individual photo, since each photo is the thing to be
    recognized.
    """
    labels = list(sp.labels)
    items = []
    if len(sp.classes) == 1:
        for i in range(len(labels)):
            items.append((i, f"{sp.classes[0]}/{sp.stems[i]}"))
    else:
        seen = set()
        for i, ci in enumerate(labels):
            if ci in seen:
                continue
            seen.add(ci)
            items.append((i, sp.classes[ci]))
    return items


def canon_class(name):
    """Canonical key for matching class names ACROSS stores.

    Different runs name the same building differently: the packed ZuBuD store
    uses 'house0001', while house_control_r1's label_map uses the bare building
    id '101'. Both denote ZuBuD building 101, so matching the raw strings silently
    finds nothing and every trained class survives the held-out filter. Reduce
    '<letters><zero-padded digits>' to its integer part so the two forms agree;
    anything else (face identities, ImageNet synsets) is returned unchanged and
    still matches itself exactly.
    """
    m = re.fullmatch(r"[A-Za-z_]*0*(\d+)", name)
    return m.group(1) if m else name


def seen_classes(label_map_path, category):
    """The class names under `category` in a run's label_map.json -- i.e. exactly
    what that model was trained on. Everything else in the packed store is
    held-out for it.

    Returns canonical keys (see canon_class), so compare with canon_class(...)
    on the other side. The label_map prefix need not equal the packed category
    name: house_control_r1 trained on a 40-building ZuBuD subset it filed under
    'houses/', and is scored on the packed 'houses_zubud' store, so callers pass
    --exclude-seen-category houses.
    """
    with open(label_map_path) as f:
        lm = json.load(f)
    prefix = category + "/"
    seen = {canon_class(k[len(prefix):]) for k in lm if k.startswith(prefix)}
    if not seen:
        # A model asked to drop its trained classes but matching nothing is
        # always a bug -- it silently scores the model on its own training
        # items. Fail loudly instead (this is how house_control_r1 was scored
        # on all 201 ZuBuD buildings, 40 of them trained-on, until 2026-08-21).
        raise SystemExit(
            f"--exclude-seen-from {label_map_path} lists no classes under "
            f"'{prefix}'. Prefixes present: "
            f"{sorted({k.split('/', 1)[0] for k in lm if '/' in k})}. "
            f"Pass --exclude-seen-category with the right one.")
    return seen


def drop_seen(items, sp, seen):
    """Keep only the items whose class the model never saw during training."""
    kept = [(i, iid) for i, iid in items
            if canon_class(sp.classes[sp.labels[i]]) not in seen]
    print(f"[!] {sp.category}/{len(items)} items -> {len(kept)} held-out "
          f"({len(items) - len(kept)} dropped as trained-on)")
    return kept


def load_item_fixations(sp, img_idx, num_fixations, offset=0, crop_size=180):
    """Return uint8 tensor(num_fixations, 3, crop_size, crop_size) of `img_idx`'s
    fixation crops (after `offset`), or None if that image lacks enough packed
    coords. Crops are raw; the caller applies OnTheFlyTransform on the GPU."""
    if offset + num_fixations > sp.num_coords:
        return None
    img = sp.images[img_idx]
    crops = []
    for j in range(offset, offset + num_fixations):
        x, y = sp.coords[img_idx, j]
        crops.append(_crop_at(img, x, y, crop_size))
    return torch.stack(crops, dim=0)


def apply_binomial_noise(binary_tensor, p_noise):
    if p_noise == 0.0:
        return binary_tensor
    mask = torch.rand_like(binary_tensor) < p_noise
    return torch.logical_xor(binary_tensor.bool(), mask).float()


# ------------------- intermediate-layer probes -------------------
# The default readout is the 256-bit bottleneck code `h`. --layer swaps in an
# earlier representation instead, to test WHERE in the network an effect first
# appears. The backbone is the torchvision resnet with avgpool+fc peeled off,
# so its residual stages are children [4..7] = layer1..layer4.
LAYER_INDEX = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
# Every probe is binarised to a {0,1} code before it reaches the memory bank, so
# the XOR retrieval noise and the KDE below are byte-identical to the `h` path
# and the LAYER is the only thing that changes. Thresholds are per-unit medians
# fitted ONCE on a reference sample (see fit_layer_thresholds) and then held
# fixed across study, test, and all four orientation conditions -- a per-call
# threshold would renormalise each item and destroy the distances the KDE reads.


def pooled_activations(model, x, layer):
    """Global-average-pooled activation at `layer` for a batch of crops."""
    if layer in ("h", "probs"):
        model.stochastic = (layer == "h")
        _, h, probs = model(x, return_rep=True)
        return h if layer == "h" else probs
    f = x
    for i, child in enumerate(model.backbone):
        f = child(f)
        if i == LAYER_INDEX[layer]:
            break
    return torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1)


def fit_layer_thresholds(model, transforms, crop_batches, device, layer):
    """Per-unit medians over a reference sample, pooled across the orientations
    in `transforms`. Mixing orientations keeps the code neutral: a threshold fitted
    on upright alone would define "on" relative to upright statistics and bias
    every inverted condition against it."""
    acts = []
    with torch.no_grad():
        for crops in crop_batches:
            for tf in transforms:
                acts.append(pooled_activations(model, tf(crops.to(device)), layer).cpu())
    return torch.median(torch.cat(acts, dim=0), dim=0).values


def encode(model, transform, crops, device, p_noise, layer="h", thresholds=None):
    """Raw uint8 fixation crops -> noisy binary code (on CPU)."""
    x = transform(crops.to(device))
    if layer == "h":
        model.stochastic = True
        _, h, _ = model(x, return_rep=True)
        return apply_binomial_noise(h.cpu(), p_noise)
    a = pooled_activations(model, x, layer).cpu()
    code = (a > thresholds).float()          # binarise against the fixed reference
    return apply_binomial_noise(code, p_noise)


# ----------------------- Barrington KDE -------------------------
def compute_p_f_given_c(f, M_c, sigma):
    dists = torch.sum((M_c - f) ** 2, dim=1)
    return torch.mean(torch.exp(-dists / (2 * sigma ** 2)))


def compute_familiarity_score(F_test, memory_bank, sigma):
    best = -float("inf")
    for _, M_c in memory_bank.items():
        ll = 0.0
        for i in range(F_test.size(0)):
            ll += torch.log(compute_p_f_given_c(F_test[i], M_c, sigma) + 1e-12).item()
        best = max(best, ll)
    return best


# ----------------------- Yin condition --------------------------
def run_condition(model, device, args, sp, study_items, unknown_items,
                  study_tf, test_tf, p_noise, sp_unknown=None):
    """One Yin 2AFC condition. `study_tf`/`test_tf` are OnTheFlyTransforms that
    fix the orientation (upright vs inverted) of the study and test phases.

    `sp_unknown` is the packed split the "new" distractors live in; it defaults
    to `sp`, but may be a different category/store entirely."""
    set_seed(args.seed)
    if sp_unknown is None:
        sp_unknown = sp

    memory_bank = {}
    test_old_idx = {}
    with torch.no_grad():
        # ---- study phase: build the noisy memory bank ----
        for img_idx, iid in study_items:
            study_crops = load_item_fixations(sp, img_idx, args.study_fixations, offset=0)
            test_crops = load_item_fixations(sp, img_idx, args.test_fixations, offset=0)
            if study_crops is None or test_crops is None:
                continue
            memory_bank[iid] = encode(model, study_tf, study_crops, device, p_noise,
                                      args.layer, getattr(args, '_thresholds', None))
            test_old_idx[iid] = img_idx

        # ---- new (never-studied) distractor pool ----
        unknown_idx = {}
        for img_idx, iid in unknown_items:
            if load_item_fixations(sp_unknown, img_idx, args.test_fixations, offset=0) is not None:
                unknown_idx[iid] = img_idx

        old_pool = list(test_old_idx.keys())
        new_pool = list(unknown_idx.keys())
        n_pairs = min(args.num_test, len(old_pool), len(new_pool))
        old_items = random.sample(old_pool, n_pairs)
        new_items = random.sample(new_pool, n_pairs)

        # ---- test phase: 2AFC familiarity decision ----
        correct = 0
        for i in range(n_pairs):
            old_crops = load_item_fixations(sp, test_old_idx[old_items[i]], args.test_fixations, offset=0)
            new_crops = load_item_fixations(sp_unknown, unknown_idx[new_items[i]], args.test_fixations, offset=0)
            h_old = encode(model, test_tf, old_crops, device, p_noise,
                           args.layer, getattr(args, '_thresholds', None))
            h_new = encode(model, test_tf, new_crops, device, p_noise,
                           args.layer, getattr(args, '_thresholds', None))
            sig = getattr(args, '_sigma_eff', args.sigma)
            if compute_familiarity_score(h_old, memory_bank, sig) > \
               compute_familiarity_score(h_new, memory_bank, sig):
                correct += 1
    return correct / max(n_pairs, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="faces | houses | objects")
    ap.add_argument("--run-dir", default=None,
                    help="training run dir; auto-loads config.json (variant, backbone, "
                         "temperature), best_model.pth and label_map.json unless the "
                         "matching flag is given explicitly")
    ap.add_argument("--variant", choices=["lp", "cnn", "plain"], default=None)
    ap.add_argument("--backbone", default=None, help="must match the trained checkpoint")
    ap.add_argument("--packed-root", default="fixation_data")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--label-map", default=None, help="label_map.json from the run (sets num_classes)")
    ap.add_argument("--num-classes", type=int, default=None, help="used if --label-map absent")
    ap.add_argument("--temperature", type=float, default=None)
    # ---- where the "new" (never-studied) distractors come from ----
    # By default they are the leftover items of --category itself. Pointing these
    # at another packed store lets the study set and the distractor set be drawn
    # from different stimulus sets (e.g. study the new Set_A faces, distract with
    # the older 128-identity set).
    ap.add_argument("--unknown-category", default=None,
                    help="draw the 'new' distractors from this category instead "
                         "of the leftover items of --category")
    ap.add_argument("--unknown-packed-root", default=None,
                    help="packed root for --unknown-category (defaults to --packed-root)")
    ap.add_argument("--unknown-split", default="valid",
                    help="split to draw --unknown-category distractors from")
    # ---- restricting items to classes the model never trained on ----
    ap.add_argument("--exclude-seen-from", default=None,
                    help="label_map.json; drop every class it lists for the "
                         "category being sampled, leaving only held-out items")
    ap.add_argument("--exclude-seen-category", default=None,
                    help="category prefix to read out of --exclude-seen-from "
                         "(defaults to the category being sampled)")
    ap.add_argument("--shuffle-items", action="store_true",
                    help="shuffle the item list (seeded) before splitting it into "
                         "study and never-studied pools. Off by default so the "
                         "existing categories keep their packed-order split; "
                         "needed for stores whose classes are grouped by "
                         "sub-population (faces_mix64 packs its 24 celeb "
                         "identities before its 40 Set_A ones, which would "
                         "otherwise put every celeb in study and make the "
                         "distractor pool purely Set_A)")
    ap.add_argument("--num-study", type=int, default=40)
    ap.add_argument("--num-test", type=int, default=24)
    ap.add_argument("--study-fixations", type=int, default=10)
    ap.add_argument("--test-fixations", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--layer", default="h",
                    choices=["h", "probs", "layer1", "layer2", "layer3", "layer4"],
                    help="which representation the memory model reads. 'h' (default) is "
                         "the 256-bit bottleneck code and reproduces the original "
                         "pipeline exactly. Anything else is pooled and binarised "
                         "against per-unit medians fitted on a mixed-orientation "
                         "reference sample; --sigma is rescaled by sqrt(D/256).")
    ap.add_argument("--noise", type=float, default=None,
                    help="fixed retrieval-noise p to use; skips calibration when set")
    ap.add_argument("--calib-target", type=float, default=0.96,
                    help="upright-upright accuracy to match when calibrating noise")
    ap.add_argument("--calib-max", type=float, default=0.75)
    ap.add_argument("--calib-step", type=float, default=0.05)
    ap.add_argument("--calib-grid", nargs="+", type=float, default=None,
                    help="report upright-upright accuracy at exactly these noise "
                         "levels and exit, instead of running the conditions")
    ap.add_argument("--calib-seeds", type=int, default=1,
                    help="seeds averaged per --calib-grid point; >1 smooths the "
                         "4.17-point-per-pair quantisation of a 24-pair score")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def resolve_from_run_dir(args):
    """Fill in checkpoint / label-map / variant / backbone / temperature from a
    training run dir, without overriding anything passed explicitly."""
    if not args.run_dir:
        return
    cfg_path = os.path.join(args.run_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.run_dir, "best_model.pth")
    if args.label_map is None:
        lm = os.path.join(args.run_dir, "label_map.json")
        if os.path.isfile(lm):
            args.label_map = lm
    if args.variant is None:
        args.variant = cfg.get("variant", "lp")
    if args.backbone is None:
        args.backbone = cfg.get("backbone", "resnet18")
    if args.temperature is None:
        args.temperature = cfg.get("temperature", 2.0)


def main():
    args = parse_args()
    resolve_from_run_dir(args)

    # defaults for anything still unset (no --run-dir given)
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

    # orientation transforms, applied on the GPU on the fly (packed pipeline)
    upright_tf = OnTheFlyTransform("valid", args.variant, device).to(device)
    inverted_tf = OnTheFlyTransform("test", args.variant, device).to(device)

    # packed 'valid' split for this category; disjoint study / unknown items
    sp = _PackedSplit(args.packed_root, args.category, "valid")
    if args.test_fixations > sp.num_coords or args.study_fixations > sp.num_coords:
        raise SystemExit(f"{args.category}/valid packed with {sp.num_coords} coords, but need "
                         f"study={args.study_fixations}, test={args.test_fixations}.")
    all_items = build_items(sp)
    if args.exclude_seen_from:
        all_items = drop_seen(all_items, sp,
                              seen_classes(args.exclude_seen_from,
                                           args.exclude_seen_category or args.category))
    if args.shuffle_items:
        # Its own RNG, so the item draw varies with --seed without disturbing the
        # global streams run_condition re-seeds per condition.
        random.Random(args.seed).shuffle(all_items)

    if args.unknown_category:
        # Study and distractor pools come from different stimulus sets, so they
        # are disjoint by construction and each is sized independently.
        sp_unknown = _PackedSplit(args.unknown_packed_root or args.packed_root,
                                  args.unknown_category, args.unknown_split)
        if args.test_fixations > sp_unknown.num_coords:
            raise SystemExit(f"{args.unknown_category}/{args.unknown_split} packed with "
                             f"{sp_unknown.num_coords} coords, need {args.test_fixations}.")
        unknown_all = build_items(sp_unknown)
        if args.exclude_seen_from:
            unknown_all = drop_seen(unknown_all, sp_unknown,
                                    seen_classes(args.exclude_seen_from,
                                                 args.unknown_category))
        num_study = min(args.num_study, len(all_items))
        num_test = min(args.num_test, len(unknown_all))
        if num_study < args.num_study or num_test < args.num_test:
            print(f"[!] scaling to study={num_study} ({args.category}), "
                  f"test={num_test} ({args.unknown_category}).")
        study_items = all_items[:num_study]
        unknown_items = unknown_all[:num_test]
    else:
        sp_unknown = sp
        need = args.num_study + args.num_test
        if len(all_items) < need:
            # Small categories (the 40-class house set) can't fill the 40/24
            # design, so scale the split down while keeping the study:test ratio.
            num_study = (len(all_items) * args.num_study) // (args.num_study + args.num_test)
            num_test = len(all_items) - num_study
            print(f"[!] '{args.category}' has {len(all_items)} items (< {need}); "
                  f"scaling to study={num_study}, test={num_test}.")
        else:
            num_study, num_test = args.num_study, args.num_test
        study_items = all_items[:num_study]
        unknown_items = all_items[num_study:need]

    # 0a) --layer probe setup. Fit the binarisation thresholds ONCE, here, so
    # every condition and every calibration seed reads the same code, then scale
    # the KDE bandwidth: the kernel is exp(-||f-m||^2 / 2 sigma^2) and on binary
    # codes ||f-m||^2 is Hamming distance, which grows ~linearly with the code
    # width. Holding sigma fixed while the width changes 64 -> 512 would silently
    # saturate the kernel, so sigma is scaled by sqrt(D/256) against the 256-bit
    # `h` the default sigma was tuned on.
    if args.layer != "h":
        ref = [c for c in (load_item_fixations(sp, i, args.study_fixations)
                           for i, _ in study_items[:16]) if c is not None]
        if not ref:
            raise SystemExit(f"--layer {args.layer}: no usable reference crops.")
        args._thresholds = fit_layer_thresholds(
            model, [upright_tf, inverted_tf], ref, device, args.layer)
        D = args._thresholds.numel()
        args._sigma_eff = args.sigma * (D / 256.0) ** 0.5
        print(f"[!] --layer {args.layer}: D={D} binary units, "
              f"sigma {args.sigma:.2f} -> {args._sigma_eff:.2f} "
              f"(x sqrt({D}/256)); thresholds fitted on {len(ref)} items "
              f"x 2 orientations\n")

    # 0) grid probe: report upright-upright accuracy at each requested p and
    # stop. Used by run_sim_seeds.py to fit ONE noise level jointly across
    # categories, which needs the same p measured on every category.
    if args.calib_grid:
        # Averaged over seeds: a single seed scores only 24 pairs, so accuracy is
        # quantised to 4.17-point steps and the curve is dominated by +-1-pair
        # jitter -- non-monotonic, and it may never even land on the value being
        # calibrated to. Averaging over item samples gives a curve smooth enough
        # to pick a p from.
        base_seed = args.seed
        for p in args.calib_grid:
            accs = []
            for k in range(args.calib_seeds):
                args.seed = base_seed + k
                accs.append(run_condition(model, device, args, sp, study_items,
                                          unknown_items, upright_tf, upright_tf, p,
                                          sp_unknown=sp_unknown))
            args.seed = base_seed
            print(f"  noise {p:.2f} -> {100 * sum(accs) / len(accs):.2f}%")
        return

    # 1) calibrate noise on the upright-upright condition (skipped if --noise given)
    if args.noise is not None:
        ideal_noise = args.noise
        print(f"[!] Using fixed noise p={ideal_noise:.2f} (calibration skipped)\n")
    else:
        print(f"--- Calibrating noise on {args.category} (upright-upright) ---")
        ideal_noise = None
        for p in np.arange(0.0, args.calib_max, args.calib_step):
            acc = run_condition(model, device, args, sp, study_items, unknown_items,
                                upright_tf, upright_tf, p, sp_unknown=sp_unknown)
            print(f"  noise {p:.2f} -> {acc*100:.2f}%")
            if acc <= args.calib_target and ideal_noise is None:
                ideal_noise = p
                break
        if ideal_noise is None:
            ideal_noise = 0.25
        print(f"[!] Using noise p={ideal_noise:.2f}\n")

    # 2) all 4 Yin conditions (orientation now set by the on-the-fly transform)
    conditions = [
        ("Upright", "Upright", upright_tf, upright_tf),
        ("Inverted", "Inverted", inverted_tf, inverted_tf),
        ("Upright", "Inverted", upright_tf, inverted_tf),
        ("Inverted", "Upright", inverted_tf, upright_tf),
    ]
    rows = []
    for s_cond, t_cond, s_tf, t_tf in conditions:
        acc = run_condition(model, device, args, sp, study_items, unknown_items,
                            s_tf, t_tf, ideal_noise, sp_unknown=sp_unknown)
        rows.append({"Study": s_cond, "Test": t_cond, "Model Accuracy": f"{acc*100:.2f}%"})

    print("=====================================================")
    print(f" YIN (1969) SIMULATION — {args.category} (noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
