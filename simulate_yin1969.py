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


def encode(model, transform, crops, device, p_noise):
    """Raw uint8 fixation crops -> noisy stochastic code h (on CPU)."""
    x = transform(crops.to(device))
    model.stochastic = True
    _, h, _ = model(x, return_rep=True)
    return apply_binomial_noise(h.cpu(), p_noise)


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
                  study_tf, test_tf, p_noise):
    """One Yin 2AFC condition. `study_tf`/`test_tf` are OnTheFlyTransforms that
    fix the orientation (upright vs inverted) of the study and test phases."""
    set_seed(args.seed)

    memory_bank = {}
    test_old_idx = {}
    with torch.no_grad():
        # ---- study phase: build the noisy memory bank ----
        for img_idx, iid in study_items:
            study_crops = load_item_fixations(sp, img_idx, args.study_fixations, offset=0)
            test_crops = load_item_fixations(sp, img_idx, args.test_fixations, offset=0)
            if study_crops is None or test_crops is None:
                continue
            memory_bank[iid] = encode(model, study_tf, study_crops, device, p_noise)
            test_old_idx[iid] = img_idx

        # ---- new (never-studied) distractor pool ----
        unknown_idx = {}
        for img_idx, iid in unknown_items:
            if load_item_fixations(sp, img_idx, args.test_fixations, offset=0) is not None:
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
            new_crops = load_item_fixations(sp, unknown_idx[new_items[i]], args.test_fixations, offset=0)
            h_old = encode(model, test_tf, old_crops, device, p_noise)
            h_new = encode(model, test_tf, new_crops, device, p_noise)
            if compute_familiarity_score(h_old, memory_bank, args.sigma) > \
               compute_familiarity_score(h_new, memory_bank, args.sigma):
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
    ap.add_argument("--num-study", type=int, default=40)
    ap.add_argument("--num-test", type=int, default=24)
    ap.add_argument("--study-fixations", type=int, default=10)
    ap.add_argument("--test-fixations", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=None,
                    help="fixed retrieval-noise p to use; skips calibration when set")
    ap.add_argument("--calib-target", type=float, default=0.96,
                    help="upright-upright accuracy to match when calibrating noise")
    ap.add_argument("--calib-max", type=float, default=0.75)
    ap.add_argument("--calib-step", type=float, default=0.05)
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
    need = args.num_study + args.num_test
    if len(all_items) < need:
        raise SystemExit(f"Category '{args.category}' has {len(all_items)} usable items; "
                         f"need >= {need} (num_study + num_test).")
    study_items = all_items[:args.num_study]
    unknown_items = all_items[args.num_study:need]

    # 1) calibrate noise on the upright-upright condition (skipped if --noise given)
    if args.noise is not None:
        ideal_noise = args.noise
        print(f"[!] Using fixed noise p={ideal_noise:.2f} (calibration skipped)\n")
    else:
        print(f"--- Calibrating noise on {args.category} (upright-upright) ---")
        ideal_noise = None
        for p in np.arange(0.0, args.calib_max, args.calib_step):
            acc = run_condition(model, device, args, sp, study_items, unknown_items,
                                upright_tf, upright_tf, p)
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
                            s_tf, t_tf, ideal_noise)
        rows.append({"Study": s_cond, "Test": t_cond, "Model Accuracy": f"{acc*100:.2f}%"})

    print("=====================================================")
    print(f" YIN (1969) SIMULATION — {args.category} (noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
