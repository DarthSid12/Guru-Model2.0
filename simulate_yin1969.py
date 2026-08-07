"""
simulate_yin1969.py

Replicate Yin (1969) for a single category (faces / houses / objects) using the
trained network and the Barrington-NIMBLE KDE familiarity model on the shared
256-d binary code `h`.

Design (per the report):
  - study : 40 items, first 10 fixations each -> noisy memory bank
  - test  : 24 OLD (subset of studied) vs 24 NEW (never-studied distractors),
            32 fixations each, retrieval noise applied, 2AFC familiarity decision
  - orientation conditions select the preprocessed split:
        upright  -> "valid"            (preprocessed upright)
        inverted -> "valid_inverted"   (same photos as "valid", 180-deg
                                         inverted -- NOT the raw "test" split,
                                         which holds different photos)

"Items" are class folders for multi-class categories (faces identities,
object categories) — one representative base image per class, same as
before. For single-class categories (houses: one generic "house" class,
many photos), items are instead individual photos within that class, since
each photo is its own memory item to recognize.

Example:
    python simulate_yin1969.py --category faces --variant lp \
        --checkpoint runs/faces_objects_lp_16fix_lr0.001/best_model.pth \
        --label-map  runs/faces_objects_lp_16fix_lr0.001/label_map.json
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import Model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- data -----------------------------
def list_classes(processed_root, category, variant, split):
    d = os.path.join(processed_root, category, variant, split)
    return sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))


def _base_stems(cls_dir):
    """Sorted base-image stems (filenames before '_proc<n>.png') in a class dir."""
    stems = set()
    for fname in os.listdir(cls_dir):
        if fname.endswith(".png") and "_proc" in fname:
            stems.add(fname.split("_proc")[0])
    return sorted(stems)


def list_items(processed_root, category, variant, split):
    """Return the study/test "items" for a category as a list of (class, base_stem).

    Multi-class categories (faces identities, object categories): one item per
    class, using that class's first available base image — matches the
    original per-class behaviour.

    Single-class categories (houses: one generic "house" class with many
    photos): one item per individual photo, since each photo is the thing to
    be recognized.
    """
    split_dir = os.path.join(processed_root, category, variant, split)
    classes = list_classes(processed_root, category, variant, split)
    items = []
    if len(classes) == 1:
        cls = classes[0]
        for stem in _base_stems(os.path.join(split_dir, cls)):
            items.append((cls, stem))
    else:
        for cls in classes:
            stems = _base_stems(os.path.join(split_dir, cls))
            if stems:
                items.append((cls, stems[0]))
    return items


def item_id(item):
    cls, stem = item
    return f"{cls}/{stem}"


def load_item_fixations(
        processed_root,
        category,
        variant,
        split,
        item,
        num_fixations,
        offset=0
):
    """
    Return tensor(num_fixations, C, H, W).

    If split == "valid_inverted", load from "valid"
    and rotate every fixation crop 180 degrees.
    """

    load_split = "valid" if split == "valid_inverted" else split

    cls, stem = item
    cls_dir = os.path.join(
        processed_root,
        category,
        variant,
        load_split,
        cls
    )

    if not os.path.isdir(cls_dir):
        return None

    proc_list = sorted(
        f for f in os.listdir(cls_dir)
        if f.startswith(stem + "_proc") and f.endswith(".png")
    )

    chosen = proc_list[offset: offset + num_fixations]

    if len(chosen) < num_fixations:
        return None

    imgs = []

    for fname in chosen:

        img = TF.to_tensor(
            Image.open(
                os.path.join(cls_dir, fname)
            ).convert("RGB")
        )

        if split == "valid_inverted":
            img = TF.rotate(img, 180)

        imgs.append(img)

    return torch.stack(imgs, dim=0)

def load_trial_data(processed_root, category, variant, split, items,
                    num_fixations, offset=0):
    """Return {item_id: tensor(num_fixations, C, H, W)} for the requested items."""
    samples = {}
    for item in items:
        imgs = load_item_fixations(processed_root, category, variant, split, item,
                                   num_fixations, offset=offset)
        if imgs is not None:
            samples[item_id(item)] = imgs
    return samples


def apply_binomial_noise(binary_tensor, p_noise):
    if p_noise == 0.0:
        return binary_tensor
    mask = torch.rand_like(binary_tensor) < p_noise
    return torch.logical_xor(binary_tensor.bool(), mask).float()


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
def run_condition(model, device, args, study_items, unknown_items,
                  study_split, test_split, p_noise):
    set_seed(args.seed)
    study_data = load_trial_data(args.processed_root, args.category, args.variant,
                                 study_split, study_items, args.study_fixations, offset=0)
    test_old_all = load_trial_data(args.processed_root, args.category, args.variant,
                                   test_split, study_items, args.test_fixations, offset=0)
    unknown_data = load_trial_data(args.processed_root, args.category, args.variant,
                                   test_split, unknown_items, args.test_fixations, offset=0)

    memory_bank = {}
    with torch.no_grad():
        for item_key, imgs in study_data.items():
            model.stochastic = True
            _, h, _ = model(imgs.to(device), return_rep=True)
            memory_bank[item_key] = apply_binomial_noise(h.cpu(), p_noise)

        old_pool = [k for k in study_data if k in test_old_all]
        new_pool = list(unknown_data.keys())
        n_pairs = min(args.num_test, len(old_pool), len(new_pool))
        old_items = random.sample(old_pool, n_pairs)
        new_items = random.sample(new_pool, n_pairs)

        correct = 0
        for i in range(n_pairs):
            model.stochastic = True
            _, h_old, _ = model(test_old_all[old_items[i]].to(device), return_rep=True)
            _, h_new, _ = model(unknown_data[new_items[i]].to(device), return_rep=True)
            h_old = apply_binomial_noise(h_old.cpu(), p_noise)
            h_new = apply_binomial_noise(h_new.cpu(), p_noise)
            if compute_familiarity_score(h_old, memory_bank, args.sigma) > \
               compute_familiarity_score(h_new, memory_bank, args.sigma):
                correct += 1
    return correct / max(n_pairs, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="faces | houses | objects")
    ap.add_argument("--variant", choices=["lp", "cnn"], default="lp")
    ap.add_argument("--processed-root", default="processed_data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default=None, help="label_map.json from the run (sets num_classes)")
    ap.add_argument("--num-classes", type=int, default=None, help="used if --label-map absent")
    ap.add_argument("--temperature", type=float, default=2.0)
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.label_map:
        with open(args.label_map) as f:
            num_classes = len(json.load(f))
    elif args.num_classes:
        num_classes = args.num_classes
    else:
        raise SystemExit("Provide --label-map or --num-classes to size the model head.")

    model = Model(size=180, num_classes=num_classes, pretrained=False, T=args.temperature).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    # disjoint study / unknown items within the category (class-level for
    # faces/objects, individual photos for single-class categories like houses)
    all_items = list_items(args.processed_root, args.category, args.variant, "valid")
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
            acc = run_condition(model, device, args, study_items, unknown_items, "valid", "valid", p)
            print(f"  noise {p:.2f} -> {acc*100:.2f}%")
            if acc <= args.calib_target and ideal_noise is None:
                ideal_noise = p
                break
        if ideal_noise is None:
            ideal_noise = 0.25
        print(f"[!] Using noise p={ideal_noise:.2f}\n")

    # 2) all 4 Yin conditions
    conditions = [
        ("Upright", "Upright", "valid", "valid"),
        ("Inverted", "Inverted", "valid_inverted", "valid_inverted"),
        ("Upright", "Inverted", "valid", "valid_inverted"),
        ("Inverted", "Upright", "valid_inverted", "valid"),
    ]
    rows = []
    for s_cond, t_cond, s_split, t_split in conditions:
        acc = run_condition(model, device, args, study_items, unknown_items, s_split, t_split, ideal_noise)
        rows.append({"Study": s_cond, "Test": t_cond, "Model Accuracy": f"{acc*100:.2f}%"})

    print("=====================================================")
    print(f" YIN (1969) SIMULATION — {args.category} (noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()