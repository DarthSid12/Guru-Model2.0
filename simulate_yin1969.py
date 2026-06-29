"""
simulate_yin1969.py

Replicate Yin (1969) for a single category (faces / houses / objects) using the
trained network and the Barrington-NIMBLE KDE familiarity model on the shared
256-d binary code `h`.

Design (per the report):
  - study : 40 classes, first 10 fixations each -> noisy memory bank
  - test  : 24 OLD (subset of studied) vs 24 NEW (never-studied distractors),
            32 fixations each, retrieval noise applied, 2AFC familiarity decision
  - orientation conditions select the preprocessed split:
        upright  -> "valid"   (preprocessed upright)
        inverted -> "test"    (preprocessed 180-deg inverted)

Example:
    python simulate_yin1969.py --category faces --variant lp \
        --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
        --label-map  runs/faces_objects_lp_32fix_lr0.001/label_map.json
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
def load_trial_data(processed_root, category, variant, split, classes,
                    num_fixations, offset=0):
    """Return {class: tensor(num_fixations, C, H, W)} for the requested classes,
    using one base image's first `num_fixations` proc crops (after `offset`)."""
    split_dir = os.path.join(processed_root, category, variant, split)
    samples = {}
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        base_dict = {}
        for fname in sorted(os.listdir(cls_dir)):
            if fname.endswith(".png") and "_proc" in fname:
                base = fname.split("_proc")[0]
                base_dict.setdefault(base, []).append(os.path.join(cls_dir, fname))
        if not base_dict:
            continue
        proc_list = sorted(base_dict[sorted(base_dict)[0]])
        chosen = proc_list[offset: offset + num_fixations]
        if len(chosen) < num_fixations:
            continue
        imgs = torch.stack([TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen], dim=0)
        samples[cls] = imgs
    return samples


def list_classes(processed_root, category, variant, split):
    d = os.path.join(processed_root, category, variant, split)
    return sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))


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
def run_condition(model, device, args, study_classes, unknown_classes,
                  study_split, test_split, p_noise):
    set_seed(args.seed)
    study_data = load_trial_data(args.processed_root, args.category, args.variant,
                                 study_split, study_classes, args.study_fixations, offset=0)
    test_old_all = load_trial_data(args.processed_root, args.category, args.variant,
                                   test_split, study_classes, args.test_fixations, offset=0)
    unknown_data = load_trial_data(args.processed_root, args.category, args.variant,
                                   test_split, unknown_classes, args.test_fixations, offset=0)

    memory_bank = {}
    with torch.no_grad():
        for cls, imgs in study_data.items():
            model.stochastic = True
            _, h, _ = model(imgs.to(device), return_rep=True)
            memory_bank[cls] = apply_binomial_noise(h.cpu(), p_noise)

        old_pool = [c for c in study_data if c in test_old_all]
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

    # disjoint study / unknown class sets within the category
    all_classes = list_classes(args.processed_root, args.category, args.variant, "valid")
    need = args.num_study + args.num_test
    if len(all_classes) < need:
        raise SystemExit(f"Category '{args.category}' has {len(all_classes)} classes; "
                         f"need >= {need} (num_study + num_test).")
    study_classes = all_classes[:args.num_study]
    unknown_classes = all_classes[args.num_study:need]

    # 1) calibrate noise on the upright-upright condition
    print(f"--- Calibrating noise on {args.category} (upright-upright) ---")
    ideal_noise = None
    for p in np.arange(0.0, args.calib_max, args.calib_step):
        acc = run_condition(model, device, args, study_classes, unknown_classes, "valid", "valid", p)
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
        ("Inverted", "Inverted", "test", "test"),
        ("Upright", "Inverted", "valid", "test"),
        ("Inverted", "Upright", "test", "valid"),
    ]
    rows = []
    for s_cond, t_cond, s_split, t_split in conditions:
        acc = run_condition(model, device, args, study_classes, unknown_classes, s_split, t_split, ideal_noise)
        rows.append({"Study": s_cond, "Test": t_cond, "Model Accuracy": f"{acc*100:.2f}%"})

    print("=====================================================")
    print(f" YIN (1969) SIMULATION — {args.category} (noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
