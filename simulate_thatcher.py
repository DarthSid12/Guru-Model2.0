"""
simulate_thatcher.py

Verifies the emergence of the thatcher effect on Model 2.0 as follows:

1. Train the model

2. Have it study normal upright faces

3. Probe it with thatcher-effect vs non-thatcher-effect images, 
and calculate how the thatcherization affects the familiarity score 
(thatcherized = lower familiarity if it looks less like a face to the model)

4. Do this for both upright and inverted probe images and compare effects

HYPOTHESIS - larger difference for upright faces
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

# ==================================================
# Helpers (TODO: add to utils ?
# ==================================================

# --------------------------------------------------
# Data
# --------------------------------------------------

# TODO: Implement actual Thatcherization load - right now image_type doesn't do anything yet
def load_trial_data(processed_root, category, variant, image_type, split, classes,
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

# --------------------------------------------------
# Barrington KDE
# --------------------------------------------------

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

# ==================================================
# ARG PARSING FOR THATCHER EXP
# ==================================================

"""
Note: for --category, only faces makes sense as a category as of yet.

However, will maintain further categories in case we later decide to
use a generic high-variance-point measure to try and create an 
artificial Thatcher-like effect for houses and objects too.
"""

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
    ap.add_argument("--noise", type=float, default=0.25)
    return ap.parse_args()

# ==================================================
# THATCHER EXPERIMENT
# ==================================================

"""
Input:

model
device
args (from command line)
p_noise

Condition ("Upright" / "Inverted" for how to load data; boolean representing whether or not we thatcherize the test)
format: ("Upright"/"Inverted", True/False)

Returns:
B-KDE familiarity score
"""

def familiarity(model, device, args, study_classes, p_noise, condition):

    orientation, thatcher = condition

    set_seed(args.seed)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------   
    study_data = load_trial_data(
        args.processed_root,
        args.category,
        args.variant,
        image_type="normal",
        split="valid",
        classes=study_classes,
        num_fixations=args.study_fixations,
        offset=0,
    )

    probe_data = load_trial_data(
        args.processed_root,
        args.category,
        args.variant,
        image_type="thatcher" if thatcher else "normal",
        split = "valid" if orientation == "Upright" else "test",
        classes=study_classes,
        num_fixations=args.test_fixations,
        offset=0,
    )

    with torch.no_grad():

        # --------------------------------------------------
        # Memory bank
        # --------------------------------------------------
        memory_bank = {}

        for cls, imgs in study_data.items():
            model.stochastic = True

            _, h, _ = model(
                imgs.to(device),
                return_rep=True
            )

            memory_bank[cls] = apply_binomial_noise(
                h.cpu(),
                p_noise
            )

        # identities present in both memory and probe
        probe_items = sorted(
            set(study_data.keys()) & # & operator used for set intersection here
            set(probe_data.keys())
        )

        if len(probe_items) == 0: # hopefully not ...
            raise RuntimeError("No identities shared between study and probe data.")

        if args.num_test is not None:
            probe_items = random.sample(
                probe_items,
                min(args.num_test, len(probe_items))
            )

        # --------------------------------------------------
        # Familiarity calculation
        # --------------------------------------------------
        familiarity_scores = []

        for cls in probe_items:

            model.stochastic = True

            _, h_probe, _ = model(
                probe_data[cls].to(device),
                return_rep=True
            )

            h_probe = apply_binomial_noise(
                h_probe.cpu(),
                p_noise
            )

            familiarity = compute_familiarity_score(
                h_probe,
                memory_bank,
                args.sigma
            )

            familiarity_scores.append(familiarity)

    return float(np.mean(familiarity_scores))

"""
Helper method to return a good "difference" between two statistics
As of yet we simply return an ACTUAL difference - I'm just concerned this may not be the best metric
"""

def diff(stat_a, stat_b):
    return stat_a - stat_b

"""
Runs above function on both upright and inverted faces, with both normal and "thatcherized" data

Compares diff(familiarity(upright, normal), familiarity(upright, thatcher)) to the same but inverted
OUR HYPOTHESIS: higher in upright case
"""

def main():

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------   
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # shuffle all_classes when studying for fair representation?
    shuffle_classes = True

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
    if len(all_classes) < args.num_study:
        raise SystemExit(f"Category '{args.category}' has {len(all_classes)} classes; "
                         f"need >= {args.num_study} study identities.")

    # pick study classes (with a random shuffle, depending on shuffle_classes)
    if shuffle_classes:
        random.shuffle(all_classes)

    study_classes = all_classes[:args.num_study]

    """
    Without a run condition this doesn't work anymore
    Maybe simply use the same noise as Yin?

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
    """

    ideal_noise = args.noise
    print(f"[!] Using noise p={ideal_noise:.2f}\n")

    # --------------------------------------------------
    # Get familiarity for each condition
    # -------------------------------------------------- 
    conditions = (
        ("Upright", False),
        ("Upright", True),
        ("Inverted", False),
        ("Inverted", True),       
    )   

    fam = [familiarity(model, device, args, study_classes, ideal_noise, c) for c in conditions]

    # --------------------------------------------------
    # Compare thatcher effect for upright vs inverted
    # --------------------------------------------------     
    upright_thatcher_effect = diff(fam[0], fam[1])
    inverted_thatcher_effect = diff(fam[2], fam[3])

    # debug output

    print(f"Upright Normal: {fam[0]}")
    print(f"Upright Thatcher: {fam[1]}")

    print()

    print(f"Inverted Normal: {fam[2]}")
    print(f"Inverted Thatcher: {fam[3]}")

    print()

    print(f"Upright Thatcher Effect: {upright_thatcher_effect}")
    print(f"Inverted Thatcher Effect: {inverted_thatcher_effect}")

    print()

    # and now for the big one
    if upright_thatcher_effect > inverted_thatcher_effect:
        print("Observed Thatcher effect!")
    else:
        print("Did not observe Thatcher effect.")

if __name__ == "__main__":
    main()