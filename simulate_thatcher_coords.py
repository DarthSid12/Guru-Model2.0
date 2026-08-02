"""
simulate_thatcher_coords.py

Verifies the emergence of the thatcher effect on "Model 2.1" (with coordinates of fixation crop as input) as follows:

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

from model_coords import ModelCoords

from salience_trans import ThatcherTransform

from utils import normalize_coords_tensor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

NUM_STUDY_IMAGES = 2 # number of images per person (identity) that go into model memory
NUM_PROBE_IMAGES = 2

# ==================================================
# Helpers
# ==================================================

# --------------------------------------------------
# Data
# --------------------------------------------------

"""
Loads:
data/thatcher_data/<upright|inverted>/<normal|thatcher>/<identity>/*.png
BOTH images AND coords (coords are None for cnn case)
"""
def load_trial_data(device,
                    processed_root,
                    category,
                    variant,
                    image_type,
                    split,
                    classes,
                    num_images,
                    offset):
    """
    Returns
        samples[class] = (images, coords)

    images:
        cnn -> (N,3,H,W)
        lp  -> (N_fixations_total,3,H,W)

    coords:
        cnn -> (N,2) zeros
        lp  -> (N_fixations_total,2)
    """

    samples = {}

    # CNN images
    cnn_dir = os.path.join(
        processed_root,
        category,
        "cnn",
        image_type,
        split,
    )

    # LP coordinate txt files
    lp_dir = os.path.join(
        processed_root,
        category,
        "lp",
        image_type,
        split,
    )

    for cls in classes:

        cls_cnn = os.path.join(cnn_dir, cls)
        cls_lp = os.path.join(lp_dir, cls)

        if not os.path.isdir(cls_cnn):
            continue

        image_files = sorted(
            f for f in os.listdir(cls_cnn)
            if f.lower().endswith(".png")
        )

        chosen = image_files[offset:offset + num_images]

        # -------------------------------------------------------
        # CNN
        # -------------------------------------------------------
        if variant == "cnn":

            imgs = torch.stack([
                TF.to_tensor(
                    Image.open(os.path.join(cls_cnn, f)).convert("RGB")
                )
                for f in chosen
            ])

            coords = torch.zeros(len(chosen), 2)

            samples[cls] = (imgs, coords)
            continue

        # -------------------------------------------------------
        # LP
        # -------------------------------------------------------

        all_imgs = []
        all_coords = []

        transformer = ThatcherTransform(variant="lp", device=device)

        for fname in chosen:

            stem = os.path.splitext(fname)[0]

            img = TF.to_tensor(
                Image.open(
                    os.path.join(cls_cnn, fname)
                ).convert("RGB")
            )

            txt_path = os.path.join(
                cls_lp,
                stem + ".txt"
            )

            coords = np.loadtxt(txt_path)

            if coords.ndim == 1:
                coords = coords[None]

            for x, y in coords:

                crop = TF.crop(
                    img,
                    top=int(y) - 90,
                    left=int(x) - 90,
                    height=180,
                    width=180,
                )

                all_imgs.append(crop)

            all_coords.append(
                torch.tensor(coords, dtype=torch.float32)
            )

        imgs = torch.stack(all_imgs)
        imgs = transformer(imgs)

        coords = torch.cat(all_coords, dim=0)
        normalize_coords_tensor(coords, 224)

        samples[cls] = (imgs, coords)

    return samples

def list_classes(processed_root, category, variant, split):
    d = os.path.join(processed_root, category, variant, split)
    classes = sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))

    #print("classes[0]:", classes[0])

    return classes


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
    ap.add_argument("--processed-root", default="data/thatcher_data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default=None, help="label_map.json from the run (sets num_classes)")
    ap.add_argument("--num-classes", type=int, default=None, help="used if --label-map absent")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--num-study", type=int, default=117)
    ap.add_argument("--num-test", type=int, default=117)
    ap.add_argument("--study-images", type=int, default=NUM_STUDY_IMAGES)
    ap.add_argument("--probe-images", type=int, default=NUM_PROBE_IMAGES)
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
# DEBUG OUTPUT HELPERS
# ==================================================

import matplotlib.pyplot as plt

"""
To check whether study and probe are indeed paired by the same identity
"""
def show_example_pairs(processed_root, category, classes,
                       study_orientation="upright",
                       study_type="normal",
                       probe_orientation="upright",
                       probe_type="thatcher",
                       study_offset=0,
                       probe_offset=4,
                       num_examples=5):

    base = os.path.join(processed_root, category)

    for cls in classes[:num_examples]:

        study_dir = os.path.join(base, study_orientation, study_type, cls)
        probe_dir = os.path.join(base, probe_orientation, probe_type, cls)

        study_files = sorted(
            f for f in os.listdir(study_dir)
            if f.lower().endswith(".png")
        )

        probe_files = sorted(
            f for f in os.listdir(probe_dir)
            if f.lower().endswith(".png")
        )

        study_img = Image.open(
            os.path.join(study_dir, study_files[study_offset])
        )

        probe_img = Image.open(
            os.path.join(probe_dir, probe_files[probe_offset])
        )

        fig, ax = plt.subplots(1, 2, figsize=(6,3))

        ax[0].imshow(study_img)
        ax[0].set_title(f"{cls}\nStudy")

        ax[1].imshow(probe_img)
        ax[1].set_title(f"{cls}\nProbe")

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()

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

    # apply noise to simulate imperfect human vision?
    noisy_study = False
    noisy_probe = False

    orientation, thatcher = condition

    set_seed(args.seed)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------   
    study_data = load_trial_data(
        device,
        args.processed_root,
        args.category,
        args.variant,
        image_type="upright",
        split="normal",
        classes=study_classes,
        num_images=args.study_images,
        offset=0
    )

    probe_data = load_trial_data(
        device,
        args.processed_root,
        args.category,
        args.variant,
        image_type = orientation,
        split = thatcher,
        classes=study_classes,
        num_images=args.probe_images,
        offset=args.study_images # so the images are new
    )

    with torch.no_grad():

        # --------------------------------------------------
        # Memory bank
        # --------------------------------------------------
        memory_bank = {}

        for cls, (imgs, coords) in study_data.items():
            model.stochastic = True

            _, h, _ = model(
                imgs.to(device),
                coords.to(device),
                return_rep=True
            )

            if noisy_study:
                memory_bank[cls] = apply_binomial_noise(
                    h.cpu(),
                    p_noise
                )
            else:
                memory_bank[cls] = h.cpu()

        # identities present in both memory and probe
        probe_items = sorted(
            set(study_data.keys()) & # & operator used for set intersection here
            set(probe_data.keys())
        )

        if len(probe_items) == 0: # hopefully not ...
            raise RuntimeError("No identities shared between study and probe data.")

        probe_items = sorted(probe_items)

        if args.num_test is not None:
            probe_items = probe_items[:min(args.num_test, len(probe_items))]

        # --------------------------------------------------
        # Familiarity calculation
        # --------------------------------------------------
        familiarity_scores = {}

        for cls in probe_items:

            model.stochastic = True

            imgs, coords = probe_data[cls]

            _, h_probe, _ = model(
                imgs.to(device),
                coords.to(device),
                return_rep=True
            )

            if noisy_probe:
                h_probe = apply_binomial_noise(
                    h_probe.cpu(),
                    p_noise
                )
            else:
                h_probe = h_probe.cpu()

            familiarity = compute_familiarity_score(
                h_probe,
                memory_bank,
                args.sigma
            )

            familiarity_scores[cls] = familiarity

    return familiarity_scores, float(np.mean(list(familiarity_scores.values()))) 

"""
Helper method to return a good "difference" between two statistics
As of yet we simply return an ACTUAL difference - I'm just concerned this may not be the best metric
"""

def diff(stat_a, stat_b):
    return stat_a - stat_b

# assume same keys for both maps; same value types
# recursive in case of nested maps
def diff_map(map_a, map_b):
    if not isinstance(map_a, dict):
        return diff(map_a, map_b)

    result = {}
    for k in map_a.keys():
        result[k] = diff_map(map_a[k], map_b[k])

    return result
    

"""
Runs above function on both upright and inverted faces, with both normal and "thatcherized" data

Compares diff(familiarity(upright, normal), familiarity(upright, thatcher)) to the same but inverted
OUR HYPOTHESIS: higher in upright case
"""

def main(debug=False):

    # This is just for me in case I don't have GPU access to train --David
    pretrain = False

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

    model = ModelCoords(size=180, num_classes=num_classes, pretrained=pretrain, T=args.temperature).to(device)

    if not pretrain:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)

    model.eval()

    # disjoint study / unknown class sets within the category
    all_classes = list_classes(args.processed_root, args.category, args.variant, "upright/normal")
    if len(all_classes) < args.num_study:
        raise SystemExit(f"Category '{args.category}' has {len(all_classes)} classes; "
                         f"need >= {args.num_study} study identities.")

    # pick study classes (with a random shuffle, depending on shuffle_classes)
    if shuffle_classes:
        random.shuffle(all_classes)

    study_classes = all_classes[:args.num_study]

    if debug:
        show_example_pairs(
            args.processed_root,
            args.category,
            study_classes,
            study_orientation="upright",
            study_type="normal",
            probe_orientation="upright", # or "inverted"
            probe_type="normal", # or "thatcher"
            study_offset=0,
            probe_offset=args.study_images,
            num_examples=5
        )

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
        ("upright", "normal"),
        ("upright", "thatcher"),
        ("inverted", "normal"),
        ("inverted", "thatcher"),       
    )  

    # get scores and whole dicts for statistical inference
    fam_scores = {}
    fam_dicts = {}

    for c in conditions:
        fam_dicts[c], fam_scores[c] = familiarity(model, device, args, study_classes, ideal_noise, c)

    # --------------------------------------------------
    # Compare thatcher effect for upright vs inverted
    # --------------------------------------------------     
    E_u = diff(fam_scores[conditions[0]], fam_scores[conditions[1]]) # upright effect
    E_i = diff(fam_scores[conditions[2]], fam_scores[conditions[3]]) # inverted effect

    # output

    print(f"Upright Normal: {fam_scores[conditions[0]]}")
    print(f"Upright Thatcher: {fam_scores[conditions[1]]}")

    print()

    print(f"Inverted Normal: {fam_scores[conditions[2]]}")
    print(f"Inverted Thatcher: {fam_scores[conditions[3]]}")

    print()

    print(f"Upright Thatcher Effect: {E_u}")
    print(f"Inverted Thatcher Effect: {E_i}")

    print()

    # and now for the big one:

    PVALUE = 0.05
    stat_results = stat_sig_thatcher_effect(fam_dicts, conditions)
    print("statistical test of results yields:")
    print(stat_results)

    if stat_results.pvalue < PVALUE:
        if E_u > E_i:
            print("Observed Thatcher effect!")
        else:
            print("Observed opposite of Thatcher effect ...")
    else:
        print("Did not observe Thatcher effect.")

"""
This function runs a t-test over identities on the results of the experimental run
to determine if we saw a statistically significant thatcher effect.
"""
def stat_sig_thatcher_effect(fam_dicts, conditions):

    from scipy.stats import ttest_1samp

    Emap_u = diff_map(fam_dicts[conditions[0]], fam_dicts[conditions[1]]) # map with keys as identities
    Emap_i = diff_map(fam_dicts[conditions[2]], fam_dicts[conditions[3]]) # also map with keys as identities

    Dmap = diff_map(Emap_u, Emap_i) # D for difference

    return ttest_1samp(list(Dmap.values()), 0.0)

if __name__ == "__main__":
    main()