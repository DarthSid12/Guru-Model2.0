"""
simulate_yin1969_coords.py

simulate yin with a coordinate model
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
from salience_trans import OnTheFlyTransform

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- packed fixation data -----------------------------

class PackedCategory:
    """
    Reads one category/split from fixation_data.

    Structure:
        fixation_data/
            faces/
                valid/
                    images.npy
                    coords.npy
                    meta.json
    """

    def __init__(self, fixation_root, category, split):
        d = os.path.join(fixation_root, category, split)

        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)

        self.images = np.load(
            os.path.join(d, "images.npy"),
            mmap_mode="r"
        )

        self.coords = np.load(
            os.path.join(d, "coords.npy"),
            mmap_mode="r"
        )

        self.category = category
        self.classes = self.meta["classes"]
        self.labels = self.meta["labels"]


def list_items(fixation_root, category, split):
    """
    Return item IDs.

    For faces/objects:
        one item = one class identity/category

    For houses:
        one item = one image
    """

    sp = PackedCategory(fixation_root, category, split)

    counts = len(sp.classes)

    items = []

    if counts == 1:
        # houses style
        for i in range(len(sp.labels)):
            items.append(i)

    else:
        # faces/objects style
        seen = set()

        for i, label in enumerate(sp.labels):
            if label not in seen:
                items.append(i)
                seen.add(label)

    return items


def item_label(sp, index):
    return sp.classes[sp.labels[index]]


def load_item_fixations(
        fixation_root,
        category,
        split,
        item,
        num_fixations,
        offset=0
):
    """
    Returns:
        images: (N,C,H,W)
        coords: (N,4)
    """

    sp = PackedCategory(
        fixation_root,
        category,
        split
    )

    imgs = []
    coords = []

    for j in range(
        offset,
        offset + num_fixations
    ):

        x, y = sp.coords[item, j]

        crop = torch.from_numpy(
            np.array(sp.images[item])
        ).permute(2,0,1)

        imgs.append(crop)

        # same coordinate convention as training
        xy = torch.tensor([
            x / sp.images.shape[2] - 0.5,
            y / sp.images.shape[1] - 0.5
        ])

        coords.append(xy)


    imgs = torch.stack(imgs)

    coords = torch.stack(coords)

    return imgs, coords

def load_trial_data(
        fixation_root,
        category,
        split,
        items,
        num_fixations,
        offset=0
):
    """
    Returns:

    {
        item_id:
            {
             "images": (N,C,H,W),
             "coords": (N,4)
            }
    }
    """

    sp = PackedCategory(
        fixation_root,
        category,
        split
    )

    samples = {}

    for item in items:

        imgs, coords = load_item_fixations(
            fixation_root,
            category,
            split,
            item,
            num_fixations,
            offset
        )

        samples[item] = {
            "images": imgs,
            "coords": coords,
            "label": item_label(sp,item)
        }

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

    # Runtime LP/CNN transform
    study_transform = OnTheFlyTransform(
        study_split,
        args.variant,
        device
    ).to(device)

    test_transform = OnTheFlyTransform(
        test_split,
        args.variant,
        device
    ).to(device)

    # Load packed fixation data
    study_data = load_trial_data(
        args.fixation_root,
        args.category,
        study_split,
        study_items,
        args.study_fixations,
        offset=0
    )

    test_old_all = load_trial_data(
        args.fixation_root,
        args.category,
        test_split,
        study_items,
        args.test_fixations,
        offset=0
    )

    unknown_data = load_trial_data(
        args.fixation_root,
        args.category,
        test_split,
        unknown_items,
        args.test_fixations,
        offset=0
    )


    memory_bank = {}

    with torch.no_grad():

        # -----------------------
        # Build memory
        # -----------------------
        for item_key, sample in study_data.items():

            imgs = sample["images"].to(device)
            coords = sample["coords"].to(device).float()

            # apply lp/cnn transform
            imgs = study_transform(imgs)

            model.stochastic = False

            logits, h, _ = model(
                imgs,
                coords,
                return_rep=True
            )
            '''
            logits2, h2, _ = model(
                imgs,
                torch.zeros_like(coords),
                return_rep=True
            )
            '''
            #print((logits2 - logits).abs().mean())

            memory_bank[item_key] = apply_binomial_noise(
                h.cpu(),
                p_noise
            )


        # -----------------------
        # 2AFC OLD vs NEW
        # -----------------------

        old_pool = [
            k for k in study_data.keys()
            if k in test_old_all
        ]

        new_pool = list(unknown_data.keys())

        n_pairs = min(
            args.num_test,
            len(old_pool),
            len(new_pool)
        )

        old_items = random.sample(old_pool, n_pairs)
        new_items = random.sample(new_pool, n_pairs)

        correct = 0


        for i in range(n_pairs):

            model.stochastic = False

            # OLD item
            old = test_old_all[old_items[i]]

            old_imgs = old["images"].to(device)
            old_coords = old["coords"].to(device).float()

            old_imgs = test_transform(old_imgs)

            logits_old, h_old, _ = model(
                old_imgs,
                old_coords,
                return_rep=True
            )
            '''
            logits2_old, h2_old, _ = model(
                old_imgs,
                torch.zeros_like(old_coords),
                return_rep=True
            )
            '''
            #print((logits2_old - logits_old).abs().mean())

            # NEW item
            new = unknown_data[new_items[i]]

            new_imgs = new["images"].to(device)
            new_coords = new["coords"].to(device).float()

            new_imgs = test_transform(new_imgs)

            logits_new, h_new, _ = model(
                new_imgs,
                new_coords,
                return_rep=True
            )
            '''
            logits2_new, h2_new, _ = model(
                new_imgs,
                torch.zeros_like(new_coords),
                return_rep=True
            )
            '''
            #print((logits2_new - logits_new).abs().mean())

            # retrieval noise
            h_old = apply_binomial_noise(
                h_old.cpu(),
                p_noise
            )

            h_new = apply_binomial_noise(
                h_new.cpu(),
                p_noise
            )


            old_score = compute_familiarity_score(
                h_old,
                memory_bank,
                args.sigma
            )

            new_score = compute_familiarity_score(
                h_new,
                memory_bank,
                args.sigma
            )


            if old_score > new_score:
                correct += 1

    return correct / max(n_pairs, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="faces | houses | objects")
    ap.add_argument("--variant", choices=["lp", "cnn"], default="lp")
    ap.add_argument("--fixation-root", default="fixation_data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default=None, help="label_map.json from the run (sets num_classes)")
    ap.add_argument("--num-classes", type=int, default=None, help="used if --label-map absent")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--num-study", type=int, default=40)
    ap.add_argument("--num-test", type=int, default=24)
    ap.add_argument("--study-fixations", type=int, default=10)
    ap.add_argument("--test-fixations", type=int, default=16)
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

    model = ModelCoords(size=180, num_classes=num_classes, pretrained=False, T=args.temperature).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    #print(model.coord_embed[0].weight.norm())

    # disjoint study / unknown items within the category (class-level for
    # faces/objects, individual photos for single-class categories like houses)
    all_items = list_items(args.fixation_root, args.category, "valid")
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
        ("Inverted", "Inverted", "test", "test"),
        ("Upright", "Inverted", "valid", "test"),
        ("Inverted", "Upright", "test", "valid"),
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
