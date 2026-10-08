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


NUM_STUDY_IMAGES = 2
NUM_PROBE_IMAGES = 2

CROP_SIZE = 180


# ==================================================
# Helpers
# ==================================================

# --------------------------------------------------
# Coordinate handling
# --------------------------------------------------

def read_fixation_coords(txt_path):
    """
    Read fixation coordinates from an LP coordinate file.

    Expected format:

        x y
        x y
        x y
        ...

    Returns:
        list of (x, y) integer pixel coordinates
    """

    coords = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            x = int(float(parts[0]))
            y = int(float(parts[1]))

            coords.append((x, y))

    return coords


def extract_fixation_crops(img, coords, crop_size=CROP_SIZE):
    """
    Extract fixation crops centered on supplied fixation coordinates.

    img:
        Tensor [C,H,W], uint8 [0,255] or float [0,1]

    coords:
        list of (x,y)

    Returns:
        Tensor [N,C,crop_size,crop_size]
    """

    half = crop_size // 2

    crops = []

    for x, y in coords:
        crop = TF.crop(
            img,
            top=y - half,
            left=x - half,
            height=crop_size,
            width=crop_size,
        )

        crops.append(crop)

    if not crops:
        raise ValueError(
            f"No fixation coordinates were found."
        )

    return torch.stack(crops, dim=0)


# --------------------------------------------------
# Image / coordinate loading
# --------------------------------------------------

def load_condition_image(
    processed_root,
    category,
    orientation,
    condition,
    identity,
    filename,
):
    """
    Load one condition-specific CNN base image.

    Directory structure:

        processed_root/
            category/
                cnn/
                    orientation/
                        condition/
                            identity/
                                image.png
    """

    path = os.path.join(
        processed_root,
        category,
        "cnn",
        orientation,
        condition,
        identity,
        filename,
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    img = Image.open(path).convert("RGB")

    # Keep uint8 because fixation cropping happens before
    # the CNN tensor conversion.
    img = TF.pil_to_tensor(img)

    return img


def get_coords_path(
    processed_root,
    category,
    orientation,
    condition,
    identity,
    filename,
):
    """
    Return the LP coordinate txt path corresponding to the
    condition-specific CNN image.

    Directory structure:

        processed_root/
            category/
                lp/
                    orientation/
                        condition/
                            identity/
                                image.txt
    """

    stem = os.path.splitext(filename)[0]

    path = os.path.join(
        processed_root,
        category,
        "lp",
        orientation,
        condition,
        identity,
        stem + ".txt",
    )

    return path


# --------------------------------------------------
# Data
# --------------------------------------------------

"""
Loads:

    data/thatcher_data/<category>/<variant>/
        <upright|inverted>/<normal|thatcher>/<identity>/*.png

For CNN:

    CNN base image
        ↓
    matching LP fixation coordinates
        ↓
    180x180 fixation crops

For LP:

    existing _proc fixation crops
"""

def load_trial_data(
    processed_root,
    category,
    variant,
    image_type,
    split,
    classes,
    num_images,
    offset
):
    samples = {}

    base_dir = os.path.join(
        processed_root,
        category,
        variant
    )

    orient_dir = os.path.join(
        base_dir,
        image_type
    )

    split_dir = os.path.join(
        orient_dir,
        split
    )

    for cls in classes:

        cls_dir = os.path.join(
            split_dir,
            cls
        )

        if not os.path.isdir(cls_dir):
            print(
                f"{cls_dir} not a valid directory!"
            )
            continue

        # --------------------------------------------------
        # CNN
        # --------------------------------------------------

        if variant == "cnn":

            files = sorted([
                f
                for f in os.listdir(cls_dir)
                if f.lower().endswith(".png")
            ])

            chosen = files[
                offset: offset + num_images
            ]

            if len(chosen) < num_images:
                continue

            imgs = []

            for filename in chosen:

                # Load CNN base image.
                img = load_condition_image(
                    processed_root,
                    category,
                    image_type,
                    split,
                    cls,
                    filename,
                )

                # Find corresponding LP coordinate file.
                coords_path = get_coords_path(
                    processed_root,
                    category,
                    image_type,
                    split,
                    cls,
                    filename,
                )

                if not os.path.isfile(coords_path):
                    raise FileNotFoundError(
                        f"Could not find fixation coordinates:\n"
                        f"{coords_path}"
                    )

                coords = read_fixation_coords(
                    coords_path
                )

                # Crop the CNN image at exactly the same
                # fixation locations used by the LP pipeline.
                fixation_crops = extract_fixation_crops(
                    img,
                    coords,
                    crop_size=CROP_SIZE,
                )

                imgs.append(
                    fixation_crops
                )

            imgs = torch.cat(
                imgs,
                dim=0
            )

        # --------------------------------------------------
        # LP
        # --------------------------------------------------

        else:

            base_dict = {}

            for fname in sorted(
                os.listdir(cls_dir)
            ):

                if (
                    fname.endswith(".png")
                    and "_proc" in fname
                ):

                    base = fname.split(
                        "_proc"
                    )[0]

                    base_dict.setdefault(
                        base,
                        []
                    ).append(
                        os.path.join(
                            cls_dir,
                            fname
                        )
                    )

            base_names = sorted(
                base_dict
            )

            chosen_bases = base_names[
                offset: offset + num_images
            ]

            if len(chosen_bases) < num_images:
                continue

            imgs = []

            for base in chosen_bases:

                proc_list = sorted(
                    base_dict[base]
                )

                imgs.extend([
                    TF.to_tensor(
                        Image.open(p).convert("RGB")
                    )
                    for p in proc_list
                ])

            imgs = torch.stack(
                imgs,
                dim=0
            )

        samples[cls] = imgs

    return samples


def list_classes(
    processed_root,
    category,
    variant,
    split
):
    d = os.path.join(
        processed_root,
        category,
        variant,
        split
    )

    classes = sorted(
        c
        for c in os.listdir(d)
        if os.path.isdir(
            os.path.join(d, c)
        )
    )

    return classes


def apply_binomial_noise(
    binary_tensor,
    p_noise
):
    if p_noise == 0.0:
        return binary_tensor

    mask = (
        torch.rand_like(binary_tensor)
        < p_noise
    )

    return torch.logical_xor(
        binary_tensor.bool(),
        mask
    ).float()


# --------------------------------------------------
# Barrington KDE
# --------------------------------------------------

def compute_p_f_given_c(
    f,
    M_c,
    sigma
):
    dists = torch.sum(
        (M_c - f) ** 2,
        dim=1
    )

    return torch.mean(
        torch.exp(
            -dists /
            (2 * sigma ** 2)
        )
    )


def compute_familiarity_score(
    F_test,
    memory_bank,
    sigma
):
    best = -float("inf")

    for _, M_c in memory_bank.items():

        ll = 0.0

        for i in range(
            F_test.size(0)
        ):

            ll += torch.log(
                compute_p_f_given_c(
                    F_test[i],
                    M_c,
                    sigma
                )
                + 1e-12
            ).item()

        best = max(
            best,
            ll
        )

    return best


# ==================================================
# ARG PARSING FOR THATCHER EXP
# ==================================================

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--category",
        required=True,
        help="faces | houses | objects"
    )

    ap.add_argument(
        "--variant",
        choices=["lp", "cnn"],
        default="lp"
    )

    ap.add_argument(
        "--processed-root",
        default="data/thatcher_data"
    )

    ap.add_argument(
        "--checkpoint",
        required=True
    )

    ap.add_argument(
        "--label-map",
        default=None,
        help="label_map.json from the run (sets num_classes)"
    )

    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="used if --label-map absent"
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=2.0
    )

    ap.add_argument(
        "--num-study",
        type=int,
        default=117
    )

    ap.add_argument(
        "--num-test",
        type=int,
        default=117
    )

    ap.add_argument(
        "--study-images",
        type=int,
        default=NUM_STUDY_IMAGES
    )

    ap.add_argument(
        "--probe-images",
        type=int,
        default=NUM_PROBE_IMAGES
    )

    ap.add_argument(
        "--sigma",
        type=float,
        default=2.0
    )

    ap.add_argument(
        "--calib-target",
        type=float,
        default=0.96,
        help="upright-upright accuracy to match when calibrating noise"
    )

    ap.add_argument(
        "--calib-max",
        type=float,
        default=0.75
    )

    ap.add_argument(
        "--calib-step",
        type=float,
        default=0.05
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    ap.add_argument(
        "--noise",
        type=float,
        default=0.25
    )

    return ap.parse_args()


# ==================================================
# DEBUG OUTPUT HELPERS
# ==================================================

import matplotlib.pyplot as plt


def show_example_pairs(
    processed_root,
    category,
    classes,
    study_orientation="upright",
    study_type="normal",
    probe_orientation="upright",
    probe_type="thatcher",
    study_offset=0,
    probe_offset=4,
    num_examples=5
):

    base = os.path.join(
        processed_root,
        category
    )

    for cls in classes[:num_examples]:

        study_dir = os.path.join(
            base,
            "cnn",
            study_orientation,
            study_type,
            cls
        )

        probe_dir = os.path.join(
            base,
            "cnn",
            probe_orientation,
            probe_type,
            cls
        )

        study_files = sorted(
            f
            for f in os.listdir(study_dir)
            if f.lower().endswith(".png")
        )

        probe_files = sorted(
            f
            for f in os.listdir(probe_dir)
            if f.lower().endswith(".png")
        )

        study_img = Image.open(
            os.path.join(
                study_dir,
                study_files[study_offset]
            )
        )

        probe_img = Image.open(
            os.path.join(
                probe_dir,
                probe_files[probe_offset]
            )
        )

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(6, 3)
        )

        ax[0].imshow(
            study_img
        )

        ax[0].set_title(
            f"{cls}\nStudy"
        )

        ax[1].imshow(
            probe_img
        )

        ax[1].set_title(
            f"{cls}\nProbe"
        )

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()


# ==================================================
# THATCHER EXPERIMENT
# ==================================================

def familiarity(
    model,
    device,
    args,
    study_classes,
    p_noise,
    condition
):

    noisy_study = False
    noisy_probe = False

    orientation, thatcher = condition

    set_seed(
        args.seed
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    study_data = load_trial_data(
        args.processed_root,
        args.category,
        args.variant,
        image_type="upright",
        split="normal",
        classes=study_classes,
        num_images=args.study_images,
        offset=0,
    )

    probe_data = load_trial_data(
        args.processed_root,
        args.category,
        args.variant,
        image_type=orientation,
        split=thatcher,
        classes=study_classes,
        num_images=args.probe_images,
        offset=args.study_images,
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

            if noisy_study:

                memory_bank[cls] = (
                    apply_binomial_noise(
                        h.cpu(),
                        p_noise
                    )
                )

            else:

                memory_bank[cls] = (
                    h.cpu()
                )

        # identities present in both memory and probe

        probe_items = sorted(
            set(study_data.keys())
            &
            set(probe_data.keys())
        )

        if len(probe_items) == 0:
            raise RuntimeError(
                "No identities shared between study and probe data."
            )

        probe_items = sorted(
            probe_items
        )

        if args.num_test is not None:

            probe_items = probe_items[
                :min(
                    args.num_test,
                    len(probe_items)
                )
            ]

        # --------------------------------------------------
        # Familiarity calculation
        # --------------------------------------------------

        familiarity_scores = {}

        for cls in probe_items:

            model.stochastic = True

            _, h_probe, _ = model(
                probe_data[cls].to(device),
                return_rep=True
            )

            if noisy_probe:

                h_probe = (
                    apply_binomial_noise(
                        h_probe.cpu(),
                        p_noise
                    )
                )

            else:

                h_probe = (
                    h_probe.cpu()
                )

            familiarity = (
                compute_familiarity_score(
                    h_probe,
                    memory_bank,
                    args.sigma
                )
            )

            familiarity_scores[
                cls
            ] = familiarity

    return (
        familiarity_scores,
        float(
            np.mean(
                list(
                    familiarity_scores.values()
                )
            )
        )
    )


def diff(
    stat_a,
    stat_b
):
    return stat_a - stat_b


def diff_map(
    map_a,
    map_b
):

    if not isinstance(
        map_a,
        dict
    ):

        return diff(
            map_a,
            map_b
        )

    result = {}

    for k in map_a.keys():

        result[k] = diff_map(
            map_a[k],
            map_b[k]
        )

    return result


def main(debug=False):

    pretrain = False

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    shuffle_classes = True

    if args.label_map:

        with open(args.label_map) as f:

            num_classes = len(
                json.load(f)
            )

    elif args.num_classes:

        num_classes = args.num_classes

    else:

        raise SystemExit(
            "Provide --label-map or --num-classes to size the model head."
        )

    model = Model(
        size=180,
        num_classes=num_classes,
        pretrained=pretrain,
        T=args.temperature
    ).to(device)

    if not pretrain:

        model.load_state_dict(
            torch.load(
                args.checkpoint,
                map_location=device
            ),
            strict=False
        )

    model.eval()

    # --------------------------------------------------
    # Classes
    # --------------------------------------------------

    all_classes = list_classes(
        args.processed_root,
        args.category,
        args.variant,
        "upright/normal"
    )

    if len(all_classes) < args.num_study:

        raise SystemExit(
            f"Category '{args.category}' has "
            f"{len(all_classes)} classes; "
            f"need >= {args.num_study} study identities."
        )

    if shuffle_classes:

        random.shuffle(
            all_classes
        )

    study_classes = all_classes[
        :args.num_study
    ]

    if debug:

        show_example_pairs(
            args.processed_root,
            args.category,
            study_classes,
            study_orientation="upright",
            study_type="normal",
            probe_orientation="upright",
            probe_type="normal",
            study_offset=0,
            probe_offset=args.study_images,
            num_examples=5
        )

    ideal_noise = args.noise

    print(
        f"[!] Using noise p={ideal_noise:.2f}\n"
    )

    # --------------------------------------------------
    # Conditions
    # --------------------------------------------------

    conditions = (
        ("upright", "normal"),
        ("upright", "thatcher"),
        ("inverted", "normal"),
        ("inverted", "thatcher"),
    )

    fam_scores = {}
    fam_dicts = {}

    for c in conditions:

        (
            fam_dicts[c],
            fam_scores[c]
        ) = familiarity(
            model,
            device,
            args,
            study_classes,
            ideal_noise,
            c
        )

    # --------------------------------------------------
    # Compare Thatcher effect
    # --------------------------------------------------

    E_u = diff(
        fam_scores[
            conditions[0]
        ],
        fam_scores[
            conditions[1]
        ]
    )

    E_i = diff(
        fam_scores[
            conditions[2]
        ],
        fam_scores[
            conditions[3]
        ]
    )

    print(
        f"Upright Normal: "
        f"{fam_scores[conditions[0]]}"
    )

    print(
        f"Upright Thatcher: "
        f"{fam_scores[conditions[1]]}"
    )

    print()

    print(
        f"Inverted Normal: "
        f"{fam_scores[conditions[2]]}"
    )

    print(
        f"Inverted Thatcher: "
        f"{fam_scores[conditions[3]]}"
    )

    print()

    print(
        f"Upright Thatcher Effect: "
        f"{E_u}"
    )

    print(
        f"Inverted Thatcher Effect: "
        f"{E_i}"
    )

    print()

    PVALUE = 0.05

    stat_results = (
        stat_sig_thatcher_effect(
            fam_dicts,
            conditions
        )
    )

    print(
        "statistical test of results yields:"
    )

    print(
        stat_results
    )

    if stat_results.pvalue < PVALUE:

        if E_u > E_i:

            print(
                "Observed Thatcher effect!"
            )

        else:

            print(
                "Observed opposite of Thatcher effect ..."
            )

    else:

        print(
            "Did not observe Thatcher effect."
        )


def stat_sig_thatcher_effect(
    fam_dicts,
    conditions
):

    from scipy.stats import ttest_1samp

    Emap_u = diff_map(
        fam_dicts[
            conditions[0]
        ],
        fam_dicts[
            conditions[1]
        ]
    )

    Emap_i = diff_map(
        fam_dicts[
            conditions[2]
        ],
        fam_dicts[
            conditions[3]
        ]
    )

    Dmap = diff_map(
        Emap_u,
        Emap_i
    )

    return ttest_1samp(
        list(Dmap.values()),
        0.0
    )


if __name__ == "__main__":
    main()