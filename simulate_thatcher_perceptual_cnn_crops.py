"""
simulate_thatcher_perceptual.py

Purely perceptual analysis of the Thatcher Effect in Model 2.0.

For every base image, four conditions are considered:

    upright-normal
    upright-Thatcherized
    inverted-normal
    inverted-Thatcherized

The same fixation coordinates stored in the LP coordinate .txt files are
used to extract 180x180 crops for BOTH representations.

CNN representation:
    fixation crop -> foveation -> Model

LP representation:
    fixation crop -> foveation -> log-polar -> Model

No fixation coordinates are passed to the model.
No memory/KDE familiarity model is used.

For each image:

    D_U = mean Hamming distance(
              h(upright-normal),
              h(upright-Thatcher)
          )

    D_I = mean Hamming distance(
              h(inverted-normal),
              h(inverted-Thatcher)
          )

    Delta = D_U - D_I

The Thatcher Effect hypothesis is:

    Delta > 0

i.e. Thatcherization should produce a larger representational disruption
when the face is upright than when it is inverted.

A one-sample t-test of Delta against zero is performed across images.

Usage example:

    python simulate_thatcher_perceptual.py \
        --category faces \
        --variant lp \
        --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
        --label-map runs/faces_objects_lp_32fix_lr0.001/label_map.json

For the CNN representation:

    python simulate_thatcher_perceptual.py \
        --category faces \
        --variant cnn \
        --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
        --label-map runs/faces_objects_lp_32fix_lr0.001/label_map.json
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
from scipy.stats import ttest_1samp

from model import Model
from salience_trans import OnTheFlyTransform


# ============================================================
# Constants
# ============================================================

CROP_SIZE = 180


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Coordinate handling
# ============================================================

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
    Extract 180x180 fixation crops centered on the supplied
    fixation coordinates.

    img:
        Tensor [C,H,W], uint8 [0,255] or float [0,1]

    coords:
        list of (x,y)

    Returns:
        Tensor [N,C,180,180]
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
        raise ValueError("No fixation coordinates were found.")

    return torch.stack(crops, dim=0)


# ============================================================
# Image / coordinate loading
# ============================================================

def load_condition_image(
    processed_root,
    category,
    orientation,
    condition,
    identity,
    filename,
):
    """
    Load one condition-specific CNN image.

    Directory structure:

        processed_root/
            category/
                cnn/
                    orientation/
                        condition/
                            identity/
                                image.png

    where:

        orientation = upright / inverted
        condition  = normal / thatcher
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

    # Keep uint8 because OnTheFlyTransform explicitly handles
    # uint8 -> float [0,1].
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
    condition-specific image.

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


# ============================================================
# Dataset discovery
# ============================================================

def list_identities(
    processed_root,
    category,
    orientation,
    condition,
):
    """
    List identities present in:

        category/cnn/orientation/condition/
    """

    directory = os.path.join(
        processed_root,
        category,
        "cnn",
        orientation,
        condition,
    )

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Could not find directory:\n{directory}"
        )

    return sorted(
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    )


def list_images(
    processed_root,
    category,
    orientation,
    condition,
    identity,
):
    """
    List PNG images for one identity.
    """

    directory = os.path.join(
        processed_root,
        category,
        "cnn",
        orientation,
        condition,
        identity,
    )

    if not os.path.isdir(directory):
        return []

    return sorted(
        f
        for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )


def find_matched_images(
    processed_root,
    category,
    orientation,
    identity,
):
    """
    Find images that exist in BOTH:

        normal
        thatcher

    for the same identity and orientation.

    Returns:
        sorted list of filenames
    """

    normal = set(
        list_images(
            processed_root,
            category,
            orientation,
            "normal",
            identity,
        )
    )

    thatcher = set(
        list_images(
            processed_root,
            category,
            orientation,
            "thatcher",
            identity,
        )
    )

    return sorted(normal & thatcher)


# ============================================================
# Representation extraction
# ============================================================

def get_representation(
    model,
    transformer,
    img,
    coords,
    device,
):
    """
    Extract fixation crops at the supplied coordinates, apply the
    requested representation transform, and obtain the model's
    deterministic binary representation.

    IMPORTANT:
        No coordinates are passed to Model.

    Returns:
        h: [N_fixations, 256]
    """

    crops = extract_fixation_crops(
        img,
        coords,
        crop_size=CROP_SIZE,
    )

    # OnTheFlyTransform handles:
    #
    #   uint8 -> float [0,1]
    #   foveation
    #   optional log-polar transform
    #
    # The crops are already in their desired orientation, so the
    # transform itself uses type="valid".
    transformed = transformer(crops.to(device))

    # Model expects [B,C,H,W].
    #
    # Each fixation is treated as one observation.
    model.stochastic = False

    _, h, _ = model(
        transformed,
        return_rep=True,
    )

    return h.detach()


# ============================================================
# Hamming distance
# ============================================================

def mean_hamming_distance(h1, h2):
    """
    Mean Hamming distance between corresponding fixation
    representations.

    h1:
        [N, D]

    h2:
        [N, D]

    Returns:
        scalar float

    The distance is normalized by representation dimensionality,
    so the result is the proportion of binary units that differ.

    Thus:

        0.0 = identical binary representations
        1.0 = every binary unit differs
    """

    if h1.shape != h2.shape:
        raise ValueError(
            f"Representation shape mismatch: "
            f"{h1.shape} vs {h2.shape}"
        )

    distance = (h1 != h2).float().mean()

    return float(distance.item())


# ============================================================
# Single-image analysis
# ============================================================

def analyze_image(
    model,
    transformer,
    processed_root,
    category,
    identity,
    filename,
    orientation,
    device,
):
    """
    Compute the normal-vs-Thatcher representational distance
    for one image in one orientation.

    Returns:
        float distance
    """

    # --------------------------------------------------------
    # Load normal image
    # --------------------------------------------------------

    normal_img = load_condition_image(
        processed_root,
        category,
        orientation,
        "normal",
        identity,
        filename,
    )

    # --------------------------------------------------------
    # Load Thatcherized image
    # --------------------------------------------------------

    thatcher_img = load_condition_image(
        processed_root,
        category,
        orientation,
        "thatcher",
        identity,
        filename,
    )

    # --------------------------------------------------------
    # Coordinates
    #
    # These come from the LP folder.
    # CNN uses these exact same coordinates.
    # --------------------------------------------------------

    normal_coords_path = get_coords_path(
        processed_root,
        category,
        orientation,
        "normal",
        identity,
        filename,
    )

    thatcher_coords_path = get_coords_path(
        processed_root,
        category,
        orientation,
        "thatcher",
        identity,
        filename,
    )

    if not os.path.isfile(normal_coords_path):
        raise FileNotFoundError(
            f"Missing normal LP coordinate file:\n"
            f"{normal_coords_path}"
        )

    if not os.path.isfile(thatcher_coords_path):
        raise FileNotFoundError(
            f"Missing Thatcher LP coordinate file:\n"
            f"{thatcher_coords_path}"
        )

    normal_coords = read_fixation_coords(
        normal_coords_path
    )

    thatcher_coords = read_fixation_coords(
        thatcher_coords_path
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We compare normal and Thatcher images using their
    # respective LP fixation coordinates.
    #
    # Those coordinates are used for CNN as well.
    #
    # Thus CNN and LP see the same fixation locations for a
    # given image.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Representations
    # --------------------------------------------------------

    h_normal = get_representation(
        model,
        transformer,
        normal_img,
        normal_coords,
        device,
    )

    h_thatcher = get_representation(
        model,
        transformer,
        thatcher_img,
        thatcher_coords,
        device,
    )

    # --------------------------------------------------------
    # Check fixation counts
    # --------------------------------------------------------

    if h_normal.shape[0] != h_thatcher.shape[0]:
        raise ValueError(
            f"Different number of fixations for "
            f"{identity}/{filename} ({orientation}): "
            f"{h_normal.shape[0]} vs {h_thatcher.shape[0]}"
        )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = mean_hamming_distance(
        h_normal,
        h_thatcher,
    )

    return distance


# ============================================================
# Full experiment
# ============================================================

def run_experiment(
    model,
    processed_root,
    category,
    variant,
    identities,
    max_images,
    device,
):
    """
    Run the complete perceptual Thatcher experiment.

    Returns:
        pandas.DataFrame with one row per image.
    """

    # --------------------------------------------------------
    # Representation transform
    #
    # The images are already stored as upright/inverted
    # condition images, so DO NOT rotate them again here.
    #
    # type="valid" therefore means:
    #
    #     crop -> foveate -> [logpolar]
    # --------------------------------------------------------

    transformer = OnTheFlyTransform(
        type="valid",
        variant=variant,
        device=device,
        crop_size=CROP_SIZE,
        output_shape=(CROP_SIZE, CROP_SIZE),
    ).to(device)

    transformer.eval()

    results = []

    # --------------------------------------------------------
    # Process identities
    # --------------------------------------------------------

    for identity_idx, identity in enumerate(identities):

        print(
            f"[{identity_idx + 1}/{len(identities)}] "
            f"{identity}"
        )

        # ----------------------------------------------------
        # Find images available in both normal and Thatcher
        # ----------------------------------------------------

        upright_images = find_matched_images(
            processed_root,
            category,
            "upright",
            identity,
        )

        inverted_images = find_matched_images(
            processed_root,
            category,
            "inverted",
            identity,
        )

        # ----------------------------------------------------
        # Optional image limit
        # ----------------------------------------------------

        if max_images is not None:
            upright_images = upright_images[:max_images]
            inverted_images = inverted_images[:max_images]

        # ----------------------------------------------------
        # Analyze each image
        # ----------------------------------------------------

        all_filenames = sorted(
            set(upright_images) |
            set(inverted_images)
        )

        for filename in all_filenames:

            # -----------------------------------------------
            # Upright
            # -----------------------------------------------

            upright_distance = np.nan

            if filename in upright_images:

                try:
                    upright_distance = analyze_image(
                        model,
                        transformer,
                        processed_root,
                        category,
                        identity,
                        filename,
                        "upright",
                        device,
                    )

                except Exception as e:
                    print(
                        f"  WARNING: failed upright "
                        f"{filename}: {e}"
                    )

            # -----------------------------------------------
            # Inverted
            # -----------------------------------------------

            inverted_distance = np.nan

            if filename in inverted_images:

                try:
                    inverted_distance = analyze_image(
                        model,
                        transformer,
                        processed_root,
                        category,
                        identity,
                        filename,
                        "inverted",
                        device,
                    )

                except Exception as e:
                    print(
                        f"  WARNING: failed inverted "
                        f"{filename}: {e}"
                    )

            # -----------------------------------------------
            # Thatcher Effect score
            #
            # Delta = upright disruption
            #       - inverted disruption
            # -----------------------------------------------

            delta = np.nan

            if (
                not np.isnan(upright_distance)
                and not np.isnan(inverted_distance)
            ):
                delta = (
                    upright_distance
                    - inverted_distance
                )

            results.append({
                "identity": identity,
                "image": filename,
                "upright_distance": upright_distance,
                "inverted_distance": inverted_distance,
                "delta": delta,
                "variant": variant,
            })

    return pd.DataFrame(results)


# ============================================================
# Statistics
# ============================================================

def summarize_results(df):
    """
    Compute the image-level Thatcher Effect statistics.

    The primary test is:

        H0: mean(delta) = 0
        H1: mean(delta) > 0

    scipy's ttest_1samp gives the two-sided p-value, so for the
    directional hypothesis we divide by 2 when the observed
    mean is positive.
    """

    valid = df["delta"].dropna().to_numpy()

    if len(valid) < 2:
        raise RuntimeError(
            "Not enough complete images for a t-test."
        )

    mean_delta = float(np.mean(valid))
    std_delta = float(np.std(valid, ddof=1))
    n = len(valid)

    t_stat, p_two_sided = ttest_1samp(
        valid,
        popmean=0.0,
    )

    # One-sided hypothesis:
    #
    #     H1: mean(delta) > 0
    #
    if t_stat > 0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - (p_two_sided / 2.0)

    # -----------------------------------------------
    # Also report raw upright/inverted distances
    # -----------------------------------------------

    upright = df["upright_distance"].dropna().to_numpy()
    inverted = df["inverted_distance"].dropna().to_numpy()

    summary = {
        "n_images": n,

        "mean_upright_distance": (
            float(np.mean(upright))
            if len(upright) > 0 else np.nan
        ),

        "sd_upright_distance": (
            float(np.std(upright, ddof=1))
            if len(upright) > 1 else np.nan
        ),

        "mean_inverted_distance": (
            float(np.mean(inverted))
            if len(inverted) > 0 else np.nan
        ),

        "sd_inverted_distance": (
            float(np.std(inverted, ddof=1))
            if len(inverted) > 1 else np.nan
        ),

        "mean_delta": mean_delta,
        "sd_delta": std_delta,

        "t": float(t_stat),

        "p_two_sided": float(p_two_sided),
        "p_one_sided": float(p_one_sided),

        "hypothesis": "upright > inverted",
    }

    return summary


# ============================================================
# Argument parsing
# ============================================================

def parse_args():

    ap = argparse.ArgumentParser(
        description=(
            "Purely perceptual Thatcher Effect analysis "
            "using matched LP fixation coordinates."
        )
    )

    ap.add_argument(
        "--processed-root",
        default="data/thatcher_data",
        help=(
            "Root directory containing the Thatcher dataset."
        ),
    )

    ap.add_argument(
        "--category",
        default="faces",
        help="Category to analyze.",
    )

    ap.add_argument(
        "--variant",
        choices=["lp", "cnn"],
        required=True,
        help=(
            "Representation to analyze. "
            "'cnn' = foveated only; "
            "'lp' = foveated + log-polar."
        ),
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained Model checkpoint.",
    )

    ap.add_argument(
        "--label-map",
        default=None,
        help=(
            "Path to label_map.json. "
            "Used to determine the model's number of classes."
        ),
    )

    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Number of classes, if --label-map is not supplied."
        ),
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Model temperature parameter.",
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--num-identities",
        type=int,
        default=None,
        help=(
            "Maximum number of identities to analyze. "
            "Default: all identities."
        ),
    )

    ap.add_argument(
        "--num-images",
        type=int,
        default=None,
        help=(
            "Maximum number of images per identity. "
            "Default: all matched images."
        ),
    )

    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. If omitted, a name is generated."
        ),
    )

    return ap.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(args.seed)

    device = torch.device(args.device)

    print(
        f"Device: {device}"
    )

    print(
        f"Category: {args.category}"
    )

    print(
        f"Variant: {args.variant}"
    )

    # --------------------------------------------------------
    # Determine model size
    # --------------------------------------------------------

    if args.label_map is not None:

        with open(args.label_map, "r") as f:
            label_map = json.load(f)

        num_classes = len(label_map)

    elif args.num_classes is not None:

        num_classes = args.num_classes

    else:

        raise SystemExit(
            "Provide either --label-map or --num-classes."
        )

    print(
        f"Number of model classes: {num_classes}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = Model(
        size=180,
        num_classes=num_classes,
        pretrained=False,
        T=args.temperature,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint,
        strict=False,
    )

    model.eval()

    # --------------------------------------------------------
    # Deterministic representation
    # --------------------------------------------------------

    model.stochastic = False

    # --------------------------------------------------------
    # Identities
    #
    # We use the upright-normal directory as the initial
    # identity source, then require matched data when actually
    # processing images.
    # --------------------------------------------------------

    identities = list_identities(
        args.processed_root,
        args.category,
        "upright",
        "normal",
    )

    if args.num_identities is not None:
        identities = identities[:args.num_identities]

    print(
        f"Identities: {len(identities)}"
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    with torch.no_grad():

        df = run_experiment(
            model=model,
            processed_root=args.processed_root,
            category=args.category,
            variant=args.variant,
            identities=identities,
            max_images=args.num_images,
            device=device,
        )

    # --------------------------------------------------------
    # Remove incomplete image pairs
    # --------------------------------------------------------

    complete_df = df.dropna(
        subset=[
            "upright_distance",
            "inverted_distance",
            "delta",
        ]
    ).copy()

    print()
    print(
        f"Complete image pairs: {len(complete_df)}"
    )

    if len(complete_df) < 2:
        raise SystemExit(
            "Fewer than two complete image pairs; "
            "cannot perform t-test."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    summary = summarize_results(
        complete_df
    )

    print()
    print("=" * 60)
    print("THATCHER EFFECT — PURELY PERCEPTUAL")
    print("=" * 60)

    print(
        f"Variant:                 {args.variant}"
    )

    print(
        f"N images:                {summary['n_images']}"
    )

    print()

    print(
        "Mean upright disruption: "
        f"{summary['mean_upright_distance']:.6f}"
    )

    print(
        "SD upright disruption:   "
        f"{summary['sd_upright_distance']:.6f}"
    )

    print()

    print(
        "Mean inverted disruption: "
        f"{summary['mean_inverted_distance']:.6f}"
    )

    print(
        "SD inverted disruption:   "
        f"{summary['sd_inverted_distance']:.6f}"
    )

    print()

    print(
        "Mean Delta (U - I):      "
        f"{summary['mean_delta']:.6f}"
    )

    print(
        "SD Delta:                "
        f"{summary['sd_delta']:.6f}"
    )

    print()

    print(
        "t-statistic:             "
        f"{summary['t']:.6f}"
    )

    print(
        "p-value (two-sided):     "
        f"{summary['p_two_sided']:.6g}"
    )

    print(
        "p-value (one-sided):     "
        f"{summary['p_one_sided']:.6g}"
    )

    print()

    if (
        summary["mean_delta"] > 0
        and summary["p_one_sided"] < 0.05
    ):
        print(
            "RESULT: significant Thatcher Effect "
            "(upright disruption > inverted disruption)."
        )

    else:
        print(
            "RESULT: no significant Thatcher Effect "
            "under the directional hypothesis."
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Save image-level results
    # --------------------------------------------------------

    if args.output is None:

        args.output = (
            f"thatcher_perceptual_"
            f"{args.category}_"
            f"{args.variant}.csv"
        )

    complete_df.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Image-level results saved to: {args.output}"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        os.path.splitext(args.output)[0]
        + "_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Summary saved to: {summary_path}"
    )


if __name__ == "__main__":
    main()
