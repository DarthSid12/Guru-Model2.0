"""
simulate_thatcher_perceptual.py

Purely perceptual / representational test of the Thatcher effect.

For each individual image:

    Upright disruption:
        D_U = distance(normal_upright, thatcher_upright)

    Inverted disruption:
        D_I = distance(normal_inverted, thatcher_inverted)

    Thatcher effect:
        Delta = D_U - D_I

The statistical test is a one-sample t-test across individual images:

    H0: mean(Delta) = 0
    H1: mean(Delta) > 0

A positive Delta means Thatcherization changes the model
representation more for upright than inverted faces.

No memory bank, KDE, familiarity calculation, or retrieval
noise is used.

For the lp variant, each original image has multiple fixation
crops. The image-level distance is the mean Hamming distance
across corresponding fixation representations.
"""

import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import ttest_1samp

from model import Model


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
# Data loading
# ============================================================

"""
Expected directory structure:

data/thatcher_data/
    <category>/
        <variant>/
            upright/
                normal/
                    <identity>/
                thatcher/
                    <identity>/
            inverted/
                normal/
                    <identity>/
                thatcher/
                    <identity>/

For cnn:

    identity/
        image1.png
        image2.png
        ...

For lp:

    identity/
        image1_proc0.png
        image1_proc1.png
        image1_proc2.png
        ...
        image2_proc0.png
        ...
"""


def get_image_groups(
    processed_root,
    category,
    variant,
    orientation,
    split,
    identity,
):
    """
    Return a dictionary:

        {
            base_image_name: [path_to_fixation_0,
                              path_to_fixation_1,
                              ...]
        }

    For cnn, each image simply has one path.
    """

    directory = os.path.join(
        processed_root,
        category,
        variant,
        orientation,
        split,
        identity,
    )

    if not os.path.isdir(directory):
        return {}

    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".png")
    )

    groups = {}

    if variant == "cnn":

        for fname in files:
            base = os.path.splitext(fname)[0]
            groups[base] = [
                os.path.join(directory, fname)
            ]

    else:

        # lp:
        # image123_proc0.png
        # image123_proc1.png
        # ...
        for fname in files:

            if "_proc" not in fname:
                continue

            base = fname.split("_proc")[0]

            groups.setdefault(base, []).append(
                os.path.join(directory, fname)
            )

        for base in groups:
            groups[base] = sorted(groups[base])

    return groups


def get_common_images(
    processed_root,
    category,
    variant,
    orientation,
    identity,
):
    """
    Find images which exist in BOTH normal and Thatcherized
    versions for a particular identity.
    """

    normal = get_image_groups(
        processed_root,
        category,
        variant,
        orientation,
        "normal",
        identity,
    )

    thatcher = get_image_groups(
        processed_root,
        category,
        variant,
        orientation,
        "thatcher",
        identity,
    )

    common = sorted(
        set(normal.keys()) & set(thatcher.keys())
    )

    return [
        (name, normal[name], thatcher[name])
        for name in common
    ]


def load_tensor(paths):
    """
    Load one image or a collection of fixation crops.

    Returns:
        [N, C, H, W]
    """

    return torch.stack([
        TF.to_tensor(
            Image.open(p).convert("RGB")
        )
        for p in paths
    ], dim=0)


# ============================================================
# Representation distance
# ============================================================

def hamming_distance(h_a, h_b):
    """
    Mean Hamming distance between two binary representations.

    h_a, h_b:
        [N, D]

    Returns one scalar.
    """

    assert h_a.shape == h_b.shape

    return torch.mean(
        (h_a != h_b).float()
    ).item()


# ============================================================
# Model representation
# ============================================================

def get_representation(
    model,
    images,
    device,
):
    """
    Convert images into the model's binary representation h.

    Returns:
        [N, D]
    """

    with torch.no_grad():

        model.stochastic = False

        _, h, _ = model(
            images.to(device),
            return_rep=True,
        )

    return h.cpu()


# ============================================================
# Per-image Thatcher distances
# ============================================================

def compute_image_distances(
    model,
    device,
    processed_root,
    category,
    variant,
    identity,
):
    """
    Compute D_U and D_I for every image belonging to one identity.

    Returns a list of dictionaries:

        {
            "identity": ...,
            "image": ...,
            "upright_distance": ...,
            "inverted_distance": ...,
            "thatcher_difference": ...
        }
    """

    upright_images = get_common_images(
        processed_root,
        category,
        variant,
        "upright",
        identity,
    )

    inverted_images = get_common_images(
        processed_root,
        category,
        variant,
        "inverted",
        identity,
    )

    # Match the same underlying image names across orientations.
    upright_names = {
        name for name, _, _ in upright_images
    }

    inverted_names = {
        name for name, _, _ in inverted_images
    }

    common_names = sorted(
        upright_names & inverted_names
    )

    results = []

    for image_name in common_names:

        # --------------------------------------------------
        # Upright
        # --------------------------------------------------

        upright_dict = {
            name: (normal, thatcher)
            for name, normal, thatcher in upright_images
        }

        normal_upright_paths, thatcher_upright_paths = \
            upright_dict[image_name]

        normal_upright = load_tensor(
            normal_upright_paths
        )

        thatcher_upright = load_tensor(
            thatcher_upright_paths
        )

        h_normal_upright = get_representation(
            model,
            normal_upright,
            device,
        )

        h_thatcher_upright = get_representation(
            model,
            thatcher_upright,
            device,
        )

        D_U = hamming_distance(
            h_normal_upright,
            h_thatcher_upright,
        )

        # --------------------------------------------------
        # Inverted
        # --------------------------------------------------

        inverted_dict = {
            name: (normal, thatcher)
            for name, normal, thatcher in inverted_images
        }

        normal_inverted_paths, thatcher_inverted_paths = \
            inverted_dict[image_name]

        normal_inverted = load_tensor(
            normal_inverted_paths
        )

        thatcher_inverted = load_tensor(
            thatcher_inverted_paths
        )

        h_normal_inverted = get_representation(
            model,
            normal_inverted,
            device,
        )

        h_thatcher_inverted = get_representation(
            model,
            thatcher_inverted,
            device,
        )

        D_I = hamming_distance(
            h_normal_inverted,
            h_thatcher_inverted,
        )

        # --------------------------------------------------
        # Thatcher effect for this individual image
        # --------------------------------------------------

        delta = D_U - D_I

        results.append({
            "identity": identity,
            "image": image_name,
            "upright_distance": D_U,
            "inverted_distance": D_I,
            "thatcher_difference": delta,
        })

    return results


# ============================================================
# Main experiment
# ============================================================

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--category",
        required=True,
        help="faces | houses | objects",
    )

    ap.add_argument(
        "--variant",
        choices=["lp", "cnn"],
        default="lp",
    )

    ap.add_argument(
        "--processed-root",
        default="data/thatcher_data",
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--num-classes",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--num-identities",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return ap.parse_args()


def list_identities(
    processed_root,
    category,
    variant,
):
    """
    Find identities which have upright/normal data.
    """

    directory = os.path.join(
        processed_root,
        category,
        variant,
        "upright",
        "normal",
    )

    return sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(
            os.path.join(directory, d)
        )
    )


def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    print()
    print("==========================================")
    print("Purely Perceptual Thatcher Effect")
    print("==========================================")
    print(f"Category:       {args.category}")
    print(f"Variant:        {args.variant}")
    print(f"Device:         {device}")
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Model(
        size=180,
        num_classes=args.num_classes,
        pretrained=False,
        T=args.temperature,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
        ),
        strict=False,
    )

    model.eval()

    # --------------------------------------------------------
    # Identities
    # --------------------------------------------------------

    identities = list_identities(
        args.processed_root,
        args.category,
        args.variant,
    )

    if args.num_identities is not None:
        identities = identities[
            :args.num_identities
        ]

    print(
        f"Testing {len(identities)} identities..."
    )
    print()

    # --------------------------------------------------------
    # Compute image-level distances
    # --------------------------------------------------------

    all_results = []

    for i, identity in enumerate(identities):

        results = compute_image_distances(
            model=model,
            device=device,
            processed_root=args.processed_root,
            category=args.category,
            variant=args.variant,
            identity=identity,
        )

        all_results.extend(results)

        print(
            f"[{i + 1:3d}/{len(identities)}] "
            f"{identity}: "
            f"{len(results)} images"
        )

    if len(all_results) == 0:
        raise RuntimeError(
            "No matched images were found."
        )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    upright = np.array([
        r["upright_distance"]
        for r in all_results
    ])

    inverted = np.array([
        r["inverted_distance"]
        for r in all_results
    ])

    delta = np.array([
        r["thatcher_difference"]
        for r in all_results
    ])

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    print()
    print("==========================================")
    print("Descriptive results")
    print("==========================================")

    print(
        f"Images:                  {len(delta)}"
    )

    print(
        f"Mean upright distance:    {np.mean(upright):.6f}"
    )

    print(
        f"Mean inverted distance:   {np.mean(inverted):.6f}"
    )

    print(
        f"Mean Thatcher difference: {np.mean(delta):.6f}"
    )

    print(
        f"SD Thatcher difference:   {np.std(delta, ddof=1):.6f}"
    )

    # --------------------------------------------------------
    # Statistical test
    # --------------------------------------------------------

    """
    One-sample t-test:

        H0: mean(delta) = 0
        H1: mean(delta) > 0

    scipy's ttest_1samp is two-sided by default, so we
    explicitly convert it to a one-sided p-value.
    """

    t_stat, p_two_sided = ttest_1samp(
        delta,
        popmean=0.0,
    )

    if t_stat > 0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - p_two_sided / 2.0

    print()
    print("==========================================")
    print("Statistical test")
    print("==========================================")

    print(
        f"t({len(delta) - 1}) = {t_stat:.6f}"
    )

    print(
        f"one-sided p = {p_one_sided:.6g}"
    )

    print()

    if p_one_sided < 0.05 and np.mean(delta) > 0:

        print(
            "Significant Thatcher effect: "
            "upright representational disruption "
            "is greater than inverted disruption."
        )

    else:

        print(
            "No significant Thatcher effect."
        )

    # --------------------------------------------------------
    # Save individual-image results
    # --------------------------------------------------------

    output_name = (
        f"thatcher_perceptual_"
        f"{args.category}_"
        f"{args.variant}.csv"
    )

    import csv

    with open(
        output_name,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "identity",
                "image",
                "upright_distance",
                "inverted_distance",
                "thatcher_difference",
            ],
        )

        writer.writeheader()

        writer.writerows(all_results)

    print()
    print(
        f"Saved individual-image results to "
        f"{output_name}"
    )


if __name__ == "__main__":
    main()

