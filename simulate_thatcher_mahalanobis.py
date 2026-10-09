"""
simulate_thatcher_mahalanobis.py

Measures the Thatcher effect using Mahalanobis distance rather than Barrington-KDE familiarity.

For each identity i:

    UN = upright normal
    UT = upright thatcherized
    IN = inverted normal
    IT = inverted thatcherized

We compute:

    D_U(i) = Mahalanobis(UN_i, UT_i)
    D_I(i) = Mahalanobis(IN_i, IT_i)

and test whether:

    D_U(i) > D_I(i)

across identities.

The covariance matrix is POOLED ACROSS CONDITIONS and regularized before inversion 
so that the same representational geometry is used for upright and inverted comparisons.

Important:
- The covariance is estimated in the Model 2.0 representation space.
- Distances are computed on the model's binary representation h.
- Two probe images are averaged within identity/condition before computing
  the identity-level Mahalanobis distance.
- The statistical unit is therefore identity, not individual image.
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
from salience_trans import OnTheFlyTransform

NUM_STUDY_IMAGES = 2
NUM_PROBE_IMAGES = 2

CROP_SIZE = 180


# ==================================================
# SEED
# ==================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# COORDINATE HANDLING
# ==================================================

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

            coords.append(
                (x, y)
            )

    return coords


def extract_fixation_crops(
    img,
    coords,
    crop_size=CROP_SIZE
):
    """
    Extract 180x180 fixation crops centered on supplied
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

        crops.append(
            crop
        )

    if not crops:
        raise ValueError(
            "No fixation coordinates were found."
        )

    return torch.stack(
        crops,
        dim=0
    )


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

    img = Image.open(
        path
    ).convert("RGB")

    img = TF.pil_to_tensor(
        img
    )

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

    stem = os.path.splitext(
        filename
    )[0]

    return os.path.join(
        processed_root,
        category,
        "lp",
        orientation,
        condition,
        identity,
        stem + ".txt"
    )


# ==================================================
# DATA
# ==================================================

def load_trial_data(
    processed_root,
    category,
    variant,
    image_type,
    split,
    classes,
    num_images,
    offset,
    transformer,
    device,
):
    """
    Load identical base images and fixation coordinates for CNN and LP.

    Expected structure:
        data/thatcher_data/
            <category>/
                cnn/
                    <upright|inverted>/
                        <normal|thatcher>/
                            <identity>/
                                image.png
                lp/
                    <upright|inverted>/
                        <normal|thatcher>/
                            <identity>/
                                image.txt

    The CNN PNGs are used as the source images for BOTH variants.
    Corresponding LP .txt files provide the fixation coordinates.

    For each selected base image:
        1. Load the base PNG.
        2. Load its matching fixation coordinates.
        3. Extract fixation crops.
        4. Apply the requested OnTheFlyTransform variant.

    Returns:
        samples: dict mapping each class to a tensor shaped
                 [num_images * num_fixations, C, H, W],
                 assuming each image has the same number of fixations.
    """
    samples = {}

    for cls in classes:
        # Base images always come from the CNN directory.
        cls_dir = os.path.join(
            processed_root,
            category,
            "cnn",
            image_type,
            split,
            cls,
        )

        if not os.path.isdir(cls_dir):
            print(f"{cls_dir} is not a valid directory!")
            continue

        files = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(".png")
        )

        chosen = files[offset:offset + num_images]

        if len(chosen) < num_images:
            print(
                f"Skipping {cls}: requested {num_images} images, "
                f"but only found {len(chosen)} after offset {offset}."
            )
            continue

        image_batches = []

        for filename in chosen:
            # Load the source image as uint8 [C, H, W].
            img = load_condition_image(
                processed_root,
                category,
                image_type,
                split,
                cls,
                filename,
            )

            # Load the corresponding LP fixation-coordinate file.
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
                    f"Missing fixation coordinates for {filename}: "
                    f"{coords_path}"
                )

            coords = read_fixation_coords(coords_path)

            # Extract the same crops for both variants.
            crops = extract_fixation_crops(
                img,
                coords,
                crop_size=CROP_SIZE,
            )

            # Apply foveation alone for CNN, or foveation + log-polar
            # transformation for LP.
            with torch.no_grad():
                transformed = transformer(crops.to(device))

            image_batches.append(transformed.cpu())

        # Concatenate fixation crops from all selected base images.
        samples[cls] = torch.cat(image_batches, dim=0)

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

    return sorted([
        c
        for c in os.listdir(d)
        if os.path.isdir(
            os.path.join(d, c)
        )
    ])


# ==================================================
# ARGUMENTS
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
        default=None
    )

    ap.add_argument(
        "--num-classes",
        type=int,
        default=None
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
        "--ridge",
        type=float,
        default=1e-2,
        help="regularization added to covariance diagonal"
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

    return ap.parse_args()


# ==================================================
# MODEL
# ==================================================

def load_model(
    args,
    device
):

    if args.label_map:

        with open(args.label_map) as f:

            num_classes = len(
                json.load(f)
            )

    elif args.num_classes:

        num_classes = args.num_classes

    else:

        raise SystemExit(
            "Provide --label-map or --num-classes."
        )

    model = Model(
        size=180,
        num_classes=num_classes,
        pretrained=False,
        T=args.temperature
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device
        ),
        strict=False
    )

    model.eval()

    return model


# ==================================================
# REPRESENTATIONS
# ==================================================

def get_representations(
    model,
    device,
    data
):
    """
    Converts each identity's images into Model 2.0
    representations.

    Returns:

        reps[identity] = tensor(n_images, 256)
    """

    reps = {}

    with torch.no_grad():

        for cls, imgs in data.items():

            model.stochastic = True

            _, h, _ = model(
                imgs.to(device),
                return_rep=True
            )

            reps[cls] = h.cpu()

    return reps


# ==================================================
# CONDITION DATA
# ==================================================

def load_all_conditions(
    args,
    classes,
    device
):

    transformer = OnTheFlyTransform(
        type="valid",
        variant=variant,  # "cnn" or "lp"
        device=device,
        crop_size=CROP_SIZE,
        output_shape=(CROP_SIZE, CROP_SIZE),
    ).to(device)

    transformer.eval()

    conditions = {

        ("upright", "normal"):
            load_trial_data(
                args.processed_root,
                args.category,
                args.variant,
                image_type="upright",
                split="normal",
                classes=classes,
                num_images=args.probe_images,
                offset=args.study_images,
                transformer=transformer,
                device=device,
            ),

        ("upright", "thatcher"):
            load_trial_data(
                args.processed_root,
                args.category,
                args.variant,
                image_type="upright",
                split="thatcher",
                classes=classes,
                num_images=args.probe_images,
                offset=args.study_images,
                transformer=transformer,
                device=device,
            ),

        ("inverted", "normal"):
            load_trial_data(
                args.processed_root,
                args.category,
                args.variant,
                image_type="inverted",
                split="normal",
                classes=classes,
                num_images=args.probe_images,
                offset=args.study_images,
                transformer=transformer,
                device=device,
            ),

        ("inverted", "thatcher"):
            load_trial_data(
                args.processed_root,
                args.category,
                args.variant,
                image_type="inverted",
                split="thatcher",
                classes=classes,
                num_images=args.probe_images,
                offset=args.study_images,
                transformer=transformer,
                device=device,
            ),
    }

    return conditions


# ==================================================
# MAHALANOBIS COVARIANCE
# ==================================================

def estimate_covariance(
    representations,
    ridge # add to diagonal entries to make inversion more numerically stable
):
    """
    Estimate a shared covariance matrix given representation space.
    Will use for Upright Normal and Inverted Normal

    representations:
        list/array with shape (N, D)

    Returns:

        covariance and inverse

    We use:

        Sigma_reg = Sigma + lambda I

    because D=256 is large relative to the number of
    available observations.
    """

    X = np.asarray(
        representations,
        dtype=np.float64
    )

    print(
        f"Estimating covariance from "
        f"{X.shape[0]} observations "
        f"with dimension {X.shape[1]}"
    )

    covariance = np.cov(
        X,
        rowvar=False
    )

    # to guarantee that to the exact precision of python, the matrix is completely symmetric (as it should be):
    covariance = (
        covariance +
        covariance.T
    ) / 2.0

    covariance += (
        ridge *
        np.eye(
            covariance.shape[0]
        )
    )

    covariance_inv = np.linalg.inv(
        covariance
    )

    return (
        covariance,
        covariance_inv
    )


# ==================================================
# MAHALANOBIS DISTANCE
# ==================================================

def mahalanobis_distance(
    x,
    y,
    covariance_inv
):
    """
    Mahalanobis distance:

        d_M(x,y)
        =
        sqrt(
            (x-y)^T Sigma^{-1} (x-y)
        )
    """

    delta = (
        np.asarray(
            x,
            dtype=np.float64
        )
        -
        np.asarray(
            y,
            dtype=np.float64
        )
    )

    squared = (
        delta.T
        @ covariance_inv
        @ delta
    )

    squared = max(
        float(squared),
        0.0
    )

    return np.sqrt(
        squared
    )


# ==================================================
# MAIN ANALYSIS
# ==================================================

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Category: {args.category}"
    )

    print(
        f"Variant: {args.variant}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = load_model(
        args,
        device
    )

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
            f"need >= {args.num_study}."
        )

    random.shuffle(
        all_classes
    )

    study_classes = all_classes[
        :args.num_study
    ]

    # --------------------------------------------------
    # Load all probe conditions
    # --------------------------------------------------

    print(
        "\nLoading probe data..."
    )

    condition_data = load_all_conditions(
        args,
        study_classes,
        device
    )

    # --------------------------------------------------
    # Get Model 2.0 representations
    # --------------------------------------------------

    print(
        "\nComputing Model 2.0 representations..."
    )

    condition_reps = {}

    for condition, data in condition_data.items():

        print(
            f"  {condition}: "
            f"{len(data)} identities"
        )

        condition_reps[condition] = (
            get_representations(
                model,
                device,
                data
            )
        )

    # --------------------------------------------------
    # Common identities
    # --------------------------------------------------

    conditions = [
        ("upright", "normal"),
        ("upright", "thatcher"),
        ("inverted", "normal"),
        ("inverted", "thatcher"),
    ]

    common_classes = set(
        condition_reps[
            conditions[0]
        ].keys()
    )

    for condition in conditions[1:]:

        common_classes &= set(
            condition_reps[
                condition
            ].keys()
        )

    common_classes = sorted(
        common_classes
    )

    if args.num_test is not None:

        common_classes = common_classes[
            :min(
                args.num_test,
                len(common_classes)
            )
        ]

    print(
        f"\nUsing {len(common_classes)} "
        f"common identities."
    )

    # ==================================================
    # BUILD UPRIGHT AND INVERTED NORMAL COVARIANCES
    # ==================================================

    print("\nBuilding upright-normal covariance...")

    upright_normal_representations = []

    for cls in common_classes:

        h = condition_reps[
            ("upright", "normal")
        ][cls]

        for row in h:
            upright_normal_representations.append(
                row.numpy()
            )

    upright_covariance, upright_covariance_inv = (
        estimate_covariance(
            upright_normal_representations,
            args.ridge
        )
    )


    print("\nBuilding inverted-normal covariance...")

    inverted_normal_representations = []

    for cls in common_classes:

        h = condition_reps[
            ("inverted", "normal")
        ][cls]

        for row in h:
            inverted_normal_representations.append(
                row.numpy()
            )

    inverted_covariance, inverted_covariance_inv = (
        estimate_covariance(
            inverted_normal_representations,
            args.ridge
        )
    )

    # ==================================================
    # MAHALANOBIS DISTANCES
    # ==================================================

    upright_distances = {}
    inverted_distances = {}

    results = []

    for cls in common_classes:

        UN = condition_reps[
            ("upright", "normal")
        ][cls].numpy()

        UT = condition_reps[
            ("upright", "thatcher")
        ][cls].numpy()

        IN = condition_reps[
            ("inverted", "normal")
        ][cls].numpy()

        IT = condition_reps[
            ("inverted", "thatcher")
        ][cls].numpy()

        # --------------------------------------------------
        # Identity-level condition representations
        # --------------------------------------------------

        UN_mean = UN.mean(axis=0)
        UT_mean = UT.mean(axis=0)

        IN_mean = IN.mean(axis=0)
        IT_mean = IT.mean(axis=0)

        # --------------------------------------------------
        # Mahalanobis distances
        # Use the covariance for the corresponding orientation
        # --------------------------------------------------

        D_U = mahalanobis_distance(
            UN_mean,
            UT_mean,
            upright_covariance_inv
        )

        D_I = mahalanobis_distance(
            IN_mean,
            IT_mean,
            inverted_covariance_inv
        )

        upright_distances[cls] = D_U
        inverted_distances[cls] = D_I

        results.append({
            "identity": cls,
            "upright_distance": D_U,
            "inverted_distance": D_I,
            "difference_U_minus_I": D_U - D_I,
        })

    results_df = pd.DataFrame(results)

    # ==================================================
    # SUMMARY
    # ==================================================

    mean_U = (
        results_df[
            "upright_distance"
        ].mean()
    )

    mean_I = (
        results_df[
            "inverted_distance"
        ].mean()
    )

    mean_difference = (
        results_df[
            "difference_U_minus_I"
        ].mean()
    )

    print(
        "\n=========================================="
    )

    print(
        "MAHALANOBIS THATCHER ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        f"\nMean UN-UT Mahalanobis distance: "
        f"{mean_U:.6f}"
    )

    print(
        f"Mean IN-IT Mahalanobis distance: "
        f"{mean_I:.6f}"
    )

    print(
        f"\nMean (UN-UT) - (IN-IT): "
        f"{mean_difference:.6f}"
    )

    # ==================================================
    # STATISTICAL TEST
    # ==================================================

    from scipy.stats import ttest_1samp

    t_result = ttest_1samp(
        results_df[
            "difference_U_minus_I"
        ].values,
        0.0
    )

    print(
        "\nPaired identity-level comparison:"
    )

    print(
        f"t = {t_result.statistic:.6f}"
    )

    print(
        f"p = {t_result.pvalue:.8g}"
    )

    print(
        f"df = {len(results_df) - 1}"
    )

    # ==================================================
    # EFFECT SIZE
    # ==================================================

    differences = results_df[
        "difference_U_minus_I"
    ].values

    sd_difference = np.std(
        differences,
        ddof=1
    )

    cohen_dz = (
        mean_difference /
        sd_difference
    )

    print(
        f"Cohen's dz = {cohen_dz:.6f}"
    )

    # ==================================================
    # SAVE
    # ==================================================

    output_name = (
        f"mahalanobis_thatcher_"
        f"{args.category}_"
        f"{args.variant}.csv"
    )

    results_df.to_csv(
        output_name,
        index=False
    )

    print(
        f"\nSaved identity-level results to:"
        f"\n  {output_name}"
    )

    # ==================================================
    # INTERPRETATION
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=========================================="
    )

    if t_result.pvalue < 0.05:

        if mean_U > mean_I:

            print(
                "Upright Thatcherization produces "
                "a significantly larger Mahalanobis "
                "displacement than inverted Thatcherization."
            )

        else:

            print(
                "Inverted Thatcherization produces "
                "a significantly larger Mahalanobis "
                "displacement than upright Thatcherization."
            )

    else:

        print(
            "No significant difference between "
            "upright and inverted Thatcherization "
            "distances."
        )


if __name__ == "__main__":
    main()