"""
thatcher_download_bigjobby_2.py

Create a dataset for the Thatcher illusion experiment from the existing
faces dataset.

Unlike the training dataset, this dataset does NOT use train/valid/test
splits. Instead, every image is duplicated into the four experimental
conditions needed for the Thatcher paradigm:
    
    thatcher_data/
        faces/
            lp/
                upright/
                    normal/
                        <identity>/<image>.txt # contains the locations of all the fixation crops - same base image as cnn case
                    thatcher/
                        <identity>/<image>.txt
                inverted/
                    normal/
                        <identity>/<image>.txt
                    thatcher/
                        <identity>/<image>.txt
            cnn/
                upright/
                    normal/
                        <identity>/<image>.png # contains the images
                    thatcher/
                        <identity>/<image>.png 
                inverted/
                    normal/
                        <identity>/<image>.png
                    thatcher/
                        <identity>/<image>.png

Command line bash:

    python thatcher_download_bigjobby_2.py \
        --source data/faces_cleaned \
        --dest data/thatcher_data/faces
"""

import argparse
import os

import torch
import torchvision.transforms.functional as TF

from PIL import Image
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp") # I noticed both .jpg and .png show up sometimes so I put a lot of file types in for whatever future dataset we may pull from

import numpy as np

from salience_trans import SaliencePipeline

def find_split_root(raw_root):
    """
    Locate the directory containing train/valid/test.

    Example:

        data/faces_cleaned/
            128_identities/
                train/
                valid/
                test/
    """
    best = None

    for dirpath, dirnames, _ in os.walk(raw_root):
        if {"train", "valid", "test"} <= set(dirnames):
            n_classes = sum(
                len(
                    [
                        d
                        for d in os.listdir(os.path.join(dirpath, split))
                        if os.path.isdir(os.path.join(dirpath, split, d))
                    ]
                )
                for split in ("train", "valid", "test")
            )

            if best is None or n_classes > best[0]:
                best = (n_classes, dirpath)

    if best is None:
        raise RuntimeError("Could not locate train/valid/test folders.")

    return best[1]

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# saves both the lp and cnn version of the image
def save_lp_and_cnn(img, out_root, orientation, condition, identity, stem, pipeline, device):
    save_image(
        img,
        os.path.join(
            out_root,
            "cnn",
            orientation,
            condition,
            identity,
            stem
        )
    )

    img_t = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        coords = pipeline.sample_salience_points(img_t)[0].cpu().numpy()

    lp_dir = os.path.join(
        out_root,
        "lp",
        orientation,
        condition,
        identity
    )

    os.makedirs(lp_dir, exist_ok=True)

    stem_base = os.path.splitext(stem)[0]

    np.savetxt(
        os.path.join(lp_dir, f"{stem_base}.txt"),
        coords,
        fmt="%d",
    )

def process_split(split_dir, out_root, pipeline, device):
    from thatcherize_bigjobby_2 import thatcherize

    identities = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )

    for identity in identities:

        src_dir = os.path.join(split_dir, identity)

        files = sorted(
            f for f in os.listdir(src_dir)
            if f.lower().endswith(IMG_EXTS)
        )

        print(f"{identity}: {len(files)} images")

        for fname in files:

            src_path = os.path.join(src_dir, fname)

            img = Image.open(src_path).convert("RGB")

            stem = os.path.splitext(fname)[0] + ".png"

            upright_normal = img
            upright_thatcher = thatcherize(src_path)

            if upright_thatcher:
                inverted_normal = img.rotate(180, expand=False)
                inverted_thatcher = upright_thatcher.rotate(180, expand=False)

                save_lp_and_cnn(
                    upright_normal,
                    out_root,
                    "upright",
                    "normal",
                    identity,
                    stem,
                    pipeline,
                    device,
                )    

                save_lp_and_cnn(
                    upright_thatcher,
                    out_root,
                    "upright",
                    "thatcher",
                    identity,
                    stem,
                    pipeline,
                    device,
                )

                save_lp_and_cnn(
                    inverted_normal,
                    out_root,
                    "inverted",
                    "normal",
                    identity,
                    stem,
                    pipeline,
                    device,
                )

                save_lp_and_cnn(
                    inverted_thatcher,
                    out_root,
                    "inverted",
                    "thatcher",
                    identity,
                    stem,
                    pipeline,
                    device,
                )
                #return # for testing one image quickly
            else:
                print("thatcherization failed - facemesh unsuccessful")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="data/thatcher_base",
        help="Root of downloaded faces dataset.",
    )

    parser.add_argument(
        "--dest",
        default="data/thatcher_data/faces",
        help="Output directory.",
    )

    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test"],
        help=(
            "Which split to use as the source images. "
            "Default is valid."
        ),
    )

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """
    This part was changed to use my (David's) manually curated set of thatcher-experiment-acceptable pictures;
    to run on full faces_cleaned dataset, input different source file and switch the commenting on the following four lines
    """

    #split_root = find_split_root(args.source)
    split_root = args.source

    #src = os.path.join(split_root, args.split)
    src = args.source

    if not os.path.isdir(src):
        raise RuntimeError(f"Missing split: {src}")

    print(f"Source: {src}")
    print(f"Destination: {args.dest}")

    pipeline = SaliencePipeline(
        type="valid",
        device=device,
        num_salient_points=32,
    ).to(device)

    process_split(src, args.dest, pipeline, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
