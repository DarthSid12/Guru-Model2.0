"""
preprocess.py

Convert raw category datasets into preprocessed fixation crops using the
(category-agnostic, Gabor-saliency) SaliencePipeline, producing both the
log-polar (lp) and plain-crop (cnn) variants.

Output layout:
    processed_data/<category>/<lp|cnn>/<split>/<class>/<img>_proc<n>.png

Split -> presentation orientation (set by the pipeline):
    train -> random rotation augmentation
    valid -> upright
    test  -> inverted (180 deg)        <- used as the "inverted" condition in Yin

All raw images are resized (shorter side) + center-cropped to a square
--input-size so every category is fed to the pipeline on equal footing.

Usage:
    python preprocess.py --categories faces objects --num-fixations 32
    python preprocess.py --categories faces --splits valid test   # e.g. for Yin only
"""

import argparse
import os

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from salience_trans import SaliencePipeline

# Where each category's raw (downloaded) data lives. preprocess auto-discovers
# the directory containing train/valid/test underneath these roots.
RAW_ROOTS = {
    "faces": "data/faces_cleaned",
    "objects": "data/ImageNet_objects",
    "houses": "data/houses",
}

SPLITS = ("train", "valid", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_split_root(raw_dir):
    """Find the directory under raw_dir that directly contains train/valid/test
    subfolders. If several exist (e.g. multiple {N}_identities folders), pick the
    one covering the most splits, then the most classes."""
    best = None
    for dirpath, dirnames, _ in os.walk(raw_dir):
        present = [s for s in SPLITS if s in dirnames]
        if not present:
            continue
        n_classes = 0
        for s in present:
            sd = os.path.join(dirpath, s)
            n_classes += sum(1 for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d)))
        key = (len(present), n_classes)
        if best is None or key > best[0]:
            best = (key, dirpath)
    if best is None:
        raise FileNotFoundError(
            f"Could not find train/valid/test under {raw_dir!r}. "
            f"Run download.py or pass the correct raw root."
        )
    return best[1]


def split_type(split):
    if split == "train":
        return "train"      # random rotation
    if split == "test":
        return "test"       # inverted
    return "valid"          # identity / upright


def preprocess_category(category, raw_root, processed_root, num_fixations,
                        input_size, splits, device, limit_per_class=None):
    split_src_root = find_split_root(raw_root)
    print(f"\n=== {category}: raw splits found at {split_src_root} ===")

    resize = T.Compose([T.Resize(input_size), T.CenterCrop(input_size)])

    for split in splits:
        src_split = os.path.join(split_src_root, split)
        if not os.path.isdir(src_split):
            print(f"  [skip] no '{split}' split for {category}")
            continue

        pipeline = SaliencePipeline(type=split_type(split), device=device,
                                    num_salient_points=num_fixations).to(device)

        classes = sorted(d for d in os.listdir(src_split)
                         if os.path.isdir(os.path.join(src_split, d)))
        print(f"  [{split}] {len(classes)} classes")

        for cls_name in classes:
            src_cls = os.path.join(src_split, cls_name)
            lp_dir = os.path.join(processed_root, category, "lp", split, cls_name)
            cnn_dir = os.path.join(processed_root, category, "cnn", split, cls_name)
            os.makedirs(lp_dir, exist_ok=True)
            os.makedirs(cnn_dir, exist_ok=True)

            imgs = [f for f in sorted(os.listdir(src_cls))
                    if f.lower().endswith(IMG_EXTS)]
            if limit_per_class:
                imgs = imgs[:limit_per_class]

            for fname in imgs:
                try:
                    img = Image.open(os.path.join(src_cls, fname)).convert("RGB")
                except Exception as e:
                    print(f"    [warn] cannot open {fname}: {e}")
                    continue
                img = resize(img)
                img_t = TF.to_tensor(img).unsqueeze(0).to(device)  # (1,C,H,W)

                with torch.no_grad():
                    lp, cnn = pipeline(img_t)  # each (1, N, C, H, W)

                stem = os.path.splitext(fname)[0]
                for n, t in enumerate(lp[0]):
                    TF.to_pil_image(t.cpu().clamp(0, 1)).save(os.path.join(lp_dir, f"{stem}_proc{n}.png"))
                for n, t in enumerate(cnn[0]):
                    TF.to_pil_image(t.cpu().clamp(0, 1)).save(os.path.join(cnn_dir, f"{stem}_proc{n}.png"))

            print(f"    {cls_name}: {len(imgs)} base imgs x {num_fixations} fixations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects"])
    ap.add_argument("--processed-root", default="processed_data")
    ap.add_argument("--num-fixations", type=int, default=32)
    ap.add_argument("--input-size", type=int, default=224,
                    help="raw images are resized + center-cropped to this square size")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap base images per class (for quick tests)")
    ap.add_argument("--raw-root", action="append", default=[],
                    help="override raw root, e.g. --raw-root objects=/path/to/objects")
    args = ap.parse_args()

    raw_roots = dict(RAW_ROOTS)
    for ov in args.raw_root:
        cat, path = ov.split("=", 1)
        raw_roots[cat] = path

    for category in args.categories:
        if category not in raw_roots:
            print(f"[skip] no raw root configured for '{category}'")
            continue
        preprocess_category(
            category, raw_roots[category], args.processed_root,
            args.num_fixations, args.input_size, args.splits, args.device,
            args.limit_per_class,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
