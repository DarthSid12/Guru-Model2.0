"""
download_houses_individualized.py

Download and extract the raw datasets for each category into data/.

Usage:
    python download.py                       # downloads all categories
    python download.py faces objects houses   # only the listed categories

Faces/objects come from Google Drive archives (see DATASETS below). Houses come from ZuBud - 5 imgs per house
"""

import os
import random
import re
import shutil
import sys
import subprocess
import urllib.request
import zipfile
import tarfile

try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
    import gdown


# category -> (google-drive share url, archive filename, extraction target dir)
DATASETS = {
    "faces": (
        "https://drive.google.com/file/d/137YEYgzi6qH5hXqpJOdqtyxe7hWgjAxZ/view?usp=drive_link",
        "128_faces_manually_cleaned.zip",
        "data/faces",
    ),
    "objects": (
        "https://drive.google.com/file/d/1kOhJXRKaGpoQHKazFPG3QA8tLhBGNKOa/view",
        "ImageNet_objects.zip",
        "data/ImageNet_objects",
    ),

    "houses": (
        "https://drive.google.com/file/d/1pqkvfEmWolxR_1QT-tFoqJPwLu3hzayz/view?usp=sharing",
        "houses.zip",
        "data/houses",
    ),
}

"""
40 houses picked via random number generator: 
[5, 9, 11, 16, 19, 
24, 31, 32, 34, 35, 
37, 38, 51, 59, 72, 
81, 86, 94, 95, 96, 
97, 101, 103, 106, 107, 
110, 121, 135, 144, 148, 
158, 171, 172, 173, 175, 
181, 183, 191, 193, 201]
"""

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OBJECTS_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, valid, test -- matches faces' ~80/10/10
OBJECTS_SPLIT_SEED = 42


def download_and_extract(category):
    if category not in DATASETS or DATASETS[category][0] is None:
        print(f"[skip] No download URL configured for '{category}'.")
        return

    url, name, extract_dir = DATASETS[category]
    os.makedirs("data", exist_ok=True)
    archive_path = os.path.join("data", name)

    if not os.path.exists(archive_path):
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Downloading {category} -> {name} ...")
        gdown.download(download_url, archive_path, quiet=False)
    else:
        print(f"[skip] {name} already downloaded.")

    if os.path.isdir(extract_dir) and any(os.scandir(extract_dir)):
        print(f"[skip] {category} already extracted to {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {name} -> {extract_dir} ...")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=extract_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(path=extract_dir)
    else:
        print(f"[warn] Unknown archive format: {name}")


def split_objects():
    """ImageNet_objects.zip and the houses dataset extracts to one folder of class subdirs with no
    train/valid/test split. preprocess.py needs that split to exist (an
    "item" for objects is a whole class, unlike houses, so a plain random
    per-image split per class is fine -- no study/test photo pairing needed).
    Lays out data/ImageNet_objects/{train,valid,test}/<synset>/<img>,
    ~80/10/10 per class to match the faces split.
    """
    extract_dirs = [DATASETS["objects"][2], DATASETS["houses"][2]]  # data/ImageNet_objects, data/houses
    for extract_dir in extract_dirs:
        if all(os.path.isdir(os.path.join(extract_dir, s)) and any(os.scandir(os.path.join(extract_dir, s)))
           for s in ("train", "valid", "test")):
                print(f"[skip] objects already split under {extract_dir}.")
                continue

        class_root = None
        for dirpath, dirnames, _ in os.walk(extract_dir):
            classes = [d for d in dirnames if d not in ("train", "valid", "test")]
            if not classes:
                continue
            sample_dir = os.path.join(dirpath, classes[0])
            if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(sample_dir)):
                class_root = dirpath
                break
        if class_root is None:
            raise FileNotFoundError(f"Could not find per-class image folders under {extract_dir!r}.")

        classes = sorted(d for d in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, d)))
        rng = random.Random(OBJECTS_SPLIT_SEED)
        train_r, valid_r, test_r = OBJECTS_SPLIT_RATIOS

        print(f"Splitting {len(classes)} object classes from {class_root!r} into "
              f"train/valid/test ({OBJECTS_SPLIT_RATIOS}) ...")
        for cls in classes:
            src_cls = os.path.join(class_root, cls)
            imgs = sorted(f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS))
            rng.shuffle(imgs)
            n = len(imgs)
            n_valid = max(int(n * valid_r), 1) 
            n_test = max(int(n * test_r), 1)
            n_train = n-n_valid-n_train
            split_files = {
                "train": imgs[:n_train],
                "valid": imgs[n_train:n_train + n_valid],
                "test": imgs[n_train + n_valid:],
            }
            for split, files in split_files.items():
                cls_dir = os.path.join(extract_dir, split, cls)
                os.makedirs(cls_dir, exist_ok=True)
                for fname in files:
                    src_path = os.path.abspath(os.path.join(src_cls, fname))
                    dst_path = os.path.join(cls_dir, fname)
                    if os.path.exists(dst_path):
                        continue
                    try:
                        os.symlink(src_path, dst_path)
                    except OSError:
                        shutil.copy2(src_path, dst_path)

        print(f"Built objects split at {extract_dir}: {len(classes)} classes across train/valid/test.")

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys()) + ["houses"]
    for cat in requested:
        download_and_extract(cat)
    split_objects()
    print("\nDone.")
