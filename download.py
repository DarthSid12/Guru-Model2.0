"""
download.py

Download and extract the raw datasets for each category into data/.

Usage:
    python download.py                       # downloads all categories
    python download.py faces objects houses   # only the listed categories

Faces/objects come from Google Drive archives (see DATASETS below). Houses
comes from the public emanhamed/Houses-dataset GitHub repo: one folder of
images per house (bedroom/bathroom/kitchen/frontal); we only keep each
house's "<id>_frontal.jpg" image. Unlike faces (folder = identity), houses
is a single generic class ("house") for classification.

Each house has only one photo (no second exemplar like faces have per
identity), so the Yin upright/inverted comparison needs the *same* photo
present in both valid and test (preprocess.py derives orientation from the
split a file sits in). Photos are therefore split into:
  - a train-only pool (classifier training, held out from valid/test)
  - a shared "Yin pool" duplicated into both valid/ and test/ (same
    filenames in both, so simulate_yin1969.py can draw study/old items that
    exist upright in valid and inverted in test)
laid out as data/houses/{train,valid,test}/house/<id>_frontal.jpg.

(TODO per project owner: replace this single-photo-per-house approximation
with a real study/test pair once a set of ~40 houses with 2 photos each is
sourced.)
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
        "data/faces_cleaned",
    ),
    "objects": (
        "https://drive.google.com/file/d/1kOhJXRKaGpoQHKazFPG3QA8tLhBGNKOa/view",
        "ImageNet_objects.zip",
        "data/ImageNet_objects",
    ),
}

HOUSES_REPO_ZIP = "https://github.com/emanhamed/Houses-dataset/archive/refs/heads/master.zip"
HOUSES_ARCHIVE = "data/Houses-dataset.zip"
HOUSES_EXTRACT_DIR = "data/houses_raw"
HOUSES_FINAL_DIR = "data/houses"
HOUSES_CLASS_NAME = "house"
HOUSES_YIN_POOL_SIZE = 100  # photos duplicated into both valid/ and test/ for the Yin sim
HOUSES_SPLIT_SEED = 42
_FRONTAL_RE = re.compile(r"^(\d+)_frontal\.(jpg|jpeg)$", re.IGNORECASE)

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
    """ImageNet_objects.zip extracts to one folder of class subdirs with no
    train/valid/test split. preprocess.py needs that split to exist (an
    "item" for objects is a whole class, unlike houses, so a plain random
    per-image split per class is fine -- no study/test photo pairing needed).
    Lays out data/ImageNet_objects/{train,valid,test}/<synset>/<img>,
    ~80/10/10 per class to match the faces split.
    """
    extract_dir = DATASETS["objects"][2]  # data/ImageNet_objects
    if all(os.path.isdir(os.path.join(extract_dir, s)) and any(os.scandir(os.path.join(extract_dir, s)))
           for s in ("train", "valid", "test")):
        print(f"[skip] objects already split under {extract_dir}.")
        return

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
    train_r, valid_r, _ = OBJECTS_SPLIT_RATIOS

    print(f"Splitting {len(classes)} object classes from {class_root!r} into "
          f"train/valid/test ({OBJECTS_SPLIT_RATIOS}) ...")
    for cls in classes:
        src_cls = os.path.join(class_root, cls)
        imgs = sorted(f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS))
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_r)
        n_valid = int(n * valid_r)
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


def download_houses():
    """Fetch emanhamed/Houses-dataset and lay frontal house photos out as a
    single 'house' class pooled across train/valid/test:
        data/houses/{train,valid,test}/house/<id>_frontal.jpg
    """
    if os.path.isdir(HOUSES_FINAL_DIR) and any(os.scandir(HOUSES_FINAL_DIR)):
        print(f"[skip] houses already built at {HOUSES_FINAL_DIR}.")
        return

    os.makedirs("data", exist_ok=True)
    if not os.path.exists(HOUSES_ARCHIVE):
        print("Downloading houses dataset repo ...")
        urllib.request.urlretrieve(HOUSES_REPO_ZIP, HOUSES_ARCHIVE)
    else:
        print(f"[skip] {HOUSES_ARCHIVE} already downloaded.")

    if not (os.path.isdir(HOUSES_EXTRACT_DIR) and any(os.scandir(HOUSES_EXTRACT_DIR))):
        os.makedirs(HOUSES_EXTRACT_DIR, exist_ok=True)
        print(f"Extracting houses repo -> {HOUSES_EXTRACT_DIR} ...")
        with zipfile.ZipFile(HOUSES_ARCHIVE, "r") as zf:
            zf.extractall(path=HOUSES_EXTRACT_DIR)

    src_dir = None
    for dirpath, _, filenames in os.walk(HOUSES_EXTRACT_DIR):
        if any(_FRONTAL_RE.match(f) for f in filenames):
            src_dir = dirpath
            break
    if src_dir is None:
        raise FileNotFoundError(
            f"Could not find any '<id>_frontal.jpg' images under {HOUSES_EXTRACT_DIR}."
        )

    frontal_files = sorted(f for f in os.listdir(src_dir) if _FRONTAL_RE.match(f))
    rng = random.Random(HOUSES_SPLIT_SEED)
    rng.shuffle(frontal_files)

    n = len(frontal_files)
    n_yin = min(HOUSES_YIN_POOL_SIZE, n)
    yin_pool = frontal_files[:n_yin]
    train_pool = frontal_files[n_yin:]

    # train gets its own held-out photos; valid/test share the same yin_pool
    # filenames (duplicated) so the same house photo exists upright (valid)
    # and inverted (test) for the Yin study/old-vs-new comparison.
    split_files = {"train": train_pool, "valid": yin_pool, "test": yin_pool}

    print(f"Pooling {n} frontal house photos from {src_dir!r} into one 'house' class "
          f"({len(train_pool)} train-only, {n_yin} shared valid/test Yin pool) ...")
    for split, files in split_files.items():
        cls_dir = os.path.join(HOUSES_FINAL_DIR, split, HOUSES_CLASS_NAME)
        os.makedirs(cls_dir, exist_ok=True)
        for fname in files:
            src_path = os.path.abspath(os.path.join(src_dir, fname))
            dst_path = os.path.join(cls_dir, fname)
            if os.path.exists(dst_path):
                continue
            try:
                os.symlink(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)
        print(f"  {split}: {len(files)} photos -> {cls_dir}")

    print(f"Built houses category: 1 class ('{HOUSES_CLASS_NAME}'), {n} total frontal photos.")


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys()) + ["houses"]
    for cat in requested:
        if cat == "houses":
            download_houses()
        else:
            download_and_extract(cat)
            if cat == "objects":
                split_objects()
    print("\nDone.")
