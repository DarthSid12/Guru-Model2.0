"""
download.py

Download and extract the raw datasets for each category into data/.

Usage:
    python download.py                # downloads all categories that have a URL
    python download.py faces objects  # only the listed categories

Houses currently have no source URL (to be added once sourced). Add the file id
to DATASETS["houses"] and it will be picked up automatically.
"""

import os
import sys
import subprocess
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
        "https://drive.google.com/file/d/1RjE9vBeAoWrd9vrIeDOCBpvaK6zyrowO/view?usp=drive_link",
        "ImageNet_objects.zip",
        "data/ImageNet_objects",
    ),
    "houses": (
        "https://drive.google.com/drive/folders/1GY3yPoQWg-8Dlw5YQ2n0dB8IhNQP-lAE?usp=sharing",
        "houses.zip",
        "data/houses",
    )
}


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


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    for cat in requested:
        download_and_extract(cat)
    print("\nDone.")
