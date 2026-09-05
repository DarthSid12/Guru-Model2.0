#!/usr/bin/env python3

import io
import tarfile
import random
from pathlib import Path

import requests
from PIL import Image


# ============================================================
# Configuration
# ============================================================

ID_FILE = Path("white_male_id_nums.txt")
OUTPUT_DIR = Path("white_male_faces")

IMAGES_PER_IDENTITY = 10

# VGGFace2 Hugging Face archive
TRAIN_URL = (
    "https://huggingface.co/datasets/ProgramComputer/VGGFace2/"
    "resolve/main/data/vggface2_train.tar.gz"
)

TEST_URL = (
    "https://huggingface.co/datasets/ProgramComputer/VGGFace2/"
    "resolve/main/data/vggface2_test.tar.gz"
)


# ============================================================
# Read selected identities
# ============================================================

with open(ID_FILE, "r", encoding="utf-8") as f:
    wanted_ids = {
        line.strip()
        for line in f
        if line.strip()
    }

if len(wanted_ids) != 64:
    raise ValueError(
        f"Expected 64 identities in {ID_FILE}, "
        f"but found {len(wanted_ids)}"
    )

print(f"Looking for {len(wanted_ids)} identities.")


# ============================================================
# Prepare output
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Store up to 10 images for each identity
selected = {
    identity_id: []
    for identity_id in wanted_ids
}


# ============================================================
# Process one TAR archive
# ============================================================

def process_archive(url, archive_name):

    print()
    print("=" * 70)
    print(f"Streaming {archive_name}")
    print("=" * 70)
    print()
    print("This may take a while because the archive is read sequentially.")
    print()

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    # Open the gzip-compressed TAR as a streaming archive.
    with tarfile.open(
        fileobj=response.raw,
        mode="r|gz"
    ) as tar:

        for member in tar:

            if not member.isfile():
                continue

            # VGGFace2 paths look approximately like:
            #
            # n000001/0001_01.jpg
            #
            parts = Path(member.name).parts

            if len(parts) < 2:
                continue

            identity_id = parts[0]

            # Not one of our 64 identities
            if identity_id not in wanted_ids:
                continue

            # Already have enough images
            if len(selected[identity_id]) >= IMAGES_PER_IDENTITY:
                continue

            # Only accept image files
            suffix = Path(member.name).suffix.lower()

            if suffix not in {".jpg", ".jpeg", ".png"}:
                continue

            # Extract image into memory
            file_object = tar.extractfile(member)

            if file_object is None:
                continue

            image_bytes = file_object.read()

            # Verify that it is actually a readable image
            try:
                image = Image.open(
                    io.BytesIO(image_bytes)
                )

                image.verify()

            except Exception:
                print(
                    f"Skipping invalid image: {member.name}"
                )
                continue

            selected[identity_id].append(
                (
                    member.name,
                    image_bytes
                )
            )

            count = len(selected[identity_id])

            print(
                f"{identity_id}: "
                f"{count}/{IMAGES_PER_IDENTITY}"
            )

    response.close()


# ============================================================
# Download train archive
# ============================================================

process_archive(
    TRAIN_URL,
    "VGGFace2 TRAIN"
)


# ============================================================
# Check whether anything is missing
# ============================================================

missing = {
    identity_id: IMAGES_PER_IDENTITY - len(images)
    for identity_id, images in selected.items()
    if len(images) < IMAGES_PER_IDENTITY
}


# ============================================================
# If necessary, check TEST archive
# ============================================================

if missing:

    print()
    print("Some identities did not have enough images in TRAIN.")
    print()

    for identity_id, count in missing.items():
        print(
            f"{identity_id}: missing {count}"
        )

    process_archive(
        TEST_URL,
        "VGGFace2 TEST"
    )


# ============================================================
# Save images
# ============================================================

print()
print("=" * 70)
print("Saving images")
print("=" * 70)

total_saved = 0

for identity_id in sorted(selected):

    images = selected[identity_id]

    if len(images) < IMAGES_PER_IDENTITY:
        print(
            f"WARNING: {identity_id} only has "
            f"{len(images)} images"
        )
        continue

    # Randomly choose 10 if more were encountered.
    chosen = random.sample(
        images,
        IMAGES_PER_IDENTITY
    )

    identity_dir = OUTPUT_DIR / identity_id
    identity_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for original_name, image_bytes in chosen:

        filename = Path(original_name).name

        destination = identity_dir / filename

        with open(destination, "wb") as f:
            f.write(image_bytes)

        total_saved += 1

    print(
        f"{identity_id}: saved 10 images"
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Images saved: {total_saved}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

if total_saved == 640:
    print("SUCCESS: 64 identities × 10 images = 640 images")
else:
    print(
        f"WARNING: expected 640 images, "
        f"but saved {total_saved}"
    )
