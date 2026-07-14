import argparse
import os

import torch
import torchvision.transforms.functional as TF

from PIL import Image

from salience_trans import SaliencePipeline

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


CONDITIONS = [
    ("upright", "normal"),
    ("upright", "thatcher"),
    ("inverted", "normal"),
    ("inverted", "thatcher"),
]


def save_lp(img, out_root, orientation, condition,
            identity, stem, pipeline, device):

    img_t = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        lp, _ = pipeline(img_t)

    lp_dir = os.path.join(
        out_root,
        "lp",
        orientation,
        condition,
        identity,
    )

    os.makedirs(lp_dir, exist_ok=True)

    stem_base = os.path.splitext(stem)[0]

    for n, crop in enumerate(lp[0]):

        TF.to_pil_image(
            crop.cpu().clamp(0, 1)
        ).save(
            os.path.join(
                lp_dir,
                f"{stem_base}_proc{n}.png"
            )
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data/thatcher_data/faces",
        help="Root of the Thatcher dataset."
    )

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipeline = SaliencePipeline(
        type="valid",
        device=device,
        num_salient_points=32,
    ).to(device)

    for orientation, condition in CONDITIONS:

        cnn_root = os.path.join(
            args.root,
            "cnn",
            orientation,
            condition
        )

        identities = sorted(
            d for d in os.listdir(cnn_root)
            if os.path.isdir(os.path.join(cnn_root, d))
        )

        for identity in identities:

            identity_dir = os.path.join(
                cnn_root,
                identity
            )

            files = sorted(
                f for f in os.listdir(identity_dir)
                if f.lower().endswith(IMG_EXTS)
            )

            print(
                f"{orientation}/{condition} "
                f"{identity}: {len(files)}"
            )

            for fname in files:

                path = os.path.join(
                    identity_dir,
                    fname
                )

                img = Image.open(path).convert("RGB")

                stem = os.path.splitext(fname)[0] + ".png"

                save_lp(
                    img,
                    args.root,
                    orientation,
                    condition,
                    identity,
                    stem,
                    pipeline,
                    device,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()