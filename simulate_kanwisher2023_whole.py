"""
simulate_kanwisher2023_whole.py

Replicate kanwisher experiment, but without fixations, to better model subjects' unlimited observation time
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF

from model import Model
from salience_trans import OnTheFlyWholeTransform


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------- data -----------------------------
def collect_identities(data_root, category, splits):
    """
    Group whole images by identity/class.

    Returns:
        {
            class_name: [
                (split, image_path),
                ...
            ]
        }
    """
    by_class = {}

    for split in splits:
        split_dir = Path(data_root) / category / split

        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png",
                    ".bmp", ".webp"
                }:
                    by_class.setdefault(class_dir.name, []).append(
                        (split, img_path)
                    )

    return by_class

def sample_identities(by_class, num_identities, images_per_identity, seed):
    """Keep identities with >= 2 photos (a trial needs a target and a same-identity
    match), then subsample identities and photos to the requested sizes."""
    rng = random.Random(seed)
    usable = {c: imgs for c, imgs in by_class.items() if len(imgs) >= 2}
    if len(usable) < 2:
        raise SystemExit(f"Need >= 2 identities with >= 2 photos each; got {len(usable)}. "
                         "For one-photo-per-class categories pass --splits valid test.")
    names = sorted(usable)
    if 0 < num_identities < len(names):
        names = rng.sample(names, num_identities)
    out = {}
    for c in sorted(names):
        imgs = sorted(usable[c], key=lambda rec: (rec[0], str(rec[1])))
        if 0 < images_per_identity < len(imgs):
            imgs = rng.sample(imgs, images_per_identity)
        out[c] = imgs
    return out

def load_whole_image(path):
    return TF.to_tensor(
        Image.open(path).convert("RGB")
    )

# --------------------- representation & distance --------------------
def apply_binomial_noise(binary_tensor, p_noise):
    if p_noise == 0.0:
        return binary_tensor
    mask = torch.rand_like(binary_tensor) < p_noise
    return torch.logical_xor(binary_tensor.bool(), mask).float()

def encode_image(model, transform, image, device, p_noise):
    """
    Whole image -> binary representation.
    """

    x = image.unsqueeze(0).to(device)

    x = transform(x)

    model.stochastic = True

    _, h, _ = model(x, return_rep=True)

    h_noisy = apply_binomial_noise(
        h.cpu(),
        p_noise
    )

    return h_noisy.squeeze(0).float().numpy()

def correlation_distance(a, b):
    """1 - Pearson r, the distance Dobs et al. use to model the network's choice."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


# ------------------------ the matching task -------------------------
def run_condition(model, device,
                  target_transform,
                  match_transform,
                  distractor_transform,
                  identities,
                  p_noise):

    """
    Image-to-image matching condition.

    A trial:
        target image
        same-identity match
        different-identity distractor

    Correct if:
        distance(target, match)
        <
        distance(target, distractor)
    """

    feats = {}

    # encode all images once
    with torch.no_grad():

        for cls, imgs in identities.items():

            for split, path in imgs:

                image = load_whole_image(path)

                key = (cls, split, str(path))

                feats[key] = {}

                feats[key]["target"] = encode_image(
                    model,
                    target_transform,
                    image,
                    device,
                    p_noise
                )

                feats[key]["match"] = encode_image(
                    model,
                    match_transform,
                    image,
                    device,
                    p_noise
                )

                feats[key]["distractor"] = encode_image(
                    model,
                    distractor_transform,
                    image,
                    device,
                    p_noise
                )


    keys = {
        cls: [
            (cls, split, str(path))
            for split, path in imgs
        ]
        for cls, imgs in identities.items()
    }


    correct = 0
    trials = 0


    for cls, cls_keys in keys.items():

        other_keys = [
            k
            for other_cls, ks in keys.items()
            if other_cls != cls
            for k in ks
        ]


        for target_key in cls_keys:

            f_target = feats[target_key]["target"]


            for match_key in cls_keys:

                if match_key == target_key:
                    continue


                d_match = correlation_distance(
                    f_target,
                    feats[match_key]["match"]
                )


                for distractor_key in other_keys:

                    d_dist = correlation_distance(
                        f_target,
                        feats[distractor_key]["distractor"]
                    )


                    if d_match < d_dist:
                        correct += 1

                    trials += 1


    return correct / max(trials,1), trials


def calibrate_noise(model, device, upright_tf, identities, args):
    """Two-stage search for the retrieval-noise p whose Upright matching
    accuracy first drops to/below args.calib_target: a coarse sweep
    (args.calib_step) finds the bracket where accuracy crosses the target,
    then a fine sweep (args.calib_fine_step) refines within that bracket.
    A full fine-resolution scan from 0 (as simulate_yin1969.py does) is too
    slow here -- one Kanwisher condition eval is O(identities^2 * images^3)
    triplets, far more expensive than Yin's 24-pair test.
    """
    print(f"--- Calibrating noise on {args.category} (Upright) ---")
    prev_p, hit_p = 0.0, None
    prev_acc = None
    for p in np.arange(0.0, args.calib_max, args.calib_step):
        set_seed(args.seed)
        acc, _ = run_condition(model, device, upright_tf, upright_tf, upright_tf, identities, p)
        print(f"  noise {p:.2f} -> {acc*100:.2f}%")
        if acc <= args.calib_target:
            hit_p = p
            break
        prev_p = p
        prev_acc = acc
    if hit_p is None:
        print(f"[!] Coarse sweep never reached target; using noise p={args.calib_max:.2f}\n")
        return args.calib_max
    if hit_p == 0.0:
        print(f"[!] Using noise p={hit_p:.2f}\n")
        return hit_p

    print(f"--- Refining between {prev_p:.2f} and {hit_p:.2f} ---")
    best_p = hit_p
    prev_p = hit_p
    for p in np.arange(prev_p + args.calib_fine_step, hit_p, args.calib_fine_step):
        set_seed(args.seed)
        acc, _ = run_condition(model, device, upright_tf, upright_tf, upright_tf, identities, p)
        print(f"  noise {p:.2f} -> {acc*100:.2f}%")
        if acc <= args.calib_target:
            best_p = p if abs(acc - args.calib_target) <= abs(prev_acc - args.calib_target) else prev_p
            break
        prev_p = p
        prev_acc = acc
    print(f"[!] Using noise p={best_p:.2f}\n")
    return best_p


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="faces | houses_zubud | objects")
    ap.add_argument("--run-dir", default=None,
                    help="training run dir; fills in checkpoint/label-map/variant/backbone")
    ap.add_argument("--variant", choices=["lp", "cnn", "plain"], default=None)
    ap.add_argument("--backbone", default=None, help="must match the trained checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--label-map", default=None, help="label_map.json from the run (sets num_classes)")
    ap.add_argument("--num-classes", type=int, default=None, help="used if --label-map absent")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--splits", nargs="+", default=["valid"],
                    help="packed splits to draw photos from; houses_zubud needs 'valid test' "
                         "(one held-out view each) to get 2 photos per house")
    ap.add_argument("--num-identities", type=int, default=40,
                    help="identities per condition (Dobs et al. Exp. 1 used 40); -1 = all")
    ap.add_argument("--images-per-identity", type=int, default=5,
                    help="photos per identity (Dobs et al. used 5); -1 = all")
    ap.add_argument("--noise", type=float, default=None,
                    help="fixed retrieval-noise p to use; skips calibration when set")
    ap.add_argument("--calib-target", type=float, default=0.875,
                    help="Upright matching accuracy to match when calibrating noise "
                         "(default 0.875 = Dobs et al. Exp. 5 human upright accuracy; "
                         "the Kanwisher analogue of Yin's upright-upright calibration)")
    ap.add_argument("--calib-max", type=float, default=0.75)
    ap.add_argument("--calib-step", type=float, default=0.05)
    ap.add_argument("--calib-fine-step", type=float, default=0.01,
                    help="refinement step within the coarse bracket that crosses --calib-target")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", default="data", help="raw whole-image dataset root")
    return ap.parse_args()


def resolve_from_run_dir(args):
    """Fill in checkpoint / label-map / variant / backbone / temperature from a
    training run dir, without overriding anything passed explicitly."""
    if not args.run_dir:
        return
    cfg_path = os.path.join(args.run_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.run_dir, "best_model.pth")
    if args.label_map is None:
        lm = os.path.join(args.run_dir, "label_map.json")
        if os.path.isfile(lm):
            args.label_map = lm
    if args.variant is None:
        args.variant = cfg.get("variant", "lp")
    if args.backbone is None:
        args.backbone = cfg.get("backbone", "resnet18")
    if args.temperature is None:
        args.temperature = cfg.get("temperature", 2.0)


def main():
    args = parse_args()
    resolve_from_run_dir(args)

    if args.variant is None:
        args.variant = "lp"
    if args.backbone is None:
        args.backbone = "resnet18"
    if args.temperature is None:
        args.temperature = 2.0
    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint or --run-dir.")

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.label_map:
        with open(args.label_map) as f:
            num_classes = len(json.load(f))
    elif args.num_classes:
        num_classes = args.num_classes
    else:
        raise SystemExit("Provide --label-map (or --run-dir) or --num-classes to size the model head.")

    model = Model(size=224, num_classes=num_classes, pretrained=False,
                  T=args.temperature, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    by_class = collect_identities(args.data_root, args.category, args.splits)
    identities = sample_identities(by_class, args.num_identities,
                                   args.images_per_identity, args.seed)
    n_photos = sum(len(v) for v in identities.values())
    print(f"--- {args.category}: {len(identities)} identities, {n_photos} photos "
          f"from splits {args.splits} ---")

    upright_tf = OnTheFlyWholeTransform("valid", args.variant, device).to(device)
    inverted_tf = OnTheFlyWholeTransform("test", args.variant, device).to(device)

    # 1) calibrate noise on the Upright condition (skipped if --noise given) --
    # the Kanwisher analogue of Yin's upright-upright calibration, since this
    # paradigm has no separate study phase to cross.
    if args.noise is not None:
        ideal_noise = args.noise
        print(f"[!] Using fixed noise p={ideal_noise:.2f} (calibration skipped)\n")
    else:
        ideal_noise = calibrate_noise(model, device, upright_tf, identities, args)

    # 2) both conditions at the calibrated noise level
    rows = []
    for cond, target_tf, match_tf, distractor_tf in [("Upright", upright_tf, upright_tf, upright_tf), ("Inverted", inverted_tf, inverted_tf, inverted_tf)]:
        set_seed(args.seed)
        acc, n_trials = run_condition(model, device, target_tf, match_tf, distractor_tf, identities, ideal_noise)
        rows.append({"Presentation": cond, "Matching Accuracy": f"{acc*100:.2f}%",
                     "Trials": n_trials})
        print(f"  {cond:8s}: {acc*100:.2f}%  ({n_trials} triplets)")

    drop = (float(rows[0]["Matching Accuracy"].rstrip('%'))
            - float(rows[1]["Matching Accuracy"].rstrip('%')))

    print("\n=====================================================")
    print(f" DOBS/KANWISHER (2023) INVERSION SIMULATION — {args.category} (noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n Inversion effect (upright - inverted): {drop:+.2f} points")


if __name__ == "__main__":
    main()
