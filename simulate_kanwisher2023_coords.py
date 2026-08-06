"""
simulate_kanwisher2023_coords.py
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from datasets_coords import _PackedSplit, _crop_at
from model_coords import ModelCoords
from salience_trans import OnTheFlyTransform


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- data -----------------------------
def collect_identities(packed_root, category, splits, num_fixations):
    """Group every packed photo of `category` by class across `splits`.

    Returns {class_name: [(split_name, split_obj, img_idx), ...]}. Photos whose
    packed coord budget is smaller than num_fixations are dropped, since a trial
    needs the same number of fixations for every image.
    """
    by_class = {}
    for split in splits:
        sp = _PackedSplit(packed_root, category, split)
        if sp.num_coords < num_fixations:
            raise SystemExit(f"{category}/{split} packed with {sp.num_coords} coords, "
                             f"but --num-fixations is {num_fixations}.")
        for img_idx, ci in enumerate(sp.labels):
            by_class.setdefault(sp.classes[ci], []).append((split, sp, img_idx))
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
        imgs = sorted(usable[c], key=lambda rec: (rec[0], rec[2]))
        if 0 < images_per_identity < len(imgs):
            imgs = rng.sample(imgs, images_per_identity)
        out[c] = imgs
    return out


def load_fixations(sp, img_idx, num_fixations, crop_size=180):
    """
    Returns:
        crops: (N,3,H,W)
        coords: (N,2)
    """

    img = sp.images[img_idx]

    crops = []
    coords = []

    H = sp.images.shape[1]
    W = sp.images.shape[2]

    for j in range(num_fixations):

        x, y = sp.coords[img_idx, j]

        crops.append(
            _crop_at(img, x, y, crop_size)
        )

        coords.append([
            x / W - 0.5,
            y / H - 0.5
        ])

    return (
        torch.stack(crops, dim=0),
        torch.tensor(coords, dtype=torch.float32)
    )

# --------------------- representation & distance --------------------
def apply_binomial_noise(binary_tensor, p_noise):
    if p_noise == 0.0:
        return binary_tensor
    mask = torch.rand_like(binary_tensor) < p_noise
    return torch.logical_xor(binary_tensor.bool(), mask).float()

def encode_image(model, transform, crops, coords, device, p_noise):

    x = transform(crops.to(device))

    coords = coords.to(device).float()

    model.stochastic = True

    _, h, _ = model(
        x,
        coords,
        return_rep=True
    )

    h_noisy = apply_binomial_noise(
        h.cpu(),
        p_noise
    )

    return h_noisy.mean(dim=0).float().numpy()

def correlation_distance(a, b):
    """1 - Pearson r, the distance Dobs et al. use to model the network's choice."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


# ------------------------ the matching task -------------------------
def run_condition(model, device, transform, identities, num_fixations, p_noise):
    """One orientation condition of the target-matching task.

    Returns (accuracy, n_trials). A trial is a (target, same-identity match,
    different-identity distractor) triplet, scored correct when the match is
    closer to the target than the distractor is.
    """
    # features first: every photo is encoded once, then reused across all trials
    feats = {}
    with torch.no_grad():
        for cls, imgs in identities.items():
            for split, sp, img_idx in imgs:
                crops, coords = load_fixations(
                    sp,
                    img_idx,
                    num_fixations
                )

                feats[(cls, split, img_idx)] = encode_image(
                    model,
                    transform,
                    crops,
                    coords,
                    device,
                    p_noise
                )

    keys = {cls: [(cls, split, i) for split, _, i in imgs]
            for cls, imgs in identities.items()}
    all_other = {cls: [k for c2, ks in keys.items() if c2 != cls for k in ks]
                 for cls in keys}

    correct = trials = 0
    for cls, ks in keys.items():
        others = all_other[cls]
        for t_key in ks:                       # target
            f_t = feats[t_key]
            for a_key in ks:                   # same-identity match
                if a_key == t_key:
                    continue
                d_match = correlation_distance(f_t, feats[a_key])
                for b_key in others:           # different-identity distractor
                    if correlation_distance(f_t, feats[b_key]) > d_match:
                        correct += 1
                    trials += 1
    return correct / max(trials, 1), trials


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
    for p in np.arange(0.0, args.calib_max, args.calib_step):
        set_seed(args.seed)
        acc, _ = run_condition(model, device, upright_tf, identities, args.num_fixations, p)
        print(f"  noise {p:.2f} -> {acc*100:.2f}%")
        if acc <= args.calib_target:
            hit_p = p
            break
        prev_p = p
    if hit_p is None:
        print(f"[!] Coarse sweep never reached target; using noise p={args.calib_max:.2f}\n")
        return args.calib_max
    if hit_p == 0.0:
        print(f"[!] Using noise p={hit_p:.2f}\n")
        return hit_p

    print(f"--- Refining between {prev_p:.2f} and {hit_p:.2f} ---")
    best_p = hit_p
    for p in np.arange(prev_p + args.calib_fine_step, hit_p, args.calib_fine_step):
        set_seed(args.seed)
        acc, _ = run_condition(model, device, upright_tf, identities, args.num_fixations, p)
        print(f"  noise {p:.2f} -> {acc*100:.2f}%")
        if acc <= args.calib_target:
            best_p = p
            break
    print(f"[!] Using noise p={best_p:.2f}\n")
    return best_p


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="faces | houses_zubud | objects")
    ap.add_argument("--run-dir", default=None,
                    help="training run dir; fills in checkpoint/label-map/variant/backbone")
    ap.add_argument("--variant", choices=["lp", "cnn", "plain"], default=None)
    ap.add_argument("--backbone", default=None, help="must match the trained checkpoint")
    ap.add_argument("--packed-root", default="fixation_data")
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
    ap.add_argument("--num-fixations", type=int, default=16)
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

    model = ModelCoords(size=180, num_classes=num_classes, pretrained=False,
                  T=args.temperature).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    by_class = collect_identities(args.packed_root, args.category, args.splits,
                                  args.num_fixations)
    identities = sample_identities(by_class, args.num_identities,
                                   args.images_per_identity, args.seed)
    n_photos = sum(len(v) for v in identities.values())
    print(f"--- {args.category}: {len(identities)} identities, {n_photos} photos "
          f"from splits {args.splits} ({args.num_fixations} fixations each) ---")

    upright_tf = OnTheFlyTransform("valid", args.variant, device).to(device)
    inverted_tf = OnTheFlyTransform("test", args.variant, device).to(device)

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
    for cond, tf in [("Upright", upright_tf), ("Inverted", inverted_tf)]:
        set_seed(args.seed)
        acc, n_trials = run_condition(model, device, tf, identities, args.num_fixations, ideal_noise)
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