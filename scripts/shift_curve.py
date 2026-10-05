"""theta-shift curve: how far does the 256-d code move when the log-polar image
is rolled along theta (rows), relative to how far apart two identities sit?

A rotation about the fixation point is a cyclic row shift in log-polar space, so
a shift-equivariant backbone should move the code very little for small k. The
baseline resnet18 does not: 1 row already costs 17% of the between-identity
distance, and 90 rows (=180 degrees) costs ~95%, i.e. an inverted probe is
almost as far from its own trace as a stranger is.

    python scripts/shift_curve.py --run-dir runs/..._r20_cyl_s42 [--category ...]
"""
import argparse, glob, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import _PackedSplit
from model import Model
from salience_trans import OnTheFlyTransform
from simulate_yin1969_bothnoise import build_items, load_item_fixations

SHIFTS = [0, 1, 2, 5, 10, 20, 45, 90]


def codes(model, x):
    """Deterministic binary bottleneck code (no Bernoulli sampling)."""
    model.stochastic = False
    with torch.no_grad():
        _, _, probs = model(x, return_rep=True)
    return (probs > 0.5).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--category", default="faces_cfdWM64")
    ap.add_argument("--packed-root", default="fixation_data")
    ap.add_argument("--num-items", type=int, default=64)
    ap.add_argument("--fixations", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    backbone = cfg.get("backbone", "resnet18")
    variant = cfg.get("variant", "lp")
    ckpt = args.checkpoint or sorted(glob.glob(os.path.join(args.run_dir, "final_model_*.pth")))[-1:]
    if not ckpt:
        ckpt = [os.path.join(args.run_dir, "best_model.pth")]
    ckpt = ckpt[0] if isinstance(ckpt, list) else ckpt
    num_classes = len(json.load(open(os.path.join(args.run_dir, "label_map.json"))))

    device = torch.device(args.device)
    model = Model(size=180, num_classes=num_classes, pretrained=False,
                  T=cfg.get("temperature", 1.0), backbone=backbone).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()

    tf = OnTheFlyTransform("valid", variant, device).to(device)
    sp = _PackedSplit(args.packed_root, args.category, "valid")
    items = build_items(sp)[: args.num_items]

    # one log-polar tensor per identity
    xs = []
    for img_idx, _ in items:
        crops = load_item_fixations(sp, img_idx, args.fixations, offset=0)
        if crops is None:
            continue
        with torch.no_grad():
            xs.append(tf(crops.to(device))[:1])
    x = torch.cat(xs, 0)                                  # [N, 3, H, W]
    n, _, H, _ = x.shape

    h0 = codes(model, x)
    D = h0.shape[1]
    # between-identity: mean pairwise hamming fraction at zero shift
    pair = (h0[:, None, :] != h0[None, :, :]).float().mean(-1)
    between = pair[~torch.eye(n, dtype=bool, device=pair.device)].mean().item()

    print(f"backbone {backbone}   ckpt {os.path.basename(ckpt)}")
    print(f"{args.category}: n={n} identities, {D}-d code, theta rows={H}")
    print(f"between-identity hamming = {between:.4f}\n")
    print(f"{'shift(rows)':>11} {'deg':>6} {'hamming':>9} {'/between':>9}")
    for k in SHIFTS:
        hk = codes(model, torch.roll(x, k, dims=-2))
        d = (h0 != hk).float().mean().item()
        print(f"{k:>11} {360.0*k/H:>6.0f} {d:>9.4f} {d/between:>9.3f}")


if __name__ == "__main__":
    main()
