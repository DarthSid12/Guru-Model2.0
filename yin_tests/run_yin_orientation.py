"""Yin with independent UU/II calibration and noise following each orientation.

Reuses the existing bothnoise encoder, task and familiarity calculation. Fits
one pair per checkpoint/category over all requested seeds. No p2 > p1 constraint.
All calibration draws use the same per-seed item splits as evaluation.
"""

# Support direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import torch
from yin_tests import simulate_yin1969_bothnoise as yin
CONDITIONS = (("U", "U"), ("I", "I"), ("U", "I"), ("I", "U"))
from training.paths import repository_root, source_path
ROOT = repository_root(__file__)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(f)


def _hash_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def split_items(items, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 64:
        raise ValueError(f"Expected at least 64 items, got {len(shuffled)}")
    return shuffled[:40], shuffled[40:64]


def noise_pair(study, test, p1, p2):
    rates = {"U": p1, "I": p2}
    return rates[study], rates[test]


def choose_fit(curve, target):
    p = min(curve, key=lambda x: (abs(curve[x] - target), x))
    return {"p": p, "target": target, "achieved": curve[p],
            "residual_pp": 100 * (curve[p] - target),
            "status": ("within_sampled_range" if min(curve.values()) <= target <= max(curve.values())
                       else "target_outside_sampled_range")}


class Experiment:
    def __init__(self, run_dir, checkpoint, category, device):
        cfg = json.loads((run_dir / "config.json").read_text())
        labels = json.loads((run_dir / "label_map.json").read_text())
        self.device = torch.device(device)
        yin.set_seed(101)
        self.model = yin.Model(size=180, num_classes=len(labels), pretrained=False,
                               T=cfg.get("temperature", 2.0), backbone=cfg["backbone"])
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.transforms = {o: yin.OnTheFlyTransform(phase, cfg.get("variant", "lp"), self.device).to(self.device)
                           for o, phase in (("U", "valid"), ("I", "test"))}
        self.sp = yin._PackedSplit(str(ROOT / "fixation_data"), category, "valid")
        if self.sp.num_coords < 32:
            raise ValueError("The packed data must contain at least 32 fixations")
        self.items = yin.build_items(self.sp)
        split_items(self.items, 101)
        # Cache only immutable raw crops; encoding and all random draws still
        # run through the existing simulator on every condition.
        yin.load_item_fixations = functools.lru_cache(maxsize=256)(yin.load_item_fixations)

    def measure(self, seed, study, test, p_study, p_test):
        args = SimpleNamespace(seed=seed, noise_phase="both", layer="h", sigma=2.0,
                               study_fixations=10, test_fixations=32, num_test=24)
        studied, new = split_items(self.items, seed)
        return yin.run_condition(self.model, self.device, args, self.sp, studied, new,
                                 self.transforms[study], self.transforms[test], p_study,
                                 p_test_in=p_test)


def calibrate(experiment, seeds, orientation, target, args, cache, cache_path):
    samples = cache.setdefault(orientation, {})

    def probe(p):
        key = f"{p:.6f}"
        values = samples.setdefault(key, {})
        for seed in seeds:
            if str(seed) not in values:
                values[str(seed)] = experiment.measure(seed, orientation, orientation, p, p)
                atomic_json(cache_path, cache)
        mean = statistics.mean(values[str(s)] for s in seeds)
        print(f"[{orientation}{orientation}] p={p:.3f} mean={mean*100:.3f}% n={len(seeds)}", flush=True)
        return mean

    coarse = [round(i * args.coarse_step, 6) for i in range(int(0.5 / args.coarse_step) + 1)]
    coarse = sorted(set(coarse + [0.0, 0.5]))
    curve = {p: probe(p) for p in coarse}
    winner = choose_fit(curve, target)["p"]
    n = round(args.fine_radius / args.fine_step)
    fine = {round(winner + i * args.fine_step, 6) for i in range(-n, n + 1)}
    for p in sorted(fine):
        if 0 <= p <= 0.5 and p not in curve:
            curve[p] = probe(p)
    fit = choose_fit(curve, target)
    fit["curve"] = {str(p): a for p, a in sorted(curve.items())}
    print(f"[fit {orientation}] {json.dumps({k:v for k,v in fit.items() if k != 'curve'})}", flush=True)
    return fit


def write_rows(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(buffer.getvalue())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--category", required=True, choices=["faces_rfwWM64", "faces_cfdWM64"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--epoch", required=True, type=int)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="101-150")
    ap.add_argument("--target-uu", type=float, default=0.9629)
    ap.add_argument("--target-ii", type=float, default=0.8188)
    ap.add_argument("--coarse-step", type=float, default=0.05)
    ap.add_argument("--fine-step", type=float, default=0.01)
    ap.add_argument("--fine-radius", type=float, default=0.04)
    ap.add_argument("--noise-up", type=float)
    ap.add_argument("--noise-inv", type=float)
    args = ap.parse_args()
    if not (0 < args.coarse_step <= 0.5 and 0 < args.fine_step <= 0.5 and 0 <= args.fine_radius <= 0.5):
        ap.error("Invalid calibration grid")
    if (args.noise_up is None) != (args.noise_inv is None):
        ap.error("Both fixed noise levels must be supplied together")
    if args.noise_up is not None and not all(0 <= p <= 0.5 for p in (args.noise_up, args.noise_inv)):
        ap.error("Noise probabilities must lie in [0, 0.5]")
    if "-" in args.seeds:
        lo, hi = map(int, args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = list(map(int, args.seeds.split(",")))
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error("Seeds must be nonempty and unique")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signature = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "device"}
    signature["seeds"] = seeds
    signature["checkpoint_sha256"] = sha256(args.checkpoint)
    signature["code_sha256"] = {f: sha256(source_path(ROOT, f)) for f in (
        "run_yin_orientation.py", "simulate_yin1969_bothnoise.py", "model.py", "datasets.py", "salience_trans.py", "trans.py", "cylconv.py")}
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError(f"Settings changed: use a new output directory, {args.out_dir}")
    atomic_json(manifest_path, signature)
    if (args.out_dir / "DONE.json").exists():
        print("Already complete", flush=True)
        return
    print(f"Starting {args.model} {args.stage} epoch {args.epoch} {args.category}; seeds={seeds}", flush=True)
    experiment = Experiment(args.run_dir, args.checkpoint, args.category, args.device)
    cache_path = args.out_dir / "calibration_samples.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    fits = {}
    for orient, target, fixed in (("U", args.target_uu, args.noise_up), ("I", args.target_ii, args.noise_inv)):
        fits[orient] = (calibrate(experiment, seeds, orient, target, args, cache, cache_path) if fixed is None
                       else {"p": fixed, "target": target, "status": "fixed_smoke_probe"})
        atomic_json(args.out_dir / "calibration.json", fits)
    p1, p2 = fits["U"]["p"], fits["I"]["p"]
    result_path = args.out_dir / "results.csv"
    rows = list(csv.DictReader(result_path.open())) if result_path.exists() else []
    done = {(int(r["seed"]), r["study"], r["test"]) for r in rows}
    expected = {(s, u, v) for s in seeds for u, v in CONDITIONS}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Duplicate or unexpected existing result rows")
    for seed in seeds:
        for study, test in CONDITIONS:
            if (seed, study, test) in done:
                continue
            ps, pt = noise_pair(study, test, p1, p2)
            acc = experiment.measure(seed, study, test, ps, pt)
            rows.append(dict(model=args.model, stage=args.stage, epoch=args.epoch, category=args.category,
                             seed=seed, study=study, test=test, p_study=ps, p_test=pt,
                             accuracy_pct=100 * acc, checkpoint_sha256=signature["checkpoint_sha256"]))
            write_rows(result_path, rows)
        print(f"[eval] seed={seed} saved rows={len(rows)}/{len(expected)}", flush=True)
    summary = {"model": args.model, "stage": args.stage, "epoch": args.epoch,
               "category": args.category, "calibration": fits, "conditions": {}}
    for u, v in CONDITIONS:
        values = [float(r["accuracy_pct"]) for r in rows if r["study"] == u and r["test"] == v]
        summary["conditions"][u+v] = {"mean_pct": statistics.mean(values), "n": len(values),
                                     "sd_pct": statistics.stdev(values) if len(values) > 1 else 0.0}
        # Calibration and evaluation must agree on the actual matched samples.
        if u == v and "achieved" in fits[u]:
            if abs(statistics.mean(values) / 100 - fits[u]["achieved"]) > 1e-9:
                raise RuntimeError("Calibration/evaluation disagreement on matched seeds")
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "DONE.json", {"rows": len(rows), "checkpoint_sha256": signature["checkpoint_sha256"]})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
