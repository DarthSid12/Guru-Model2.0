"""Fit one study/test noise rate to the category's human UU value.

The final epoch-124 VGG checkpoints, packed 40/24 object and house splits,
50 seeds, and simulator are the same as the earlier Yin matrix. One p is used
for both phases in all four orientation conditions. Prior matching UU samples
are reused during fitting, and winning calibration samples supply evaluation UU.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

import torch

import run_yin_orientation as base
import run_single_noise as single

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "runs/yin_single_noise_20260925/results"
CHECKPOINTS = ROOT / "runs/yin_orientation_20260925/checkpoints/final"
MODELS = ("r21dev_vgg16_bn_aa_s42", "r21vgg2k_vgg16_bn_aa_s42")
TARGETS = {"objects": 0.8479, "houses_yin64": 0.9071}
CONDITIONS = base.CONDITIONS


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_hashes():
    names = ("run_yin_category_uu.py", "run_single_noise.py", "run_yin_orientation.py",
             "simulate_yin1969_bothnoise.py", "model.py", "datasets.py",
             "salience_trans.py", "trans.py", "cylconv.py", "utils.py")
    return {name: base.sha256(Path(__file__).parent / name) for name in names}


def prior_samples(model, category, seeds, checkpoint):
    """Read identical-checkpoint UU observations at the two old face-fitted p's."""
    result = {}
    origins = {}
    new_weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for face in ("faces_rfwWM64", "faces_cfdWM64"):
        folder = PRIOR / model / face
        manifest = json.loads((folder / "manifest.json").read_text())
        prior_weights = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)
        if new_weights.keys() != prior_weights.keys() or any(
                not torch.equal(new_weights[k], prior_weights[k]) for k in new_weights):
            raise ValueError(f"Prior weights differ: {model}/{face}")
        if manifest["epoch"] != 124 or manifest["seeds"] != "101-150":
            raise ValueError(f"Prior protocol differs: {folder}")
        if manifest["source_sha256"]["run_single_noise.py"] != base.sha256(Path(__file__).parent / "run_single_noise.py"):
            raise ValueError("Prior category split code differs")
        for name in ("simulate_yin1969_bothnoise.py", "model.py", "run_yin_orientation.py"):
            if manifest["source_sha256"][name] != base.sha256(Path(__file__).parent / name):
                raise ValueError(f"Prior source differs: {name}")
        fit = json.loads((folder / "calibration.json").read_text())
        p = float(fit["p"])
        path = folder / category / "results.csv"
        if not (folder / category / "DONE.json").exists():
            raise ValueError(f"Prior category incomplete: {path}")
        rows = read_rows(path)
        uu = {int(r["seed"]): r for r in rows if r["condition"] == "UU"}
        if len(uu) != 50 or len(rows) != 200:
            raise ValueError(f"Prior category rows incomplete: {path}")
        key = f"{p:.6f}"
        values = result.setdefault(key, {})
        for seed in seeds:
            row = uu[seed]
            if float(row["p_noise"]) != p or int(row["epoch"]) != 124:
                raise ValueError(f"Prior noise/epoch mismatch: {path}")
            value = float(row["accuracy_pct"]) / 100
            if str(seed) in values and values[str(seed)] != value:
                raise ValueError(f"Prior UU observations disagree at p={p}, seed={seed}")
            values[str(seed)] = value
        origins[key] = str(path)
    return result, origins


def evaluate(model, category, seeds, fit, exp, ck_hash, output, cache):
    p = float(fit["p"])
    path = output / "results.csv"
    rows = read_rows(path) if path.exists() else []
    by_key = {(int(row["seed"]), row["condition"]): row for row in rows}
    expected = {(seed, study + test) for seed in seeds for study, test in CONDITIONS}
    if len(by_key) != len(rows) or not set(by_key) <= expected:
        raise ValueError(f"Duplicate or unexpected rows: {path}")
    calibrated_uu = cache["U"][f"{p:.6f}"]
    for seed in seeds:
        for study, test in CONDITIONS:
            condition = study + test
            if (seed, condition) in by_key:
                continue
            if condition == "UU":
                accuracy = float(calibrated_uu[str(seed)]) * 100
                origin = "calibration"
            else:
                accuracy = 100 * exp.measure(seed, study, test, p, p)
                origin = "evaluation"
            row = dict(model=model, epoch=124, calibration_dataset=category,
                       category=category, seed=seed, condition=condition,
                       p_study=p, p_test=p, accuracy_pct=accuracy,
                       origin=origin, checkpoint_sha256=ck_hash)
            rows.append(row)
            by_key[(seed, condition)] = row
            base.write_rows(path, rows)
        print(f"[{model} {category}] seed={seed} rows={len(rows)}/{len(expected)}", flush=True)
    summary = dict(model=model, epoch=124, calibration_dataset=category,
                   category=category, p_noise=p, target_uu=fit["target"],
                   achieved_uu=fit["achieved"], residual_pp=fit["residual_pp"],
                   fit_status=fit["status"], n_seeds=len(seeds), conditions={})
    for study, test in CONDITIONS:
        condition = study + test
        values = [float(by_key[(seed, condition)]["accuracy_pct"]) for seed in seeds]
        summary["conditions"][condition] = dict(mean_pct=statistics.mean(values),
            sd_pct=statistics.stdev(values) if len(values) > 1 else 0.0, n=len(values))
    if abs(summary["conditions"]["UU"]["mean_pct"] / 100 - fit["achieved"]) > 1e-9:
        raise ValueError("Calibration and UU evaluation disagree")
    base.atomic_json(output / "summary.json", summary)
    base.atomic_json(output / "DONE.json", dict(rows=len(rows), checkpoint_sha256=ck_hash))
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--category", choices=tuple(TARGETS), required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seeds", default="101-150")
    ap.add_argument("--coarse-step", type=float, default=0.05)
    ap.add_argument("--fine-step", type=float, default=0.01)
    ap.add_argument("--fine-radius", type=float, default=0.04)
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if not seeds or not set(seeds) <= set(range(101, 151)):
        ap.error("Seeds must be a nonempty range within 101-150")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    ck_dir = CHECKPOINTS / args.model
    checkpoint = ck_dir / "weights.pth"
    ck_hash = base.sha256(checkpoint)
    output = args.out_root / "results" / args.model / args.category
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=args.model, category=args.category, target_uu=TARGETS[args.category],
        epoch=124, seeds=seeds, coarse_step=args.coarse_step, fine_step=args.fine_step,
        fine_radius=args.fine_radius, checkpoint=str(checkpoint), checkpoint_sha256=ck_hash,
        source_sha256=source_hashes(), noise_phase="both", layer="h",
        study_fixations=10, test_fixations=32, num_test=24, sigma=2.0)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Settings changed: {manifest_path}")
    base.atomic_json(manifest_path, manifest)
    if (output / "DONE.json").exists():
        print(f"Already complete: {output}", flush=True)
        return
    exp = single.Experiment(ck_dir, checkpoint, args.category, args.device)
    cache_path = output / "calibration_samples.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    prior, origins = prior_samples(args.model, args.category, seeds, checkpoint)
    current = cache.setdefault("U", {})
    for p, values in prior.items():
        existing = current.setdefault(p, {})
        for seed, value in values.items():
            if seed in existing and existing[seed] != value:
                raise ValueError(f"Cached sample changed at p={p}, seed={seed}")
            existing[seed] = value
    base.atomic_json(cache_path, cache)
    base.atomic_json(output / "reused_samples.json", origins)
    fit_path = output / "calibration.json"
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
    else:
        fit = base.calibrate(exp, seeds, "U", TARGETS[args.category], args, cache, cache_path)
        base.atomic_json(fit_path, fit)
    evaluate(args.model, args.category, seeds, fit, exp, ck_hash, output, cache)


if __name__ == "__main__":
    main()
