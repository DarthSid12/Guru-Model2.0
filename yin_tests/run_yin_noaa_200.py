"""Extend the frozen no-AA VGG16 Yin trials to 200 seeds.

The rates fitted on seeds 101-150 remain fixed. Seven distinct noise cells
cover the three published protocols; each 50-seed cell is reused verbatim.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import torch
from yin_tests import run_yin_orientation as base
import run_single_noise as single
ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "runs/yin_latest_vgg_20260927"
OUT = ROOT / "runs/yin_noaa_200_20260928"
MODEL = "r21vgg2k_vgg16_bn_s42"
CHECKPOINT_DIR = ROOT / "runs/yin_noaa_two_cross_20260927/checkpoint"
CHECKPOINT = CHECKPOINT_DIR / "weights.pth"
FACES = ("faces_rfwWM64", "faces_cfdWM64")
CELLS = {
    "UU": ("U", "U", "p1", "p1"),
    "II_single": ("I", "I", "p1", "p1"),
    "UI_shared": ("U", "I", "p1", "p1"),
    "IU_single": ("I", "U", "p1", "p1"),
    "II_cross": ("I", "I", "p2", "p2"),
    "UI_cross": ("U", "I", "p1", "p2"),
    "IU_cross": ("I", "U", "p2", "p1"),
}
SINGLE_KEYS = {"UU": "UU", "II_single": "II", "UI_shared": "UI", "IU_single": "IU"}
CROSS_KEYS = {"II_cross": "II", "UI_cross": "UI_cross", "IU_cross": "IU"}


def read_index(path, key):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(int(row["seed"]), row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate rows in {path}")
    return result


def inputs(face, category):
    one = OLD / "single_noise" / MODEL / face
    cross = OLD / "transfer/results" / MODEL / face / category
    one_manifest = json.loads((one / "manifest.json").read_text())
    cross_manifest = json.loads((OLD / "transfer/results" / MODEL / face / "manifest.json").read_text())
    one_fit = json.loads((one / "calibration.json").read_text())
    two_fit = json.loads((ROOT / "runs/yin_noaa_two_cross_20260927/results" / face / "calibration.json").read_text())
    p1, p2 = float(one_fit["p"]), float(two_fit["I"]["p"])
    if p1 != float(two_fit["U"]["p"]):
        raise ValueError("Calibration rates disagree")
    ck_hash = base.sha256(CHECKPOINT)
    if one_manifest["checkpoint_sha256"] != ck_hash or cross_manifest["checkpoint_sha256"] != ck_hash:
        raise ValueError("Prior checkpoint hashes disagree")
    one_path = one / category / "results.csv"
    cross_path = cross / "results.csv"
    if not (one / category / "DONE.json").exists() or not (cross / "DONE.json").exists():
        raise ValueError("Prior results incomplete")
    old_one = read_index(one_path, "condition")
    old_cross = read_index(cross_path, "cell")
    if len(old_one) != 200 or len(old_cross) != 250:
        raise ValueError("Expected 50 complete seeds in prior data")
    return p1, p2, ck_hash, old_one, old_cross, one_path, cross_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face", choices=FACES, required=True)
    ap.add_argument("--category", choices=("face", "objects"), required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    category = args.face if args.category == "face" else args.category
    p1, p2, ck_hash, old_one, old_cross, one_path, cross_path = inputs(args.face, category)
    out = OUT / "results" / args.face / category
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=MODEL, epoch=124, face=args.face, category=category,
                    seeds=list(range(101, 301)), calibration_seeds=list(range(101, 151)),
                    p1=p1, p2=p2, checkpoint_sha256=ck_hash,
                    source_sha256={name: base.sha256(Path(__file__).parent / name)
                                   for name in ("run_yin_noaa_200.py", "run_yin_orientation.py",
                                                "run_single_noise.py", "simulate_yin1969_bothnoise.py",
                                                "model.py")})
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Manifest differs: {manifest_path}")
    base.atomic_json(manifest_path, manifest)
    path = out / "results.csv"
    rows = read_index(path, "cell") if path.exists() else {}
    expected = {(seed, cell) for seed in range(101, 301) for cell in CELLS}
    if not set(rows) <= expected:
        raise ValueError("Unexpected existing rows")

    def make_row(seed, cell, accuracy, source, source_path):
        study, test, ps, pt = CELLS[cell]
        return dict(model=MODEL, calibration_face=args.face, category=category,
                    seed=seed, cell=cell, study=study, test=test,
                    p_study=p1 if ps == "p1" else p2, p_test=p1 if pt == "p1" else p2,
                    accuracy_pct=float(accuracy), source_kind=source,
                    source_path=str(source_path), checkpoint_sha256=ck_hash)

    for seed in range(101, 151):
        for cell in CELLS:
            if cell in SINGLE_KEYS:
                prior = old_one[(seed, SINGLE_KEYS[cell])]
                source, source_path = "prior_single_noise", one_path
            else:
                prior = old_cross[(seed, CROSS_KEYS[cell])]
                source, source_path = "prior_two_cross", cross_path
            row = make_row(seed, cell, prior["accuracy_pct"], source, source_path)
            if (seed, cell) in rows and rows[(seed, cell)] != {k: str(v) for k, v in row.items()}:
                raise ValueError(f"Reused row changed: {seed}, {cell}")
            rows[(seed, cell)] = row
    base.write_rows(path, [rows[key] for key in sorted(rows)])
    missing = expected - set(rows)
    if missing:
        experiment = single.Experiment(CHECKPOINT_DIR, CHECKPOINT, category, args.device)
        for seed in range(151, 301):
            for cell, (study, test, ps, pt) in CELLS.items():
                if (seed, cell) in rows:
                    continue
                accuracy = experiment.measure(seed, study, test,
                    p1 if ps == "p1" else p2, p1 if pt == "p1" else p2)
                rows[(seed, cell)] = make_row(seed, cell, 100 * accuracy,
                                               "new_evaluation", path)
                base.write_rows(path, [rows[key] for key in sorted(rows)])
            print(f"[{args.face} {category}] seed={seed} rows={len(rows)}/1400", flush=True)
    if set(rows) != expected:
        raise ValueError("Incomplete result matrix")
    summary = dict(model=MODEL, face=args.face, category=category,
                   calibration_seeds=50, evaluation_seeds=200, p1=p1, p2=p2,
                   checkpoint_sha256=ck_hash, cells={})
    for cell in CELLS:
        values = [float(rows[(seed, cell)]["accuracy_pct"]) for seed in range(101, 301)]
        summary["cells"][cell] = dict(mean_pct=statistics.mean(values),
                                       sd_pct=statistics.stdev(values), n=len(values))
    base.atomic_json(out / "summary.json", summary)
    base.atomic_json(out / "DONE.json", dict(rows=len(rows), checkpoint_sha256=ck_hash))
    print(f"[{args.face} {category}] complete", flush=True)


if __name__ == "__main__":
    main()
