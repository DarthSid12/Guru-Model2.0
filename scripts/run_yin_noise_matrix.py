"""Fill the missing final-checkpoint Yin two-noise transfer cells.

The one-noise run supplies UU and UI at p1/p1. The completed two-noise face
run supplies UU, II, UI at p1/p2, and IU at p2/p1. Only II, UI at p1/p2,
and IU on objects/houses require new model evaluations. All five distinct
cells are materialized per dataset so both requested two-noise rules can be
assembled without rerunning identical Monte Carlo trials.
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
SINGLE = ROOT / "runs/yin_single_noise_20260925/results"
TWO = ROOT / "runs/yin_orientation_20260925/results/final"
CHECKPOINTS = ROOT / "runs/yin_orientation_20260925/checkpoints/final"
FACES = ("faces_rfwWM64", "faces_cfdWM64")
MODELS = ("r21dev_vgg16_bn_aa_s42", "r21vgg2k_vgg16_bn_aa_s42")
CATEGORIES = ("objects", "houses_yin64")
CELLS = ("UU", "II", "UI_cross", "IU", "UI_shared")
FIELDS = ("model", "calibration_face", "category", "seed", "cell", "study",
          "test", "p_study", "p_test", "accuracy_pct", "source_kind",
          "source_path", "checkpoint_sha256")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(path, condition_key):
    rows = read_csv(path)
    result = {(int(row["seed"]), condition_key(row)): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate seed/condition in {path}")
    return result


def cell_rates(cell, p1, p2):
    return {
        "UU": ("U", "U", p1, p1),
        "II": ("I", "I", p2, p2),
        "UI_cross": ("U", "I", p1, p2),
        "IU": ("I", "U", p2, p1),
        "UI_shared": ("U", "I", p1, p1),
    }[cell]


def output_row(model, face, category, seed, cell, p1, p2, accuracy, kind, path, ck_hash):
    study, test, ps, pt = cell_rates(cell, p1, p2)
    return dict(model=model, calibration_face=face, category=category,
                seed=seed, cell=cell, study=study, test=test, p_study=ps,
                p_test=pt, accuracy_pct=float(accuracy), source_kind=kind,
                source_path=str(path), checkpoint_sha256=ck_hash)


def checked_inputs(model, face, seeds):
    one = SINGLE / model / face
    two = TWO / model / face
    checkpoint_dir = CHECKPOINTS / model
    single_manifest = json.loads((one / "manifest.json").read_text())
    two_manifest = json.loads((two / "manifest.json").read_text())
    single_done = json.loads((one / "DONE.json").read_text())
    two_done = json.loads((two / "DONE.json").read_text())
    one_fit = json.loads((one / "calibration.json").read_text())
    two_fit = json.loads((two / "calibration.json").read_text())
    p1, p2 = float(two_fit["U"]["p"]), float(two_fit["I"]["p"])
    if one_fit["p"] != p1:
        raise ValueError(f"Single-noise p != two-noise p1 for {model}/{face}")
    if single_manifest["epoch"] != two_manifest["epoch"] or two_manifest["epoch"] != 124:
        raise ValueError(f"Checkpoint epochs disagree for {model}/{face}")
    if single_done["checkpoint_sha256"] != single_manifest["checkpoint_sha256"]:
        raise ValueError(f"Invalid one-noise completion record: {one}")
    if two_done["checkpoint_sha256"] != two_manifest["checkpoint_sha256"]:
        raise ValueError(f"Invalid two-noise completion record: {two}")
    ck = checkpoint_dir / "weights.pth"
    if base.sha256(ck) != two_manifest["checkpoint_sha256"]:
        raise ValueError(f"Final snapshot changed: {ck}")
    a = torch.load(ck, map_location="cpu", weights_only=True)
    b = torch.load(single_manifest["checkpoint"], map_location="cpu", weights_only=True)
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise ValueError(f"Two experiment checkpoints contain different tensors: {model}")
    del a, b
    face_one_path = one / face / "results.csv"
    face_two_path = two / "results.csv"
    face_one = index_rows(face_one_path, lambda r: r["condition"])
    face_two = index_rows(face_two_path, lambda r: r["study"] + r["test"])
    if set(face_one) != set(face_two) or len(face_one) != 200:
        raise ValueError("Prior face runs do not have the same 50 seeds and four cells")
    for seed in range(101, 151):
        if face_one[(seed, "UU")]["accuracy_pct"] != face_two[(seed, "UU")]["accuracy_pct"]:
            raise ValueError(f"Seed {seed} face UU differs between prior protocols")
    if not set(seeds) <= set(range(101, 151)):
        raise ValueError("New seeds require new calibration; only 101-150 may be reused")
    return dict(one=one, two=two, checkpoint_dir=checkpoint_dir, checkpoint=ck,
                checkpoint_sha256=two_manifest["checkpoint_sha256"], p1=p1, p2=p2,
                face_one=face_one, face_two=face_two,
                face_one_path=face_one_path, face_two_path=face_two_path)


def preload(model, face, category, seeds, inputs):
    p1, p2 = inputs["p1"], inputs["p2"]
    ck_hash = inputs["checkpoint_sha256"]
    if category == face:
        from_one = inputs["face_one"]
        from_two = inputs["face_two"]
        one_path = inputs["face_one_path"]
        two_path = inputs["face_two_path"]
    else:
        one_path = inputs["one"] / category / "results.csv"
        if not (inputs["one"] / category / "DONE.json").exists():
            raise ValueError(f"Single-noise category incomplete: {category}")
        from_one = index_rows(one_path, lambda r: r["condition"])
        from_two = {}
        two_path = None
    result = {}
    for seed in seeds:
        for cell, prior, kind, path, condition in (
            ("UU", from_two if category == face else from_one,
             "prior_two_noise" if category == face else "prior_single_noise",
             two_path if category == face else one_path, "UU"),
            ("UI_shared", from_one, "prior_single_noise", one_path, "UI"),
        ):
            old = prior[(seed, condition)]
            result[(seed, cell)] = output_row(model, face, category, seed, cell,
                p1, p2, old["accuracy_pct"], kind, path, ck_hash)
        if category == face:
            for cell, condition in (("II", "II"), ("UI_cross", "UI"), ("IU", "IU")):
                old = from_two[(seed, condition)]
                result[(seed, cell)] = output_row(model, face, category, seed,
                    cell, p1, p2, old["accuracy_pct"], "prior_two_noise",
                    two_path, ck_hash)
    return result


def run_category(model, face, category, seeds, inputs, output, device):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "results.csv"
    expected = {(seed, cell) for seed in seeds for cell in CELLS}
    prefilled = preload(model, face, category, seeds, inputs)
    old = index_rows(path, lambda r: r["cell"]) if path.exists() else {}
    if not set(old) <= expected:
        raise ValueError(f"Unexpected existing rows in {path}")
    for key, row in prefilled.items():
        if key in old and any(str(old[key][k]) != str(row[k]) for k in FIELDS):
            raise ValueError(f"Existing reused row disagrees at {key}: {path}")
        old[key] = row
    rows = old
    base.write_rows(path, [rows[k] for k in sorted(rows)])
    missing = expected - set(rows)
    if missing:
        exp = single.Experiment(inputs["checkpoint_dir"], inputs["checkpoint"], category, device)
        for seed in seeds:
            for cell in ("II", "UI_cross", "IU"):
                key = seed, cell
                if key not in missing:
                    continue
                study, test, ps, pt = cell_rates(cell, inputs["p1"], inputs["p2"])
                acc = exp.measure(seed, study, test, ps, pt)
                rows[key] = output_row(model, face, category, seed, cell,
                    inputs["p1"], inputs["p2"], 100 * acc,
                    "new_evaluation", path, inputs["checkpoint_sha256"])
                base.write_rows(path, [rows[k] for k in sorted(rows)])
            print(f"[{model} {face} {category}] seed={seed} {len(rows)}/{len(expected)}", flush=True)
    if set(rows) != expected:
        raise ValueError(f"Incomplete results: {path}")
    summary = dict(model=model, epoch=124, calibration_face=face,
                   category=category, p1=inputs["p1"], p2=inputs["p2"],
                   n_seeds=len(seeds), cells={})
    for cell in CELLS:
        values = [float(rows[(seed, cell)]["accuracy_pct"]) for seed in seeds]
        summary["cells"][cell] = dict(mean_pct=statistics.mean(values),
            sd_pct=statistics.stdev(values) if len(values) > 1 else 0.0, n=len(values))
    base.atomic_json(output / "summary.json", summary)
    base.atomic_json(output / "DONE.json", dict(rows=len(rows),
        checkpoint_sha256=inputs["checkpoint_sha256"], cells=list(CELLS)))
    print(f"[{model} {face} {category}] complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--face", choices=FACES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101-150")
    parser.add_argument("--categories", nargs="+", choices=("face",) + CATEGORIES,
                        default=["face", *CATEGORIES])
    args = parser.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if not seeds or not set(seeds) <= set(range(101, 151)):
        parser.error("Seeds must be a nonempty range within 101-150")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    inputs = checked_inputs(args.model, args.face, seeds)
    output = args.out_root / "results" / args.model / args.face
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=args.model, face=args.face, categories=args.categories,
        seeds=seeds, p1=inputs["p1"], p2=inputs["p2"],
        checkpoint_sha256=inputs["checkpoint_sha256"],
        runner_sha256=base.sha256(Path(__file__)),
        frozen_source_sha256={name: base.sha256(Path(__file__).parent / name)
                              for name in ("run_yin_orientation.py", "run_single_noise.py",
                                           "simulate_yin1969_bothnoise.py", "model.py")})
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Settings changed: {manifest_path}")
    base.atomic_json(manifest_path, manifest)
    for label in args.categories:
        category = args.face if label == "face" else label
        run_category(args.model, args.face, category, seeds, inputs,
                     output / category, args.device)


if __name__ == "__main__":
    main()
