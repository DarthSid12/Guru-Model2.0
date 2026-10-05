"""Reuse two-cross UU calibration and seed results in a single-noise run."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/yin_latest_vgg_20260927"
NOAA_CROSS = ROOT / "runs/yin_noaa_two_cross_20260927/results"
MODELS = {"noaa": "r21vgg2k_vgg16_bn_s42",
          "aa5": "r21vgg2k_vgg16_bn_aa5_s42"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--face", required=True,
                    choices=("faces_rfwWM64", "faces_cfdWM64"))
    args = ap.parse_args()
    model = MODELS[args.model]
    cross = (NOAA_CROSS / args.face if args.model == "noaa" else
             OUT / "two_cross" / model / args.face)
    single = OUT / "single_noise" / model / args.face
    if not (cross / "DONE.json").exists():
        raise RuntimeError(f"Two-cross run is incomplete: {cross}")
    cross_manifest = json.loads((cross / "manifest.json").read_text())
    cross_done = json.loads((cross / "DONE.json").read_text())
    if cross_manifest["checkpoint_sha256"] != cross_done["checkpoint_sha256"]:
        raise RuntimeError("Two-cross checkpoint hash mismatch")
    fit = json.loads((cross / "calibration.json").read_text())["U"]
    samples = json.loads((cross / "calibration_samples.json").read_text())["U"]
    with (cross / "results.csv").open(newline="") as handle:
        source = [row for row in csv.DictReader(handle)
                  if row["study"] == row["test"] == "U"]
    if len(source) != 50 or {int(row["seed"]) for row in source} != set(range(101, 151)):
        raise RuntimeError("Expected 50 completed UU seeds")
    if any(float(row["p_study"]) != fit["p"] or
           float(row["p_test"]) != fit["p"] for row in source):
        raise RuntimeError("UU noise differs from calibration")
    single.mkdir(parents=True, exist_ok=True)
    for name, value in (("calibration.json", fit),
                        ("calibration_samples.json", {"U": samples})):
        path = single / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f"Existing data differs: {path}")
        path.write_text(json.dumps(value, indent=2) + "\n")
    result = single / args.face / "results.csv"
    result.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(model=model, epoch=124, calibration_face=args.face,
                 category=args.face, seed=row["seed"], condition="UU",
                 p_noise=fit["p"], accuracy_pct=row["accuracy_pct"],
                 checkpoint_sha256=cross_done["checkpoint_sha256"])
            for row in source]
    if result.exists():
        with result.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
        for row in existing:
            if row["condition"] == "UU" and row != {k: str(v) for k, v in
                                                    rows[int(row["seed"]) - 101].items()}:
                raise RuntimeError(f"Existing UU result differs: {result}")
        rows = existing or rows
    with result.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reused UU calibration and 50 UU rows: {model} {args.face}")


if __name__ == "__main__":
    main()
