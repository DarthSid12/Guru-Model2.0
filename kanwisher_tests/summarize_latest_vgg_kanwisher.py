"""Summarize the nine final-checkpoint Kanwisher face runs once complete."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kanw_latest_vgg_20260927"
MODELS = {
    "noaa": "r21vgg2k_vgg16_bn_s42",
    "bin5": "r21vgg2k_vgg16_bn_aa5_s42",
    "aa1331": "r21vgg2k_vgg16_bn_aa_s42",
}
DATASETS = {"rfw": "faces_rfwWM64", "cfd": "faces_cfdWMk", "setA": "faces_setA"}


def fmt(values):
    return f"{mean(values):.2f} ± {stdev(values)/math.sqrt(len(values)):.2f}"


def main():
    lines = ["# Kanwisher matching on three held out face stores", "",
             "Final epoch VGG2k checkpoints; 20 seeds (101–120); retrieval noise fitted",
             "separately for each model and store to the 87.5% upright target.", "",
             "| Model | Store | Noise p | Upright % ± SEM | Inverted % ± SEM | Cost pp ± SEM |",
             "|---|---|---:|---:|---:|---:|"]
    for model, tag in MODELS.items():
        for dataset, category in DATASETS.items():
            folder = OUT / dataset
            with (folder / f"results_{tag}.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_seed = {}
            for row in rows:
                if row["sim"] != "kanwisher" or row["category"] != category:
                    continue
                by_seed.setdefault(int(row["seed"]), {})[row["test"]] = float(row["accuracy_pct"])
            if set(by_seed) != set(range(101, 121)) or any(set(v) != {"Upright", "Inverted"} for v in by_seed.values()):
                raise RuntimeError(f"Incomplete result: {model}/{dataset}: {len(by_seed)} seeds")
            noise = json.loads((folder / f"noise_{tag}.json").read_text())["kanwisher"]
            upright = [by_seed[s]["Upright"] for s in sorted(by_seed)]
            inverted = [by_seed[s]["Inverted"] for s in sorted(by_seed)]
            costs = [u - i for u, i in zip(upright, inverted)]
            lines.append(f"| {model} | {dataset} | {noise:.2f} | {fmt(upright)} | {fmt(inverted)} | {fmt(costs)} |")
    lines += ["", "RFW uses 40 sampled identities from `faces_rfwWM64` (four photos each);",
              "CFD uses all 36 multi photo identities in `faces_cfdWMk`; Set A uses",
              "all 40 identities in `faces_setA` (five photos each). Each trial compares",
              "a target with a different photo of the same person and a different person.",
              "The inversion cost is paired within each seed. Checkpoint hashes and exact",
              "paths are recorded in the per cell manifest files.", ""]
    report = OUT / "summary.md"
    report.write_text("\n".join(lines))
    print(report)
    print("\n".join(lines[5:15]))


if __name__ == "__main__":
    main()
