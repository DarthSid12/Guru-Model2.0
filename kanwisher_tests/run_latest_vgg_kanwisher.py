"""Run the existing Dobs/Kanwisher seed driver on the three final VGG2k models.

Example: python kanwisher_tests/run_latest_vgg_kanwisher.py --model noaa --dataset rfw --gpu 0
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kanwisher_tests import run_sim_seeds as driver
RUN_PREFIX = "faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_"
MODELS = {
    "noaa": ("vgg16_bn", "r21vgg2k_vgg16_bn_s42", "final_model_20260926_230735.pth"),
    "bin5": ("vgg16_bn_aa5", "r21vgg2k_vgg16_bn_aa5_s42", "final_model_20260927_145006.pth"),
    "aa1331": ("vgg16_bn_aa", "r21vgg2k_vgg16_bn_aa_s42", "final_model_20260925_110304.pth"),
}
DATASETS = {"rfw": "faces_rfwWM64", "cfd": "faces_cfdWMk", "setA": "faces_setA"}
OUT = ROOT / "runs/kanw_latest_vgg_20260927"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seeds", default="101-120")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / f"{args.model}_{args.dataset}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    backbone, tag, filename = MODELS[args.model]
    run_dir = ROOT / "runs" / (RUN_PREFIX + backbone + "_" + tag)
    checkpoint = run_dir / filename
    category = DATASETS[args.dataset]
    assert checkpoint.is_file() and (run_dir / "config.json").is_file()
    assert (ROOT / "fixation_data" / category / "valid" / "meta.json").is_file()
    assert json.loads((run_dir / "config.json").read_text())["backbone"] == backbone

    out_dir = OUT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{tag}.json"
    manifest = {
        "model": tag, "backbone": backbone, "run_dir": str(run_dir),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "category": category, "seeds": args.seeds,
        "calibration": "upright 87.5% target; seed 42; coarse 0.05, fine 0.01",
        "num_identities": 40, "images_per_identity": 5, "num_fixations": 16,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Existing manifest differs: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    requested = driver.expand_seeds(args.seeds)
    result_path = out_dir / f"results_{tag}.csv"
    completed = set()
    if result_path.exists():
        by_seed = {}
        with result_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["sim"] != "kanwisher" or row["category"] != category:
                    raise RuntimeError(f"Unexpected result row in {result_path}")
                seed = int(row["seed"])
                by_seed.setdefault(seed, []).append(row["test"])
        if any(sorted(conditions) != ["Inverted", "Upright"] for conditions in by_seed.values()):
            raise RuntimeError(f"Incomplete or duplicate seed rows in {result_path}")
        completed = set(by_seed)
    missing = [seed for seed in requested if seed not in completed]
    if not missing:
        print(f"Already complete: {tag} {category}, {len(completed)} seeds", flush=True)
        return

    driver.MODELS[tag] = str(run_dir)
    driver.CHECKPOINT[tag] = str(checkpoint)
    experiment = f"latest_vgg_{args.dataset}"
    driver.EXPERIMENTS[experiment] = {
        "out_dir": str(out_dir), "sims": ["kanwisher"],
        "categories": {"kanwisher": [category]},
        "calib_categories": {"kanwisher": [category]},
        "yin_shuffle": set(), "kanw_units": {},
        "kanw_calib_label": None, "label_col": "n_identities",
    }
    sys.argv = [sys.argv[0], "--model", tag, "--experiment", experiment,
                "--gpu", args.gpu, "--seeds", ",".join(map(str, missing)),
                "--threads", "1"]
    os.chdir(ROOT)
    driver.main()


if __name__ == "__main__":
    main()
