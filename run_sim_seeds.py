"""
run_sim_seeds.py

Multi-seed driver for the Yin (1969) and Dobs/Kanwisher (2023) simulations.

Single-seed sims are noisy -- Yin scores 24 test pairs per condition, so one pair
is 4.2 points and two seeds of the same checkpoint can differ by 8 points. This
runs both simulations over many seeds and writes one tidy CSV, so conditions can
be reported as mean +/- SEM instead of a single draw.

Two phases:
  1. calibrate: for each (model, category) the retrieval-noise level p is swept
     ONCE (Yin with a fine 0.01 step; Kanwisher does coarse->fine internally) and
     recorded. Calibrating per seed would fold calibration jitter into the seed
     variance, which is exactly what we are trying to measure.
  2. seeds: every seed is then run at that fixed p, varying only the item sample.

Results are appended to the CSV as they land, so a partial run is still usable.

    python run_sim_seeds.py --model r7_curriculum --gpu 1 --seeds 101-150
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

MODELS = {
    "r7_curriculum": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r7_curriculum",
    "r5b_allatonce": "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r5b_houses_zubud_30ep",
}
CATEGORIES = ["faces", "objects", "houses_zubud"]
# Yin (1969) human upright-upright accuracy, per category (Tables 1 & 2)
YIN_TARGET = {"faces": 0.9629, "objects": 0.8479, "houses_zubud": 0.9071}
# The Kanwisher triplet task needs two photos of the same identity. ZuBuD has
# only one house photo per class in valid, so it borrows test as well (the same
# splits the earlier runs/kanwisher logs used).
KANW_SPLITS = {"houses_zubud": ["valid", "test"]}

RE_NOISE = re.compile(r"Using noise p=([\d.]+)")
RE_YIN = re.compile(r"^\s*(Upright|Inverted)\s+(Upright|Inverted)\s+([\d.]+)%", re.M)
RE_KANW = re.compile(r"^\s*(Upright|Inverted)\s+([\d.]+)%\s+(\d+)\s*$", re.M)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seeds", default="101-150",
                    help="inclusive range 'a-b' or a comma list")
    ap.add_argument("--out-dir", default="runs/sim_seeds")
    ap.add_argument("--sims", nargs="+", default=["yin", "kanwisher"])
    return ap.parse_args()


def expand_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def run(cmd, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def calibrate(model, category, sim, gpu, log_dir):
    """Sweep the retrieval noise once and return the p the sim settled on."""
    script = "simulate_yin1969.py" if sim == "yin" else "simulate_kanwisher2023.py"
    cmd = [sys.executable, script, "--category", category,
           "--run-dir", MODELS[model], "--device", "cuda:0"]
    if sim == "yin":  # kanwisher already refines coarse->fine on its own
        cmd += ["--calib-target", str(YIN_TARGET[category]), "--calib-step", "0.01"]
    elif category in KANW_SPLITS:
        cmd += ["--splits", *KANW_SPLITS[category]]
    rc, out = run(cmd, gpu)
    with open(os.path.join(log_dir, f"calib_{sim}_{model}_{category}.log"), "w") as f:
        f.write(out)
    m = RE_NOISE.search(out)
    if not m:
        print(f"[warn] no noise line for {sim}/{model}/{category} (rc={rc})", flush=True)
        return None
    return float(m.group(1))


def parse_rows(sim, out):
    """(study, test, accuracy) triples for yin; (presentation, accuracy) for kanwisher."""
    if sim == "yin":
        return [(s, t, float(a)) for s, t, a in RE_YIN.findall(out)]
    return [(p, p, float(a)) for p, a, _ in RE_KANW.findall(out)]


def main():
    args = parse_args()
    seeds = expand_seeds(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- phase 1: calibrate once per (sim, category) -------------
    noise_path = os.path.join(args.out_dir, f"noise_{args.model}.json")
    noise = json.load(open(noise_path)) if os.path.isfile(noise_path) else {}
    for sim in args.sims:
        for cat in CATEGORIES:
            key = f"{sim}/{cat}"
            if key in noise:
                continue
            t0 = time.time()
            noise[key] = calibrate(args.model, cat, sim, args.gpu, log_dir)
            print(f"[calib] {args.model} {key} -> p={noise[key]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            with open(noise_path, "w") as f:
                json.dump(noise, f, indent=2)

    # ---------------- phase 2: seeds at fixed noise ---------------------------
    csv_path = os.path.join(args.out_dir, f"results_{args.model}.csv")
    fresh = not os.path.isfile(csv_path)
    fh = open(csv_path, "a", newline="")
    w = csv.writer(fh)
    if fresh:
        w.writerow(["model", "sim", "category", "seed", "noise",
                    "study", "test", "accuracy_pct"])
        fh.flush()

    total = len(args.sims) * len(CATEGORIES) * len(seeds)
    done = 0
    t_start = time.time()
    for seed in seeds:
        for sim in args.sims:
            script = ("simulate_yin1969.py" if sim == "yin"
                      else "simulate_kanwisher2023.py")
            for cat in CATEGORIES:
                p = noise.get(f"{sim}/{cat}")
                done += 1
                if p is None:
                    print(f"[skip] {sim}/{cat}: no calibrated noise", flush=True)
                    continue
                cmd = [sys.executable, script, "--category", cat,
                       "--run-dir", MODELS[args.model], "--device", "cuda:0",
                       "--noise", str(p), "--seed", str(seed)]
                if sim == "kanwisher" and cat in KANW_SPLITS:
                    cmd += ["--splits", *KANW_SPLITS[cat]]
                rc, out = run(cmd, args.gpu)
                rows = parse_rows(sim, out)
                if rc != 0 or not rows:
                    print(f"[warn] {sim}/{cat}/seed{seed} rc={rc}, {len(rows)} rows",
                          flush=True)
                    with open(os.path.join(
                            log_dir, f"fail_{sim}_{args.model}_{cat}_{seed}.log"), "w") as f:
                        f.write(out)
                    continue
                for study, test, acc in rows:
                    w.writerow([args.model, sim, cat, seed, p, study, test, acc])
                fh.flush()
                el = time.time() - t_start
                print(f"[{done}/{total}] {sim} {cat} seed {seed} "
                      f"({el/60:.1f}m elapsed, eta {el/done*(total-done)/60:.0f}m)",
                      flush=True)
    fh.close()
    print(f"DONE {args.model} in {(time.time()-t_start)/60:.1f}m -> {csv_path}")


if __name__ == "__main__":
    main()
