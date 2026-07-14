"""
run_experiments.py

Launch several train.py runs simultaneously, one per GPU, each in its own
auto-assigned output folder. Free GPUs are detected via nvidia-smi (or given
explicitly); if there are more runs than GPUs, runs are queued and started as
GPUs free up.

Each run's folder gets:
    config.json    the exact input configuration (written by train.py)
    summary.json   concise input + output: best/final accuracies, wall time
    train.log      full stdout/stderr of the run
plus the usual best_model.pth / label_map.json / history CSV / accuracy.png.
The experiment folder itself gets a manifest.json indexing all runs.

Examples:
    # 3 learning rates in parallel on whatever GPUs are free
    python run_experiments.py \
        --base "--categories faces objects houses --variant lp --epochs 50" \
        --run "--lr 1e-3" --run "--lr 3e-4" --run "--lr 1e-4"

    # explicit GPUs, runs read from a file (one train.py arg-string per line)
    python run_experiments.py --gpus 0 1 2 --runs-file my_grid.txt

my_grid.txt example:
    --lr 1e-3 --variant lp
    --lr 1e-3 --variant cnn
    # comments and blank lines are ignored
    --lr 3e-4 --variant lp --num-fixations 8
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time


def detect_free_gpus(max_mem_mib=2048):
    """GPUs currently using less than max_mem_mib of memory."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    free = []
    for line in out.strip().splitlines():
        idx, used = (int(v) for v in line.split(","))
        if used < max_mem_mib:
            free.append(idx)
    return free


def slug_for(run_args):
    """Short filesystem-safe tag summarizing a run's extra args."""
    s = run_args.replace("--", "").strip().replace("=", "").replace(" ", "-")
    s = "".join(c for c in s if c.isalnum() or c in "-._")
    return s[:60] or "run"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="",
                    help="train.py args shared by every run, e.g. "
                         "\"--categories faces objects houses --epochs 50\"")
    ap.add_argument("--run", action="append", default=[],
                    help="extra train.py args for one run; repeat for several runs")
    ap.add_argument("--runs-file", default=None,
                    help="file with one run's extra args per line ('#' comments allowed)")
    ap.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="GPU indices to use (default: auto-detect free GPUs)")
    ap.add_argument("--jobs-per-gpu", type=int, default=1,
                    help="concurrent runs per GPU (ResNet18@180 uses ~6-8 GB of a "
                         "48 GB A6000, so 2 can time-share one GPU)")
    ap.add_argument("--exp-dir", default=None,
                    help="experiment folder (default: runs/exp_<timestamp>)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--poll-sec", type=float, default=10.0)
    args = ap.parse_args()

    runs = list(args.run)
    if args.runs_file:
        with open(args.runs_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    runs.append(line)
    if not runs:
        ap.error("no runs given; use --run and/or --runs-file")

    gpus = args.gpus if args.gpus is not None else detect_free_gpus()
    if not gpus:
        raise SystemExit("No free GPUs detected; pass --gpus explicitly.")
    slots = [(g, k) for g in gpus for k in range(args.jobs_per_gpu)]
    print(f"Using GPUs {gpus} ({len(slots)} slot(s)) for {len(runs)} run(s)")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = args.exp_dir or f"runs/exp_{ts}"
    os.makedirs(exp_dir, exist_ok=True)

    manifest = {"started": datetime.datetime.now().isoformat(timespec="seconds"),
                "base_args": args.base, "gpus": gpus, "runs": []}
    for k, extra in enumerate(runs, 1):
        out_dir = os.path.join(exp_dir, f"run{k:02d}_{slug_for(extra)}")
        manifest["runs"].append({"id": k, "args": extra, "out_dir": out_dir,
                                 "status": "queued", "gpu": None, "returncode": None})

    def save_manifest():
        with open(os.path.join(exp_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    save_manifest()

    queue = list(manifest["runs"])
    running = {}  # (gpu, slot) -> (Popen, run_entry, log_file)

    def launch(gpu, entry):
        os.makedirs(entry["out_dir"], exist_ok=True)
        cmd = ([args.python, "train.py"] + shlex.split(args.base) + shlex.split(entry["args"])
               + ["--device", "cuda:0", "--output-dir", entry["out_dir"]])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(os.path.join(entry["out_dir"], "train.log"), "w")
        log.write(f"# GPU {gpu}\n# {' '.join(cmd)}\n\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        entry.update(status="running", gpu=gpu, pid=p.pid,
                     cmd=" ".join(cmd))
        print(f"[launch] run{entry['id']:02d} on GPU {gpu} -> {entry['out_dir']}")
        save_manifest()
        return p, log

    try:
        while queue or running:
            for slot in slots:
                if slot not in running and queue:
                    entry = queue.pop(0)
                    p, log = launch(slot[0], entry)
                    running[slot] = (p, entry, log)
            time.sleep(args.poll_sec)
            for slot in list(running):
                p, entry, log = running[slot]
                rc = p.poll()
                if rc is not None:
                    log.close()
                    entry.update(status="done" if rc == 0 else "failed", returncode=rc)
                    print(f"[{'done' if rc == 0 else 'FAILED'}] run{entry['id']:02d} "
                          f"(GPU {slot[0]}, rc={rc}) -> {entry['out_dir']}")
                    save_manifest()
                    del running[slot]
    except KeyboardInterrupt:
        print("\nInterrupted; terminating running jobs...")
        for slot, (p, entry, log) in running.items():
            p.terminate()
            entry.update(status="interrupted")
            log.close()
        save_manifest()
        raise

    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_manifest()

    # concise cross-run report
    print("\n=== Experiment summary ===")
    for entry in manifest["runs"]:
        s_path = os.path.join(entry["out_dir"], "summary.json")
        if os.path.isfile(s_path):
            with open(s_path) as f:
                r = json.load(f)["results"]
            print(f"run{entry['id']:02d} [{entry['status']}] {entry['args']}: "
                  f"best valid {r['best_valid_acc']*100:.2f}% (epoch {r['best_epoch']}), "
                  f"test(inv) {r['test_acc_at_best_epoch']*100:.2f}%, "
                  f"{r['sec_per_epoch']:.0f}s/epoch")
        else:
            print(f"run{entry['id']:02d} [{entry['status']}] {entry['args']}: no summary.json")
    print(f"\nManifest: {os.path.join(exp_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
