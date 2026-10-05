#!/usr/bin/env python
"""Exercise every scheduled checkpoint/face set and verify resumable outputs."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    manifest = json.loads((out / "manifest.json").read_text())

    def worker(gpu, category):
        results = []
        for m in manifest["models"]:
            phase = "model2" if m["pid"] is None else "latest"
            ck = m[phase]
            dest = out / "smoke" / m["model"] / category
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [manifest["python"], "-u", str(Path(manifest["source"]) / "run_yin_orientation.py"),
                   "--model", m["model"], "--stage", "smoke", "--epoch", str(ck["epoch"]),
                   "--run-dir", ck["run_dir"], "--checkpoint", ck["checkpoint"],
                   "--out-dir", str(dest), "--category", category, "--device", "cuda:0"]
            if phase == "model2":
                cmd += ["--seeds", "101-102", "--coarse-step", "0.25", "--fine-radius", "0"]
                expected = 8
            else:
                cmd += ["--seeds", "101", "--noise-up", "0.1", "--noise-inv", "0.3"]
                expected = 4
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            start = time.monotonic()
            with (dest / "smoke.log").open("a") as log:
                subprocess.run(cmd, env=env, cwd=manifest["source"], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
            rows = list(csv.DictReader((dest / "results.csv").open()))
            assert len(rows) == expected
            fits = json.loads((dest / "calibration.json").read_text())
            rates = {orient: fit["p"] for orient, fit in fits.items()}
            for row in rows:
                assert float(row["p_study"]) == rates[row["study"]]
                assert float(row["p_test"]) == rates[row["test"]]
                assert 0 <= float(row["accuracy_pct"]) <= 100
            before = (dest / "results.csv").read_bytes()
            resumed = subprocess.run(cmd, env=env, cwd=manifest["source"], capture_output=True,
                                     text=True, check=True, timeout=120)
            assert "Already complete" in resumed.stdout
            assert (dest / "results.csv").read_bytes() == before
            results.append(dict(model=m["model"], category=category, rows=len(rows),
                                seconds=round(time.monotonic()-start, 1), resume_verified=True))
            print(f"PASS gpu{gpu}: {m['model']} {category} ({results[-1]['seconds']}s)", flush=True)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, cat) for gpu, cat in zip(manifest["gpus"], manifest["categories"])]
        results = [item for future in futures for item in future.result()]
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("All eight real-checkpoint smoke runs passed", flush=True)


if __name__ == "__main__":
    main()
