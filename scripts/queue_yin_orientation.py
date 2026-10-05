#!/usr/bin/env python
"""Prepare immutable Yin checkpoints, then run a resumable detached queue."""
import argparse
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_yin_orientation import atomic_json, sha256

PYTHON = Path("/home/siagrawal/miniconda3/envs/themodel2/bin/python")
CATEGORIES = ["faces_rfwWM64", "faces_cfdWM64"]
MODELS = [
    dict(model="r21dev_aa_s42", suffix="resnet18_aa_r21dev_aa_s42", pid=None, log="logs/r21dev_aa_s42.log"),
    dict(model="r20_cylblur_s42", suffix="resnet18_cylblur_r20_cylblur_s42", pid=241963, log="runs/logs_r20_cylblur_s42.log"),
    dict(model="r21dev_vgg16_bn_aa_s42", suffix="vgg16_bn_aa_r21dev_vgg16_bn_aa_s42", pid=253782, log="logs/r21dev_vgg16bnaa_s42.log"),
    dict(model="r21vgg2k_vgg16_bn_aa_s42", suffix="vgg16_bn_aa_r21vgg2k_vgg16_bn_aa_s42", pid=269328, log="logs/r21vgg2k_vgg16bnaa_s42.log"),
]


def process_token(pid):
    try:
        # The comm field can contain spaces; fields after its final ')' start at 3.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # starttime, field 22
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(proc):
    return proc.get("token") is not None and process_token(proc["pid"]) == proc["token"]


def all_trainings_finished(models):
    return not any(alive(m) for m in models if m.get("pid"))


def snapshot(run_dir, dest, kind):
    import torch
    dest.mkdir(parents=True, exist_ok=True)
    metadata = dest / "checkpoint.json"
    if metadata.exists():
        return json.loads(metadata.read_text())
    for name in ("config.json", "label_map.json"):
        shutil.copy2(run_dir / name, dest / name)
    if kind == "best":
        summary = json.loads((run_dir / "summary.json").read_text())
        epoch = summary["results"]["best_epoch"]
        source = run_dir / "best_model.pth"
        weights = torch.load(source, map_location="cpu", weights_only=True)
    elif kind == "latest":
        source = run_dir / "checkpoint_last.pth"
        # Training replaces this file atomically. An open descriptor pins one
        # complete epoch even if training saves the next epoch during our read.
        with source.open("rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)
        epoch, weights = state["epoch"], state["model"]
    elif kind == "final":
        if not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"Training has no completion summary: {run_dir}")
        source = max(run_dir.glob("final_model_*.pth"), key=lambda p: p.stat().st_mtime_ns)
        with (run_dir / "checkpoint_last.pth").open("rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)
        epoch = state["epoch"]
        weights = torch.load(source, map_location="cpu", weights_only=True)
    else:
        raise ValueError(kind)
    checkpoint = dest / "weights.pth"
    torch.save(weights, checkpoint.with_suffix(".tmp"))
    os.replace(checkpoint.with_suffix(".tmp"), checkpoint)
    result = dict(epoch=int(epoch), source=str(source), source_run_dir=str(run_dir),
                  kind=kind, checkpoint=str(checkpoint), run_dir=str(dest),
                  sha256=sha256(checkpoint), captured_at=datetime.now().astimezone().isoformat())
    atomic_json(metadata, result)
    del weights
    gc.collect()
    return result


def prepare(out):
    if (out / "manifest.json").exists():
        raise RuntimeError("Queue already prepared; use its run command")
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source"
    source.mkdir(exist_ok=True)
    files = ["run_yin_orientation.py", "simulate_yin1969_bothnoise.py", "model.py", "datasets.py",
             "salience_trans.py", "trans.py", "cylconv.py", "utils.py", "requirements.txt"]
    for name in files:
        shutil.copy2(ROOT / name, source / name)
    script_files = ["queue_yin_orientation.py", "supervise_yin_orientation.sh", "smoke_yin_orientation.py"]
    for name in script_files:
        shutil.copy2(ROOT / "scripts" / name, source / name)
    (source / "fixation_data").symlink_to(ROOT / "fixation_data", target_is_directory=True)
    models = []
    for spec in MODELS:
        matches = list((ROOT / "runs").glob("*" + spec["suffix"]))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one run for {spec['model']}: {matches}")
        m = {**spec, "source_run_dir": str(matches[0]), "log": str(ROOT / spec["log"]),
             "token": process_token(spec["pid"]) if spec["pid"] else None}
        phase = "model2" if spec["pid"] is None else "latest"
        m[phase] = snapshot(matches[0], out / "checkpoints" / phase / spec["model"],
                            "best" if phase == "model2" else "latest")
        models.append(m)
        print(f"Captured {phase}: {m['model']} epoch {m[phase]['epoch']}", flush=True)
    # Capture any pre-existing Yin experiment, without relying on recyclable PIDs.
    blockers = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            args = (p / "cmdline").read_bytes().split(b"\0")
            if any(Path(os.fsdecode(a)).name in ("run_sim_seeds.py", "simulate_yin1969.py", "simulate_yin1969_bothnoise.py") for a in args):
                blockers.append(dict(pid=int(p.name), token=process_token(int(p.name))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), models=models, initial_wait=blockers,
                    categories=CATEGORIES, seeds="101-150", target_uu=0.9629, target_ii=0.8188,
                    gpus=[0, 3], min_free_mb=4500, python=str(PYTHON), source=str(source),
                    source_hashes={n: sha256(source / n) for n in files + script_files})
    atomic_json(out / "manifest.json", manifest)
    print(f"Prepared {out}; waiting on {blockers}", flush=True)


def free_gpu_memory():
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       text=True, capture_output=True, check=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in p.stdout.strip().splitlines())}


def command_for(manifest, out, job):
    ck = job["snapshot"]
    return [manifest["python"], "-u", str(Path(manifest["source"]) / "run_yin_orientation.py"),
            "--model", job["model"], "--stage", job["phase"], "--epoch", str(ck["epoch"]),
            "--run-dir", ck["run_dir"], "--checkpoint", ck["checkpoint"],
            "--category", job["category"], "--seeds", manifest["seeds"],
            "--target-uu", str(manifest["target_uu"]), "--target-ii", str(manifest["target_ii"]),
            "--out-dir", str(out / "results" / job["id"]), "--device", "cuda:0"]


def add_jobs(state, phase, model, ck, categories):
    for category in categories:
        key = f"{phase}/{model}/{category}"
        state["jobs"].setdefault(key, dict(id=key, phase=phase, model=model, category=category,
                                         snapshot=ck, status="pending", attempts=0))


def terminal(jobs):
    return all(j["status"] in ("done", "failed") for j in jobs)


def aggregate(out, state):
    rows = []
    for job in state["jobs"].values():
        p = out / "results" / job["id"] / "summary.json"
        if job["status"] == "done" and p.exists():
            rows.append(json.loads(p.read_text()))
    atomic_json(out / "all_results.json", rows)


def run_queue(out, once=False):
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if sha256(Path(manifest["source"]) / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    state_path = out / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}, "status": "starting", "final_prepared": False}
    for m in manifest["models"]:
        phase = "model2" if m["pid"] is None else "latest"
        add_jobs(state, phase, m["model"], m[phase], manifest["categories"])
    children = {}
    while True:
        for key, child in list(children.items()):
            if child.poll() is not None:
                state["jobs"][key]["returncode"] = child.returncode
                del children[key]
        for job in state["jobs"].values():
            if job["status"] == "running" and not alive(job):
                done_path = out / "results" / job["id"] / "DONE.json"
                if done_path.exists():
                    done = json.loads(done_path.read_text())
                    if done["checkpoint_sha256"] != job["snapshot"]["sha256"] or done["rows"] != 200:
                        raise RuntimeError(f"Invalid completion record: {done_path}")
                    job["status"] = "done"
                else:
                    job["status"] = "pending" if job["attempts"] < 3 else "failed"
                print(f"[{datetime.now().isoformat()}] {job['id']} -> {job['status']}", flush=True)
        old_active = [p for p in manifest["initial_wait"] if alive(p)]
        first = [j for j in state["jobs"].values() if j["phase"] == "model2"]
        latest = [j for j in state["jobs"].values() if j["phase"] == "latest"]
        if old_active:
            phase = None
            state["status"] = "waiting_for_current_experiment"
        elif not terminal(first):
            phase = "model2"
        elif not terminal(latest):
            phase = "latest"
        elif not all_trainings_finished(manifest["models"]):
            phase = None
            state["status"] = "waiting_for_all_training"
        else:
            phase = "final"
            if not state["final_prepared"]:
                for m in manifest["models"]:
                    if m["pid"] is None:
                        continue
                    try:
                        log = ROOT / m["log"]
                        with log.open("rb") as f:
                            f.seek(max(0, log.stat().st_size - 65536))
                            end = f.read().decode(errors="replace")
                        if "Done. Best valid acc" not in end:
                            raise RuntimeError("Training did not report successful completion")
                        ck = snapshot(Path(m["source_run_dir"]), out / "checkpoints" / "final" / m["model"], "final")
                        add_jobs(state, "final", m["model"], ck, manifest["categories"])
                    except Exception as exc:
                        state.setdefault("training_errors", {})[m["model"]] = str(exc)
                        print(f"Final snapshot unavailable for {m['model']}: {exc}", flush=True)
                state["final_prepared"] = True
        if phase:
            state["status"] = f"running_{phase}"
            occupied = {j["gpu"] for j in state["jobs"].values() if j["status"] == "running"}
            free = free_gpu_memory()
            for gpu in manifest["gpus"]:
                if gpu in occupied or free.get(gpu, 0) < manifest["min_free_mb"]:
                    continue
                job = next((j for j in state["jobs"].values() if j["phase"] == phase and j["status"] == "pending"), None)
                if job is None:
                    break
                dest = out / "results" / job["id"]
                dest.mkdir(parents=True, exist_ok=True)
                cmd = command_for(manifest, out, job)
                atomic_json(dest / "command.json", {"argv": cmd, "gpu": gpu})
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
                with (dest / "worker.log").open("a") as f:
                    child = subprocess.Popen(cmd, cwd=manifest["source"], env=env, stdout=f, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                children[job["id"]] = child
                job.update(status="running", pid=child.pid, token=process_token(child.pid), gpu=gpu,
                           attempts=job["attempts"] + 1, started_at=datetime.now().astimezone().isoformat())
                print(f"Started gpu{gpu}: {job['id']} pid={child.pid}", flush=True)
                atomic_json(state_path, state)
        if state["final_prepared"] and terminal(state["jobs"].values()):
            state["status"] = "complete_with_errors" if state.get("training_errors") or any(j["status"] == "failed" for j in state["jobs"].values()) else "complete"
        state["updated_at"] = datetime.now().astimezone().isoformat()
        atomic_json(state_path, state)
        aggregate(out, state)
        if state["status"].startswith("complete") or once:
            print(state["status"], flush=True)
            break
        time.sleep(30)
    lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "run", "status"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--once", action="store_true", help="one scheduling iteration (for verification)")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.action == "prepare":
        prepare(out)
    elif args.action == "status":
        print((out / "queue_state.json").read_text())
    else:
        run_queue(out, args.once)


if __name__ == "__main__":
    main()
