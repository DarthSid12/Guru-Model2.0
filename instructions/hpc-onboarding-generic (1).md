# GPU training on NCSA Delta, NCSA DeltaAI, and Purdue Anvil AI: a practical guide

Notes from running long PyTorch training jobs on these three ACCESS/NCSA Slurm systems. Nothing
here is specific to a project. Placeholders:

- `<user>`: your username on the cluster. Anvil usernames are ACCESS-style, `x-<user>`.
- `<proj>`: your allocation's project code on Delta/DeltaAI (e.g. `abcd`).
- `<alloc>`: your ACCESS allocation on Anvil (e.g. `abc123456`).
- `<account>`: the Slurm `--account` name. **It is often not the same as the allocation name**
  (see §3.3).

Hardware and policy facts were correct as of late 2026. Check anything you depend on against
the official docs: [Delta](https://docs.ncsa.illinois.edu/systems/delta) ·
[DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai) ·
[Anvil](https://www.rcac.purdue.edu/knowledge/anvil).

---

## 1. The three systems

| | **Delta** (NCSA) | **DeltaAI** (NCSA) | **Anvil AI** (Purdue, ACCESS) |
|---|---|---|---|
| Login node | `login.delta.ncsa.illinois.edu` (round-robin; pin one node, e.g. `dt-login01`) | `gh-login01.delta.ncsa.illinois.edu` | `anvil.rcac.purdue.edu` |
| Auth | NCSA password + Duo | NCSA password + Duo | SSH public key |
| CPU arch | x86-64 (AMD EPYC Milan) | **aarch64** (NVIDIA Grace) | x86-64 |
| GPU node | 4× A100-SXM4 **40 GB** (sm_80), 64 cores, 256 GB RAM | 4× GH200 **120 GB** (Hopper sm_90), ~288 Arm cores | 4× H100 SXM **80 GB** (sm_90), 96 cores, 1 TB RAM |
| Main GPU partition | `gpuA100x4` (also `gpuA40x4`, A40 48 GB) | `ghx4` | `ai` (the `gpu` partition is the older A100-40 GB subsystem) |
| Debug partition | `gpuA100x4-interactive` (1 h) | `ghx4-interactive` (2 h) | — |
| Max walltime | 48 h | 48 h | 48 h, **12 GPUs per user** at once |
| Node sharing | yes (you can request part of a node) | yes | yes |
| Mixed precision | `bf16` | `bf16` | `bf16` |
| Container runtime | Apptainer 1.5 in `/usr/bin`, no module needed | Apptainer 1.4 in `/usr/bin` | Apptainer 1.4 in `/usr/bin` |
| Internet from compute nodes | yes (direct) | yes (direct) | check yourself (curl test, §4) |
| What drives queue priority | **fairshare** | **fairshare** | **job age (≈ first-come-first-served)** |
| Support | help.ncsa.illinois.edu | help.ncsa.illinois.edu | **ACCESS ticket** (not RCAC directly) |

**Delta and DeltaAI are different machines** with separate login nodes, schedulers and
allocations. They **share the `/projects` filesystem**, so data staged on one is visible on the
other. Their binaries and container images are **not** interchangeable, because DeltaAI is ARM.

---

## 2. Storage

| | Delta / DeltaAI | Anvil |
|---|---|---|
| Home | `/u/<user>`, 100 GB, not purged | `/home/x-<user>`, **25 GB** |
| Project (persistent, shared with your group) | `/projects/<proj>/`, quota set by the allocation (Taiga, spinning disk) | `/anvil/projects/x-<alloc>/`, quota set by the allocation |
| Work / scratch | `/work/nvme/<proj>`, `/work/hdd/<proj>` (if your allocation has them) | `/anvil/scratch/x-<user>` (large; check the purge policy) |
| Node-local | `/tmp` NVMe, about 1.5 TB on Delta GPU nodes and 1.8 TB on DeltaAI, **wiped at job end** | `/tmp`, size not reported by Slurm; check `df -h /tmp` inside a job |

Run `quota` (NCSA) or `myquota` (Anvil) to see your limits.

Rules:
- **Put datasets, checkpoints, containers and logs on project storage**, not home. On Anvil the
  25 GB home fills up with one dataset unzip or a few checkpoints.
- **Copy the training dataset to node-local `/tmp` at the start of each job** if your data loader
  does many small random reads (Arrow/Parquet shards, image folders, small files). Delta's
  `/projects` is spinning disk and is very slow for that pattern. Ship the data as one archive and
  unzip it to `/tmp`, which takes a minute or two. Write checkpoints to project storage, because
  `/tmp` is wiped when the job ends.
- Project quotas are shared by the whole group. Delete checkpoints you no longer need.

---

## 3. One-time setup

### 3.1 SSH config

NCSA requires Duo on every new connection. Open one authenticated "master" connection and
reuse it for every later `ssh`, `scp` and `rsync`:

```sshconfig
Host delta
    HostName dt-login01.delta.ncsa.illinois.edu   # use a fixed node so tmux sessions are still there
    User <user>
    ControlMaster auto
    ControlPath ~/.ssh/delta.sock
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host deltaai
    HostName gh-login01.delta.ncsa.illinois.edu
    User <user>
    ControlMaster auto
    ControlPath ~/.ssh/deltaai.sock
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host anvil
    HostName anvil.rcac.purdue.edu
    User x-<user>
```

To use it: start a `tmux` session, run `ssh delta`, and complete the password and Duo prompts
(`1` = push). Detach with `Ctrl-b d`. **Don't press `Ctrl-c`**, because that closes the master
connection. For the next 8 hours, `ssh delta …` won't ask for Duo again. Check the connection
with `ssh -O check delta`.

On Anvil, register your public key through the Anvil / ACCESS key process. No master connection
is needed.

### 3.2 Git on the login nodes

To clone your code on the cluster (recommended, §7), create a key on each login node
(`ssh-keygen -t ed25519`), add it to GitHub as a user key or a repo deploy key, and test with
`ssh -T git@github.com`. Compute nodes don't need git if you clone on the login node and
bind-mount the clone into the job.

### 3.3 Accounts and balances

```bash
ssh delta   accounts      # Delta: lists e.g. <proj>-delta-gpu and <proj>-delta-cpu
ssh deltaai accounts      # DeltaAI: e.g. <proj>-dtai-gh
ssh anvil   mybalance     # Anvil: lists the Slurm account name(s) and remaining SUs
```

- On NCSA, **GPU and CPU are separate Slurm accounts.** A GPU partition rejects the CPU account
  with "Invalid account".
- On Anvil, **the Slurm account name can differ from the allocation name.** For example, an
  `x-<alloc>` allocation on the AI partition may show up as `<alloc>-ai`. Use the name that
  `mybalance` prints.
- **Billing is by what you reserve, not what you use.** A full 4-GPU node costs 4 GPU-hours per
  wall-hour even if your code uses one GPU. All three systems let you request part of a node
  (`--gpus-per-node=1`), which costs 1 GPU-hour per hour.

### 3.4 Containers

Apptainer images are the most reliable way to get a fixed CUDA/PyTorch environment. Build the
environment into the image and **bind-mount your code at runtime** instead of baking it in, so
a code change doesn't need a rebuild.

- **Delta (x86-64):** in our experience you **can't build images on Delta** (no root/fakeroot).
  Build on a machine where you have root, e.g. `sudo apptainer build env.sif env.def` from a
  `nvidia/cuda:12.x-cudnn-runtime` base, then `scp` it to project storage. The same image runs
  on Anvil.
- **DeltaAI (aarch64):** you need an **ARM image**. Your x86 image will fail with
  `exec format error`. Cross-building with QEMU usually fails on rootless workstations, so build
  **on the DeltaAI login node**:
  ```bash
  export APPTAINER_CACHEDIR=/projects/<proj>/<user>/apptainer-cache
  export APPTAINER_TMPDIR=/projects/<proj>/<user>/apptainer-tmp
  nohup apptainer build --fakeroot env-arm64.sif env.def > build.log 2>&1 &
  ```
  Use **`--fakeroot`**, not `--ignore-fakeroot-command`. Without a subuid mapping, apt can't
  drop privileges and every package install fails with `setgroups: Operation not permitted`.
  aarch64 wheels can lag behind x86. In late 2026 there was no aarch64 torch 2.8.0 wheel (2.7.1
  and 2.9.0 existed), and some rdkit and other scientific packages lacked aarch64 builds for
  every Python version. Pin versions that actually exist for aarch64.
- In the `%post` section of the definition file, **assert that the GPU architecture you need is
  present**, e.g. `python -c "import torch; assert 'sm_90' in torch.cuda.get_arch_list()"`. A bad
  image then fails at build time instead of after a long queue wait.
- Run with `apptainer exec --nv --cleanenv …`. `--cleanenv` stops login-shell variables from
  leaking into the job, so pass anything you need explicitly with `--env`.

---

## 4. Running long training: the self-resubmitting chain

All three systems cap jobs at 48 hours. For longer runs, have each job **submit its own
successor before it starts training**, and have every chunk resume from the latest checkpoint.

Key points:
1. **Submit the successor first**, with `--dependency=afterany:$SLURM_JOB_ID`. A walltime kill
   is SIGKILL with no cleanup, but a dependency registered up front still holds.
2. **Use a fixed run ID for the whole chain** (pass it to the successor via `--export`), so every
   chunk resumes the same `last.ckpt` and logs to the same experiment-tracker run. For W&B,
   pass a fixed `id=` with `resume="allow"`.
3. **Use a sentinel to stop.** When training exits 0 (finished or early-stopped), write a
   `COMPLETE` file. The successor that is already queued sees it and exits immediately.
4. Set a **`MAX_CHUNKS` limit** so a job that crashes at startup can't resubmit forever.
5. Pass partition, account, GPUs, CPUs and memory down the chain explicitly, so successors run
   on the same hardware.

Template (edit the `#SBATCH` lines and the training command):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --account=<account>
#SBATCH --partition=<partition>          # gpuA100x4 | ghx4 | ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=<PROJECT_DIR>/slurm-logs/%x-%j.out
set -euo pipefail

PROJECT_DIR=<PROJECT_DIR>
IMAGE=$PROJECT_DIR/images/env.sif
CODE_DIR=${CODE_DIR:?}                 # a git clone checked out at a fixed SHA
EXP=${EXP:?}
SCRIPT=${SCRIPT:?absolute path to this file}  # Slurm runs a spooled copy, so $0 is not this file
MAX_CHUNKS=${MAX_CHUNKS:-10}
CHAIN_INDEX=${CHAIN_INDEX:-1}
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=$PROJECT_DIR/runs/$EXP/$RUN_ID
mkdir -p "$RUN_DIR"

[[ -f $RUN_DIR/COMPLETE ]] && { echo "already complete"; exit 0; }

if (( CHAIN_INDEX < MAX_CHUNKS )); then
  sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
    --partition=$SLURM_JOB_PARTITION --account=$SLURM_JOB_ACCOUNT \
    --gpus-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --mem=$SLURM_MEM_PER_NODE --job-name=$SLURM_JOB_NAME \
    --export=ALL,RUN_ID=$RUN_ID,CHAIN_INDEX=$((CHAIN_INDEX+1)) "$SCRIPT"
fi

DATA=/tmp/$USER/job_$SLURM_JOB_ID; mkdir -p "$DATA"
trap 'rm -rf "$DATA"' EXIT
unzip -q "$PROJECT_DIR/datasets/mydata.zip" -d "$DATA"

NPROC=$(nvidia-smi -L | wc -l)          # follow what was actually allocated
set +e
apptainer exec --nv --cleanenv --pwd /code \
  -B "$CODE_DIR":/code -B "$DATA":/data -B "$RUN_DIR":/run_dir \
  -B "$PROJECT_DIR/wandb_key":/wandb_key:ro \
  --env RUN_ID=$RUN_ID --env WANDB_DIR=/run_dir \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node $NPROC train.py \
    --data /data --out /run_dir --resume-from-last --precision bf16-mixed
rc=$?
set -e
(( rc == 0 )) && touch "$RUN_DIR/COMPLETE"
exit $rc
```

**Stopping a chain:** cancel the running job **and** its queued successor
(`scancel <id> <successor-id>`), or `touch $RUN_DIR/COMPLETE`. If you cancel only the running
job, the successor (`afterany`) starts and resumes training.

**Internet on compute nodes:** Delta and DeltaAI compute nodes have direct internet access, so
online W&B logging works. On Anvil, check at the start of each job, e.g.
`curl -sI --max-time 10 https://api.wandb.ai`. If it fails, use `WANDB_MODE=offline` and run
`wandb sync` from the login node afterwards.

---

## 5. Sizing and throughput

- **Keep the global batch size fixed across clusters:** global batch = per-GPU batch × number of
  GPUs × gradient-accumulation steps. Then runs on different hardware are comparable. Example
  with a global batch of 512:

  | Cluster | Job shape | Batch config |
  |---|---|---|
  | Delta | full node, `--gpus-per-node=4 --cpus-per-task=64 --mem=240G` | 32 × 4 × 4 |
  | DeltaAI | 1 GPU, `--cpus-per-task=16 --mem=110G` | 512 × 1 × 1 (120 GB holds a lot) |
  | Anvil | 1 GPU, `--cpus-per-task=16 --mem=110G` | 512 × 1 × 1 if it fits in 80 GB, else 256 × 1 × 2 |

- **One large-memory GPU often beats a node of smaller GPUs per GPU-hour.** In our workload, one
  GH200 was faster per epoch than 4× A100-40 GB with DDP, at a quarter of the GPU-hour cost.
  Single-GPU jobs also get scheduled much sooner.
- **Measure memory before scaling.** Don't assume a configuration from the 120 GB GH200 fits on
  an 80 GB H100 or a 40 GB A100.
- Get the GPU count from the allocation (`nvidia-smi -L`) rather than hardcoding it, and log the
  actual global batch, so a job that received fewer GPUs is easy to spot.
- If each DataLoader worker holds its own copy of a large in-memory dataset, host RAM usage grows
  with the worker count. Memory-mapped arrays share pages through the OS cache and let you
  request much less `--mem`.

---

## 6. Scheduling

- **Delta and DeltaAI rank pending jobs mainly by fairshare.** Your account's recent usage
  pushes your jobs down, and job age counts for much less. **Anvil ranks mainly by job age**
  (fairshare has close to zero weight), so jobs run roughly in submission order: submit early.
  Policies change; check with `sprio -w` and `scontrol show config | grep -i ^Priority`.
- Before submitting, check free GPUs with `sinfo -p <partition> -o '%n %G %t'`, and pending jobs
  with `squeue -p <partition> -t PD | wc -l` and `sprio -p <partition>`.
- Your usage counts against everyone on the allocation, both GPU-hours and fairshare. Plan large
  sweeps with your group.
- Use the interactive partitions (`srun --partition=gpuA100x4-interactive --account=<account>
  --gpus-per-node=1 --time=00:30:00 --pty bash`) for debugging. **Don't run real work on login
  nodes.** These short sessions still use GPU-hours.

---

## 7. Practices

1. **Pin experiments to an exact git SHA, not a branch.** Clone on the login node and run
   `git checkout --detach <sha>`, then bind that clone into the job. Branch tips move between
   seeds and between chain chunks, and rsyncing a local working tree can silently ship stale or
   uncommitted files. Write the SHA into the run directory.
2. **Before running at scale:** run an import test of the image on the login node, then submit
   one short job and read its log (GPU visible, data path correct, logger online, global batch
   as expected). Then submit the full set.
3. **Put a timestamp in job names** (`--job-name=myexp-s0-20261002-1415`). Keep the experiment
   name stable and meaningful so relaunches are easy to tell apart.
4. **Stage data once and reuse it.** Copy datasets to project storage ahead of time and verify
   them with `sha256sum`. Use Globus for large transfers (each site has an endpoint).
5. **Keep secrets in files with `chmod 600`** (e.g. the W&B key) and bind them read-only.
   Exporting them in the shell environment doesn't survive `--cleanenv`.
6. Watch your jobs with `squeue -u <user>` and `tail -f` on the Slurm log. With the chain setup,
   a running job plus a pending successor marked `(Dependency)` is normal.

## 8. Common problems

| Symptom | Cause / fix |
|---|---|
| `Invalid account` / `Project not found` | Wrong account. Use the GPU account (NCSA) or the name `mybalance` shows (Anvil). |
| `exec format error` on DeltaAI | x86 image on ARM. Build an aarch64 image on the DeltaAI login node. |
| Image build fails with apt `setgroups … not permitted` | Build with `--fakeroot` (DeltaAI). Delta doesn't allow builds at all; build elsewhere. |
| Duo prompts again and again | Master connection expired. Check with `ssh -O check <alias>` and re-authenticate in tmux. |
| tmux session gone | You landed on a different login node. Use a specific login node in your SSH config. |
| Training slow, GPU utilization low | Data read straight from `/projects`. Copy it to node-local `/tmp`. |
| Anvil home "disk quota exceeded" | 25 GB limit. Move everything to `/anvil/projects` or scratch. |
| A list argument like `{a,b,c}` reaches your script as one literal string | Bash brace expansion doesn't apply inside variables. Separate values with spaces. |
| Chain keeps resubmitting | Training never exits 0, so `COMPLETE` is never written. Read the log; `MAX_CHUNKS` limits it. |
| W&B creates a new run for each chunk | No fixed run ID. Pass a fixed `id` with `resume="allow"`. |
