# combined_lpnet

A single log-polar / foveated ResNet18 trained jointly on **faces + houses + objects**,
used to replicate Yin's (1969) face-inversion effect (and the same test for houses
and objects) via a Barrington-NIMBLE KDE memory model.

The model (`model.py`) comes from the `TheModel2.0` `familiar-faces` branch:
ResNet18 backbone → `fc1` (512→256) → temperature-scaled sigmoid → Bernoulli binary
code `h` → `fc2` classifier. The Yin/NIMBLE simulation operates on the shared 256-d
binary code `h`; the classifier head is only the training signal.

Key design points:
- One unified softmax over every class of every category (face identities +
  object categories + a single generic `house` class).
- Fixations come from bottom-up Gabor-variance saliency (V1-like), the same
  mechanism for all categories.
- All raw images are resized + center-cropped to a square `--input-size`
  (default 224).

## Setup (conda)

Requires a CUDA-capable GPU for practical training speed.

```bash
conda create -y -n lpnet python=3.10
conda activate lpnet
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt

# verify
python -c "import torch, cv2; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## Pipeline

Run each step yourself, in order. Do **not** point at anyone else's
pre-built `data/` / `fixation_data/` — the steps below are cheap enough to
run fresh and guarantee your data matches your code version.

### 1. Download raw data

```bash
python download.py faces objects houses
```

Faces/objects come from Google Drive archives; houses from the public
[emanhamed/Houses-dataset](https://github.com/emanhamed/Houses-dataset) repo
(frontal photos only, pooled into one generic `house` class). Re-running skips
anything already present.

### 2. Preprocess into packed fixation data

```bash
python preprocess_fixations.py \
    --categories faces objects houses \
    --num-coords 32 \
    --devices cuda:0 cuda:1
```

This stores only the expensive-to-compute Gabor-saliency fixation
*coordinates* plus the resized raw images as a few large memory-mappable
arrays (`fixation_data/<category>/<split>/{images.npy, coords.npy, meta.json}`,
~25 GB total). Cropping, rotation, foveation and the log-polar transform then
happen on the fly on the GPU at train time — ~22 min/epoch vs ~3 h/epoch for
the old pre-rendered-PNG path, and numerically identical output.

`--num-coords 32` lets training use any `--num-fixations <= 32`. Re-runs skip
already-packed `(category, split)` units (`--force` to re-pack).

<details>
<summary>Legacy PNG path (not recommended)</summary>

`preprocess.py` pre-renders every fixation crop to disk under
`processed_data/<category>/<lp|cnn>/<split>/<class>/` (~246 GB, ~6M PNGs,
I/O-bound training). Only use it if you specifically need the rendered crops:

```bash
python preprocess.py --categories faces objects houses \
    --num-fixations 16 --input-size 224 --devices cuda:0
```
</details>

### 3. Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_coords.py \
    --categories faces objects houses \
    --variant lp \
    --lr 1e-3 \
    --epochs 50 \
    --batch-size 256 \
    --num-fixations 16
```

Key flags:
- `--lr` — learning rate (required tunable)
- `--categories` — which categories to train jointly
- `--variant lp|cnn` — log-polar/foveated vs. plain-crop control
- `--data-mode auto|packed|png` — `auto` (default) picks `packed` when
  `fixation_data/` exists
- `--epochs`, `--batch-size`, `--num-fixations`, `--patience`, `--temperature`
- `--pretrained-path PATH` — warm-start from a checkpoint
- `--device auto|cuda:N|cpu`

Each run writes `config.json`, `summary.json` (accuracies, wall time),
`best_model.pth`, `label_map.json` (needed by the Yin simulation), a history
CSV, and an accuracy plot into its `runs/...` directory.

### 3b. Run several experiments at once — `run_experiments.py`

`run_experiments.py` is a parallel experiment launcher: it takes a list of
`train.py` argument strings and runs them simultaneously, **one per free GPU**
(auto-detected via nvidia-smi, or pinned with `--gpus 0 1 2`). Extra runs
queue and start as GPUs free up. Everything lands under an auto-named
`runs/exp_<timestamp>/` folder — one subfolder per run (`config.json`,
`summary.json`, `train.log`) plus a `manifest.json` index — and a cross-run
accuracy table is printed at the end.

```bash
python run_experiments.py \
    --base "--categories faces objects houses --variant lp --epochs 50" \
    --run "--lr 1e-3" --run "--lr 3e-4" --run "--lr 1e-4 --variant cnn"

# or keep the grid in a file (one train.py arg-string per line, '#' comments)
python run_experiments.py --gpus 0 1 2 --runs-file my_grid.txt
```

### 4. Run the Yin (1969) simulation

Once per category, pointing at the trained checkpoint:

```bash
python simulate_yin1969.py --category faces --variant lp \
    --checkpoint runs/<run>/best_model.pth \
    --label-map  runs/<run>/label_map.json
```

Repeat with `--category objects` and `--category houses`. Key flags:
`--num-study`, `--num-test`, `--study-fixations`, `--test-fixations`,
`--sigma` (NIMBLE kernel bandwidth), `--calib-target`.

Note on houses: there is no per-house identity, so an "item" is an individual
photo drawn from a shared valid/test "Yin pool" built by `download.py`
(default 100 photos, same photo upright vs. inverted). This is a stopgap
until a set of ~40 houses with 2 photos each is sourced.

## Files
| file | purpose |
|------|---------|
| `model.py` | ResNet18 + binary-code head (from `familiar-faces`) |
| `trans.py` | Rotate / Foveate / LogPolar / Pipeline transforms (from `familiar-faces`) |
| `salience_trans.py` | Gabor-saliency fixation pipeline |
| `datasets.py` | combined multi-category datasets, one global label space |
| `utils.py` | global label map + helpers |
| `download.py` | fetch raw data per category |
| `preprocess_fixations.py` | raw → packed images + saliency coords (fast path) |
| `preprocess.py` | raw → pre-rendered lp/cnn fixation crop PNGs (legacy path) |
| `train.py` | joint training |
| `run_experiments.py` | parallel multi-GPU experiment launcher |
| `simulate_yin1969.py` | per-category Yin/NIMBLE 2AFC simulation |
