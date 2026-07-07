# combined_lpnet

A single log-polar / foveated ResNet18 trained jointly on **faces + houses + objects**,
used to replicate Yin's (1969) face-inversion effect (and the same test for houses
and objects) via a Barrington-NIMBLE KDE memory model.

The model (`model.py`) is unchanged from `TheModel2.0` `familiar-faces` branch:
ResNet18 backbone → `fc1` (512→256) → temperature-scaled sigmoid → Bernoulli binary
code `h` → `fc2` classifier. The NIMBLE/Yin simulation operates on the shared 256-d
binary code `h`; the classifier head is only the training signal.

## Design decisions
- **Single unified softmax** over every class of every category (faces identities +
  object categories + house classes), one global label space.
- **Folder = class** (faces = person identities, ImageNet objects = object categories).
- **Use all available classes** per category (configurable via CLI).
- **Categories are configurable** — runs today with `faces objects houses`.
- **Fixations** are chosen by bottom-up **Gabor-variance saliency** (V1-like), which is
  category-agnostic — the same fixation mechanism is used for all three categories.
- All raw images are resized + center-cropped to a square `--input-size` (default 224)
  so every category is fed to the pipeline on equal footing.

## Environment setup (conda)

Requires a CUDA-capable GPU for practical training speed (CPU works for testing).
This recipe matches the CUDA 12.1 build used elsewhere in the project.

```bash
# 1. create and activate the environment
conda create -y -n lpnet python=3.10
conda activate lpnet

# 2. install torch/torchvision against CUDA 12.1 wheels
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1

# 3. install the rest of the pinned dependencies
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch, cv2; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

To remove the environment later: `conda env remove -n lpnet`.

## Pipeline

All commands below assume `conda activate lpnet`.

### 1. Download raw data

```bash
# faces + objects (the two categories currently sourced)
python download.py faces objects

# or a single category
python download.py faces
```

### 2. Preprocess into fixation crops

Produces both log-polar (`lp`) and plain-crop (`cnn`) variants under
`processed_data/<category>/<lp|cnn>/<split>/<class>/`.
`train` gets random-rotation augmentation, `valid` is upright, `test` is
inverted (180°) — `valid`/`test` are what the Yin simulation compares.

```bash
python preprocess.py \
    --categories faces objects \
    --num-fixations 32 \
    --input-size 224
```

Useful flags:
- `--categories <names>` — which categories to preprocess (default: `faces objects`)
- `--num-fixations N` — fixations per base image (default 32)
- `--input-size N` — square resize/crop applied before fixation sampling (default 224)
- `--splits train valid test` — restrict to specific splits
- `--limit-per-class N` — cap base images per class, useful for a quick dry run
- `--device cuda:0|cpu` — defaults to GPU if available

### 3. Train the single network

The learning rate (and everything else) is a CLI flag:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --categories faces objects \
    --variant lp \
    --lr 1e-3 \
    --epochs 50 \
    --batch-size 256 \
    --num-fixations 32
```

Key flags:
- `--lr` — learning rate (required tunable; no fixed default to rely on blindly)
- `--categories` — which categories to train jointly, e.g. `faces objects` or `faces objects houses`
- `--variant lp|cnn` — log-polar/foveated vs. plain-crop control condition
- `--epochs`, `--batch-size`, `--num-fixations`, `--patience`, `--temperature`
- `--pretrained-path PATH` — warm-start from an existing checkpoint (loaded `strict=False`)
- `--output-dir DIR` — override the auto-named `runs/...` directory

Each run writes `best_model.pth`, a final checkpoint, `label_map.json` (the
global class→index mapping needed by the Yin simulation), a training-history
CSV, and an accuracy plot into its output directory.

### 4. Run the Yin (1969) simulation per category

```bash
python simulate_yin1969.py --category faces --variant lp \
    --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
    --label-map  runs/faces_objects_lp_32fix_lr0.001/label_map.json

python simulate_yin1969.py --category objects --variant lp \
    --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
    --label-map  runs/faces_objects_lp_32fix_lr0.001/label_map.json

python simulate_yin1969.py --category houses --variant lp \
    --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
    --label-map  runs/faces_objects_lp_32fix_lr0.001/label_map.json
```

Key flags: `--num-study`, `--num-test`, `--study-fixations`, `--test-fixations`,
`--sigma` (NIMBLE kernel bandwidth), `--calib-target` (upright-upright accuracy
to calibrate noise against).

### 5. Run the Thatcher Effect experiment (currently only supports category faces)

```bash
# preprocess data for the experiment
python thatcher_download.py --source data/faces_cleaned \
    --dest data/thatcher_data/faces \
    --split valid

# run experiment
python simulate_thatcher.py --category faces --variant lp \
    --checkpoint runs/faces_objects_lp_32fix_lr0.001/best_model.pth \
    --label-map  runs/faces_objects_lp_32fix_lr0.001/label_map.json
```

## Files
| file | purpose |
|------|---------|
| `model.py` | ResNet18 + binary-code head (verbatim from `familiar-faces`) |
| `trans.py` | Rotate / Foveate / LogPolar / Pipeline transforms (verbatim) |
| `salience_trans.py` | Gabor-saliency fixation pipeline (MediaPipe code removed) |
| `datasets.py` | combined multi-category datasets, one global label space |
| `utils.py` | global label map + helpers |
| `download.py` | fetch raw data per category |
| `preprocess.py` | raw → lp/cnn fixation crops |
| `train.py` | joint training (`--lr`, `--categories`, `--variant`, …) |
| `simulate_yin1969.py` | per-category Yin/NIMBLE 2AFC simulation |
| `simulate_thatcher.py` | tests whether Thatcher effect emerges from Model 2.0 training |
| `thatcherize.py` | applies the thatcher effect to an upright, non-log polarized image of a face |
| `thatcher_download.py` | processes a raw dataset of faces (could do non-faces in the future if a category-agnostic "thatcherization" model is implemented - unlikely we do this) and formats it for the simulate_thatcher experiment script |

## Adding houses
1. Dataset added - first run of `preprocess.py` and `train.py` with `--categories faces houses objects` pending.
