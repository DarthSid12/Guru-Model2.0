"""
train.py

Train a single network (model.py, unchanged from the familiar-faces branch) on
the combined faces + houses + objects data under one unified softmax head.

Two data modes:
  - packed (default when ./fixation_data exists): raw images + precomputed
    Gabor-saliency fixation coords (preprocess_fixations.py); crops are cut on
    the CPU and rotated/foveated/log-polar-transformed on the GPU on the fly.
    ~10 GB-scale sequential I/O per epoch instead of millions of PNG reads.
  - png: legacy pre-rendered crops from preprocess.py under ./processed_data.

Example:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --categories faces objects houses \
        --lr 1e-3 --epochs 50 --variant lp

Each run writes into its output dir:
    config.json      the exact input configuration of the run
    summary.json     concise input + output (best/final metrics, wall time)
    best_model.pth / final_model_<ts>.pth / label_map.json /
    training_history_<ts>.csv / accuracy.png

The classifier head (fc2) is just the training signal; the Yin/NIMBLE
simulation operates on the shared 256-d binary code `h` from fc1.
"""

import argparse
import datetime
import json
import os
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Model, BACKBONES
from datasets import build_packed_label_map, make_datasets, make_packed_datasets
from salience_trans import OnTheFlyTransform

# On DHONI, faces/objects/houses are already downloaded + preprocessed here.
# If no local ./processed_data exists, fall back to the shared copy instead
# of forcing a fresh (slow, disk-heavy) download.py + preprocess.py run.
DHONI_PROCESSED_ROOT = "/home/siagrawal/combined_lpnet/processed_data"
DHONI_FIXATION_ROOT = "/home/siagrawal/combined_lpnet/fixation_data"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects"],
                    help="categories to train on jointly")
    ap.add_argument("--variant", choices=["lp", "cnn", "plain"], default="lp",
                    help="lp = log-polar/foveated, cnn = plain crop")
    ap.add_argument("--data-mode", choices=["auto", "packed", "png"], default="auto",
                    help="packed = fixation_data (fast, on-the-fly transforms); "
                         "png = legacy pre-rendered processed_data crops; "
                         "auto = packed if fixation_data exists, else png")
    ap.add_argument("--fixation-root", default=None,
                    help="packed data root (default: ./fixation_data, falling back to "
                         f"{DHONI_FIXATION_ROOT} on DHONI)")
    ap.add_argument("--processed-root", default=None,
                    help="png-mode data root (default: ./processed_data, falling back to "
                         f"{DHONI_PROCESSED_ROOT} on DHONI)")
    ap.add_argument("--num-fixations", type=int, default=16)
    ap.add_argument("--max-images-per-class", nargs="+", default=[],
                    help="optional per-category cap on training base images, e.g. "
                         "--max-images-per-class objects=200 (train split only; "
                         "valid/test are unaffected)")
    ap.add_argument("--max-classes-per-category", nargs="+", default=[],
                    help="optional per-category cap on the number of classes, e.g. "
                         "--max-classes-per-category houses_zubud=40. The first N classes "
                         "in the packed meta order are kept and the rest are dropped from "
                         "the label map and from every split, so the softmax head is sized "
                         "to the classes that are actually trainable.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="cosine = CosineAnnealingLR decaying to 0 over --epochs")
    ap.add_argument("--weight-decay", type=float, default=5e-2,
                    help="AdamW weight decay, applied to conv/linear weights only "
                         "(biases and norm params are excluded); 0 = plain Adam behaviour")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast for model forward/backward (--no-amp to disable)")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="channels_last memory format for model + inputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--curriculum", action="store_true",
                    help="sequential ('learn a few at a time') training: start from a small "
                         "nested subset of classes per category and double it stage by stage "
                         "until all classes are active. Packed data mode only.")
    ap.add_argument("--curriculum-stages", nargs="+",
                    default=["4", "8", "16", "32", "64", "128", "all"],
                    help="classes active at each stage. Either one number applied to every "
                         "category ('4', or 'all' = every class), or per-category counts "
                         "'faces=4,objects=4,houses_zubud=0' (every --categories entry must "
                         "appear; each value may itself be 'all'). A category with fewer "
                         "classes than requested is simply capped at its own count.")
    ap.add_argument("--curriculum-epochs", nargs="+", type=int,
                    default=[6, 6, 6, 6, 8, 10, 18],
                    help="epochs to spend in each stage (same length as --curriculum-stages). "
                         "Their sum overrides --epochs.")
    ap.add_argument("--category-weights", nargs="+", default=[],
                    help="target share of each training batch per category, e.g. "
                         "--category-weights faces=0.45 objects=0.45 houses_zubud=0.10. "
                         "Values are normalised, and renormalised over whichever categories "
                         "are active in the current curriculum stage. Without this the diet "
                         "follows natural frequency (objects ~1026 img/class vs faces ~130 "
                         "vs zubud ~3, i.e. ~90%% objects). Epoch length is unchanged, so "
                         "compute stays comparable to an unweighted run.")
    ap.add_argument("--acuity-sigmas", nargs="+", type=float, default=[],
                    help="Gaussian blur sigma (in crop pixels) per curriculum stage, "
                         "standing in for low neonatal visual acuity relaxed over "
                         "development (Vogelsang et al. 2018 PNAS): "
                         "--acuity-sigmas 8 4 2 1 0 0. Train-time only -- valid/test "
                         "are always run at full acuity. Must have one entry per "
                         "stage; omit the flag for a full-acuity run.")
    ap.add_argument("--curriculum-seed", type=int, default=0,
                    help="seed for the nested random class ordering (which classes come first)")
    ap.add_argument("--curriculum-warmup-steps", type=int, default=200,
                    help="linear LR warm-up over this many steps after each class introduction, "
                         "to absorb the new-class loss spike (0 = off)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dataloader-sharing", choices=["auto", "file_descriptor", "file_system"],
                    default="auto",
                    help="how DataLoader workers hand batches to the parent. auto = the torch "
                         "default unless /dev/shm is nearly full; file_system = never use "
                         "/dev/shm (pick this on a box where other users' jobs can fill it "
                         "mid-run, which has killed runs here before)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout applied to the binary code h before fc2 (0 = off)")
    ap.add_argument("--invert-p", type=float, default=0.0,
                    help="fraction of TRAIN crops shown upside down (per-sample, "
                         "composed with the normal rotation jitter). Default 0.0 "
                         "reproduces every model up to r17, which never saw an "
                         "inverted view in training. Recorded in config.json.")
    ap.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True,
                    help="ImageNet-recipe train augmentation (random resized crop, "
                         "color jitter); packed data mode only. --no-aug to disable. "
                         "Random erasing and the horizontal flip were removed 2026-09-04.")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="resnet18",
                    choices=list(BACKBONES),
                    help="conv feature extractor (small->large): mobilenet_v3_small, "
                         "resnet18 (default), resnet34, resnet50, convnext_tiny")
    ap.add_argument("--pretrained", action="store_true",
                    help="initialise the backbone from ImageNet weights. NOTE: breaks the "
                         "'purely log-polar trained' assumption; diagnostic use only.")
    ap.add_argument("--pretrained-path", default=None,
                    help="optional checkpoint to warm-start from (loaded strict=False)")
    ap.add_argument("--pretrained-label-map", default=None,
                    help="label_map.json of the --pretrained-path run. When the new class "
                         "set is a superset of the old one, the old fc2 rows are copied "
                         "into the resized head BY CLASS NAME and only genuinely-new rows "
                         "are randomly initialised, so known classes keep their classifier "
                         "instead of having to relearn it.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-tag", default=None,
                    help="suffix appended to the auto-named output dir (used by run_experiments.py)")
    ap.add_argument("--device", default="auto",
                    help="cuda:N | cpu | auto (auto picks the CUDA device with the most free memory)")
    return ap.parse_args()


def use_file_system_sharing_if_shm_full(min_free_gb=4.0):
    """DataLoader workers hand collated batches to the parent through /dev/shm.

    DHONI's /dev/shm is a shared 252 GB tmpfs, and other users' stale joblib
    memmap folders have filled it to 100% before now, which kills a run mid-epoch
    with "No space left on device". Torch's 'file_system' strategy passes the
    same tensors through TMPDIR-backed files instead, so a full /dev/shm that we
    do not own cannot take the run down. Only switch when we have to: the default
    'file_descriptor' strategy is the one that cleans up after a hard crash.
    """
    try:
        st = os.statvfs("/dev/shm")
        free_gb = st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return
    if free_gb < min_free_gb:
        torch.multiprocessing.set_sharing_strategy("file_system")
        print(f"[warn] /dev/shm has only {free_gb:.1f} GB free; using the 'file_system' "
              f"sharing strategy (TMPDIR={os.environ.get('TMPDIR', '/tmp')}) so DataLoader "
              f"workers do not depend on it.")


def pick_free_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True)
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
        # nvidia-smi enumerates physical GPUs; respect CUDA_VISIBLE_DEVICES if set
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            allowed = [int(v) for v in visible.split(",") if v.strip() != ""]
            rows = [(allowed.index(i), m) for i, m in rows if i in allowed]
        idx = min(rows, key=lambda r: r[1])[0]
        return f"cuda:{idx}"
    except Exception:
        return "cuda:0"


@torch.no_grad()
def evaluate(model, loader, device, num_classes, transform=None, amp=False, channels_last=False,
             active_mask=None):
    """Sum per-fixation logits over a base image's fixations, then argmax.

    Returns (mean_batch_acc, std_batch_acc, correct_per_class, total_per_class),
    the last two as LongTensors of shape [num_classes] for error analysis.

    `active_mask` (bool [num_classes]) restricts the decision to the classes the
    model has been introduced to so far: a class it has never seen cannot be
    predicted. It is all-True (a no-op) outside curriculum training and in the
    final curriculum stage.
    """
    model.eval()
    accs = []
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels
        B, n, C, H, W = inputs.shape
        inputs = inputs.reshape(-1, C, H, W)
        if transform is not None:
            inputs = transform(inputs)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(inputs)
        logits = logits.float().reshape(B, n, -1).sum(dim=1)
        if active_mask is not None:
            logits = logits.masked_fill(~active_mask, float("-inf"))
        preds = logits.argmax(dim=1)
        hits = preds == label_ids
        accs.append(hits.float().mean().item())
        lids = label_ids.cpu()
        total_per_class += torch.bincount(lids, minlength=num_classes)
        correct_per_class += torch.bincount(lids[hits.cpu()], minlength=num_classes)
    return (float(np.mean(accs)), float(np.std(accs)),
            correct_per_class, total_per_class)


def per_category_accuracy(correct, total, id_to_category):
    """Aggregate per-class correct/total counts up to category level."""
    agg = {}
    for i, cat in enumerate(id_to_category):
        c, t = agg.get(cat, (0, 0))
        agg[cat] = (c + int(correct[i]), t + int(total[i]))
    return {cat: (c / t if t else float("nan")) for cat, (c, t) in agg.items()}


def resolve_root(explicit, local_default, dhoni_fallback):
    if explicit is not None:
        return explicit
    if not os.path.isdir(local_default) and "dhoni" in socket.gethostname().lower() \
            and os.path.isdir(dhoni_fallback):
        print(f"[info] No local ./{local_default} found; using shared DHONI data at {dhoni_fallback}.")
        return dhoni_fallback
    return local_default


def parse_stage_spec(spec, categories):
    """One --curriculum-stages entry -> {category: size or None}, None meaning 'all'.

    Accepts a scalar applied to every category ("4", "all") or explicit
    per-category counts ("faces=4,objects=4,houses_zubud=0"), which is what a
    developmental schedule needs: faces, objects and places do not grow at the
    same rate, and places start at zero.
    """
    spec = str(spec)
    if "=" not in spec:
        size = None if spec.lower() == "all" else int(spec)
        return {c: size for c in categories}

    sizes = {}
    for part in spec.split(","):
        cat, _, n = part.partition("=")
        cat, n = cat.strip(), n.strip()
        if not n:
            raise ValueError(f"stage spec {spec!r}: expected category=N, got {part!r}")
        if cat not in categories:
            raise ValueError(f"stage spec {spec!r}: unknown category {cat!r} "
                             f"(--categories is {categories})")
        sizes[cat] = None if n.lower() == "all" else int(n)
    missing = [c for c in categories if c not in sizes]
    if missing:
        raise ValueError(f"stage spec {spec!r} does not mention {missing}; per-category "
                         f"specs must list every category (use 0 to keep one out of a stage)")
    return sizes


def build_curriculum_stages(args, label_map, categories):
    """Nested class subsets, one per stage.

    Each category gets a single seeded random ordering of its classes; stage k
    activates the first `size_k` of that ordering, so stage k's classes are
    always a subset of stage k+1's ("learn a few at a time", never forgetting).
    A category with fewer classes than `size_k` is capped at its own count, so
    e.g. objects (64 classes) saturates while faces keep growing.
    """
    if len(args.curriculum_stages) != len(args.curriculum_epochs):
        raise ValueError(f"--curriculum-stages has {len(args.curriculum_stages)} entries but "
                         f"--curriculum-epochs has {len(args.curriculum_epochs)}")

    ids_by_category = {c: [] for c in categories}
    for name, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        ids_by_category[name.split("/", 1)[0]].append(idx)

    rng = np.random.default_rng(args.curriculum_seed)
    order = {c: rng.permutation(ids).tolist() for c, ids in ids_by_category.items()}

    stages, prev = [], {c: 0 for c in categories}
    for spec, epochs in zip(args.curriculum_stages, args.curriculum_epochs):
        sizes = parse_stage_spec(spec, categories)
        active, per_cat = [], {}
        for c in categories:
            size = sizes[c]
            take = order[c] if size is None else order[c][:size]
            if len(take) < prev[c]:
                raise ValueError(f"stage spec {spec!r} shrinks {c} from {prev[c]} to "
                                 f"{len(take)} classes; stages must be nested (never forget)")
            per_cat[c] = len(take)
            active.extend(take)
        if not active:
            raise ValueError(f"stage spec {spec!r} activates no classes at all")
        prev = per_cat
        stages.append({"spec": str(spec), "epochs": int(epochs),
                       "classes_per_category": per_cat, "active_ids": sorted(active)})
    return stages


def subset_indices(dataset, active_ids):
    """Positions in dataset.samples whose global label is currently active.

    Both packed datasets store the global label last in each sample tuple
    (train: (split, image, fixation, label); eval: (split, image, label)).
    """
    active = set(active_ids)
    return [i for i, s in enumerate(dataset.samples) if s[-1] in active]


def curriculum_lr(base_lr, epoch_frac, warmup_frac):
    """Global cosine over the whole run, times a linear per-stage warm-up factor.

    The cosine spans every stage rather than restarting per stage: a per-stage
    schedule would drive the LR to zero five times over and freeze the features
    before the hard, many-class stages ever start.
    """
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * min(max(epoch_frac, 0.0), 1.0))) * warmup_frac


def main():
    args = parse_args()
    args.processed_root = resolve_root(args.processed_root, "processed_data", DHONI_PROCESSED_ROOT)
    args.fixation_root = resolve_root(args.fixation_root, "fixation_data", DHONI_FIXATION_ROOT)
    if args.data_mode == "auto":
        args.data_mode = "packed" if os.path.isdir(args.fixation_root) else "png"
        print(f"[info] --data-mode auto -> {args.data_mode}")
    if args.device == "auto":
        args.device = pick_free_device()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.num_workers > 0:
        if args.dataloader_sharing == "auto":
            use_file_system_sharing_if_shm_full()
        else:
            torch.multiprocessing.set_sharing_strategy(args.dataloader_sharing)
            print(f"[info] DataLoader sharing strategy: {args.dataloader_sharing}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed 180x180 input -> autotune kernels
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    def parse_category_caps(specs, flag):
        caps = {}
        for spec in specs:
            category, _, n = spec.partition("=")
            if not n:
                raise ValueError(f"{flag} expects category=N pairs, got {spec!r}")
            if category not in args.categories:
                raise ValueError(f"{flag}: unknown category {category!r} "
                                 f"(--categories is {args.categories})")
            caps[category] = int(n)
        return caps

    max_images_per_class = parse_category_caps(args.max_images_per_class,
                                               "--max-images-per-class")
    max_classes_per_category = parse_category_caps(args.max_classes_per_category,
                                                   "--max-classes-per-category")

    category_weights = {}
    for spec in args.category_weights:
        category, _, w = spec.partition("=")
        if not w:
            raise ValueError(f"--category-weights expects category=W pairs, got {spec!r}")
        if category not in args.categories:
            raise ValueError(f"--category-weights: unknown category {category!r} "
                             f"(--categories is {args.categories})")
        if float(w) < 0:
            raise ValueError(f"--category-weights: negative weight in {spec!r}")
        category_weights[category] = float(w)
    if category_weights:
        total = sum(category_weights.values())
        if total <= 0:
            raise ValueError("--category-weights must not sum to zero")
        category_weights = {c: w / total for c, w in category_weights.items()}
    print("Device:", device, "| categories:", args.categories, "| variant:", args.variant,
          "| data-mode:", args.data_mode, "| max_images_per_class:", max_images_per_class or "none")

    out_dir = args.output_dir or (
        f"runs/{'_'.join(args.categories)}_{args.variant}_"
        f"{args.num_fixations}fix_lr{args.lr}_{args.backbone}"
        + ("_pt" if args.pretrained else "")
        + ("_" + "_".join(f"{c}{n}img" for c, n in sorted(max_images_per_class.items()))
           if max_images_per_class else "")
        + (f"_{args.run_tag}" if args.run_tag else "")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "hostname": socket.gethostname(),
                   "started": datetime.datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)

    # ----------------------------- Data -----------------------------
    if args.data_mode == "packed":
        capped_map = None
        if max_classes_per_category:
            # Trim the label map first, then re-index it: the packed datasets skip
            # any sample whose "<category>/<class>" key is absent, so dropped
            # classes vanish from train/valid/test and from the softmax head.
            full = build_packed_label_map(args.fixation_root, args.categories, split="train")
            kept, seen = [], {}
            for name in sorted(full, key=full.get):
                category = name.split("/", 1)[0]
                cap = max_classes_per_category.get(category)
                seen[category] = seen.get(category, 0) + 1
                if cap is None or seen[category] <= cap:
                    kept.append(name)
            capped_map = {name: i for i, name in enumerate(kept)}
            for category, cap in sorted(max_classes_per_category.items()):
                print(f"[info] --max-classes-per-category {category}={cap}: "
                      f"{seen.get(category, 0)} -> {min(cap, seen.get(category, 0))} classes")
        datasets, label_map = make_packed_datasets(
            categories=args.categories,
            packed_root=args.fixation_root,
            num_salient_points=args.num_fixations,
            label_map=capped_map,
            max_images_per_class=max_images_per_class,
        )
        transforms = {
            "train": OnTheFlyTransform("train", args.variant, device,
                                       imagenet_aug=args.aug).to(device),
            "valid": OnTheFlyTransform("valid", args.variant, device).to(device),
            "test": OnTheFlyTransform("test", args.variant, device).to(device),
        }
        # Inverted-exposure arm. Only the TRAIN transform is touched: valid and
        # test must stay upright/inverted by definition or the eval columns stop
        # meaning anything.
        transforms["train"].invert_p = args.invert_p
        if args.invert_p > 0:
            print(f"[!] inverted-exposure run: {args.invert_p:.1%} of train crops "
                  f"rotated 180 deg before the +-15 deg jitter. This DELIBERATELY "
                  f"breaks the upright-only training invariant every model up to "
                  f"r17 relied on -- Yin rows from this run are not comparable to "
                  f"theirs except as the intended manipulation.")
    else:
        datasets, label_map = make_datasets(
            categories=args.categories,
            processed_root=args.processed_root,
            variant=args.variant,
            num_salient_points=args.num_fixations,
            max_images_per_class=max_images_per_class,
        )
        transforms = {"train": None, "valid": None, "test": None}

    num_classes = len(label_map)
    print(f"Unified label space: {num_classes} classes across {len(args.categories)} categories")
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # id -> "category/ClassName" and id -> "category", for error analysis
    id_to_name = [None] * num_classes
    for name, i in label_map.items():
        id_to_name[i] = name
    id_to_category = [n.split("/")[0] for n in id_to_name]

    # ------------------------- Curriculum stages ---------------------
    if args.curriculum:
        if args.data_mode != "packed":
            raise ValueError("--curriculum requires --data-mode packed")
        stages = build_curriculum_stages(args, label_map, args.categories)
        args.epochs = sum(s["epochs"] for s in stages)
        if args.acuity_sigmas and len(args.acuity_sigmas) != len(stages):
            raise ValueError(f"--acuity-sigmas has {len(args.acuity_sigmas)} entries but "
                             f"there are {len(stages)} curriculum stages")
        print(f"Curriculum: {len(stages)} stages, {args.epochs} epochs total")
        for k, s in enumerate(stages, 1):
            per_cat = ", ".join(f"{n} {c}" for c, n in s["classes_per_category"].items())
            print(f"  stage {k}: {len(s['active_ids']):4d} classes ({per_cat}) "
                  f"| {s['epochs']} epochs")
    else:
        if args.acuity_sigmas:
            raise ValueError("--acuity-sigmas is a per-stage schedule and needs --curriculum")
        stages = [{"spec": "all", "epochs": args.epochs,
                   "classes_per_category": {}, "active_ids": list(range(num_classes))}]

    valid_batch_size = max(1, args.batch_size // args.num_fixations)

    def category_sampler(train_split):
        """WeightedRandomSampler giving each category its requested batch share.

        A sample's weight is share_c / n_c, so a category's expected share of the
        batch is its target regardless of how many crops it actually owns. Shares
        are renormalised over the categories present in this stage (houses are
        absent from the first stages), and the number of draws per epoch equals
        the subset size, so weighting changes the *diet* and not the compute.
        """
        if not category_weights:
            return None
        base = train_split.dataset if isinstance(train_split, Subset) else train_split
        idxs = train_split.indices if isinstance(train_split, Subset) else range(len(base))
        cats = [id_to_category[base.samples[i][-1]] for i in idxs]
        n_by_cat = {}
        for c in cats:
            n_by_cat[c] = n_by_cat.get(c, 0) + 1
        present = sum(category_weights.get(c, 0.0) for c in n_by_cat)
        if present <= 0:
            raise ValueError(f"--category-weights gives zero total weight to the categories "
                             f"present in this stage ({sorted(n_by_cat)})")
        share = {c: category_weights.get(c, 0.0) / present for c in n_by_cat}
        print("  sampling shares: " + ", ".join(
            f"{c} {100 * share[c]:.1f}% (natural {100 * n_by_cat[c] / len(cats):.1f}%, "
            f"{n_by_cat[c]} crops)" for c in sorted(n_by_cat)))
        w = [share[c] / n_by_cat[c] for c in cats]
        return torch.utils.data.WeightedRandomSampler(w, num_samples=len(cats),
                                                      replacement=True)

    def make_loaders(active_ids):
        """Loaders restricted to the currently active classes (all of them
        outside curriculum mode, where the subsets are skipped entirely)."""
        full = len(active_ids) == num_classes
        def split_of(name):
            ds = datasets[name]
            return ds if full else Subset(ds, subset_indices(ds, active_ids))
        train_split = split_of("train")
        sampler = category_sampler(train_split)
        return (
            DataLoader(train_split, batch_size=args.batch_size,
                       shuffle=sampler is None, sampler=sampler,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=args.num_workers > 0),
            DataLoader(split_of("valid"), batch_size=valid_batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=True),
            DataLoader(split_of("test"), batch_size=valid_batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=True),
        )

    # ----------------------------- Model ----------------------------
    model = Model(size=180, num_classes=num_classes, pretrained=args.pretrained,
                  T=args.temperature, dropout=args.dropout, backbone=args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {args.backbone} (feat dim {model.in_size}, {n_params/1e6:.1f}M params)"
          + (" [ImageNet-pretrained]" if args.pretrained else " [from scratch]"))
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location=device)
        # Fine-tuning onto a different class set resizes fc2, and a shape
        # mismatch raises even under strict=False, so drop the old classifier
        # and let it re-initialise. Everything the simulations read is upstream
        # of it -- they score the 256-d bottleneck `h` (fc1), not fc2 -- so the
        # transferred part is exactly the part that matters.
        params = dict(model.named_parameters())
        mismatched = [k for k in state
                      if k.startswith("fc2.") and state[k].shape != params[k].shape]
        if mismatched and args.pretrained_label_map:
            # Re-initialising the whole head makes EVERY known class relearn its
            # classifier, and categories with few images per class (houses at ~3,
            # the new faces at 1-4) cannot do that before early stopping fires on
            # an aggregate valid accuracy dominated by the big categories -- that
            # is how r8ft lost its house arm (90.0% -> 17.5%). Copy the old rows
            # across by class NAME instead: the label map is reordered by the new
            # categories, so row index alone is not a valid correspondence.
            with open(args.pretrained_label_map) as f:
                old_map = json.load(f)
            new_w = params["fc2.weight"].data.clone()
            new_b = params["fc2.bias"].data.clone()
            kept = 0
            for name, old_i in old_map.items():
                new_i = label_map.get(name)
                if new_i is not None:
                    new_w[new_i] = state["fc2.weight"][old_i].to(new_w.device, new_w.dtype)
                    new_b[new_i] = state["fc2.bias"][old_i].to(new_b.device, new_b.dtype)
                    kept += 1
            state["fc2.weight"], state["fc2.bias"] = new_w, new_b
            print(f"  [warm-start] fc2 {tuple(params['fc2.weight'].shape)}: carried over "
                  f"{kept}/{len(old_map)} old class rows by name, "
                  f"{len(label_map) - kept} newly initialised")
        else:
            for k in mismatched:
                print(f"  [warm-start] dropping {k} {tuple(state[k].shape)} "
                      f"-> {tuple(params[k].shape)} (re-init)")
                del state[k]
        missing = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {args.pretrained_path} ({missing})")
    model.stochastic = False  # deterministic expectation during training

    # standard ImageNet practice: no weight decay on biases / norm params
    decay_params, no_decay_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr)
    # curriculum mode drives the LR by hand (global cosine + per-stage warm-up)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.lr_schedule == "cosine" and not args.curriculum else None)
    ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.amp and device.type == "cuda"

    best_val = -1.0
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(out_dir, "best_model.pth")
    history = []
    t_start = time.time()

    # ----------------------------- Train ----------------------------
    # One pass per curriculum stage (a single all-classes stage when
    # --curriculum is off, which reproduces the original loop exactly).
    epoch = 0
    for stage_idx, stage in enumerate(stages, 1):
        is_final_stage = stage_idx == len(stages)
        active_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        active_mask[torch.tensor(stage["active_ids"], device=device)] = True
        masked = not bool(active_mask.all())
        train_loader, valid_loader, test_loader = make_loaders(stage["active_ids"])
        # Accuracy is only comparable across stages once the class set stops
        # growing, so best_model.pth tracks the final stage; earlier stages get
        # their own best purely for the curves and for early stopping.
        patience_counter = 0
        stage_best = -1.0
        # warm up the LR after each class introduction (not at the very start,
        # where the cosine already begins from the full base LR)
        warmup_left = args.curriculum_warmup_steps if (args.curriculum and stage_idx > 1) else 0
        # Train-time acuity for this stage. valid/test transforms are never
        # touched: the model is always evaluated at full acuity.
        if args.acuity_sigmas:
            transforms["train"].set_acuity(args.acuity_sigmas[stage_idx - 1])
        if args.curriculum:
            per_cat = ", ".join(f"{n} {c}" for c, n in stage["classes_per_category"].items())
            print(f"\n=== Stage {stage_idx}/{len(stages)}: {len(stage['active_ids'])} classes "
                  f"({per_cat}), {stage['epochs']} epochs, "
                  f"{len(train_loader.dataset)} train crops"
                  + (f", acuity sigma {args.acuity_sigmas[stage_idx - 1]:g}px"
                     if args.acuity_sigmas else "") + " ===")

        for _ in range(stage["epochs"]):
            epoch += 1
            model.train()
            correct = total = 0
            epoch_losses = []
            steps_per_epoch = max(len(train_loader), 1)
            pbar = tqdm(total=len(train_loader.dataset),
                        desc=f"Epoch {epoch}/{args.epochs}"
                             + (f" [stage {stage_idx}]" if args.curriculum else ""), unit="img")
            for step, (inputs, labels) in enumerate(train_loader):
                if args.curriculum:
                    warmup_frac = 1.0
                    if warmup_left > 0:
                        n = max(args.curriculum_warmup_steps, 1)
                        warmup_frac = (n - warmup_left + 1) / n   # 1/n -> 1.0
                        warmup_left -= 1
                    lr_now = curriculum_lr(args.lr, (epoch - 1 + step / steps_per_epoch)
                                           / max(args.epochs, 1), warmup_frac) \
                        if args.lr_schedule == "cosine" else args.lr * warmup_frac
                    for g in optimizer.param_groups:
                        g["lr"] = lr_now

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if transforms["train"] is not None:
                    with torch.no_grad():
                        inputs = transforms["train"](inputs)
                if args.channels_last:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)
                label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(inputs)
                    # classes not yet introduced take no probability mass, so
                    # their fc2 rows stay untrained until their stage arrives
                    if masked:
                        logits = logits.masked_fill(~active_mask, -1e4)
                    loss = ce_criterion(logits, label_ids)
                loss.backward()
                optimizer.step()

                correct += (logits.argmax(dim=1) == label_ids).sum().item()
                total += label_ids.size(0)
                epoch_losses.append(loss.item())
                pbar.update(inputs.size(0))
                pbar.set_postfix(acc=f"{correct/total*100:.2f}%", ce=f"{loss.item():.3f}")
            pbar.close()

            train_acc = correct / max(total, 1)
            train_loss = float(np.mean(epoch_losses))
            eval_mask = active_mask if masked else None
            valid_acc, valid_std, v_corr, v_tot = evaluate(
                model, valid_loader, device, num_classes, transforms["valid"],
                amp=use_amp, channels_last=args.channels_last, active_mask=eval_mask)
            test_acc, test_std, t_corr, t_tot = evaluate(
                model, test_loader, device, num_classes, transforms["test"],
                amp=use_amp, channels_last=args.channels_last, active_mask=eval_mask)
            valid_by_cat = per_category_accuracy(v_corr, v_tot, id_to_category)
            test_by_cat = per_category_accuracy(t_corr, t_tot, id_to_category)
            cur_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            cat_str = " ".join(f"{c} {a*100:.1f}%" for c, a in sorted(valid_by_cat.items()))
            print(f"-> Epoch {epoch}: train {train_acc*100:.2f}% | "
                  f"valid {valid_acc*100:.2f}% | test(inv) {test_acc*100:.2f}% | "
                  f"loss {train_loss:.4f} | lr {cur_lr:.2e} | valid by cat: {cat_str}")

            history.append({"epoch": epoch, "lr": cur_lr,
                            "stage": stage_idx, "stage_spec": stage["spec"],
                            "active_classes": len(stage["active_ids"]),
                            "train_acc": train_acc, "train_loss": train_loss,
                            "valid_acc": valid_acc, "valid_std": valid_std,
                            "test_acc": test_acc, "test_std": test_std,
                            **{f"valid_acc_{c}": a for c, a in valid_by_cat.items()},
                            **{f"test_acc_{c}": a for c, a in test_by_cat.items()},
                            "elapsed_sec": round(time.time() - t_start, 1)})

            improved = valid_acc > stage_best
            if improved:
                stage_best = valid_acc
            if improved and is_final_stage:
                best_val = valid_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                print(f"   saved new best (valid {valid_acc*100:.2f}%)")
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"   early stopping {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered"
                          + (f" in stage {stage_idx}; advancing." if not is_final_stage else "."))
                    break

        if args.curriculum:
            # snapshot the end of every stage, so simulate_yin1969.py can be run
            # at each point in "development": the inversion effect as a function
            # of how many identities the model has learned. Without this only
            # the fully-trained model survives.
            stage_ckpt = os.path.join(out_dir,
                                      f"stage{stage_idx}_{len(stage['active_ids'])}cls.pth")
            torch.save(model.state_dict(), stage_ckpt)
            # the class subset is needed to score that checkpoint like-for-like
            with open(os.path.join(out_dir, f"stage{stage_idx}_active_ids.json"), "w") as f:
                json.dump({"stage": stage_idx, "spec": stage["spec"],
                           "classes_per_category": stage["classes_per_category"],
                           "active_ids": stage["active_ids"]}, f)
            print(f"   saved stage checkpoint {os.path.basename(stage_ckpt)}")

    # a curriculum run that early-stopped every final-stage epoch, or a stage
    # list ending before any save, still needs a checkpoint to analyse
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        best_val, best_epoch = history[-1]["valid_acc"], history[-1]["epoch"]

    # ----------------------------- Save -----------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_csv = os.path.join(out_dir, f"training_history_{ts}.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, f"final_model_{ts}.pth"))

    # ------------------- Error analysis (best ckpt) ------------------
    # Where is the accuracy going: specific classes, or a whole category?
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    _, _, v_corr, v_tot = evaluate(model, valid_loader, device, num_classes,
                                   transforms["valid"], amp=use_amp,
                                   channels_last=args.channels_last)
    _, _, t_corr, t_tot = evaluate(model, test_loader, device, num_classes,
                                   transforms["test"], amp=use_amp,
                                   channels_last=args.channels_last)
    per_class = pd.DataFrame({
        "class_id": range(num_classes),
        "class_name": id_to_name,
        "category": id_to_category,
        "valid_n": v_tot.tolist(),
        "valid_correct": v_corr.tolist(),
        "test_n": t_tot.tolist(),
        "test_correct": t_corr.tolist(),
    })
    per_class["valid_acc"] = per_class.valid_correct / per_class.valid_n.clip(lower=1)
    per_class["test_acc"] = per_class.test_correct / per_class.test_n.clip(lower=1)
    per_class = per_class.sort_values("valid_acc")
    per_class_csv = os.path.join(out_dir, "per_class_accuracy.csv")
    per_class.to_csv(per_class_csv, index=False)

    valid_by_cat = per_category_accuracy(v_corr, v_tot, id_to_category)
    test_by_cat = per_category_accuracy(t_corr, t_tot, id_to_category)
    print("\nPer-category accuracy (best checkpoint):")
    for cat in sorted(valid_by_cat):
        print(f"  {cat:<10} valid {valid_by_cat[cat]*100:5.1f}% | "
              f"test(inv) {test_by_cat[cat]*100:5.1f}%")
    print("\nWorst 15 classes by valid accuracy (full list in per_class_accuracy.csv):")
    for _, r in per_class.head(15).iterrows():
        print(f"  {r.class_name:<45} valid {r.valid_acc*100:5.1f}% (n={r.valid_n}) | "
              f"test(inv) {r.test_acc*100:5.1f}%")

    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="train")
    plt.plot(epochs, [h["valid_acc"] for h in history], label="valid (upright)")
    plt.plot(epochs, [h["test_acc"] for h in history], label="test (inverted)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    if args.curriculum:
        # dashed line at each class introduction, taken from the history so an
        # early-stopped stage still lands in the right place
        for prev, cur in zip(history, history[1:]):
            if cur["stage"] != prev["stage"]:
                plt.axvline(cur["epoch"] - 0.5, color="0.7", lw=0.8, ls="--")
                plt.text(cur["epoch"] - 0.5, 0.02, str(cur["active_classes"]),
                         fontsize=6, color="0.4", rotation=90)
        plt.title(f"Curriculum training ({len(stages)} stages)")
    else:
        plt.title("Training")
    plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()

    best_row = next(h for h in history if h["epoch"] == best_epoch) if history else {}
    summary = {
        "config": vars(args),
        "hostname": socket.gethostname(),
        "num_classes": num_classes,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "curriculum": [{"stage": i, "spec": s["spec"], "epochs": s["epochs"],
                        "num_classes": len(s["active_ids"]),
                        "classes_per_category": s["classes_per_category"]}
                       for i, s in enumerate(stages, 1)] if args.curriculum else None,
        "results": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_valid_acc": best_val,
            "test_acc_at_best_epoch": best_row.get("test_acc"),
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_valid_acc": history[-1]["valid_acc"] if history else None,
            "final_test_acc": history[-1]["test_acc"] if history else None,
            "valid_acc_by_category": valid_by_cat,
            "test_acc_by_category": test_by_cat,
            "wall_time_sec": round(time.time() - t_start, 1),
            "sec_per_epoch": round((time.time() - t_start) / max(len(history), 1), 1),
        },
        "artifacts": {
            "best_model": best_path,
            "history_csv": hist_csv,
            "label_map": os.path.join(out_dir, "label_map.json"),
            "per_class_csv": per_class_csv,
        },
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Best valid acc {best_val*100:.2f}% (epoch {best_epoch}). Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
