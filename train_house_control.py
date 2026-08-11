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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Model, BACKBONES
from datasets import make_datasets, make_packed_datasets
from salience_trans import OnTheFlyTransform

# On DHONI, faces/objects/houses are already downloaded + preprocessed here.
# If no local ./processed_data exists, fall back to the shared copy instead
# of forcing a fresh (slow, disk-heavy) download.py + preprocess.py run.
DHONI_PROCESSED_ROOT = "/home/d1deutsch/Guru-Model2.0/processed_data"
DHONI_FIXATION_ROOT = "/home/d1deutsch/Guru-Model2.0/fixation_data" # contains 40 house data finally


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
                    help="classes per category active at each stage; 'all' = every class. "
                         "A category with fewer classes is simply capped at its own count.")
    ap.add_argument("--curriculum-epochs", nargs="+", type=int,
                    default=[6, 6, 6, 6, 8, 10, 18],
                    help="epochs to spend in each stage (same length as --curriculum-stages). "
                         "Their sum overrides --epochs.")
    ap.add_argument("--curriculum-seed", type=int, default=0,
                    help="seed for the nested random class ordering (which classes come first)")
    ap.add_argument("--curriculum-warmup-steps", type=int, default=200,
                    help="linear LR warm-up over this many steps after each class introduction, "
                         "to absorb the new-class loss spike (0 = off)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout applied to the binary code h before fc2 (0 = off)")
    ap.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True,
                    help="ImageNet-recipe train augmentation (random resized crop, "
                         "color jitter, random erasing); packed data mode only. "
                         "--no-aug to disable")
    ap.add_argument("--hflip-p", type=float, default=0.5,
                    help="probability of a random horizontal flip at train time "
                         "(0 = off; never applied vertically, which would fake inversion)")
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
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-tag", default=None,
                    help="suffix appended to the auto-named output dir (used by run_experiments.py)")
    ap.add_argument("--device", default="auto",
                    help="cuda:N | cpu | auto (auto picks the CUDA device with the most free memory)")

    # house training fine-tuning arguments
    ap.add_argument("--houses-delay-epochs", type=int, default=2,
                    help="number of initial epochs to train without houses; "
                         "houses are introduced afterward and remain active")

    ap.add_argument("--category-ratios", nargs="+",
                    default=["faces=0.40", "objects=0.40", "houses=0.20"],
                    help="target training sampling ratios by category, e.g. "
                         "faces=0.40 objects=0.40 houses=0.20; ratios are "
                         "renormalized over currently active categories")

    return ap.parse_args()

def parse_category_ratios(specs):
    """Parse category=ratio arguments into a dict."""
    ratios = {}

    for spec in specs:
        category, sep, value = spec.partition("=")
        if not sep or not category:
            raise ValueError(
                f"--category-ratios expects category=ratio, got {spec!r}"
            )

        value = float(value)
        if value < 0:
            raise ValueError(
                f"Category ratio cannot be negative: {spec!r}"
            )

        ratios[category] = value

    if not ratios or sum(ratios.values()) <= 0:
        raise ValueError("At least one positive category ratio is required.")

    return ratios

def make_category_sampler(dataset, active_ids, id_to_category, category_ratios):
    """Weighted sampler giving approximately the requested category mix.

    Sampling is with replacement, so category proportions are controlled by
    optimization steps rather than by the raw number of available images.
    """
    active_ids = set(active_ids)

    active_categories = {
        id_to_category[i]
        for i in active_ids
        if i < len(id_to_category)
    }

    # Keep only categories that are currently active.
    active_ratios = {
        cat: category_ratios.get(cat, 0.0)
        for cat in active_categories
    }

    total_ratio = sum(active_ratios.values())
    if total_ratio <= 0:
        raise ValueError(
            f"No positive sampling ratio for active categories: "
            f"{sorted(active_categories)}"
        )

    # Renormalize. Before houses are introduced, for example,
    # faces=.4 objects=.4 becomes faces=.5 objects=.5.
    active_ratios = {
        cat: ratio / total_ratio
        for cat, ratio in active_ratios.items()
    }

    # Number of eligible examples per category.
    category_counts = {cat: 0 for cat in active_categories}

    for sample in dataset.samples:
        label_id = sample[-1]
        if label_id in active_ids:
            cat = id_to_category[label_id]
            category_counts[cat] += 1

    # Each individual example gets:
    #
    #   desired_category_probability / number_of_examples_in_category
    #
    # so the total probability mass assigned to a category is the requested
    # category ratio.
    weights = torch.zeros(len(dataset), dtype=torch.double)

    for idx, sample in enumerate(dataset.samples):
        label_id = sample[-1]

        if label_id not in active_ids:
            continue

        cat = id_to_category[label_id]
        count = category_counts[cat]

        if count > 0:
            weights[idx] = active_ratios[cat] / count

    # Keep the same approximate number of optimization examples per epoch
    # as the ordinary training loader.
    num_samples = sum(category_counts.values())

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )

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


def build_curriculum_stages(args, label_map, categories):
    """Nested class subsets, one per stage.

    Each category gets a single seeded random ordering of its classes; stage k
    activates the first `size_k` of that ordering, so stage k's classes are
    always a subset of stage k+1's ("learn a few at a time", never forgetting).
    A category with fewer classes than `size_k` is capped at its own count, so
    e.g. objects (64 classes) saturates while houses keeps growing.
    """
    if len(args.curriculum_stages) != len(args.curriculum_epochs):
        raise ValueError(f"--curriculum-stages has {len(args.curriculum_stages)} entries but "
                         f"--curriculum-epochs has {len(args.curriculum_epochs)}")

    ids_by_category = {c: [] for c in categories}
    for name, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        ids_by_category[name.split("/", 1)[0]].append(idx)

    rng = np.random.default_rng(args.curriculum_seed)
    order = {c: rng.permutation(ids).tolist() for c, ids in ids_by_category.items()}

    stages = []
    for spec, epochs in zip(args.curriculum_stages, args.curriculum_epochs):
        size = None if str(spec).lower() == "all" else int(spec)
        active, per_cat = [], {}
        for c in categories:
            take = order[c] if size is None else order[c][:size]
            per_cat[c] = len(take)
            active.extend(take)
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
    category_ratios = parse_category_ratios(args.category_ratios)
    args.processed_root = resolve_root(args.processed_root, "processed_data", DHONI_PROCESSED_ROOT)
    args.fixation_root = resolve_root(args.fixation_root, "fixation_data", DHONI_FIXATION_ROOT)
    if args.data_mode == "auto":
        args.data_mode = "packed" if os.path.isdir(args.fixation_root) else "png"
        print(f"[info] --data-mode auto -> {args.data_mode}")
    if args.device == "auto":
        args.device = pick_free_device()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # fixed 180x180 input -> autotune kernels
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    max_images_per_class = {}
    for spec in args.max_images_per_class:
        category, _, n = spec.partition("=")
        if not n:
            raise ValueError(f"--max-images-per-class expects category=N pairs, got {spec!r}")
        max_images_per_class[category] = int(n)
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
        datasets, label_map = make_packed_datasets(
            categories=args.categories,
            packed_root=args.fixation_root,
            num_salient_points=args.num_fixations,
            max_images_per_class=max_images_per_class,
        )
        transforms = {
            "train": OnTheFlyTransform("train", args.variant, device, hflip_p=args.hflip_p,
                                       imagenet_aug=args.aug).to(device),
            "valid": OnTheFlyTransform("valid", args.variant, device).to(device),
            "test": OnTheFlyTransform("test", args.variant, device).to(device),
        }
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
        print(f"Curriculum: {len(stages)} stages, {args.epochs} epochs total")
        for k, s in enumerate(stages, 1):
            per_cat = " ".join(f"{c} {n}" for c, n in s["classes_per_category"].items())
            print(f"  stage {k}: {s['spec']:>4} per category -> {len(s['active_ids']):4d} classes "
                  f"({per_cat}) | {s['epochs']} epochs")
    else:
        stages = [{"spec": "all", "epochs": args.epochs,
                   "classes_per_category": {}, "active_ids": list(range(num_classes))}]

    valid_batch_size = max(1, args.batch_size // args.num_fixations)

    def make_loaders(active_ids):
        """Create loaders for the currently active developmental classes."""

        full = len(active_ids) == num_classes

        sampler = make_category_sampler(
            datasets["train"],
            active_ids,
            id_to_category,
            category_ratios,
        )

        # Important: sampler indices refer to the ORIGINAL dataset, so we
        # cannot combine this sampler with a Subset. Instead, use the full
        # dataset and give inactive classes zero sampling weight.
        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        # Validation/test remain ordinary deterministic loaders.
        def split_of(name):
            ds = datasets[name]
            return ds if full else Subset(
                ds,
                subset_indices(ds, active_ids)
            )

        valid_loader = DataLoader(
            split_of("valid"),
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            split_of("test"),
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        return train_loader, valid_loader, test_loader

    # ------------------------- House delay -------------------------

    if "houses" in args.categories and args.houses_delay_epochs > 0:
        house_delay = args.houses_delay_epochs

        delayed_stages = []

        for stage in stages:
            if house_delay <= 0:
                delayed_stages.append(stage)
                continue

            n_pre = min(stage["epochs"], house_delay)

            non_house_ids = [
                i for i in stage["active_ids"]
                if id_to_category[i] != "houses"
            ]

            if n_pre > 0:
                delayed_stages.append({
                    **stage,
                    "epochs": n_pre,
                    "spec": f"{stage['spec']}-prehouse",
                    "active_ids": non_house_ids,
                    "classes_per_category": {
                        c: n
                        for c, n in stage["classes_per_category"].items()
                        if c != "houses"
                    },
                })

            house_delay -= n_pre

            remaining = stage["epochs"] - n_pre

            if remaining > 0:
                delayed_stages.append({
                    **stage,
                    "epochs": remaining,
                })

        stages = delayed_stages
        args.epochs = sum(s["epochs"] for s in stages)

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

        for _ in range(stage["epochs"]):
            epoch += 1
            model.train()
            correct = total = 0
            epoch_losses = []
            steps_per_epoch = max(len(train_loader), 1)

            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}/{args.epochs}"
                     + (f" [stage {stage_idx}]" if args.curriculum else ""),
                unit="batch"
            )

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
                pbar.update(1)
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

    _, full_valid_loader, full_test_loader = make_loaders(list(range(num_classes)))

    _, _, v_corr, v_tot = evaluate(
        model, full_valid_loader, device, num_classes,
        transforms["valid"],
        amp=use_amp,
        channels_last=args.channels_last,
    )

    _, _, t_corr, t_tot = evaluate(
        model, full_test_loader, device, num_classes,
        transforms["test"],
        amp=use_amp,
        channels_last=args.channels_last,
    )

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