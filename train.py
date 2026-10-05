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
import math
import os
import random
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
    ap.add_argument("--random-fixations", action="store_true",
                    help="draw each training crop's fixation at random from the packed set "
                         "instead of always taking the first --num-fixations of them. Stores "
                         "hold 32 points but training uses 16, so without this the other 16 "
                         "are never seen by any model and every epoch re-presents an image "
                         "through exactly the same 16 apertures. Slot j draws from "
                         "{j, j+N, j+2N, ...}, so each image still contributes N DISTINCT "
                         "fixations per epoch and all 32 are reached over a run. Train split "
                         "only -- valid/test stay deterministic or their accuracy columns "
                         "stop being comparable across epochs.")
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
    ap.add_argument("--category-weighting", choices=["fixed", "sqrt"], default="fixed",
                    help="fixed = the hand-set --category-weights shares. sqrt = no hand-set "
                         "numbers: at every level of the hierarchy an item's sampling weight is "
                         "proportional to the square root of the number of items beneath it. "
                         "A domain's share of the batch is sqrt(its active classes) over the "
                         "stage total -- a single-class domain (the generic house category) is "
                         "counted by its active photos instead, since its ladder grows in "
                         "exemplars -- and within a domain each class is weighted by "
                         "sqrt(its photos). This is the same sqrt rule as the epoch schedule, "
                         "the geometric midpoint between equal shares per domain (which lets "
                         "four buildings take a quarter of every batch) and equal shares per "
                         "class (which lets the largest domain swamp the rest).")
    ap.add_argument("--weight-domains", nargs="+", default=[],
                    help="with --category-weighting sqrt: group categories into one domain, e.g. "
                         "--weight-domains faces=faces_vgg,faces,faces_rfwW,faces_rfwO. "
                         "Ungrouped categories are their own domain.")
    ap.add_argument("--curriculum-seed", type=int, default=0,
                    help="seed for the nested random class ordering (which classes come first)")
    ap.add_argument("--curriculum-image-caps", nargs="+", default=[],
                    help="per-stage cap on BASE IMAGES per class, one entry per stage, e.g. "
                         "--curriculum-image-caps houses=4 houses=8 houses=16. The generic "
                         "house category is a single class, so its ladder has to grow in "
                         "exemplars rather than classes (each generic house photo is a "
                         "different building, so this grows building variety too). Categories "
                         "left out of an entry are uncapped. Caps must not shrink between "
                         "stages, for the same never-forget reason class counts must not. "
                         "Applies to the train split only.")
    ap.add_argument("--curriculum-pin", nargs="+", default=[],
                    help="hand-picked classes that take the FRONT of a category's ordering, e.g. "
                         "--curriculum-pin faces_vgg=vgg_n000002,vgg_n000003, or =@file with one "
                         "name per line. Without this the "
                         "order is a --curriculum-seed permutation, so which identities the model "
                         "meets first is arbitrary; pin them when stage 1 is supposed to be a "
                         "specific set (the white-first face ladder). Unpinned classes keep their "
                         "seeded order behind the pins, so nesting is unaffected. A pinned name "
                         "that is not in the label map is an error, never a silent fallback.")
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="fixed optimizer steps per epoch (0 = one pass over the active subset). "
                         "An epoch is otherwise the subset size, so a 4-class stage gets ~30 steps "
                         "and the final stage ~8000: the early 'developmental' stages are then a "
                         "rounding error next to the last one. A fixed budget makes stages "
                         "compute-comparable and the cosine LR and --patience mean the same thing "
                         "at every stage. Sampling is with replacement.")
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
    ap.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True,
                    help="ImageNet-recipe train augmentation (random resized crop, "
                         "color jitter); packed data mode only. --no-aug to disable. "
                         "Random erasing and the horizontal flip were removed 2026-09-04.")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="vgg16_bn_aa5",
                    choices=list(BACKBONES),
                    help="convolutional backbone (default: vgg16_bn_aa5)")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from a full-state checkpoint. Bare --resume (or "
                         "'auto') picks up out_dir/checkpoint_last.pth if it exists "
                         "and starts fresh if it does not, so the same command can "
                         "be re-run after an eviction.")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="write checkpoint_last.pth every N epochs (0 disables).")
    ap.add_argument("--pretrained", action="store_true",
                    help="initialise the backbone from ImageNet weights. NOTE: breaks the "
                         "'purely log-polar trained' assumption; diagnostic use only.")
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


def parse_image_cap_spec(spec, categories):
    """One --curriculum-image-caps entry -> {category: n}; absent or 'all' = uncapped."""
    spec = str(spec).strip()
    if spec.lower() in ("", "none", "all"):
        return {}
    caps = {}
    for part in spec.split(","):
        cat, _, n = part.partition("=")
        cat, n = cat.strip(), n.strip()
        if not n:
            raise ValueError(f"--curriculum-image-caps {spec!r}: expected category=N, "
                             f"got {part!r}")
        if cat not in categories:
            raise ValueError(f"--curriculum-image-caps {spec!r}: unknown category {cat!r} "
                             f"(--categories is {categories})")
        # None = explicitly uncapped, which is NOT the same as absent: absent
        # after an earlier cap is a silent jump and is rejected below.
        caps[cat] = None if n.lower() == "all" else int(n)
    return caps


def parse_pin_spec(specs, label_map, categories):
    """--curriculum-pin entries -> {category: [global_id, ...]} in the order given.

    Names are the packed meta class names without the category prefix, i.e.
    `faces_vgg=vgg_n000002` looks up "faces_vgg/vgg_n000002". An unknown name
    raises: a pin that silently missed would hand stage 1 back to the random
    permutation, which is exactly the failure this flag exists to prevent.
    """
    pinned = {}
    for spec in specs:
        cat, _, names = str(spec).partition("=")
        cat, names = cat.strip(), names.strip()
        if not names:
            raise ValueError(f"--curriculum-pin {spec!r}: expected category=name1,name2,...")
        if cat not in categories:
            raise ValueError(f"--curriculum-pin {spec!r}: unknown category {cat!r} "
                             f"(--categories is {categories})")
        if cat in pinned:
            raise ValueError(f"--curriculum-pin names category {cat!r} more than once")
        if names.startswith("@"):
            # @file: one class name per line -- how a full hand-built order (e.g.
            # the 2048-identity White-first VGGFace2 ladder) is passed without a
            # 25 KB command line
            with open(names[1:]) as f:
                names = ",".join(line.strip() for line in f if line.strip())
        ids = []
        for n in names.split(","):
            n = n.strip()
            if not n:
                continue
            key = f"{cat}/{n}"
            if key not in label_map:
                raise ValueError(f"--curriculum-pin {spec!r}: {key!r} is not a class in this "
                                 f"run's label map (check the spelling, and that "
                                 f"--max-classes-per-category has not trimmed it away)")
            if label_map[key] in ids:
                raise ValueError(f"--curriculum-pin {spec!r}: {n!r} listed twice")
            ids.append(label_map[key])
        pinned[cat] = ids
    return pinned


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
    caps_per_stage = [parse_image_cap_spec(x, categories) for x in args.curriculum_image_caps] \
        or [{}] * len(args.curriculum_stages)
    if len(caps_per_stage) != len(args.curriculum_stages):
        raise ValueError(f"--curriculum-image-caps has {len(caps_per_stage)} entries but "
                         f"--curriculum-stages has {len(args.curriculum_stages)}")

    ids_by_category = {c: [] for c in categories}
    for name, idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        ids_by_category[name.split("/", 1)[0]].append(idx)

    rng = np.random.default_rng(args.curriculum_seed)
    order = {c: rng.permutation(ids).tolist() for c, ids in ids_by_category.items()}

    # Hand-picked classes jump to the front, so stage 1 is the set that was
    # chosen rather than whatever --curriculum-seed happened to draw. The rest
    # keep their seeded order behind the pins; nesting still holds because the
    # ordering is still a single fixed list every stage takes a prefix of.
    for c, head in parse_pin_spec(args.curriculum_pin, label_map, categories).items():
        rest = [i for i in order[c] if i not in set(head)]
        order[c] = head + rest

    stages, prev, prev_caps = [], {c: 0 for c in categories}, {}
    for spec, epochs, caps in zip(args.curriculum_stages, args.curriculum_epochs,
                                  caps_per_stage):
        sizes = parse_stage_spec(spec, categories)
        for c, n in caps.items():
            was = prev_caps.get(c)
            if was is not None and n is not None and n < was:
                raise ValueError(f"--curriculum-image-caps shrinks {c} from {was} to {n} "
                                 f"images/class; caps must be nested (never forget)")
        for c, was in prev_caps.items():
            if c not in caps:
                raise ValueError(f"--curriculum-image-caps drops the cap on {c} after "
                                 f"stage-capping it at {was}; pass {c}=all to uncap it "
                                 f"explicitly so the jump is visible in the command")
        # an explicit =all retires the cap, so later stages need not repeat it
        prev_caps = {c: n for c, n in caps.items() if n is not None}
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
                       "classes_per_category": per_cat, "active_ids": sorted(active),
                       "image_caps": caps})
    return stages


def subset_indices(dataset, active_ids, image_caps=None, id_to_category=None):
    """Positions in dataset.samples whose global label is currently active.

    Both packed datasets store the global label last in each sample tuple
    (train: (split, image, fixation, label); eval: (split, image, label)), and
    the first two entries identify the base image in both.

    `image_caps` ({category: n}) additionally keeps only the first n BASE IMAGES
    of each class in that category. Every fixation of a kept image is kept, so a
    capped class is a smaller set of photos rather than a thinner sampling of the
    same photos -- which is what an exemplar ladder means.
    """
    active = set(active_ids)
    caps = image_caps or {}
    if not caps:
        return [i for i, s in enumerate(dataset.samples) if s[-1] in active]

    keep, seen = [], {}
    for i, s in enumerate(dataset.samples):
        gl = s[-1]
        if gl not in active:
            continue
        cap = caps.get(id_to_category[gl])
        if cap is not None:
            imgs = seen.setdefault(gl, set())
            key = (s[0], s[1])
            if key not in imgs:
                if len(imgs) >= cap:
                    continue
                imgs.add(key)
        keep.append(i)
    return keep


def curriculum_lr(base_lr, epoch_frac, warmup_frac):
    """Global cosine over the whole run, times a linear per-stage warm-up factor.

    The cosine spans every stage rather than restarting per stage: a per-stage
    schedule would drive the LR to zero five times over and freeze the features
    before the hard, many-class stages ever start.
    """
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * min(max(epoch_frac, 0.0), 1.0))) * warmup_frac


RESUME_NAME = "checkpoint_last.pth"


def save_resume_state(path, model, optimizer, **state):
    """Full training state, written atomically.

    torch.save of ~11M params takes long enough that a kill mid-write leaves a
    truncated file, so write to .tmp and os.replace (atomic on POSIX). The
    previous good checkpoint survives any crash during the write.
    """
    state["model"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    state["rng"] = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(path, model, optimizer, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    rng = ck.get("rng") or {}
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() if hasattr(r, "cpu") else r for r in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception as e:                      # RNG is a nicety, not correctness
        print(f"[resume] could not restore RNG state ({e}); continuing")
    return ck


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
            random_fixations=args.random_fixations,
        )
        transforms = {
            "train": OnTheFlyTransform("train", args.variant, device,
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
            per_cat = ", ".join(f"{n} {c}" for c, n in s["classes_per_category"].items())
            print(f"  stage {k}: {len(s['active_ids']):4d} classes ({per_cat}) "
                  f"| {s['epochs']} epochs")
    else:
        stages = [{"spec": "all", "epochs": args.epochs, "image_caps": {},
                   "classes_per_category": {}, "active_ids": list(range(num_classes))}]

    valid_batch_size = max(1, args.batch_size // args.num_fixations)

    def epoch_draws(natural):
        """Samples drawn per epoch: the subset's own size, or a fixed budget.

        Without --steps-per-epoch an epoch is one pass over whatever is active,
        which makes stage 1 (~30 steps) and the final stage (~8000) differ by
        two orders of magnitude in how much they can actually shape the weights.
        """
        return args.steps_per_epoch * args.batch_size if args.steps_per_epoch else natural

    domain_of = {}
    for spec in args.weight_domains:
        dom, _, members = spec.partition("=")
        for c in members.split(","):
            c = c.strip()
            if c not in args.categories:
                raise ValueError(f"--weight-domains {spec!r}: {c!r} is not in --categories")
            if c in domain_of:
                raise ValueError(f"--weight-domains puts {c!r} in two domains")
            domain_of[c] = dom.strip()

    def sqrt_sampler(train_split):
        """WeightedRandomSampler for --category-weighting sqrt (see its help).

        A crop's weight is
            share(domain) * sqrt(photos_k) / sum_{k' in domain} sqrt(photos_k') / crops_k
        so every class k gets its sqrt-share of its domain's share, spread evenly
        over its own crops. Counts are taken from the ACTIVE subset, so the shares
        follow the curriculum and the per-stage image caps automatically.
        """
        base = train_split.dataset if isinstance(train_split, Subset) else train_split
        idxs = list(train_split.indices if isinstance(train_split, Subset) else range(len(base)))
        crops, photos = {}, {}
        for i in idxs:
            smp = base.samples[i]
            k = smp[-1]
            crops[k] = crops.get(k, 0) + 1
            photos.setdefault(k, set()).add((smp[0], smp[1]))
        photos = {k: len(v) for k, v in photos.items()}
        dom = lambda k: domain_of.get(id_to_category[k], id_to_category[k])
        classes_in = {}
        for k in crops:
            classes_in.setdefault(dom(k), []).append(k)
        # items beneath each domain: its classes, or its photos if it has only one
        items = {d: (photos[ks[0]] if len(ks) == 1 else len(ks)) for d, ks in classes_in.items()}
        tot = sum(math.sqrt(n) for n in items.values())
        dshare = {d: math.sqrt(n) / tot for d, n in items.items()}
        kshare = {}
        for d, ks in classes_in.items():
            z = sum(math.sqrt(photos[k]) for k in ks)
            for k in ks:
                kshare[k] = dshare[d] * math.sqrt(photos[k]) / z
        cat_share = {}
        for k, v in kshare.items():
            cat_share[id_to_category[k]] = cat_share.get(id_to_category[k], 0.0) + v
        print("  sqrt-rule domain shares: " + ", ".join(
            f"{d} {100 * dshare[d]:.1f}% ({items[d]} {'photos' if len(classes_in[d]) == 1 else 'classes'})"
            for d in sorted(dshare)))
        print("  sqrt-rule category shares: " + ", ".join(
            f"{c} {100 * v:.1f}%" for c, v in sorted(cat_share.items())))
        w = [kshare[base.samples[i][-1]] / crops[base.samples[i][-1]] for i in idxs]
        return torch.utils.data.WeightedRandomSampler(w, num_samples=epoch_draws(len(idxs)),
                                                      replacement=True)

    def category_sampler(train_split):
        """WeightedRandomSampler giving each category its requested batch share.

        A sample's weight is share_c / n_c, so a category's expected share of the
        batch is its target regardless of how many crops it actually owns. Shares
        are renormalised over the categories present in this stage (houses are
        absent from the first stages), and the number of draws per epoch equals
        the subset size, so weighting changes the *diet* and not the compute.
        """
        if args.category_weighting == "sqrt":
            return sqrt_sampler(train_split)
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
        return torch.utils.data.WeightedRandomSampler(w, num_samples=epoch_draws(len(cats)),
                                                      replacement=True)

    def make_loaders(active_ids, image_caps=None):
        """Loaders restricted to the currently active classes (all of them
        outside curriculum mode, where the subsets are skipped entirely).

        `image_caps` is a train-only exemplar cap: valid and test must stay the
        same set of images at every stage or the accuracy columns stop being
        comparable across stages.
        """
        full = len(active_ids) == num_classes
        def split_of(name):
            ds = datasets[name]
            caps = image_caps if name == "train" else None
            if full and not caps:
                return ds
            return Subset(ds, subset_indices(ds, active_ids, caps, id_to_category))
        train_split = split_of("train")
        sampler = category_sampler(train_split)
        if sampler is None and args.steps_per_epoch:
            # No category weights, but the epoch still has to be a fixed length,
            # so draw the same budget uniformly (with replacement, like the
            # weighted path) instead of walking the subset once.
            sampler = torch.utils.data.RandomSampler(
                train_split, replacement=True, num_samples=epoch_draws(len(train_split)))
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
    resume_path = os.path.join(out_dir, RESUME_NAME)
    history = []
    t_start = time.time()

    # ---------------------------- Resume ----------------------------
    # Restores weights, AdamW moments, LR position (via `epoch`), curriculum
    # position, early-stopping counters and RNG. Skips stages already finished
    # and epochs already done inside the stage we died in.
    resume_stage, resume_ep_in_stage, resume_epoch = 1, 0, 0
    resume_warmup_left = None
    elapsed_offset = 0.0
    if args.resume:
        rp = resume_path if args.resume in ("auto", "1", "true") else args.resume
        if os.path.exists(rp):
            ck = load_resume_state(rp, model, optimizer, device)
            resume_epoch = ck["epoch"]
            resume_stage = ck["stage_idx"]
            resume_ep_in_stage = ck["ep_in_stage"] + 1
            best_val, best_epoch = ck["best_val"], ck["best_epoch"]
            patience_counter = ck["patience_counter"]
            history = ck.get("history", [])
            resume_warmup_left = ck.get("warmup_left")
            elapsed_offset = ck.get("elapsed", 0.0)
            print(f"[resume] {os.path.basename(rp)}: continuing after epoch "
                  f"{resume_epoch} (stage {resume_stage}, epoch {resume_ep_in_stage} "
                  f"of that stage); best valid so far {best_val*100:.2f}% "
                  f"@ epoch {best_epoch}")
        else:
            print(f"[resume] no checkpoint at {rp}; starting from scratch")
    t_start = time.time() - elapsed_offset

    # ----------------------------- Train ----------------------------
    # One pass per curriculum stage (a single all-classes stage when
    # --curriculum is off, which reproduces the original loop exactly).
    epoch = 0
    for stage_idx, stage in enumerate(stages, 1):
        is_final_stage = stage_idx == len(stages)
        if stage_idx < resume_stage:
            continue          # finished before the interruption; `epoch` restored
        active_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        active_mask[torch.tensor(stage["active_ids"], device=device)] = True
        masked = not bool(active_mask.all())
        train_loader, valid_loader, test_loader = make_loaders(
            stage["active_ids"], stage.get("image_caps"))
        # Accuracy is only comparable across stages once the class set stops
        # growing, so best_model.pth tracks the final stage; earlier stages get
        # their own best purely for the curves and for early stopping.
        if stage_idx > resume_stage:
            patience_counter = 0
        stage_best = max([h["valid_acc"] for h in history
                          if h.get("stage") == stage_idx], default=-1.0)
        # warm up the LR after each class introduction (not at the very start,
        # where the cosine already begins from the full base LR)
        warmup_left = args.curriculum_warmup_steps if (args.curriculum and stage_idx > 1) else 0
        if stage_idx == resume_stage and resume_warmup_left is not None:
            warmup_left = resume_warmup_left      # warm-up already partly spent
        epoch = resume_epoch if stage_idx == resume_stage else epoch
        if args.curriculum:
            per_cat = ", ".join(f"{n} {c}" for c, n in stage["classes_per_category"].items())
            # "train crops" is the pool this stage draws from; the epoch itself is
            # len(sampler), which --steps-per-epoch pins to a fixed budget and which
            # is otherwise one pass over that pool.
            print(f"\n=== Stage {stage_idx}/{len(stages)}: {len(stage['active_ids'])} classes "
                  f"({per_cat}), {stage['epochs']} epochs, "
                  f"{len(train_loader.dataset)} train crops, "
                  f"{len(train_loader)} steps/epoch"
                  + " ===")

        for ep_in_stage in range(stage["epochs"]):
            if stage_idx == resume_stage and ep_in_stage < resume_ep_in_stage:
                continue      # already done before the interruption
            epoch += 1
            model.train()
            correct = total = 0
            epoch_losses = []
            steps_per_epoch = max(len(train_loader), 1)
            # the sampler decides epoch length once --steps-per-epoch is set,
            # and it is the dataset size in every other case
            pbar = tqdm(total=len(train_loader.sampler),
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

            # Full-state checkpoint AFTER the counters settle and BEFORE any
            # early-stopping break, so a resume never replays a finished epoch.
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_resume_state(resume_path, model, optimizer,
                                  epoch=epoch, stage_idx=stage_idx,
                                  ep_in_stage=ep_in_stage,
                                  best_val=best_val, best_epoch=best_epoch,
                                  patience_counter=patience_counter,
                                  warmup_left=warmup_left, history=history,
                                  elapsed=time.time() - t_start)

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
