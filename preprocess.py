"""
preprocess.py

Convert raw category datasets into preprocessed fixation crops using the
(category-agnostic, Gabor-saliency) SaliencePipeline, producing both the
log-polar (lp) and plain-crop (cnn) variants.

Output layout:
    processed_data/<category>/<lp|cnn>/<split>/<class>/<img>_proc<n>.png

Split -> presentation orientation (set by the pipeline):
    train          -> random rotation augmentation
    valid          -> upright
    test           -> inverted (180 deg), disjoint photos from 'valid'
    valid_inverted -> inverted (180 deg) versions of the SAME photos as
                       'valid' (not a separate raw split) -- used as the
                       "inverted" condition in Yin so the "old" test image is
                       guaranteed to be the exact photo seen at study

All raw images are resized (shorter side) + center-cropped to a square
--input-size so every category is fed to the pipeline on equal footing.

Images are loaded in batches via a DataLoader (parallel CPU workers feeding
the GPU). Each (category, split) is one unit of work; if multiple --devices
are given, units are spread round-robin across one worker process per
device, so e.g. faces/objects/houses x train/valid/test can run on 4 GPUs
at once.

Usage:
    python preprocess.py --categories faces objects --num-fixations 16
    python preprocess.py --categories faces --splits valid test   # e.g. for Yin only
    python preprocess.py --devices cuda:0 cuda:1 cuda:2 cuda:3    # spread across 4 GPUs
"""

import argparse
import os

import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from salience_trans import SaliencePipeline

# Where each category's raw (downloaded) data lives. preprocess auto-discovers
# the directory containing train/valid/test underneath these roots.
RAW_ROOTS = {
    "faces": "data/faces_cleaned",
    "objects": "data/ImageNet_objects",
    "houses": "data/houses",
}

SPLITS = ("train", "valid", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Extra output splits that reuse another split's raw photos under a different
# orientation pipeline (see split_type below). Used so Yin's "old" test image
# is guaranteed to be the exact photo seen at study, just inverted -- unlike
# the raw 'test' split, which is a disjoint set of photos from 'valid'.
EXTRA_SPLIT_SOURCE = {"valid_inverted": "valid"}


def find_split_root(raw_dir):
    """Find the directory under raw_dir that directly contains train/valid/test
    subfolders. If several exist (e.g. multiple {N}_identities folders), pick the
    one covering the most splits, then the most classes."""
    best = None
    for dirpath, dirnames, _ in os.walk(raw_dir):
        present = [s for s in SPLITS if s in dirnames]
        if not present:
            continue
        n_classes = 0
        for s in present:
            sd = os.path.join(dirpath, s)
            n_classes += sum(1 for d in os.listdir(sd) if os.path.isdir(os.path.join(sd, d)))
        key = (len(present), n_classes)
        if best is None or key > best[0]:
            best = (key, dirpath)
    if best is None:
        raise FileNotFoundError(
            f"Could not find train/valid/test under {raw_dir!r}. "
            f"Run download.py or pass the correct raw root."
        )
    return best[1]


def split_type(split):
    if split == "train":
        return "train"      # random rotation
    if split in ("test", "valid_inverted"):
        return "test"       # inverted
    return "valid"          # identity / upright


def find_done_stems(dir_path, num_fixations):
    """Stems in dir_path that already have all num_fixations '_proc<n>.png' crops,
    i.e. fully written by a previous run."""
    if not os.path.isdir(dir_path):
        return set()
    needed = [f"_proc{n}.png" for n in range(num_fixations)]
    by_stem = {}
    for fname in os.listdir(dir_path):
        for suf in needed:
            if fname.endswith(suf):
                by_stem.setdefault(fname[: -len(suf)], set()).add(suf)
                break
    return {stem for stem, sufs in by_stem.items() if len(sufs) == num_fixations}


class SplitImageDataset(Dataset):
    """Flat list of (class, image) samples under one split dir, resized to
    --input-size so they can be batched and fed to the pipeline together.
    Samples whose lp+cnn crops already exist on disk (from a prior run) are
    skipped, so re-running after an interruption resumes instead of redoing
    everything from scratch."""

    def __init__(self, src_split, input_size, processed_root, category, split,
                num_fixations, limit_per_class=None):
        self.src_split = src_split
        self.resize = T.Compose([T.Resize(input_size), T.CenterCrop(input_size)])
        self.classes = sorted(d for d in os.listdir(src_split)
                              if os.path.isdir(os.path.join(src_split, d)))
        self.samples = []  # (cls_name, fname)
        self.n_skipped = 0
        for cls_name in self.classes:
            src_cls = os.path.join(src_split, cls_name)
            imgs = sorted(f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS))
            if limit_per_class:
                imgs = imgs[:limit_per_class]

            lp_dir = os.path.join(processed_root, category, "lp", split, cls_name)
            cnn_dir = os.path.join(processed_root, category, "cnn", split, cls_name)
            done = find_done_stems(lp_dir, num_fixations) & find_done_stems(cnn_dir, num_fixations)

            for f in imgs:
                stem = os.path.splitext(f)[0]
                if stem in done:
                    self.n_skipped += 1
                else:
                    self.samples.append((cls_name, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_name, fname = self.samples[idx]
        path = os.path.join(self.src_split, cls_name, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    [warn] cannot open {fname}: {e}")
            return None
        img = self.resize(img)
        stem = os.path.splitext(fname)[0]
        return TF.to_tensor(img), cls_name, stem


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, classes, stems = zip(*batch)
    return torch.stack(imgs), list(classes), list(stems)


def process_split(category, raw_root, processed_root, split, num_fixations,
                  input_size, device, batch_size, num_workers, limit_per_class=None):
    split_src_root = find_split_root(raw_root)
    src_name = EXTRA_SPLIT_SOURCE.get(split, split)
    src_split = os.path.join(split_src_root, src_name)
    if not os.path.isdir(src_split):
        print(f"[{device}] [skip] no '{src_name}' split for {category}")
        return

    dataset = SplitImageDataset(src_split, input_size, processed_root, category, split,
                                num_fixations, limit_per_class)

    for cls_name in dataset.classes:
        os.makedirs(os.path.join(processed_root, category, "lp", split, cls_name), exist_ok=True)
        os.makedirs(os.path.join(processed_root, category, "cnn", split, cls_name), exist_ok=True)

    if dataset.n_skipped:
        print(f"[{device}] {category}/{split}: resuming, {dataset.n_skipped} images "
              f"already done, {len(dataset)} remaining")

    if len(dataset) == 0:
        print(f"[{device}] [skip] no images left to process for {category}/{split}")
        return

    # Workers must use 'spawn', not the default 'fork': by the time the loader
    # iterates, this process has already initialized a CUDA context (below),
    # and fork()-ing a CUDA-initialized process hangs.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_skip_none,
                        pin_memory=str(device).startswith("cuda"),
                        multiprocessing_context="spawn" if num_workers > 0 else None)

    pipeline = SaliencePipeline(type=split_type(split), device=device,
                                num_salient_points=num_fixations).to(device)

    print(f"[{device}] {category}/{split}: {len(dataset.classes)} classes, "
          f"{len(dataset)} images, batch_size={batch_size}")

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, classes, stems = batch
            imgs = imgs.to(device, non_blocking=True)

            lp, cnn = pipeline(imgs)  # each (B, N, C, H, W)

            for b in range(imgs.shape[0]):
                cls_name, stem = classes[b], stems[b]
                lp_dir = os.path.join(processed_root, category, "lp", split, cls_name)
                cnn_dir = os.path.join(processed_root, category, "cnn", split, cls_name)
                for n, t in enumerate(lp[b]):
                    TF.to_pil_image(t.cpu().clamp(0, 1)).save(os.path.join(lp_dir, f"{stem}_proc{n}.png"))
                for n, t in enumerate(cnn[b]):
                    TF.to_pil_image(t.cpu().clamp(0, 1)).save(os.path.join(cnn_dir, f"{stem}_proc{n}.png"))

            n_done += imgs.shape[0]
            print(f"[{device}] {category}/{split}: {n_done}/{len(dataset)} imgs "
                  f"x {num_fixations} fixations", end="\r")
    print()


def _run_jobs(jobs, processed_root, num_fixations, input_size, batch_size,
              num_workers, limit_per_class, device, omp_threads=None):
    # Each device runs in its own process; without a cap, every process's
    # torch/OMP backend defaults to using *all* CPU cores, so N processes
    # oversubscribe the machine by Nx and contend with each other.
    if omp_threads:
        torch.set_num_threads(omp_threads)
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    for category, raw_root, split in jobs:
        process_split(category, raw_root, processed_root, split, num_fixations,
                      input_size, device, batch_size, num_workers, limit_per_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects", "houses"])
    ap.add_argument("--processed-root", default="processed_data")
    ap.add_argument("--num-fixations", type=int, default=16)
    ap.add_argument("--input-size", type=int, default=224,
                    help="raw images are resized + center-cropped to this square size")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--devices", nargs="+",
                    default=["cuda:0" if torch.cuda.is_available() else "cpu"],
                    help="one or more devices, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3. "
                         "(category, split) work units are spread round-robin across them.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="images per batch fed to the pipeline")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers per device process, for parallel image decode/resize")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap base images per class (for quick tests)")
    ap.add_argument("--raw-root", action="append", default=[],
                    help="override raw root, e.g. --raw-root objects=/path/to/objects")
    args = ap.parse_args()

    raw_roots = dict(RAW_ROOTS)
    for ov in args.raw_root:
        cat, path = ov.split("=", 1)
        raw_roots[cat] = path

    jobs = []
    for category in args.categories:
        if category not in raw_roots:
            print(f"[skip] no raw root configured for '{category}'")
            continue
        for split in args.splits:
            jobs.append((category, raw_roots[category], split))

    if len(args.devices) == 1:
        _run_jobs(jobs, args.processed_root, args.num_fixations, args.input_size,
                  args.batch_size, args.num_workers, args.limit_per_class, args.devices[0])
    else:
        buckets = {d: [] for d in args.devices}
        for i, job in enumerate(jobs):
            buckets[args.devices[i % len(args.devices)]].append(job)

        n_active = sum(1 for v in buckets.values() if v)
        omp_threads = max(1, (os.cpu_count() or 1) // n_active)

        ctx = mp.get_context("spawn")
        procs = []
        for device, dev_jobs in buckets.items():
            if not dev_jobs:
                continue
            p = ctx.Process(target=_run_jobs, args=(
                dev_jobs, args.processed_root, args.num_fixations, args.input_size,
                args.batch_size, args.num_workers, args.limit_per_class, device, omp_threads,
            ))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("\nDone.")


if __name__ == "__main__":
    main()
