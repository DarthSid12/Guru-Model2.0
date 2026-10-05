"""
preprocess_fixations.py

Lightweight replacement for preprocess.py: instead of rendering every fixation
crop of every image to disk as a PNG (~246 GB, ~6M small files -- reading those
back over NFS is what made training I/O-bound), this stores only:

    fixation_data/<category>/<split>/
        images.npy   uint8 (M, S, S, 3)   resized+center-cropped raw images
        coords.npy   int16 (M, K, 2)      K Gabor-saliency fixation points (x, y)
        meta.json    classes / labels / stems / sizes

Everything downstream (crop around fixation, rotation, foveation, log-polar)
is cheap GPU math and is now done on the fly at train time (see
salience_trans.OnTheFlyTransform + datasets.make_packed_datasets). For the
full dataset this is ~25 GB of large sequential files instead of ~246 GB of
tiny PNGs, and one epoch does zero per-sample file opens after the OS page
cache warms up.

Only train/valid/test need packing. 'valid_inverted' is NOT a separate copy
anymore: in packed mode inversion is just a 180-degree rotation applied on the
fly to the 'valid' images.

Usage:
    python preprocess_fixations.py --categories faces objects houses
    python preprocess_fixations.py --devices cuda:0 cuda:1 cuda:2   # parallel
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from preprocess import RAW_ROOTS, SPLITS, IMG_EXTS, find_split_root
from salience_trans import SaliencePipeline


class RawSplitDataset(Dataset):
    """All (class, image) samples under one raw split dir, resized to a square
    input_size, in a deterministic (sorted) order."""

    def __init__(self, src_split, input_size):
        self.src_split = src_split
        self.resize = T.Compose([T.Resize(input_size), T.CenterCrop(input_size)])
        self.classes = sorted(d for d in os.listdir(src_split)
                              if os.path.isdir(os.path.join(src_split, d)))
        self.samples = []  # (cls_idx, fname)
        for ci, cls_name in enumerate(self.classes):
            src_cls = os.path.join(src_split, cls_name)
            for f in sorted(f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS)):
                self.samples.append((ci, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, fname = self.samples[idx]
        path = os.path.join(self.src_split, self.classes[ci], fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    [warn] cannot open {fname}: {e}")
            return idx, None
        return idx, TF.to_tensor(self.resize(img))


def collate_keep_idx(batch):
    batch = [(i, t) for i, t in batch if t is not None]
    if not batch:
        return None
    idxs, imgs = zip(*batch)
    return torch.tensor(idxs), torch.stack(imgs)


def pack_split(category, raw_root, out_root, split, num_coords, input_size,
               device, batch_size, num_workers, limit_per_class=None, force=False,
               edge_margin=0.0):
    out_dir = os.path.join(out_root, category, split)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.isfile(meta_path) and not force:
        print(f"[{device}] [skip] {category}/{split} already packed ({meta_path})")
        return

    src_split = os.path.join(find_split_root(raw_root), split)
    if not os.path.isdir(src_split):
        print(f"[{device}] [skip] no '{split}' split for {category}")
        return

    dataset = RawSplitDataset(src_split, input_size)
    if limit_per_class:
        keep, per_cls = [], {}
        for ci, fname in dataset.samples:
            if per_cls.get(ci, 0) < limit_per_class:
                keep.append((ci, fname))
                per_cls[ci] = per_cls.get(ci, 0) + 1
        dataset.samples = keep

    M = len(dataset)
    if M == 0:
        print(f"[{device}] [skip] no images for {category}/{split}")
        return
    print(f"[{device}] {category}/{split}: {len(dataset.classes)} classes, {M} images")

    tmp_dir = out_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    images = np.lib.format.open_memmap(os.path.join(tmp_dir, "images.npy"), mode="w+",
                                       dtype=np.uint8, shape=(M, input_size, input_size, 3))
    coords = np.lib.format.open_memmap(os.path.join(tmp_dir, "coords.npy"), mode="w+",
                                       dtype=np.int16, shape=(M, num_coords, 2))

    # Only the saliency sampler is needed; crop/rotate/foveate/logpolar happen
    # at train time.
    pipeline = SaliencePipeline(type="valid", device=device,
                                num_salient_points=num_coords,
                                edge_margin=edge_margin).to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_keep_idx,
                        pin_memory=str(device).startswith("cuda"),
                        multiprocessing_context="spawn" if num_workers > 0 else None)

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            idxs, imgs = batch
            imgs = imgs.to(device, non_blocking=True)
            pts = pipeline.sample_salience_points(imgs)  # (B, K, 2) as (x, y)
            images[idxs.numpy()] = (imgs.clamp(0, 1) * 255).round().to(torch.uint8) \
                .permute(0, 2, 3, 1).cpu().numpy()
            coords[idxs.numpy()] = pts.cpu().to(torch.int16).numpy()
            n_done += imgs.shape[0]
            print(f"[{device}] {category}/{split}: {n_done}/{M}", end="\r")
    print()

    images.flush()
    coords.flush()
    meta = {
        "classes": dataset.classes,
        "labels": [ci for ci, _ in dataset.samples],
        "stems": [os.path.splitext(f)[0] for _, f in dataset.samples],
        "input_size": input_size,
        "num_coords": num_coords,
        "edge_margin": edge_margin,
    }
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp_dir, out_dir)
    print(f"[{device}] {category}/{split}: packed -> {out_dir}")


def recompute_coords(category, out_root, split, device, batch_size, edge_margin,
                     num_coords=None):
    """Rewrite coords.npy for an already-packed split, leaving images.npy alone.

    The packed images ARE the resized/center-cropped tensors the sampler saw
    when the store was built, so re-running the sampler on them reproduces the
    original pipeline without touching raw data -- minutes instead of hours, and
    no dependence on raw roots that may no longer be mounted. The one difference
    is that packed images are uint8, so the sampler sees the 8-bit quantised
    image rather than the float one; on a Gabor variance map that is noise well
    below the sampling stochasticity, and the point here is to change the
    sampling distribution anyway.

    coords.npy is replaced atomically, so an interrupted run leaves the store
    readable with its old coordinates rather than half-written.
    """
    d = os.path.join(out_root, category, split)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"[{device}] [skip] {category}/{split} is not packed")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    K = num_coords or meta["num_coords"]

    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    M = images.shape[0]
    pipeline = SaliencePipeline(type="valid", device=device, num_salient_points=K,
                                edge_margin=edge_margin).to(device)

    tmp = os.path.join(d, "coords.npy.tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(M, K, 2))
    with torch.no_grad():
        for lo in range(0, M, batch_size):
            hi = min(lo + batch_size, M)
            batch = torch.from_numpy(np.asarray(images[lo:hi]).copy()) \
                .permute(0, 3, 1, 2).float().div_(255.0).to(device)
            out[lo:hi] = pipeline.sample_salience_points(batch).cpu().to(torch.int16).numpy()
            print(f"[{device}] {category}/{split}: {hi}/{M}", end="\r")
    out.flush()
    del out
    os.replace(tmp, os.path.join(d, "coords.npy"))

    meta["num_coords"] = K
    meta["edge_margin"] = edge_margin
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"\n[{device}] {category}/{split}: recomputed {M} x {K} coords "
          f"(edge_margin={edge_margin})")


def _run_jobs(jobs, out_root, num_coords, input_size, batch_size, num_workers,
              limit_per_class, force, device, omp_threads=None,
              edge_margin=0.0, coords_only=False):
    if omp_threads:
        torch.set_num_threads(omp_threads)
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    for category, raw_root, split in jobs:
        if coords_only:
            recompute_coords(category, out_root, split, device, batch_size,
                             edge_margin, num_coords)
        else:
            pack_split(category, raw_root, out_root, split, num_coords, input_size,
                       device, batch_size, num_workers, limit_per_class, force,
                       edge_margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["faces", "objects", "houses"])
    ap.add_argument("--out-root", default="fixation_data")
    ap.add_argument("--num-coords", type=int, default=32,
                    help="fixation points stored per image (training can use any prefix of them)")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--devices", nargs="+",
                    default=["cuda:0" if torch.cuda.is_available() else "cpu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-pack even if output exists")
    ap.add_argument("--edge-margin", type=float, default=0.0,
                    help="reject fixations within this fraction of the image on each side, "
                         "so every point falls in the central (1-2m) square. 0.15 keeps the "
                         "central 70%%. A border fixation yields a crop that is mostly zero "
                         "padding (crop 180 on a 224 image), which costs a full forward pass "
                         "and carries a label but almost no stimulus. Default 0.0 reproduces "
                         "every store packed before this flag existed.")
    ap.add_argument("--coords-only", action="store_true",
                    help="recompute coords.npy for already-packed splits and leave images.npy "
                         "untouched. This is how an existing store is moved to a new "
                         "--edge-margin without re-reading raw data. --categories/--splits "
                         "select what is rewritten, IN PLACE -- back the stores up first if "
                         "the old coordinates still matter.")
    ap.add_argument("--raw-root", action="append", default=[],
                    help="override raw root, e.g. --raw-root objects=/path/to/objects")
    args = ap.parse_args()

    raw_roots = dict(RAW_ROOTS)
    for ov in args.raw_root:
        cat, path = ov.split("=", 1)
        raw_roots[cat] = path

    jobs = []
    for category in args.categories:
        # --coords-only works off the packed store, so a category needs no raw
        # root: many of the derived stores (the ladders, the held-out sets) were
        # built by re-packing scripts and have no raw tree at all.
        if not args.coords_only and category not in raw_roots:
            print(f"[skip] no raw root configured for '{category}'")
            continue
        for split in args.splits:
            jobs.append((category, raw_roots.get(category), split))

    common = (args.out_root, args.num_coords, args.input_size, args.batch_size,
              args.num_workers, args.limit_per_class, args.force)
    if len(args.devices) == 1:
        _run_jobs(jobs, *common, args.devices[0], None, args.edge_margin, args.coords_only)
    else:
        buckets = {d: [] for d in args.devices}
        for i, job in enumerate(jobs):
            buckets[args.devices[i % len(args.devices)]].append(job)
        n_active = sum(1 for v in buckets.values() if v)
        omp_threads = max(1, (os.cpu_count() or 1) // n_active)
        ctx = mp.get_context("spawn")
        procs = []
        for device, dev_jobs in buckets.items():
            if dev_jobs:
                p = ctx.Process(target=_run_jobs,
                                args=(dev_jobs, *common, device, omp_threads,
                                      args.edge_margin, args.coords_only))
                p.start()
                procs.append(p)
        for p in procs:
            p.join()

    print("\nDone.")


if __name__ == "__main__":
    main()
