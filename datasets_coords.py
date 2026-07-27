"""
datasets_coords.py

Like datasets, but modified to save fixation crop coordinates
"""

import json
import os
import re

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from utils import build_global_label_map, label_to_one_hot, list_classes

_PROC_RE = re.compile(r"_proc(\d+)\.png$")


def _proc_index(fname):
    m = _PROC_RE.search(fname)
    return int(m.group(1)) if m else None


def make_datasets(categories, processed_root="processed_data", variant="lp",
                  num_salient_points=32, label_map=None, max_images_per_class=None, full_image=False):
    """
    Build train/valid/test datasets spanning all `categories` under one global
    label space.

    Args:
        categories (list[str]): e.g. ["faces", "objects"]
        processed_root (str): root of preprocessed data
        variant (str): "lp" (log-polar) or "cnn"
        num_salient_points (int): fixations per base image
        label_map (dict|None): global label map; built from the train split if None
        max_images_per_class (dict[str,int]|None): optional cap on the number of
            base images per class, keyed by category (e.g. {"objects": 200}).
            Only applied to the training split; valid/test are left untouched.
    """
    if label_map is None:
        label_map = build_global_label_map(processed_root, categories, variant, split="train")

    return {
        "train": CombinedSalienceDatasetBatched(
            categories, processed_root, variant, "train", num_salient_points, label_map,
            max_images_per_class=max_images_per_class, full_image=full_image),
        "valid": CombinedSalienceDataset(
            categories, processed_root, variant, "valid", num_salient_points, label_map, full_image=full_image),
        "test": CombinedSalienceDataset(
            categories, processed_root, variant, "test", num_salient_points, label_map, full_image=full_image),
    }, label_map


class CombinedSalienceDatasetBatched(Dataset):
    """Training set: one sample == one fixation patch (path, global_label)."""

    def __init__(self, categories, processed_root, variant, split,
                 num_salient_points, label_map, max_images_per_class=None, full_image=False):
        self.num_salient_points = num_salient_points
        self.map = label_map
        self.samples = []  # [(path, "<category>/<class>"), ...]
        self.full_image = full_image
        max_images_per_class = max_images_per_class or {}

        for category in categories:
            split_dir = os.path.join(processed_root, category, variant, split)
            cap = max_images_per_class.get(category)
            for cls_name in list_classes(split_dir):
                label = f"{category}/{cls_name}"
                if label not in self.map:
                    continue
                cls_dir = os.path.join(split_dir, cls_name)

                base_dict = {}
                for fname in sorted(os.listdir(cls_dir)):
                    idx = _proc_index(fname)
                    if idx is None or idx >= num_salient_points:
                        continue
                    base_num = fname.split("_proc")[0]
                    base_dict.setdefault(base_num, []).append(fname)

                base_nums = sorted(base_dict)
                if cap is not None:
                    base_nums = base_nums[:cap]
                for base_num in base_nums:
                    for fname in base_dict[base_num]:
                        self.samples.append((os.path.join(cls_dir, fname), label))

        if not self.samples:
            raise ValueError(f"No training samples found for categories={categories} ({variant}/{split}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = TF.to_tensor(Image.open(path).convert("RGB"))
        return img, label_to_one_hot(label, self.map)


class CombinedSalienceDataset(Dataset):
    """Eval set: one sample == all N fixations of a base image, stacked (N,C,H,W)."""

    def __init__(self, categories, processed_root, variant, split,
                 num_salient_points, label_map, full_image=False):
        self.num_salient_points = num_salient_points
        self.map = label_map
        self.samples = []  # [(base_id, [proc_paths...], "<category>/<class>"), ...]
        self.full_image = full_image

        for category in categories:
            split_dir = os.path.join(processed_root, category, variant, split)
            for cls_name in list_classes(split_dir):
                label = f"{category}/{cls_name}"
                if label not in self.map:
                    continue
                cls_dir = os.path.join(split_dir, cls_name)
                base_dict = {}
                for fname in sorted(os.listdir(cls_dir)):
                    if _proc_index(fname) is not None:
                        base_num = fname.split("_proc")[0]
                        base_dict.setdefault(base_num, []).append(os.path.join(cls_dir, fname))
                for base_num, proc_list in base_dict.items():
                    self.samples.append((base_num, sorted(proc_list), label))

        if not self.samples:
            raise ValueError(f"No eval samples found for categories={categories} ({variant}/{split}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_num, proc_list, label = self.samples[idx]
        chosen = proc_list[:self.num_salient_points]
        imgs = torch.stack([TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen], dim=0)
        return imgs, label_to_one_hot(label, self.map)


# --------------------------------------------------------------------------
# Packed ("fixation_data") datasets: raw images + saliency coords only, all
# rendering (rotate/foveate/logpolar) done on the fly on the GPU. Produced by
# preprocess_fixations.py; consumed together with
# salience_trans.OnTheFlyTransform. This avoids the ~6M-tiny-PNG reads per
# epoch that made the PNG pipeline I/O-bound on NFS.
# --------------------------------------------------------------------------

class _PackedSplit:
    """Memory-mapped view of one packed <category>/<split> directory."""

    def __init__(self, out_root, category, split):
        d = os.path.join(out_root, category, split)
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        self.images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
        self.coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")
        self.category = category
        self.classes = self.meta["classes"]
        self.labels = self.meta["labels"]      # class index per image
        self.stems = self.meta["stems"]
        self.num_coords = self.meta["num_coords"]


def build_packed_label_map(packed_root, categories, split="train"):
    """Same global "<category>/<class>" -> idx mapping as utils.build_global_label_map,
    but read from packed meta.json files."""
    mapping = {}
    idx = 0
    for category in categories:
        sp = _PackedSplit(packed_root, category, split)
        for cls_name in sp.classes:
            mapping[f"{category}/{cls_name}"] = idx
            idx += 1
    return mapping


def _crop_at(img_hwc, x, y, crop_size):
    """180x180 crop of a (H, W, 3) uint8 memmap centered at (x, y), zero-padded
    at the borders exactly like TF.crop in SaliencePipeline.forward."""
    t = torch.from_numpy(np.array(img_hwc)).permute(2, 0, 1)  # (3, H, W)
    return TF.crop(t, top=int(y) - crop_size // 2, left=int(x) - crop_size // 2,
                   height=crop_size, width=crop_size)


class PackedFixationTrainDataset(Dataset):
    """Training set: one sample == one (image, fixation) crop, uint8 (3,180,180)."""

    def __init__(self, categories, packed_root, split, num_salient_points,
                 label_map, crop_size=180, max_images_per_class=None, full_image=False):
        self.crop_size = crop_size
        self.map = label_map
        self.splits = []
        self.samples = []  # (split_idx, img_idx, fix_idx, global_label)
        self.full_image = full_image
        max_images_per_class = max_images_per_class or {}

        for category in categories:
            sp = _PackedSplit(packed_root, category, split)
            if num_salient_points > sp.num_coords:
                raise ValueError(f"{category}/{split} packed with {sp.num_coords} coords, "
                                 f"but num_salient_points={num_salient_points} requested")
            si = len(self.splits)
            self.splits.append(sp)
            cap = max_images_per_class.get(category)
            per_class_count = {}
            for i, ci in enumerate(sp.labels):
                label = f"{category}/{sp.classes[ci]}"
                if label not in self.map:
                    continue
                if cap is not None:
                    per_class_count[ci] = per_class_count.get(ci, 0) + 1
                    if per_class_count[ci] > cap:
                        continue
                gl = self.map[label]
                for j in range(num_salient_points):
                    self.samples.append((si, i, j, gl))

        if not self.samples:
            raise ValueError(f"No packed training samples for categories={categories} ({split}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        si, i, j, gl = self.samples[idx]
        sp = self.splits[si]

        if self.full_image:
            crop = torch.from_numpy(np.array(sp.images[i])).permute(2,0,1)

            coord = torch.zeros(2, dtype=torch.float32)
        else:
            x, y = sp.coords[i, j]
            crop = _crop_at(sp.images[i], x, y, self.crop_size)

            # return fixation crop coord too
            coord = torch.tensor([
                x / sp.images.shape[2],   # width (224)
                y / sp.images.shape[1],   # height (224)
            ], dtype=torch.float32)

        one_hot = torch.zeros(len(self.map))
        one_hot[gl] = 1.0

        return crop, coord, one_hot


class PackedFixationEvalDataset(Dataset):
    """Eval set: one sample == all N fixation crops of a base image, uint8 (N,3,180,180)."""

    def __init__(self, categories, packed_root, split, num_salient_points,
                 label_map, crop_size=180, full_image=False):
        self.crop_size = crop_size
        self.num_salient_points = num_salient_points
        self.map = label_map
        self.splits = []
        self.samples = []  # (split_idx, img_idx, global_label)
        self.full_image = full_image

        for category in categories:
            sp = _PackedSplit(packed_root, category, split)
            si = len(self.splits)
            self.splits.append(sp)
            for i, ci in enumerate(sp.labels):
                label = f"{category}/{sp.classes[ci]}"
                if label in self.map:
                    self.samples.append((si, i, self.map[label]))

        if not self.samples:
            raise ValueError(f"No packed eval samples for categories={categories} ({split}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        si, i, gl = self.samples[idx]
        sp = self.splits[si]

        if self.full_image:
            img = torch.from_numpy(np.array(sp.images[i])).permute(2, 0, 1)
            crops = img.unsqueeze(0)

            coords = torch.zeros((1, 2), dtype=torch.float32)

        else:
            crops = torch.stack([
                _crop_at(
                    sp.images[i],
                    sp.coords[i, j, 0],
                    sp.coords[i, j, 1],
                    self.crop_size,
                )
                for j in range(self.num_salient_points)
            ], dim=0)

            coords = torch.tensor(
                sp.coords[i, :self.num_salient_points],
                dtype=torch.float32,
            )

            coords[:, 0] /= sp.images.shape[2]
            coords[:, 1] /= sp.images.shape[1]

        one_hot = torch.zeros(len(self.map))
        one_hot[gl] = 1.0

        return crops, coords, one_hot

def make_packed_datasets(categories, packed_root="fixation_data",
                         num_salient_points=16, label_map=None,
                         max_images_per_class=None, full_image=False):
    """Packed-mode counterpart of make_datasets. Returns the same
    {"train","valid","test"} dict plus the global label map. The returned
    datasets yield *raw uint8 crops*; apply salience_trans.OnTheFlyTransform
    (train/valid/test type per split) on the GPU before the model."""
    if label_map is None:
        label_map = build_packed_label_map(packed_root, categories, split="train")

    return {
        "train": PackedFixationTrainDataset(
            categories, packed_root, "train", num_salient_points, label_map,
            max_images_per_class=max_images_per_class, full_image=full_image),
        "valid": PackedFixationEvalDataset(
            categories, packed_root, "valid", num_salient_points, label_map, full_image=full_image),
        "test": PackedFixationEvalDataset(
            categories, packed_root, "test", num_salient_points, label_map, full_image=full_image),
    }, label_map
