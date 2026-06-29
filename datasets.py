"""
datasets.py

Combined multi-category datasets for training a single network on faces +
houses + objects. Every class of every category lives in one unified label
space (see utils.build_global_label_map), so the model has a single softmax
head over all exemplars across categories.

Processed data layout (produced by preprocess.py):
    processed_data/<category>/<variant>/<split>/<class>/<img>_proc<n>.png
where:
    <category> in {faces, houses, objects, ...}
    <variant>  in {lp, cnn}
    <split>    in {train, valid, test}
"""

import os
import re
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
                  num_salient_points=32, label_map=None):
    """
    Build train/valid/test datasets spanning all `categories` under one global
    label space.

    Args:
        categories (list[str]): e.g. ["faces", "objects"]
        processed_root (str): root of preprocessed data
        variant (str): "lp" (log-polar) or "cnn"
        num_salient_points (int): fixations per base image
        label_map (dict|None): global label map; built from the train split if None
    """
    if label_map is None:
        label_map = build_global_label_map(processed_root, categories, variant, split="train")

    return {
        "train": CombinedSalienceDatasetBatched(
            categories, processed_root, variant, "train", num_salient_points, label_map),
        "valid": CombinedSalienceDataset(
            categories, processed_root, variant, "valid", num_salient_points, label_map),
        "test": CombinedSalienceDataset(
            categories, processed_root, variant, "test", num_salient_points, label_map),
    }, label_map


class CombinedSalienceDatasetBatched(Dataset):
    """Training set: one sample == one fixation patch (path, global_label)."""

    def __init__(self, categories, processed_root, variant, split,
                 num_salient_points, label_map):
        self.num_salient_points = num_salient_points
        self.map = label_map
        self.samples = []  # [(path, "<category>/<class>"), ...]

        for category in categories:
            split_dir = os.path.join(processed_root, category, variant, split)
            for cls_name in list_classes(split_dir):
                label = f"{category}/{cls_name}"
                if label not in self.map:
                    continue
                cls_dir = os.path.join(split_dir, cls_name)
                for fname in os.listdir(cls_dir):
                    idx = _proc_index(fname)
                    if idx is not None and idx < num_salient_points:
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
                 num_salient_points, label_map):
        self.num_salient_points = num_salient_points
        self.map = label_map
        self.samples = []  # [(base_id, [proc_paths...], "<category>/<class>"), ...]

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
