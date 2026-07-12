import math
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F


def show_images(imgs, cols=3, figsize=None):
    """Display one or more images (tensors or array-likes) in a grid."""
    if isinstance(imgs, torch.Tensor):
        if imgs.ndim == 2 or imgs.ndim == 3:
            imgs = [imgs]
        elif imgs.ndim == 4:
            imgs = list(imgs)
        else:
            raise ValueError(f"Tensor of unsupported shape {imgs.shape}")
    else:
        imgs = list(imgs)

    np_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3:
                t = t.permute(1, 2, 0)
            elif t.ndim != 2:
                raise ValueError(f"Tensor of unsupported shape {img.shape}")
            arr = t.numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
            np_imgs.append(arr)
        else:
            arr = np.asarray(img)
            if arr.ndim not in (2, 3):
                raise ValueError(f"Array of unsupported shape {arr.shape}")
            np_imgs.append(arr)

    n = len(np_imgs)
    if n == 0:
        return

    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (cols * 2, rows * 2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, im in zip(axes, np_imgs):
        ax.imshow(im, interpolation='nearest')
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def list_classes(split_dir):
    """Return the sorted list of class (folder) names under a split directory."""
    if not os.path.isdir(split_dir):
        raise ValueError(f"Directory not found: {split_dir!r}")
    result = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )

    #print(result)
    return result


def build_global_label_map(processed_root, categories, variant="lp", split="train"):
    """
    Build a single global label map spanning every class of every category, so a
    single unified softmax can be trained over faces + houses + objects.

    Classes are namespaced by category ("<category>/<class>") to avoid collisions,
    and assigned contiguous indices in the order categories are listed.

    Args:
        processed_root (str): e.g. "processed_data"
        categories (list[str]): e.g. ["faces", "objects"]
        variant (str): "lp" or "cnn"
        split (str): split used to enumerate the class set (use "train")

    Returns:
        dict: { "<category>/<class>": global_idx }
    """
    mapping = {}
    idx = 0
    for category in categories:
        split_dir = os.path.join(processed_root, category, variant, split)
        #print("split_dir:", split_dir)

        for cls_name in list_classes(split_dir):
            mapping[f"{category}/{cls_name}"] = idx
            idx += 1
    if not mapping:
        raise ValueError(
            f"No classes found under {processed_root} for categories={categories}, "
            f"variant={variant}, split={split}. Did you run preprocess.py?"
        )
    return mapping


def label_to_one_hot(label, mapping):
    """Convert a (namespaced) class label into a one-hot float tensor."""
    idx = mapping[label]
    num_classes = len(mapping)
    return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=num_classes).float()
