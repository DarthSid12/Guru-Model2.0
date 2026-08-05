import os
import torch
import torch.nn.functional as F
import random

def list_classes(split_dir):
    """Return the sorted list of class (folder) names under a split directory."""
    if not os.path.isdir(split_dir):
        raise ValueError(f"Directory not found: {split_dir!r}")
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


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

def normalize_coords_tensor(coords, img_size):
    coords /= img_size
    coords -= 0.5

def shuffle_fixations_one_image(inputs, coords):
    """
    inputs : (N, C, H, W)
    coords : (N, 4)  # (x, y, dx, dy)

    Returns:
        shuffled_inputs
        shuffled_coords  # recomputed (x, y, dx, dy)
    """

    # pair each fixation with its coordinates
    data = list(zip(inputs, coords[:, :2]))   # only keep x,y

    random.shuffle(data)

    new_inputs = torch.stack([d[0] for d in data])
    coords_xy = torch.stack([d[1] for d in data])

    deltas = torch.zeros_like(coords_xy)
    deltas[1:] = coords_xy[1:] - coords_xy[:-1]

    new_coords = torch.cat([coords_xy, deltas], dim=1)

    return new_inputs, new_coords

def shuffle_fixations(inputs, coords):
    new_inputs = []
    new_coords = []

    for x in range(len(inputs)):
        new_i, new_c = shuffle_fixations_one_image(inputs[x], coords[x])
        new_inputs.append(new_i)
        new_coords.append(new_c)

    return torch.stack(new_inputs), torch.stack(new_coords)