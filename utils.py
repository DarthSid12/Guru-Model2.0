import os
import torch
import torch.nn.functional as F


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

"""
These functions are for better noise fine-tuning;
They use functional programming to tune a certain parameter to any function.

We essentially binary search down to the required precision;
if the bounds are too small or too large, we extend them by doubling the distance.
"""

"""
Searches for a parameter value that closest leads to function(param) = target
Obviously presumes the function is monotonic, which noise functions should be
"""
def pure_binary_search(low, high, function,
                       target, # what we want function(param) to be
                       param_precision=0.01,
                       pick='LOW'): # whether we pick the bound right below or right above
    while high - low > param_precision:
        mid = (low + high) / 2
        value = function(mid)

        if value < target:
            low = mid
        else:
            high = mid

    return low if pick == 'LOW' else high

def search(low, high, function, target,
           dir = -1, # -1 means monotonically dec; 1 means inc; more noise = less accuracy leads to this default
           param_precision=0.01, pick='LOW', 
           extend_low = True, # whether we allow the lower bound to be decreased,
           extend_high = True):
    if (function(low) - target) * dir > 0:
        if not extend_low:
            return low
        low -= (high - low)
        return search(low, high, function, target, dir, param_precision, pick, extend_low, extend_high)
    if (function(high) - target) * dir < 0:
        if not extend_high:
            return high
        high += (high - low)
        return search(low, high, function, target, dir, param_precision, pick, extend_low, extend_high)
    return pure_binary_search(low, high, function, target, param_precision, pick)

