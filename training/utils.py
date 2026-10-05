
# Support direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

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
                       pick='LOW', # whether we pick the bound right below or right above
                       dir=-1): # -1 monotonically decreasing, +1 increasing
    # `dir` is REQUIRED here, not just in search(). The original bisection was
    # `if value < target: low = mid`, which is only correct for an INCREASING
    # function. Retrieval-noise accuracy DECREASES with p (which is why search()
    # defaults dir=-1), so on a real noise curve that test moved the bracket the
    # wrong way every iteration and the result converged on the upper bound
    # instead of the crossing: on a linear curve with its true answer at p=0.20,
    # search(0.2, 0.36, f, 0.96) returned 0.3550, where f is 0.929 rather than
    # the requested 0.96. Verified against an increasing function too, where the
    # original was already exact (4.9902 for a truth of 5.0) -- so this changes
    # nothing for dir=+1 callers and fixes every dir=-1 one.
    while high - low > param_precision:
        mid = (low + high) / 2
        value = function(mid)

        # Move `low` up when the parameter needs to increase to approach target.
        if (value - target) * dir < 0:
            low = mid
        else:
            high = mid

    return low if pick == 'LOW' else high

def search(low, high, function, target,
           dir = -1, # -1 means monotonically dec; 1 means inc; more noise = less accuracy leads to this default
           param_precision=0.01, pick='LOW',
           extend_low = True, # whether we allow the lower bound to be decreased,
           extend_high = True,
           max_extend = 8,                      # give up rather than recurse forever
           hard_low = None, hard_high = None):  # domain limits; never extend past these
    # Bound extension originally recursed with no termination guard. When the
    # target is unreachable -- which happens constantly here, because
    # accuracy-vs-noise is U-shaped rather than monotonic (see the note in
    # simulate_yin1969_bothnoise.py) -- it doubles the bracket forever, spending
    # one GPU forward pass per level, and finally dies on RecursionError. Seven
    # units in the first batch burned up to 4.6 h each exactly that way.
    # Now: never leave [hard_low, hard_high], cap the doublings, and raise a
    # clear error the caller can report instead of returning a bogus number.
    if max_extend <= 0:
        raise ValueError(f"search: target {target} unreachable in [{low:.3f}, {high:.3f}] "
                         f"after the allowed extensions")
    if extend_low and ((function(low) - target) * dir > 0):
        new_low = low - (high - low)
        if hard_low is not None and new_low < hard_low:
            new_low = hard_low
            if new_low >= low:
                raise ValueError(f"search: target {target} unreachable at the lower "
                                 f"domain limit {hard_low}")
        return search(new_low, high, function, target, dir, param_precision, pick,
                      extend_low, extend_high, max_extend - 1, hard_low, hard_high)
    if extend_high and ((function(high) - target) * dir < 0):
        new_high = high + (high - low)
        if hard_high is not None and new_high > hard_high:
            new_high = hard_high
            if new_high <= high:
                raise ValueError(f"search: target {target} unreachable at the upper "
                                 f"domain limit {hard_high}")
        return search(low, new_high, function, target, dir, param_precision, pick,
                      extend_low, extend_high, max_extend - 1, hard_low, hard_high)
    return pure_binary_search(low, high, function, target, param_precision, pick, dir)
