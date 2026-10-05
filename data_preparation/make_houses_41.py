"""
make_houses_41.py

Re-split a packed ZuBuD house store from 3 train / 1 valid / 1 test views per
building to 4 train / 1 valid, and write it out under a new name.

Why. ZuBuD carries 5 views per building, and the house arm is by far the most
data-starved category in the diet: houses_zubud137 trains on 137 ids x 3 views
= 411 images against ~96k for faces_vgg and ~66k for objects. Folding view05
into train makes that 548, a 33% gain, in the one arm where it matters most.

Why the test split is the one to spend. train.py's "test" split is NOT a
held-out generalisation set -- transforms["test"] is OnTheFlyTransform("test"),
which rotates 180 deg, so the test column is the per-epoch INVERTED-presentation
monitor. For ZuBuD that monitor compares upright view04 against inverted view05,
i.e. it confounds orientation with image set, and it is a training-time readout
only: every reported inversion result comes from simulate_yin1969.py, which
takes both orientations from the SAME valid images and is unaffected by this.

What this does NOT touch. simulate_kanwisher2023.py --category houses_zubud
--splits valid test needs two genuinely different views for cross-view identity
matching. It reads houses_zubud (all 201), not the store re-split here, so it
keeps working; and the better version of that test runs on held-out buildings
anyway (houses_ho64), which is a separate change.

Pure re-pack: crops and fixation coordinates are copied verbatim, no
re-rendering, so the new store is bit-identical to its source where they
overlap. The source store is left in place, so every earlier run keeps meaning
what it meant.

    python data_preparation/make_houses_41.py                       # houses_zubud137 -> houses_zubud137_41
    python data_preparation/make_houses_41.py --src houses_zubud    # any 5-view ZuBuD store
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import collections
import json
import os
import numpy as np
PACKED_ROOT = "fixation_data"


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    return (meta,
            np.load(os.path.join(d, "images.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "coords.npy"), mmap_mode="r"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="houses_zubud137")
    ap.add_argument("--out", default=None, help="default: <src>_41")
    ap.add_argument("--packed-root", default=PACKED_ROOT)
    args = ap.parse_args()
    out_name = args.out or f"{args.src}_41"

    src = os.path.join(args.packed_root, args.src)
    tr_meta, tr_img, tr_crd = load(os.path.join(src, "train"))
    te_meta, te_img, te_crd = load(os.path.join(src, "test"))
    va_meta, va_img, va_crd = load(os.path.join(src, "valid"))

    # Class INDICES are what meta["labels"] holds, so the two splits' class
    # lists must agree before their labels can be concatenated.
    if tr_meta["classes"] != te_meta["classes"]:
        raise SystemExit(f"{args.src}: train and test class lists differ; "
                         f"their label indices are not comparable")
    for k in ("input_size", "num_coords"):
        if tr_meta[k] != te_meta[k]:
            raise SystemExit(f"{args.src}: train {k}={tr_meta[k]} but test {k}={te_meta[k]}")

    before = collections.Counter(collections.Counter(tr_meta["labels"]).values())
    print(f"{args.src}: {len(tr_meta['classes'])} buildings, "
          f"train views/building {dict(before)}, "
          f"{len(tr_meta['labels'])} train + {len(te_meta['labels'])} test images")

    images = np.concatenate([np.asarray(tr_img), np.asarray(te_img)], axis=0)
    coords = np.concatenate([np.asarray(tr_crd), np.asarray(te_crd)], axis=0)
    labels = list(tr_meta["labels"]) + list(te_meta["labels"])
    stems = list(tr_meta["stems"]) + list(te_meta["stems"])
    if len(set(stems)) != len(stems):
        raise SystemExit("train and test share a stem: the same view would be packed twice")

    after = collections.Counter(collections.Counter(labels).values())
    print(f"  -> train views/building {dict(after)}, {len(labels)} train images")

    out_train = os.path.join(args.packed_root, out_name, "train")
    os.makedirs(out_train, exist_ok=True)
    np.save(os.path.join(out_train, "images.npy"), images)
    np.save(os.path.join(out_train, "coords.npy"), coords)
    json.dump({"classes": tr_meta["classes"], "labels": labels, "stems": stems,
               "input_size": tr_meta["input_size"], "num_coords": tr_meta["num_coords"]},
              open(os.path.join(out_train, "meta.json"), "w"))

    # valid copied verbatim; no test split is written at all -- datasets.py
    # skips a category that has none, and that absence is the point.
    out_valid = os.path.join(args.packed_root, out_name, "valid")
    os.makedirs(out_valid, exist_ok=True)
    np.save(os.path.join(out_valid, "images.npy"), np.asarray(va_img))
    np.save(os.path.join(out_valid, "coords.npy"), np.asarray(va_crd))
    json.dump(va_meta, open(os.path.join(out_valid, "meta.json"), "w"))

    print(f"wrote {args.packed_root}/{out_name}: "
          f"{len(labels)} train / {len(va_meta['labels'])} valid / no test")


if __name__ == "__main__":
    main()
