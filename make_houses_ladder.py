"""
make_houses_ladder.py

Build a house familiarity ladder for one model: 40 ZuBuD buildings per store,
swapping 4 trained ("known") buildings for 4 never-trained ("unknown") ones at
each rung, from 40 known / 0 unknown down to 0 known / 40 unknown.

This is the house counterpart of the face ladder in make_faces_mix64.py. The
store size is held at 40 so the Yin design is identical at every rung and only
the familiar/unfamiliar ratio moves; with 40 items simulate_yin1969.py scales
its 40 study / 24 test split to 25 / 15, keeping the same 5:3 ratio.

Which buildings count as "known" is per-model and must be read from that model's
label_map: r8 trained on ZuBuD buildings 1-40, house_control_r1 on a different
40 (the two overlap in only 12), and r7 trained on all 201 so it has no unknown
houses and no ladder.

Name canonicalisation matters here. house_control_r1's label_map files houses
under `houses/` with bare numeric ids ('101') while the packed store is
`houses_zubud` with names like 'house0101', so a plain string match finds
nothing and silently reports every building as unseen -- the bug that made its
earlier ZuBuD rows include its own training buildings. Both sides are reduced to
an integer id before comparing, and a zero-length known set is an error.

Known and unknown are both taken in sorted order, so a store is reproducible
without a seed and the rungs nest: the k=36 store's known half is the k=40
store's first 36.

    python make_houses_ladder.py --model house_control_r1
"""

import argparse
import json
import os
import re

import numpy as np

PACKED_ROOT = "fixation_data"
SRC = "houses_zubud"
STEP = 4
TOTAL = 40

_GURU = "/home/d1deutsch/Guru-Model2.0"
MODELS = {
    "r8_developmental":
        "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r8_developmental",
    "house_control_r1":
        f"{_GURU}/runs/faces_objects_houses_lp_16fix_lr0.001_resnet18_house_control_r1",
}
# Short tag that goes in the store name, so two models' ladders never collide.
TAGS = {"r8_developmental": "r8", "house_control_r1": "dv"}


def building_id(name):
    """'house0101' / '101' / 'houses/101' -> 101."""
    m = re.search(r"(\d+)", name.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else None


def trained_ids(run_dir):
    lm = json.load(open(os.path.join(run_dir, "label_map.json")))
    ids = {building_id(k) for k in lm if k.split("/")[0].startswith("houses")}
    ids.discard(None)
    if not ids:
        raise SystemExit(f"no house classes found in {run_dir}/label_map.json")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="house_control_r1")
    args = ap.parse_args()

    run_dir = MODELS[args.model]
    tag = TAGS[args.model]
    seen = trained_ids(run_dir)

    d = os.path.join(PACKED_ROOT, SRC, "valid")
    meta = json.load(open(os.path.join(d, "meta.json")))
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")

    # One row per building in this store, split by whether the model trained on it.
    rows = [(meta["classes"][ci], row) for row, ci in enumerate(meta["labels"])]
    known = sorted((c, r) for c, r in rows if building_id(c) in seen)
    unknown = sorted((c, r) for c, r in rows if building_id(c) not in seen)
    print(f"{args.model}: {len(known)} known / {len(unknown)} unknown "
          f"of {len(rows)} packed buildings")
    if len(known) < TOTAL:
        raise SystemExit(f"need {TOTAL} known buildings, found {len(known)}")
    if len(unknown) < TOTAL:
        raise SystemExit(f"need up to {TOTAL} unknown buildings, found {len(unknown)}")

    for k in range(TOTAL, -1, -STEP):
        picked = ([("known_" + c, r) for c, r in known[:k]]
                  + [("unseen_" + c, r) for c, r in unknown[:TOTAL - k]])
        out_img = np.empty((TOTAL,) + images.shape[1:], dtype=images.dtype)
        out_crd = np.empty((TOTAL,) + coords.shape[1:], dtype=coords.dtype)
        classes, labels, stems = [], [], []
        for out_row, (cls, row) in enumerate(picked):
            out_img[out_row] = images[row]
            out_crd[out_row] = coords[row]
            classes.append(cls)
            labels.append(out_row)
            stems.append(cls)

        out_dir = os.path.join(PACKED_ROOT, f"houses_{tag}lad_k{k}", "valid")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "images.npy"), out_img)
        np.save(os.path.join(out_dir, "coords.npy"), out_crd)
        json.dump({"classes": classes, "labels": labels, "stems": stems,
                   "input_size": meta["input_size"],
                   "num_coords": meta["num_coords"]},
                  open(os.path.join(out_dir, "meta.json"), "w"))
        print(f"  wrote houses_{tag}lad_k{k}: {k} known + {TOTAL - k} unknown")


if __name__ == "__main__":
    main()
