"""
make_objects_ladder.py

Object familiarity ladder: 64 ImageNet classes per store, swapping 8 trained
("known") classes for 8 never-trained ("unknown") ones at each rung, from
64 known / 0 unknown down to 0 known / 64 unknown.

The counterpart of the face ladder (make_faces_mix64.py) and the house ladder
(make_houses_ladder.py). Store size is held at 64 so the Yin 40 study / 24 test
split is exact at every rung and only the familiar/unfamiliar ratio moves.

The unknown half comes from fixation_data/objects_all128_backup, the 128-class
store that predates the 2026-07-16 prune. r7 and r8 trained on the 64 classes
that survived the prune, so the 64 pruned ones are genuinely unseen -- this is
the only held-out object set that exists. house_control_r1 trained on all 128
and therefore has NO held-out objects and cannot run this ladder.

simulate_yin1969.build_items takes one item per class (that class's first packed
image), so each store carries one image per class and 64 classes is exactly the
40 + 24 the design needs.

    python make_objects_ladder.py --model r8_developmental
"""

import argparse
import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
SRC = "objects_all128_backup"
STEP = 8
TOTAL = 64

MODELS = {
    "r7_curriculum":
        "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r7_curriculum",
    "r8_developmental":
        "runs/faces_objects_houses_zubud_lp_16fix_lr0.001_resnet18_r8_developmental",
}
TAGS = {"r7_curriculum": "r7", "r8_developmental": "r8"}


def trained_classes(run_dir):
    lm = json.load(open(os.path.join(run_dir, "label_map.json")))
    names = {k.split("/", 1)[1] for k in lm if k.split("/")[0] == "objects"}
    if not names:
        raise SystemExit(f"no object classes in {run_dir}/label_map.json")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="r8_developmental")
    args = ap.parse_args()
    run_dir, tag = MODELS[args.model], TAGS[args.model]
    seen = trained_classes(run_dir)

    d = os.path.join(PACKED_ROOT, SRC, "valid")
    meta = json.load(open(os.path.join(d, "meta.json")))
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")

    # First packed image of each class -- the one build_items would pick anyway.
    first = {}
    for row, ci in enumerate(meta["labels"]):
        first.setdefault(meta["classes"][ci], row)

    known = sorted((c, r) for c, r in first.items() if c in seen)
    unknown = sorted((c, r) for c, r in first.items() if c not in seen)
    print(f"{args.model}: {len(known)} known / {len(unknown)} unknown "
          f"of {len(first)} packed classes")
    for label, pool in (("known", known), ("unknown", unknown)):
        if len(pool) < TOTAL:
            raise SystemExit(f"need {TOTAL} {label} classes, found {len(pool)}")

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

        out_dir = os.path.join(PACKED_ROOT, f"objects_{tag}lad_k{k}", "valid")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "images.npy"), out_img)
        np.save(os.path.join(out_dir, "coords.npy"), out_crd)
        json.dump({"classes": classes, "labels": labels, "stems": stems,
                   "input_size": meta["input_size"],
                   "num_coords": meta["num_coords"]},
                  open(os.path.join(out_dir, "meta.json"), "w"))
        print(f"  wrote objects_{tag}lad_k{k}: {k} known + {TOTAL - k} unknown")


if __name__ == "__main__":
    main()
