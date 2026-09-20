"""
make_rfw_race_stores.py

Slice the packed faces_rfw store into two race-keyed categories so a curriculum
can introduce white faces before the rest:

    faces_rfwW   Caucasian identities
    faces_rfwO   African + Asian + Indian, round-robin interleaved by depth so
                 the first N classes are balanced N/3 per race

Two deviations from a plain re-slice, both decided 2026-09-01:

  * the source `test` split is folded into `train`. RFW's real test is
    held-out IDENTITIES (faces_rfwHO), so a held-out-image split for a trained
    identity is redundant, and folding it lifts training depth ~3.0 -> ~4.0
    photos/id. `valid` is kept: early stopping and best_model.pth run off
    aggregate valid accuracy, and a category with no valid rows goes invisible.
    The written `test` split is a copy of `valid`, because train.py evaluates
    `test` under a 180-degree rotation -- it is the inversion readout, not a
    second held-out set. Same images upright (valid) and inverted (test) is in
    fact the cleaner matched-image inversion contrast, and nothing in either
    split is trained on. (houses_common does the same thing.)

  * classes are written in DEPTH-DESCENDING order (ties by name). train.py's
    --max-classes-per-category keeps the first N classes in stored order, so
    `faces_rfwW=1200` takes the 1200 deepest Caucasian identities rather than
    an alphabetical slice.

This is a pure re-pack: crops and fixation coordinates are copied verbatim from
faces_rfw, which was packed by preprocess_fixations.py at input_size 224 with
32 coords. No salience model is re-run, so the coordinates stay bit-identical
(re-packing would resample them -- the sampler is not deterministic).

    python scripts/make_rfw_race_stores.py
"""
import json
import os
from collections import defaultdict

import numpy as np

PACKED_ROOT = "fixation_data"
SRC = "faces_rfw"
WHITE = "rfwCaucasian"
OTHER = ["rfwAfrican", "rfwAsian", "rfwIndian"]
# Headroom above the 1200 / 300 the r16 runs ask for, so the split can be
# re-cut with --max-classes-per-category without re-packing.
KEEP_W, KEEP_O = 1800, 900


def read_split(split):
    d = os.path.join(PACKED_ROOT, SRC, split)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")
    assert len(meta["labels"]) == len(images) == len(coords) == len(meta["stems"])
    return meta, images, coords


def rows_by_class(meta):
    """class name -> list of row indices, in packed order."""
    out = defaultdict(list)
    for row, ci in enumerate(meta["labels"]):
        out[meta["classes"][ci]].append(row)
    return out


def main():
    tr_meta, tr_img, tr_co = read_split("train")
    te_meta, te_img, te_co = read_split("test")
    va_meta, va_img, va_co = read_split("valid")
    for m in (te_meta, va_meta):
        assert m["input_size"] == tr_meta["input_size"]
        assert m["num_coords"] == tr_meta["num_coords"]

    tr_rows, te_rows, va_rows = (rows_by_class(m) for m in (tr_meta, te_meta, va_meta))

    # Training depth after folding test into train, which is what the ordering
    # and the KEEP_* cuts are based on.
    depth = defaultdict(int)
    for d in (tr_rows, te_rows):
        for cls, rows in d.items():
            depth[cls] += len(rows)

    by_race = defaultdict(list)
    for cls, n in depth.items():
        by_race[cls.split("_")[0]].append(cls)
    for race in by_race:
        by_race[race].sort(key=lambda c: (-depth[c], c))

    # White: straight depth order. Other: round-robin across the three races by
    # within-race depth rank, so any prefix of the store is race-balanced.
    order = {"faces_rfwW": by_race[WHITE][:KEEP_W]}
    ranked = [by_race[r][:KEEP_O] for r in OTHER]
    interleaved = []
    for i in range(max(len(r) for r in ranked)):
        for r in ranked:
            if i < len(r):
                interleaved.append(r[i])
    order["faces_rfwO"] = interleaved[:KEEP_O]

    for out_cat, classes in order.items():
        index = {c: i for i, c in enumerate(classes)}
        # train = source train + source test; valid = source valid
        plan = {
            "train": [(tr_img, tr_co, tr_meta, tr_rows), (te_img, te_co, te_meta, te_rows)],
            "valid": [(va_img, va_co, va_meta, va_rows)],
            "test": [(va_img, va_co, va_meta, va_rows)],
        }
        for split, sources in plan.items():
            rows_out, labels, stems = [], [], []
            for img, co, meta, rows in sources:
                for cls in classes:
                    for row in rows.get(cls, []):
                        rows_out.append((img, co, row))
                        labels.append(index[cls])
                        stems.append(meta["stems"][row])
            if not rows_out:
                raise SystemExit(f"{out_cat}/{split}: no rows selected")

            d = os.path.join(PACKED_ROOT, out_cat, split)
            os.makedirs(d, exist_ok=True)
            n = len(rows_out)
            images = np.lib.format.open_memmap(
                os.path.join(d, "images.npy"), mode="w+", dtype=tr_img.dtype,
                shape=(n,) + tr_img.shape[1:])
            coords = np.lib.format.open_memmap(
                os.path.join(d, "coords.npy"), mode="w+", dtype=tr_co.dtype,
                shape=(n,) + tr_co.shape[1:])
            for i, (src_img, src_co, row) in enumerate(rows_out):
                images[i] = src_img[row]
                coords[i] = src_co[row]
            images.flush(); coords.flush()
            del images, coords

            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"classes": classes, "labels": labels, "stems": stems,
                           "input_size": tr_meta["input_size"],
                           "num_coords": tr_meta["num_coords"]}, f)

            per_race = defaultdict(int)
            for cls in classes:
                per_race[cls.split("_")[0]] += 1
            print(f"{out_cat}/{split}: {len(classes)} classes, {n} images "
                  f"({n / len(classes):.2f}/class) "
                  + ", ".join(f"{r} {c}" for r, c in sorted(per_race.items())))


if __name__ == "__main__":
    main()
