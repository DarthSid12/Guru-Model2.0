"""
make_houses_ho64.py

Build houses_ho64: 64 Houses-dataset buildings that are held out from EVERY
model in the r15/r16 comparison, as a clean replacement for houses_yin64.

Why houses_yin64 stopped working: r16 trains `houses_gen`, which folds 97
leftover ZuBuD buildings into a single generic `house` label. That appears to
collapse ZuBuD representations together, and all three r16 models score
houses_yin64 at 33.9-43.1% upright (at or below the 50% 2AFC chance floor)
against r15_vgg's 90.6% on the identical store.

houses_ho64 avoids that entirely: its buildings come from the Houses-dataset
`valid` split, verified disjoint from
  * `houses/train`     (435 buildings, r15's generic house exposure), and
  * `houses_gen/train` (726 buildings, r16's),
so these 64 were never trained on by any model -- not even as generic "house"
exemplars -- and they are a different dataset from ZuBuD altogether.

Source is houses_common (94 buildings), itself a slice of the Houses-dataset
frontal exteriors. Cut to 64 so the store matches faces_cfdWM64, faces_rfwWM64
and objects, all of which yield exactly 64 items for the standard 40 study /
24 test Yin design.

Yin only: the packed houses_ident valid and test splits hold the SAME images, so
a Kanwisher identity match here would match an image against itself.

Pure re-pack: crops and fixation coordinates copied verbatim.

    python scripts/make_houses_ho64.py
"""
import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
SRC, OUT, N_KEEP = "houses_common", "houses_ho64", 64


def main():
    d = os.path.join(PACKED_ROOT, SRC, "valid")
    meta = json.load(open(os.path.join(d, "meta.json")))
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")

    # Assert the held-out property rather than trusting it.
    seen = set()
    for cat in ("houses", "houses_gen"):
        p = os.path.join(PACKED_ROOT, cat, "train", "meta.json")
        if os.path.exists(p):
            seen |= set(json.load(open(p))["stems"])
    overlap = set(meta["stems"]) & seen
    assert not overlap, f"{SRC} overlaps a training split: {sorted(overlap)[:5]}"
    print(f"{SRC}: {len(meta['classes'])} buildings, none in any training split "
          f"({len(seen)} training stems checked)")

    classes = meta["classes"][:N_KEEP]
    assert len(classes) == N_KEEP, f"only {len(classes)} buildings, need {N_KEEP}"
    index = {c: i for i, c in enumerate(classes)}

    rows, labels, stems = [], [], []
    for row, ci in enumerate(meta["labels"]):
        cls = meta["classes"][ci]
        if cls in index:
            rows.append(row); labels.append(index[cls]); stems.append(meta["stems"][row])

    out = os.path.join(PACKED_ROOT, OUT, "valid")
    os.makedirs(out, exist_ok=True)
    n = len(rows)
    oi = np.lib.format.open_memmap(os.path.join(out, "images.npy"), mode="w+",
                                   dtype=images.dtype, shape=(n,) + images.shape[1:])
    oc = np.lib.format.open_memmap(os.path.join(out, "coords.npy"), mode="w+",
                                   dtype=coords.dtype, shape=(n,) + coords.shape[1:])
    for i, row in enumerate(rows):
        oi[i] = images[row]; oc[i] = coords[row]
    oi.flush(); oc.flush(); del oi, oc
    json.dump({"classes": classes, "labels": labels, "stems": stems,
               "input_size": meta["input_size"], "num_coords": meta["num_coords"]},
              open(os.path.join(out, "meta.json"), "w"))
    print(f"{OUT}/valid: {len(classes)} buildings, {n} images -> {out}")


if __name__ == "__main__":
    main()
