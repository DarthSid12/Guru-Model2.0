"""
make_houses_common.py

Pack one house store that is HELD OUT for every model in the comparison.

The house arm has never been comparable across models: r7 trained on all 201
ZuBuD buildings (zero held out), r8 and house_control_r1 on 40 each. Each model
therefore had a different -- and differently contaminated -- ZuBuD test, which is
why their house numbers could not be read against each other.

This packs the frontal exterior photos of the Houses-dataset (fixation_data/
houses_ident, one image per building), which is a DIFFERENT dataset from ZuBuD:
no model in the comparison trained on any of it, so every building here is unseen
by all three and their house rows become directly comparable for the first time.

Note on the exclusion below: it drops houses_ident buildings whose numeric id
appears in house_control_r1's label_map. That label_map lists ZUBUD building ids,
so the match is a numeric coincidence across two unrelated datasets, not a real
overlap -- it was written when house_control_r1 was believed to have trained on
the Houses-dataset (corrected 2026-08-21). It is harmless (nothing here was
trained on either way) but it means the 94 buildings are an arbitrary subset of
the ~100 in the houses_ident 'valid' split rather than a principled one.

Yin only. The packed houses_ident 'valid' and 'test' splits hold the SAME images
(verified bit-identical), so a Kanwisher identity match on this store would be
matching an image against itself. Kanwisher houses must use ZuBuD, whose valid
and test splits are genuinely different views (view04 vs view05).

    python make_houses_common.py
"""

import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
SRC = "houses_ident"
OUT = "houses_common"
DAVID_LABEL_MAP = ("/home/d1deutsch/Guru-Model2.0/runs/"
                   "faces_objects_houses_lp_16fix_lr0.001_resnet18_house_control_r1/"
                   "label_map.json")


def main():
    d = os.path.join(PACKED_ROOT, SRC, "valid")
    meta = json.load(open(os.path.join(d, "meta.json")))
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")

    # house_control_r1 names its house classes by bare building id ('101'),
    # houses_ident by '<id>_frontal', so the exclusion matches on the id prefix.
    lm = json.load(open(DAVID_LABEL_MAP))
    seen = {k.split("/", 1)[1] for k in lm if k.startswith("houses/")}

    keep = [(row, meta["classes"][ci])
            for row, ci in enumerate(meta["labels"])
            if meta["classes"][ci].split("_")[0] not in seen]
    if not keep:
        raise SystemExit("nothing left after exclusions")

    n = len(keep)
    out_img = np.empty((n,) + images.shape[1:], dtype=images.dtype)
    out_crd = np.empty((n,) + coords.shape[1:], dtype=coords.dtype)
    classes, labels, stems = [], [], []
    for out_row, (row, cls) in enumerate(keep):
        out_img[out_row] = images[row]
        out_crd[out_row] = coords[row]
        classes.append(cls)
        labels.append(out_row)
        stems.append(cls)

    out_dir = os.path.join(PACKED_ROOT, OUT, "valid")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "images.npy"), out_img)
    np.save(os.path.join(out_dir, "coords.npy"), out_crd)
    json.dump({"classes": classes, "labels": labels, "stems": stems,
               "input_size": meta["input_size"], "num_coords": meta["num_coords"]},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print(f"wrote {out_dir}: {n} buildings held out from all three models "
          f"({len(meta['labels']) - n} dropped as seen by house_control_r1)")


if __name__ == "__main__":
    main()
