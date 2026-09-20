"""
make_rfw_wm64.py

Build faces_rfwWM64: 64 white male identities x 4 photos, held out from every
model, as the in-the-wild counterpart to faces_cfdWM64 (64 white male CFD
identities, studio-controlled). Same N, same race, same sex -- so a Yin battery
run over both isolates natural photo variation from everything else.

Source is faces_rfwHO, the 500-identity held-out store (125 per race, 4 photos
each), sliced to its Caucasian identities. RFW ships **no sex labels** -- only
race directories and Freebase MIDs -- so the male set below was read off contact
sheets of all 125 Caucasian held-out identities by eye on 2026-09-02. These are
visual judgements, not ground-truth metadata; CFD's own labels are coder
consensus, so the two sets are comparable in kind but not in provenance.

Two identities were dropped from the male pool for stimulus quality rather than
sex: index 109 is a Renaissance painting, index 79 is a heavily blurred profile.
75 clean males remained; the first 64 in packed order are kept (packed order is
alphabetical by Freebase MID, so it is arbitrary with respect to anything the
simulations measure).

Pure re-pack: crops and fixation coordinates copied verbatim from faces_rfwHO.

    python scripts/make_rfw_wm64.py
"""
import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
SRC, OUT, N_KEEP = "faces_rfwHO", "faces_rfwWM64", 64

# Indices into the Caucasian subset of faces_rfwHO/valid, in packed order.
MALE = [1, 2, 3, 4, 5, 8, 9, 10, 12, 14, 17, 20, 21, 22, 24, 27, 28, 29, 30, 31,
        32, 33, 34, 39, 41, 42, 43, 44, 45, 48, 50, 53, 55, 57, 60, 61, 63,
        64, 65, 66, 67, 69, 70, 71, 72, 73, 75, 78, 80, 81, 82, 83, 84, 86,
        88, 89, 92, 93, 95,
        96, 97, 98, 99, 103, 104, 106, 110, 112, 113, 114, 115, 118, 120, 121, 122]


def main():
    d = os.path.join(PACKED_ROOT, SRC, "valid")
    meta = json.load(open(os.path.join(d, "meta.json")))
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")

    cau = [c for c in meta["classes"] if c.startswith("rfwCaucasian")]
    assert len(cau) == 125, f"expected 125 Caucasian held-out ids, got {len(cau)}"
    assert max(MALE) < len(cau) and len(set(MALE)) == len(MALE)
    print(f"{len(MALE)} males identified of {len(cau)} Caucasian held-out ids")

    classes = [cau[i] for i in MALE][:N_KEEP]
    assert len(classes) == N_KEEP, f"only {len(classes)} males, need {N_KEEP}"
    index = {c: i for i, c in enumerate(classes)}

    rows_by_class = {}
    for row, ci in enumerate(meta["labels"]):
        rows_by_class.setdefault(meta["classes"][ci], []).append(row)

    rows, labels, stems = [], [], []
    for cls in classes:
        for row in rows_by_class[cls]:
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
    print(f"{OUT}/valid: {len(classes)} identities, {n} images "
          f"({n / len(classes):.1f}/id) -> {out}")


if __name__ == "__main__":
    main()
