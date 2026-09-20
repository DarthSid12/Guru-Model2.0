"""
make_houses_generic.py

Build the "generic house" category for the r16 models: one class named `house`
holding buildings the model sees but never individuates, mirroring how people
know thousands of houses at basic level while individuating a handful.

Two stores, because the choice of exemplar pool is a real design decision:

  houses_rest  the 97 ZuBuD buildings left over once the 40 trained identities
               (house0001-0040) and the 64 Yin test buildings (house0041-0104)
               are removed. Same domain as both, so the model gains ZuBuD
               domain familiarity without gaining any ZuBuD identity.

  houses_gen   houses_rest PLUS the existing `houses` store (the Houses-dataset
               frontal exteriors, 635 buildings, a different dataset). Broader
               exemplar diversity, and dilutes the ZuBuD domain exposure.

The ZuBuD split structure is inherited: 3 train / 1 valid / 1 test view per
building. Nothing here overlaps houses_yin64 -- that is asserted, not assumed.

Pure re-pack: crops and fixation coordinates are copied verbatim, so the packed
fixation coordinates stay bit-identical (re-packing would resample them).

    python scripts/make_houses_generic.py
"""
import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
N_IDENTITY = 40          # house0001-0040, trained as individual classes
SPLITS = ["train", "valid", "test"]


def read(category, split):
    d = os.path.join(PACKED_ROOT, category, split)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return (meta,
            np.load(os.path.join(d, "images.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "coords.npy"), mmap_mode="r"))


def write(category, split, sources, rows, input_size, num_coords):
    d = os.path.join(PACKED_ROOT, category, split)
    os.makedirs(d, exist_ok=True)
    n = len(rows)
    proto_img, proto_co = sources[0][1], sources[0][2]
    images = np.lib.format.open_memmap(os.path.join(d, "images.npy"), mode="w+",
                                       dtype=proto_img.dtype,
                                       shape=(n,) + proto_img.shape[1:])
    coords = np.lib.format.open_memmap(os.path.join(d, "coords.npy"), mode="w+",
                                       dtype=proto_co.dtype,
                                       shape=(n,) + proto_co.shape[1:])
    stems = []
    for i, (si, row) in enumerate(rows):
        meta, img, co = sources[si]
        images[i] = img[row]
        coords[i] = co[row]
        stems.append(meta["stems"][row])
    images.flush(); coords.flush()
    del images, coords
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"classes": ["house"], "labels": [0] * n, "stems": stems,
                   "input_size": input_size, "num_coords": num_coords}, f)
    print(f"  {category}/{split}: {n} images")


def main():
    zub = {sp: read("houses_zubud", sp) for sp in SPLITS}
    hou = {sp: read("houses", sp) for sp in SPLITS}

    all_ids = zub["train"][0]["classes"]
    yin = set(json.load(open(f"{PACKED_ROOT}/houses_yin64/valid/meta.json"))["classes"])
    identity = set(all_ids[:N_IDENTITY])
    rest = [c for c in all_ids if c not in identity and c not in yin]
    assert not (set(rest) & yin), "generic pool overlaps the Yin test buildings"
    assert not (set(rest) & identity), "generic pool overlaps the trained identities"
    print(f"ZuBuD: {len(all_ids)} buildings = {len(identity)} identities "
          f"+ {len(yin)} held out for Yin + {len(rest)} generic")
    rest = set(rest)

    input_size = zub["train"][0]["input_size"]
    num_coords = zub["train"][0]["num_coords"]
    for meta, _, _ in list(zub.values()) + list(hou.values()):
        assert meta["input_size"] == input_size and meta["num_coords"] == num_coords

    print("houses_rest (ZuBuD leftovers only):")
    for sp in SPLITS:
        meta, _, _ = zub[sp]
        rows = [(0, r) for r, ci in enumerate(meta["labels"])
                if meta["classes"][ci] in rest]
        write("houses_rest", sp, [zub[sp]], rows, input_size, num_coords)

    print("houses_gen (ZuBuD leftovers + Houses-dataset):")
    for sp in SPLITS:
        zmeta, _, _ = zub[sp]
        hmeta, _, _ = hou[sp]
        rows = [(0, r) for r, ci in enumerate(zmeta["labels"])
                if zmeta["classes"][ci] in rest]
        rows += [(1, r) for r in range(len(hmeta["labels"]))]
        write("houses_gen", sp, [zub[sp], hou[sp]], rows, input_size, num_coords)


if __name__ == "__main__":
    main()
