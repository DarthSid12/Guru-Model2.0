"""Pass 1 over the full VGGFace2 train archive: pull a few probe photos per
identity for race labelling (data_preparation/vggface2_race.py).

Candidates: every VGGFace2 TRAIN-split identity with >= 224 photos, enough for
the 200 train / 12 valid / 12 test split faces_vgg uses. Each identity's photos
are shuffled once with a fixed seed; the probes are the first N_PROBE of that
order, and the later packing pass takes its 224 from the SAME order, so probes
are a subset of the identity's own store photos and nothing is re-drawn.

Output: one pickle {"order": {id: [224 photo names]}, "probes": {id: {name: jpg bytes}}}.
Streams the .tar.gz once (it is not sorted by identity), keeping only members
in the probe set, so nothing but the probes touches disk.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import collections, pickle, random, sys, tarfile, time

ROOT = "data/vggface2_raw/hf_mirror"
N_NEED, N_PROBE, SEED = 224, 10, 20260923
OUT = "data/vggface2_raw/probe_10.pkl"

by = collections.defaultdict(list)
for line in open(f"{ROOT}/data/train_list.txt"):
    ident, name = line.strip().split("/")
    by[ident].append(name)
rng = random.Random(SEED)
order = {}
for ident in sorted(by):
    names = sorted(by[ident])
    if len(names) < N_NEED:
        continue
    rng.shuffle(names)
    order[ident] = names[:N_NEED]
want = {f"train/{i}/{n}": i for i, ns in order.items() for n in ns[:N_PROBE]}
print(f"{len(order)} candidate identities, {len(want)} probe photos to pull", flush=True)

probes = collections.defaultdict(dict)
t0, seen = time.time(), 0
with tarfile.open(f"{ROOT}/data/vggface2_train.tar.gz", "r|gz") as tar:
    for mem in tar:
        seen += 1
        ident = want.get(mem.name)
        if ident is not None:
            probes[ident][mem.name.rsplit("/", 1)[1]] = tar.extractfile(mem).read()
        if seen % 200000 == 0:
            got = sum(len(v) for v in probes.values())
            print(f"  {seen:,} members read, {got:,}/{len(want):,} probes, {time.time()-t0:.0f}s", flush=True)
got = sum(len(v) for v in probes.values())
if got != len(want):
    sys.exit(f"only found {got} of {len(want)} probe photos -- archive incomplete?")
pickle.dump({"order": order, "probes": dict(probes), "seed": SEED}, open(OUT, "wb"))
print(f"done: {got:,} probes for {len(probes)} identities -> {OUT} ({time.time()-t0:.0f}s)")
