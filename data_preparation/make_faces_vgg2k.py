"""Build faces_vgg2k: 2048 VGGFace2 identities in a fixed White-first order.

Order (the curriculum takes prefixes of it via --curriculum-pin faces_vgg2k=@file):
  1-8      the hand-pinned identities (same as every r21 run)
  9-512    White only                       -> stages 1-8 are 100% White
  513-2048 988 White + 548 NonWhite, randomly interleaved
           -> 1500 White overall, other races first appear in stage 9

Race labels come from data_preparation/vggface2_race.py (FairFace, perceived race).
Ambiguous identities are never used. All 2048 come from VGGFace2's TRAIN split,
so faces_vggHO (VGGFace2 test split) stays identity-disjoint by construction.

Photos: each identity's 224 from the seeded order in probe_10.pkl -> first 200
train, next 12 valid, last 12 test (faces_vgg's split). Raw JPEGs are written to
data/faces_vgg2k/<split>/vgg_<id>/ for preprocess_fixations.py to pack, the same
pipeline faces_vgg went through.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv, os, pickle, random, sys, tarfile, time

PINS = ["n000002", "n000003", "n000004", "n000005", "n000012", "n000016", "n000018", "n000019"]
N_TOTAL, N_WHITE_FIRST, N_WHITE, SEED = 2048, 512, 1500, 20260923
LABELS = sys.argv[1] if len(sys.argv) > 1 else "data/vggface2_raw/race_labels_x1.0.csv"
OUT_RAW, ORDER = "data/faces_vgg2k", "data/vggface2_raw/faces_vgg2k_order.txt"

# RFW's four-group taxonomy over FairFace's seven classes. FairFace alone splits
# White faces between "White" and "Middle Eastern" (7 of our 8 hand-picked White
# pins scored P(White) 0.49-0.93 with Middle Eastern runner-up); RFW files both
# under Caucasian. Validated on RFW (200 ids/group, >=0.8): Caucasian recalled
# 89%; other groups are clean (1/200 Caucasians leaked into them); the one leak
# INTO Caucasian is Indian (24/200), closed below by rejecting any identity with
# P(Indian) > 0.05 from the White pool.
def four(r):
    g = lambda k: float(r["p_" + k])
    return {"Caucasian": g("White") + g("Middle Eastern"), "African": g("Black"),
            "Asian": g("East Asian") + g("Southeast Asian"), "Indian": g("Indian")}
lab, white, nonwhite = {}, [], []
for r in csv.DictReader(open(LABELS)):
    if r["p_White"] == "nan":
        continue
    q = four(r); best = max(q, key=q.get)
    if q[best] < 0.8:
        continue
    if best == "Caucasian" and float(r["p_Indian"]) <= 0.05:
        white.append(r["id"]); lab[r["id"]] = "Caucasian"
    elif best != "Caucasian":
        nonwhite.append(r["id"]); lab[r["id"]] = best
white, nonwhite = sorted(white), sorted(nonwhite)
# Pins are hand-labelled. The seven with enough photos to be candidates were
# also confirmed Caucasian by the rule above (0.93-1.00); n000003 has 205 photos,
# below the 224-photo candidate bar, so it was never probed and keeps its pin.
for p in PINS:
    if lab.get(p, "Caucasian") != "Caucasian":
        sys.exit(f"pin {p} classified {lab[p]}")
    lab[p] = "Caucasian"
print(f"pools: {len(white)} Caucasian, {len(nonwhite)} other-race "
      f"({dict(sorted(__import__('collections').Counter(lab[i] for i in nonwhite).items()))})")
n_nw = N_TOTAL - N_WHITE
if len(white) < N_WHITE or len(nonwhite) < n_nw:
    sys.exit(f"not enough: {len(white)} White (need {N_WHITE}), {len(nonwhite)} NonWhite (need {n_nw})")

rng = random.Random(SEED)
w_rest = [i for i in white if i not in PINS]; rng.shuffle(w_rest)
nw = list(nonwhite); rng.shuffle(nw)
head = PINS + w_rest[:N_WHITE_FIRST - len(PINS)]
tail = w_rest[N_WHITE_FIRST - len(PINS):N_WHITE - len(PINS)] + nw[:n_nw]
rng.shuffle(tail)
order = head + tail
assert len(order) == N_TOTAL == len(set(order))
with open(ORDER, "w") as f:
    f.writelines(f"vgg_{i}\n" for i in order)
for cut in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
    k = sum(lab[i] != "Caucasian" for i in order[:cut])
    print(f"  first {cut:4d}: {cut-k:4d} White, {k:3d} NonWhite")

photos = pickle.load(open("data/vggface2_raw/probe_10.pkl", "rb"))["order"]
for p in PINS:
    if p not in photos:   # below the candidate bar: all its photos, same seeded shuffle
        names = sorted(l.strip().split("/")[1] for l in open(
            "data/vggface2_raw/hf_mirror/data/train_list.txt") if l.startswith(p + "/"))
        random.Random(f"{SEED}-{p}").shuffle(names)
        photos[p] = names
want = {}
for i in order:
    n_tr = len(photos[i]) - 24      # 200 for every candidate; fewer for a short pin
    for j, n in enumerate(photos[i]):
        want[f"train/{i}/{n}"] = (i, "train" if j < n_tr else "valid" if j < n_tr + 12 else "test")
with open(ORDER.replace(".txt", "_race.csv"), "w") as f:
    f.write("position,class,race\n")
    f.writelines(f"{k+1},vgg_{i},{lab[i]}\n" for k, i in enumerate(order))
t0, done = time.time(), 0
with tarfile.open("data/vggface2_raw/hf_mirror/data/vggface2_train.tar.gz", "r|gz") as tar:
    for mem in tar:
        hit = want.get(mem.name)
        if hit is None:
            continue
        i, split = hit
        d = f"{OUT_RAW}/{split}/vgg_{i}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/{mem.name.rsplit('/', 1)[1]}", "wb") as f:
            f.write(tar.extractfile(mem).read())
        done += 1
        if done % 50000 == 0:
            print(f"  {done:,}/{len(want):,} photos written, {time.time()-t0:.0f}s", flush=True)
if done != len(want):
    sys.exit(f"wrote {done} of {len(want)} photos")
print(f"done: {done:,} photos for {N_TOTAL} identities -> {OUT_RAW}; order -> {ORDER}")
