"""Build raw-tree stores from RFW (Racial Faces in-the-Wild).

RFW is 11,429 identities x ~3.6 photos across four demographically balanced
subsets. That shape is the opposite of VGGFace2's (480 ids x 200): very broad,
very shallow. Identities keep their race prefix in the class name so per-subset
accuracy can be read back out of the per-class CSV.

Held out: 125 identities per race with >=4 photos -> faces_rfwHO, a
demographically balanced store that supports Kanwisher matching (needs >=2
photos per identity) and is disjoint from training by identity.
"""
import os, random, shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

SRC = "data/drive_faces/test/data"
HO_PER_RACE, HO_PHOTOS = 125, 4
SEED = 20260901


def plan():
    rng = random.Random(SEED)
    jobs, counts = [], defaultdict(int)
    for race in sorted(os.listdir(SRC)):
        rd = os.path.join(SRC, race)
        ids = sorted(os.listdir(rd))
        photos = {m: sorted(os.listdir(os.path.join(rd, m))) for m in ids}
        deep = [m for m in ids if len(photos[m]) >= HO_PHOTOS]
        rng.shuffle(deep)
        held = set(deep[:HO_PER_RACE])

        for m in ids:
            ph = photos[m][:]
            rng.shuffle(ph)
            cls = f"rfw{race}_{m}"
            if m in held:
                for f in ph[:HO_PHOTOS]:
                    jobs.append((os.path.join(rd, m, f),
                                 f"data/faces_rfwHO/valid/{cls}/{f}"))
                counts[f"HO/{race}"] += min(HO_PHOTOS, len(ph))
                continue
            # >=3 photos: hold one back for valid and one for test.
            # 2 photos: both train, no valid row for this identity.
            if len(ph) >= 4:
                sp = {"valid": ph[:1], "test": ph[1:2], "train": ph[2:]}
            elif len(ph) == 3:
                sp = {"valid": ph[:1], "train": ph[1:]}
            else:
                sp = {"train": ph}
            for split, fs in sp.items():
                for f in fs:
                    jobs.append((os.path.join(rd, m, f),
                                 f"data/faces_rfw/{split}/{cls}/{f}"))
                counts[f"{split}/{race}"] += len(fs)
    return jobs, counts


def worker(chunk):
    made = set()
    for src, dst in chunk:
        d = os.path.dirname(dst)
        if d not in made:
            os.makedirs(d, exist_ok=True)
            made.add(d)
        shutil.copyfile(src, dst)
    return len(chunk)


if __name__ == "__main__":
    jobs, counts = plan()
    for k in sorted(counts):
        print(f"  {k:<20} {counts[k]}")
    print("total files:", len(jobs), flush=True)
    nw = 16
    with ProcessPoolExecutor(nw) as ex:
        done = sum(ex.map(worker, [jobs[i::nw] for i in range(nw)]))
    print("copied", done)
