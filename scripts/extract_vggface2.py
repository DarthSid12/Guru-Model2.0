"""Extract a training subset of the Kaggle vggface2 mirror into the raw-tree
layout preprocess_fixations.py expects.

The mirror's train/ (480 ids) and val/ (60 ids) identity sets are disjoint, so
val/ becomes a held-out test store rather than a validation split: 60 people
with natural photo variation that no model has seen. The train identities get
their own train/valid/test split drawn from their own photos.

Photos are sampled with a fixed seed rather than taken in filename order --
VGGFace2 filenames run in source-video order, so the first N of an identity
can come from a single session.
"""
import json, os, random, sys, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ZIP = "data/vggface2_raw/vggface2.zip"
N_TRAIN, N_VALID, N_TEST = 200, 12, 12   # per training identity
N_HO = 6                                  # photos per held-out identity
SEED = 20260901


def plan():
    z = zipfile.ZipFile(ZIP)
    by = defaultdict(list)
    for n in z.namelist():
        if not n.endswith(".jpg"):
            continue
        sp, ident, fn = n.split("/")
        by[(sp, ident)].append(n)
    z.close()

    rng = random.Random(SEED)
    jobs = []   # (member, dest_path)
    counts = defaultdict(int)
    for (sp, ident), names in sorted(by.items()):
        names.sort()
        rng.shuffle(names)
        if sp == "train":
            cuts = [("data/faces_vgg/train", N_TRAIN),
                    ("data/faces_vgg/valid", N_VALID),
                    ("data/faces_vgg/test", N_TEST)]
            i = 0
            for root, k in cuts:
                take = names[i:i + k]
                i += k
                for n in take:
                    jobs.append((n, f"{root}/vgg_{ident}/{os.path.basename(n)}"))
                counts[root] += len(take)
        else:
            for n in names[:N_HO]:
                jobs.append((n, f"data/faces_vggHO/valid/vgg_{ident}/{os.path.basename(n)}"))
            counts["data/faces_vggHO/valid"] += min(N_HO, len(names))
    return jobs, counts


def worker(chunk):
    z = zipfile.ZipFile(ZIP)
    made = set()
    for member, dest in chunk:
        d = os.path.dirname(dest)
        if d not in made:
            os.makedirs(d, exist_ok=True)
            made.add(d)
        with open(dest, "wb") as f:
            f.write(z.read(member))
    z.close()
    return len(chunk)


if __name__ == "__main__":
    jobs, counts = plan()
    print("planned:", json.dumps(counts, indent=2), f"\ntotal files: {len(jobs)}", flush=True)
    nw = 16
    chunks = [jobs[i::nw] for i in range(nw)]
    done = 0
    with ProcessPoolExecutor(nw) as ex:
        for r in ex.map(worker, chunks):
            done += r
            print(f"  {done}/{len(jobs)}", flush=True)
    print("done")
