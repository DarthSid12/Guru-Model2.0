"""Label every candidate VGGFace2 identity White / NonWhite / Ambiguous with
FairFace (data_preparation/fairface.py), from the probes pulled by vggface2_probe.py.

Per photo: crop VGGFace2's own loose face box, grown to a square of EXPAND x its
longer side (FairFace was trained on tight dlib chips; on RFW a loose crop sent
White recall at P>=0.8 from 60% down to 18%), and drop greyscale photos, whose
missing colour FairFace misreads. Per identity: mean race probabilities over the
remaining photos.

    White     mean P(White) >= 0.8   (0/450 false Whites on RFW, 3 photos/id)
    NonWhite  mean P(White) <= 0.2
    Ambiguous otherwise, or < MIN_COLOUR colour probes -- excluded from the store

Labels are PERCEIVED race from an automatic classifier -- the same kind of label
RFW itself carries (RFW used the Face++ API).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv, io, pickle, sys
import numpy as np
from PIL import Image
from data_preparation import fairface as ff

EXPAND, MIN_COLOUR, DEV = float(sys.argv[1]) if len(sys.argv) > 1 else 1.3, 5, "cuda:3"
probe = pickle.load(open("data/vggface2_raw/probe_10.pkl", "rb"))
bb = {}
for r in csv.DictReader(open("data/vggface2_raw/meta/bb_landmark/loose_bb_train.csv")):
    bb[r["NAME_ID"]] = tuple(int(r[k]) for k in ("X", "Y", "W", "H"))
meta = {r[0].strip(): (r[1].strip(), r[4].strip())
        for r in list(csv.reader(open("data/vggface2_raw/hf_mirror/meta/identity_meta.csv"),
                                 skipinitialspace=True))[1:]}

def crop(img, box):
    x, y, w, h = box
    s = EXPAND * max(w, h); cx, cy = x + w / 2, y + h / 2
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))

def grey(img):
    a = np.asarray(img.resize((64, 64)), dtype=np.float32)
    return np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 6

model = ff.load(DEV)
out = open(f"data/vggface2_raw/race_labels_x{EXPAND}.csv", "w", newline="")
wr = csv.writer(out)
wr.writerow(["id", "name", "gender", "n_colour", "label", "top_nonwhite"] + [f"p_{r}" for r in ff.RACES])
for k, (ident, photos) in enumerate(sorted(probe["probes"].items())):
    ims = []
    for name, data in photos.items():
        im = Image.open(io.BytesIO(data)).convert("RGB")
        c = crop(im, bb[f"{ident}/{name.rsplit('.', 1)[0]}"])
        if not grey(c):
            ims.append(c)
    if len(ims) >= MIN_COLOUR:
        p = ff.race_probs(model, ims, DEV).mean(0).numpy()
        lab = "White" if p[0] >= 0.8 else "NonWhite" if p[0] <= 0.2 else "Ambiguous"
        top = ff.RACES[1 + int(np.argmax(p[1:]))]
    else:
        p, lab, top = np.full(7, np.nan), "Ambiguous", ""
    name, gender = meta.get(ident, ("", ""))
    wr.writerow([ident, name, gender, len(ims), lab, top] + [f"{v:.4f}" for v in p])
    if k % 1000 == 0:
        print(f"  {k}/{len(probe['probes'])}", flush=True)
out.close()
print("done", out.name)
