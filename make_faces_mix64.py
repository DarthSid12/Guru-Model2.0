"""
make_faces_mix64.py

Build the packed face stores used by the celeb-based simulations. Each store is
a mix of identity groups, one group per source store, written as

    fixation_data/<out category>/valid/{images.npy, coords.npy, meta.json}

with every class name carrying its group's prefix, so a simulation can select on
sub-population (`--group-sizes celeb_=24 setA_=16`) or split familiar from
unfamiliar after the fact.

Presets (`--preset`, default `mix64`):

  mix64    24 celeb + all 40 Set_A         = 64 identities  (the original store)
  celeb24  24 celeb                        = 24 identities
  mixE64   24 celeb + first 40 Set_E       = 64 identities

The 24 celeb identities are the named list below: identities from the
128-identity training set ('faces') that every model in the sweep TRAINED on, so
they are the "familiar" group. Their photos still come from the held-out 'valid'
split, so no exact image was seen -- only the identity. Set_A and Set_E are
fully novel to every model -- the "unfamiliar" groups.

This is a pure re-pack: all source stores were produced by
preprocess_fixations.py at input_size 224 with 32 coords, so the crops and
fixation coordinates are copied verbatim and no salience model is re-run. Every
source is checked against the first one for matching input_size/num_coords
before anything is written.

Five photos per identity, because that is what the Dobs/Kanwisher matching task
uses (Exp. 1) and what Set_A provides; identities with more in 'valid' are
truncated to their first five in packed order.

    python make_faces_mix64.py --preset mixE64
"""

import argparse
import json
import os

import numpy as np

PACKED_ROOT = "fixation_data"
PHOTOS_PER_IDENTITY = 5

# The 24 CelebA identities requested for the mix, as they are named in
# data/faces_cleaned/faces/128_identities (and hence in the packed 'faces' store).
CELEB_IDENTITIES = [
    "Adele", "AshleyGraham", "Beyonce", "CardiB", "EmmaStone", "GalGadot",
    "GretaGerwig", "JenniferLopez", "KaiaGerber", "Kesha", "MalalaYousafzai",
    "MargotRobbie", "MaricaBranchesi", "MaryBarra", "MeghanMarkle",
    "MelindaGates", "MillieBobbyBrown", "MistyCopeland", "NadiaMurad",
    "OprahWinfrey", "SarahPaulson", "SherylSandberg", "SoniaFriedman",
    "TildaSwinton",
]

# One entry per store: (source category, class-name prefix, identities).
# `identities` is either an explicit list of source class names, or an int n
# meaning "the first n in sorted order" (None = all of them). Sorted order is
# used rather than a random draw so a store is reproducible without a seed.
CELEB_GROUP = ("faces", "celeb_", CELEB_IDENTITIES)


class MoreOf:
    """`n` more classes from a store, in sorted order, skipping `exclude`.

    Used for the familiar control arm of the growth sweep: it draws further
    TRAINED identities from the 128-identity 'faces' store rather than novel
    ones, so a step adds pool size without adding unfamiliarity. Classes with
    fewer than PHOTOS_PER_IDENTITY photos in 'valid' are skipped (one of the 128
    has only four), because pick() would otherwise abort on them.
    """

    def __init__(self, n, exclude=()):
        self.n = n
        self.exclude = set(exclude)


class RandomOf(MoreOf):
    """`n` classes drawn at random (seeded) from a store, skipping `exclude`.

    MoreOf takes the first n in sorted order, which is reproducible but not a
    random sample -- for the trained 'faces' store that means always the same
    alphabetical head. Where the design calls for "other random known
    identities", this draws with a fixed seed instead, so the store is both
    random and reproducible.
    """

    def __init__(self, n, exclude=(), seed=0):
        super().__init__(n, exclude)
        self.seed = seed


PRESETS = {
    # The original store: familiar celebs + every Set_A identity.
    "mix64": ("faces_mix64", [CELEB_GROUP, ("faces_setA", "setA_", None)]),
    # Familiar identities alone, for the celeb-only Yin experiment. 24 items
    # only, so simulate_yin1969.py scales its 40/24 study/test split down to
    # 15/9 -- the same 5:3 ratio.
    "celeb24": ("faces_celeb24", [CELEB_GROUP]),
    # Familiar celebs + 40 Set_E identities. Set_E has 64; the first 40 sorted
    # are taken so the unfamiliar group matches Set_A's size and the store keeps
    # the 24+40 shape faces_mix64 established.
    "mixE64": ("faces_mixE64", [CELEB_GROUP, ("faces_setE", "setE_", 40)]),
}

# ---- growth sweep -----------------------------------------------------------
# Yin on the 24 familiar celebs, then the same 24 with novel identities added 4
# at a time, up to 24 + 40 = 64. The question the sweep asks is how many
# unfamiliar identities the pool tolerates before upright-upright accuracy falls
# from the models' ~97-98% ceiling to Yin's human 96.29%, so `p` must be held at
# the value fitted on the 24-only store -- refitting per point would erase the
# very drop being measured (run_sim_seeds.py --experiment growth does this).
#
# Three arms, identical in pool size at every step, differing only in WHAT is
# added:
#   growA_<k>  + k novel Set_A identities   (the requested arm)
#   growE_<k>  + k novel Set_E identities   (Set_A is near-chance under Yin, so
#                                            this is the arm that is readable)
#   growF_<k>  + k further TRAINED celebs   (familiarity control: isolates the
#                                            set-size cost from the novelty cost)
# Fine near the start, coarse later: with Set_A the drop from the models' ~97%
# ceiling to Yin's 96.29 anchor happens within the first few added identities
# (measured: 24 + 8 Set_A already reads 83% UU at p=0.33), so the informative
# region is k <= 8. The tail is kept so the same store set also spans the full
# 24 -> 64 range the mix64 sweep used.
GROWTH_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40]
for _k in GROWTH_STEPS:
    PRESETS[f"growA_{_k}"] = (f"faces_growA{_k}",
                              [CELEB_GROUP, ("faces_setA", "setA_", _k)])
    PRESETS[f"growE_{_k}"] = (f"faces_growE{_k}",
                              [CELEB_GROUP, ("faces_setE", "setE_", _k)])
    PRESETS[f"growF_{_k}"] = (f"faces_growF{_k}",
                              [CELEB_GROUP,
                               ("faces", "extra_", MoreOf(_k, CELEB_IDENTITIES))])


# ---- 2026-08-20 round: four 64-identity face stores ------------------------
# One design, four familiarity compositions, all 64 identities x 5 photos so the
# Yin 40 study / 24 test split is identical across them and only WHO the
# identities are changes. "Known" = one of the 128 CelebA identities every model
# trained on (photos still from the held-out 'valid' split, so the identity was
# seen and the image was not); "unknown" = Set_E or Set_A, novel to all models.
# The 24 named celebs above nest inside every store that has known identities,
# so the stores are directly comparable to the earlier celeb24 runs.
PRESETS.update({
    # 64 known: the 24 named celebs + 40 more trained identities.
    "k64": ("faces_k64", [CELEB_GROUP,
                          ("faces", "known_", MoreOf(40, CELEB_IDENTITIES))]),
    # 32 known + 32 unknown: the 24 named celebs + 8 more trained, + 32 Set_E.
    "k32u32": ("faces_k32u32", [CELEB_GROUP,
                                ("faces", "known_", MoreOf(8, CELEB_IDENTITIES)),
                                ("faces_setE", "setE_", 32)]),
    # 64 unknown: all of Set_E, truncated to 5 photos per identity like the rest
    # (the packed Set_E store carries 10) so photo count is not a confound.
    "u64": ("faces_u64", [("faces_setE", "setE_", None)]),
    # 24 named celebs + 16 other random trained + 12 Set_E + 12 Set_A.
    "mixed64": ("faces_mixed64",
                [CELEB_GROUP,
                 ("faces", "known_", RandomOf(16, CELEB_IDENTITIES, seed=0)),
                 ("faces_setE", "setE_", 12),
                 ("faces_setA", "setA_", 12)]),
})


# ---- Yin's own 40/24 structure, split along familiarity --------------------
# 40 known identities packed FIRST, then 24 unknown. Run WITHOUT --shuffle-items
# the packed order is the design: all_items[:40] become the study pool and
# all_items[40:64] the never-studied distractors, so "old" is always a trained
# identity and "new" is always a novel one.
#
# That alignment is the whole point of the condition AND its main hazard. When
# familiarity lines up with the correct answer, a 2AFC can be won by recognising
# the sub-population rather than by remembering the study episode -- the failure
# that invalidated the earlier Set_A runs, where UNSTUDIED items scored as well
# as studied ones against cross-store distractors. faces_u40k24 is the control:
# identical populations, reversed roles, so familiarity now OPPOSES the correct
# answer. Memory-driven performance survives the reversal; appearance-driven
# performance collapses. Always read the two together.
PRESETS.update({
    "k40u24": ("faces_k40u24",
               [CELEB_GROUP,
                ("faces", "known_", MoreOf(16, CELEB_IDENTITIES)),
                ("faces_setE", "setE_", 24)]),
    "u40k24": ("faces_u40k24",
               [("faces_setE", "setE_", 40),
                CELEB_GROUP]),
})


# ---- familiarity ladder ----------------------------------------------------
# One 64-identity store per unknown-share, so the Yin design is identical across
# them and only the familiar/unfamiliar ratio moves. Unknown identities are split
# evenly between Set_E and Set_A; known ones are the 24 named celebs plus however
# many more trained identities the store needs, so the named 24 nest throughout.
#
# The ladder exists for two opposite reasons. Going UP (more unknown) tests how
# much unfamiliarity the models tolerate. Going DOWN (more known) buys
# CALIBRATION HEADROOM: a joint faces+houses fit can only land on both human
# anchors if the model sits ABOVE each anchor at p=0, and on faces_mixed64
# (mixu24) every model sits below 96.29 at zero noise, which pins p at the floor
# and leaves nothing to fit. The higher-known stores read above the anchor, so a
# positive p exists that brings faces down onto 96.29 while houses come down
# onto 90.71 at the same p.
UNKNOWN_SHARES = [8, 16, 24, 32, 40, 48]
for _u in UNKNOWN_SHARES:
    _known = 64 - _u
    _extra = _known - len(CELEB_IDENTITIES)
    if _extra >= 0:
        # The full named list, topped up with further trained identities.
        _groups = [CELEB_GROUP]
        if _extra > 0:
            _groups.append(("faces", "known_", MoreOf(_extra, CELEB_IDENTITIES)))
    else:
        # Fewer known slots than named celebs: take the first _known of the
        # named list, so the high-unknown stores still nest inside the lower ones.
        _groups = [("faces", "celeb_", CELEB_IDENTITIES[:_known])]
    _groups += [("faces_setE", "setE_", _u // 2), ("faces_setA", "setA_", _u - _u // 2)]
    PRESETS[f"mixu{_u}"] = (f"faces_mixu{_u}", _groups)


def load_source(category, split="valid"):
    d = os.path.join(PACKED_ROOT, category, split)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
    coords = np.load(os.path.join(d, "coords.npy"), mmap_mode="r")
    return meta, images, coords


def pick(meta, wanted, prefix):
    """(out_class_name, [source row index, ...]) for each wanted class, in the
    order `wanted` gives them. Raises if a class is missing or short of photos."""
    by_class = {}
    for row, ci in enumerate(meta["labels"]):
        by_class.setdefault(meta["classes"][ci], []).append(row)
    out = []
    for name in wanted:
        rows = by_class.get(name)
        if rows is None:
            raise SystemExit(f"'{name}' not in the packed store "
                             f"({len(by_class)} classes available)")
        if len(rows) < PHOTOS_PER_IDENTITY:
            raise SystemExit(f"'{name}' has {len(rows)} photos, "
                             f"need {PHOTOS_PER_IDENTITY}")
        out.append((prefix + name, rows[:PHOTOS_PER_IDENTITY]))
    return out


def wanted_classes(meta, spec):
    """The source class names one group asks for: an explicit list, the first n
    sorted, or all of them."""
    if isinstance(spec, list):
        return spec
    names = sorted(meta["classes"])
    if isinstance(spec, RandomOf):
        counts = {}
        for ci in meta["labels"]:
            counts[meta["classes"][ci]] = counts.get(meta["classes"][ci], 0) + 1
        pool = [n for n in names
                if n not in spec.exclude and counts[n] >= PHOTOS_PER_IDENTITY]
        if spec.n > len(pool):
            raise SystemExit(f"asked for {spec.n} random identities, "
                             f"{len(pool)} available after exclusions")
        import random
        return sorted(random.Random(spec.seed).sample(pool, spec.n))
    if isinstance(spec, MoreOf):
        counts = {}
        for ci in meta["labels"]:
            counts[meta["classes"][ci]] = counts.get(meta["classes"][ci], 0) + 1
        pool = [n for n in names
                if n not in spec.exclude and counts[n] >= PHOTOS_PER_IDENTITY]
        if spec.n > len(pool):
            raise SystemExit(f"asked for {spec.n} more identities, "
                             f"{len(pool)} available after exclusions")
        return pool[:spec.n]
    if spec is None:
        return names
    if spec > len(names):
        raise SystemExit(f"asked for {spec} identities, store has {len(names)}")
    return names[:spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="mix64")
    args = ap.parse_args()
    out_category, group_specs = PRESETS[args.preset]

    groups, size, ncoord = [], None, None
    for source, prefix, spec in group_specs:
        meta, img, crd = load_source(source)
        if size is None:
            size, ncoord = meta["input_size"], meta["num_coords"]
        for key, have in (("input_size", size), ("num_coords", ncoord)):
            if meta[key] != have:
                raise SystemExit(f"source stores disagree on {key}: "
                                 f"{source}={meta[key]} vs {have}")
        groups += [(n, r, img, crd)
                   for n, r in pick(meta, wanted_classes(meta, spec), prefix)]

    n_images = sum(len(rows) for _, rows, _, _ in groups)
    images = np.empty((n_images, size, size, 3), dtype=np.uint8)
    coords = np.empty((n_images, ncoord, 2), dtype=groups[0][3].dtype)

    classes, labels, stems, out_row = [], [], [], 0
    for cls, rows, src_img, src_crd in groups:
        ci = len(classes)
        classes.append(cls)
        for k, row in enumerate(rows):
            images[out_row] = src_img[row]
            coords[out_row] = src_crd[row]
            labels.append(ci)
            stems.append(f"{cls}_{k}")
            out_row += 1

    out_dir = os.path.join(PACKED_ROOT, out_category, "valid")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "images.npy"), images)
    np.save(os.path.join(out_dir, "coords.npy"), coords)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"classes": classes, "labels": labels, "stems": stems,
                   "input_size": size, "num_coords": ncoord}, f)

    counts = {p: sum(1 for c in classes if c.startswith(p))
              for _, p, _ in group_specs}
    per_group = " + ".join(f"{n} {p.rstrip('_')}" for p, n in counts.items())
    print(f"wrote {out_dir}: {len(classes)} identities ({per_group}), "
          f"{n_images} photos, {ncoord} coords @ {size}px")


if __name__ == "__main__":
    main()
