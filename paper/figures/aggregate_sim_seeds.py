"""Aggregate the multi-seed Yin / Kanwisher sweeps into mean +/- SEM tables.

Data source: runs/sim_seeds/results_*.csv (written by run_sim_seeds.py).
Outputs: sim_seeds_summary.csv, sim_seeds_face_specific.csv, plus a printed table.

The headline quantity is the *face-specific* inversion cost:

    (upright - inverted)faces  -  mean[(upright - inverted)objects, houses]

which is what Yin (1969) and Dobs et al. (2023) actually claim: faces lose more
from inversion than other within-category identity tasks do. A model can show a
large inversion cost everywhere and still fail that claim.

Reported as mean +/- SEM over seeds, with a bootstrap CI on the face-specific
contrast, paired by seed since every category shares a seed's sampling draw.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
ROOT = Path(__file__).parent.parent.parent
CATS = ["faces", "objects", "houses_zubud"]
PRETTY = {"faces": "Faces", "objects": "Objects", "houses_zubud": "Houses"}
# Yin (1969) Tables 1 & 2, for the reference rows
HUMAN_YIN = {"faces": (96.29, 81.88), "objects": (84.79, 83.96),
             "houses_zubud": (90.71, 85.75)}


def inversion_cost(d):
    """Per-seed (upright - inverted) for each model/category.

    Yin's analogue of Kanwisher's two conditions is UU vs II: the same
    orientation at study and test, differing only in which one. The crossed
    conditions (UI/IU) measure something else and are left out of this contrast.
    """
    up = d[(d.study == "Upright") & (d.test == "Upright")]
    inv = d[(d.study == "Inverted") & (d.test == "Inverted")]
    key = ["model", "category", "seed"]
    m = up.merge(inv, on=key, suffixes=("_up", "_inv"))
    m["cost"] = m.accuracy_pct_up - m.accuracy_pct_inv
    return m[key + ["accuracy_pct_up", "accuracy_pct_inv", "cost"]]


def boot_ci(x, n=20000, seed=0):
    x = np.asarray(x)
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main():
    files = sorted(glob.glob(str(ROOT / "runs/sim_seeds/results_*.csv")))
    if not files:
        raise SystemExit("no results_*.csv yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rows, contrasts = [], []
    for sim in ["yin", "kanwisher"]:
        d = df[df.sim == sim]
        if d.empty:
            continue
        cost = inversion_cost(d)
        for model in sorted(cost.model.unique()):
            cm = cost[cost.model == model]
            per_cat = {}
            for cat in CATS:
                c = cm[cm.category == cat]
                if c.empty:
                    continue
                per_cat[cat] = c.set_index("seed")["cost"]
                rows.append({
                    "sim": sim, "model": model, "category": PRETTY[cat],
                    "n_seeds": len(c),
                    "upright": c.accuracy_pct_up.mean(),
                    "upright_sem": c.accuracy_pct_up.sem(),
                    "inverted": c.accuracy_pct_inv.mean(),
                    "inverted_sem": c.accuracy_pct_inv.sem(),
                    "inversion_cost": c.cost.mean(),
                    "inversion_cost_sem": c.cost.sem(),
                })
            if len(per_cat) == 3:  # paired across categories by seed
                aligned = pd.DataFrame(per_cat).dropna()
                fs = aligned["faces"] - aligned[["objects", "houses_zubud"]].mean(axis=1)
                lo, hi = boot_ci(fs.values)
                contrasts.append({"sim": sim, "model": model, "n_seeds": len(fs),
                                  "face_specific": fs.mean(), "face_specific_sem": fs.sem(),
                                  "ci_lo": lo, "ci_hi": hi})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "sim_seeds_summary.csv", index=False)
    con = pd.DataFrame(contrasts)
    con.to_csv(OUT / "sim_seeds_face_specific.csv", index=False)

    for sim in summary.sim.unique():
        print(f"\n===== {sim.upper()} =====")
        print(f"{'model':16s} {'category':9s} {'n':>4} {'upright':>15} "
              f"{'inverted':>15} {'inversion cost':>17}")
        for _, r in summary[summary.sim == sim].iterrows():
            print(f"{r.model:16s} {r.category:9s} {int(r.n_seeds):>4} "
                  f"{r.upright:8.2f} ±{r.upright_sem:4.2f} "
                  f"{r.inverted:8.2f} ±{r.inverted_sem:4.2f} "
                  f"{r.inversion_cost:+9.2f} ±{r.inversion_cost_sem:4.2f}")
        if sim == "yin":
            for cat in CATS:
                u, i = HUMAN_YIN[cat]
                print(f"{'human (Yin69)':16s} {PRETTY[cat]:9s} {'-':>4} "
                      f"{u:8.2f} {'':5} {i:8.2f} {'':5} {u - i:+9.2f}")

    print("\n===== FACE-SPECIFIC INVERSION COST "
          "(faces - mean[objects, houses]), paired by seed =====")
    for _, r in con.iterrows():
        verdict = "excludes 0" if (r.ci_lo > 0 or r.ci_hi < 0) else "INCLUDES 0"
        print(f"  {r.sim:10s} {r.model:16s} n={int(r.n_seeds):3d}  "
              f"{r.face_specific:+6.2f} ±{r.face_specific_sem:4.2f}  "
              f"95% CI [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}]   ({verdict})")
    print("\n  human (Yin 1969) reference: +11.52")
    print(f"\nwrote {OUT / 'sim_seeds_summary.csv'}")


if __name__ == "__main__":
    main()
