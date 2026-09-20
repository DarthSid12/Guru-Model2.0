"""Summarise the three celeb-identity seed sweeps (run_sim_seeds.py presets).

    python paper/aggregate_celeb_experiments.py

  celeb24  Yin over the 24 trained-on celeb identities alone. 24 items cannot
           fill Yin's 40/24 design, so it runs at study=15 / test=9 -- the same
           5:3 ratio. Nine pairs is 11.1 points per pair, so read the SEM, never
           a single seed.
  mixE64   Yin over 24 celeb + 40 Set_E identities: the mix64 design with Set_E
           as the unfamiliar half instead of Set_A.
  ratio    Dobs/Kanwisher matching at a FIXED pool of 40 identities, sweeping the
           composition from 24 familiar celebs + 16 novel Set_A identities to 0 +
           40. The pool size is held constant so the sweep measures familiarity
           rather than task difficulty, which the mix64 identity-count sweep
           confounds.

Every table is mean +/- SEM over seeds. The retrieval noise p is fitted once per
(model, sim) and held fixed across seeds and across sweep points, so differences
within a table are differences in the model's representation, not in calibration.
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEPS = {"celeb24": "runs/sim_seeds_celeb24",
          "mixE64": "runs/sim_seeds_mixE64",
          "ratio": "runs/sim_seeds_ratio"}
# Model order in every table: all-at-once, developmental, no-house control.
MODEL_ORDER = ["r5b_allatonce", "r8_developmental", "house_control_r1"]

# ---- human anchors ----------------------------------------------------------
# Yin (1969), accuracy = (24 - mean errors) * 100 / 24 from the mean-error tables:
# Table 1 (Exp. I, UU and II) and Table 2 (Exp. II, UI and IU), 24 test pairs
# each. Same numbers the earlier write-ups use (paper/results_faces_mix64.md,
# paper/results_summary.md). Both celeb sweeps are face stores, so the faces row
# is the anchor for both.
YIN_HUMAN = {"faces": {"UU": 96.29, "II": 81.88, "UI": 84.13, "IU": 78.58},
             "objects": {"UU": 84.79, "II": 83.96, "UI": 86.71, "IU": 82.75},
             "houses": {"UU": 90.71, "II": 85.75, "UI": 88.08, "IU": 85.71}}
# Dobs et al. (2023) Exp. 5 target matching, and their Face-ID CNN (Fig. 3B).
KANW_HUMAN = [("Human (between-subjects, n=1,532/1,219)", 87.50, 76.80),
              ("Human (within-subject, n=364)", 87.50, 75.90),
              ("Dobs et al. Face-ID CNN (Fig. 3B)", 86.90, 66.40)]
# The inversion cost the earlier tables report is UU - UI, paired per seed: a
# consistently inverted study+test pair is about as matchable as an upright one,
# so the orientation cost lands on the MISMATCHED conditions and UU - II is not
# the informative contrast here. Both are printed, UU - UI first.
CONTRASTS = [("UU", "UI"), ("UU", "II")]


def cell(g):
    return g["mean"].round(1).astype(str) + " ±" + g["sem"].round(1).astype(str)


def by_model(df):
    """Sort on MODEL_ORDER, keeping anything unlisted at the end."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.assign(_o=df.model.map(lambda m: order.get(m, len(order)))) \
             .sort_values(["_o", "model"]).drop(columns="_o")


def load(sweep):
    files = sorted(glob.glob(os.path.join(ROOT, SWEEPS[sweep], "results_*.csv")))
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else None


def noise_table(sweep):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, SWEEPS[sweep], "noise_*.json"))):
        model = os.path.basename(f)[len("noise_"):-len(".json")]
        d = json.load(open(f))
        for sim in ("yin", "kanwisher"):
            if sim in d:
                at_p = {k: round(v * 100, 1) for k, v in
                        (d.get(f"{sim}_upright_at_p") or {}).items() if v is not None}
                rows.append({"model": model, "sim": sim, "p": d[sim],
                             "upright_at_p": at_p})
    return by_model(pd.DataFrame(rows)) if rows else None


def paired_effect(df, a, b):
    """mean and SEM of the per-seed difference cond_a - cond_b, per model.

    Paired on seed rather than differencing two independent means: both cells
    come from the same item sample, so the seed-to-seed sampling variance is
    shared and cancels."""
    w = df.pivot_table(index=["model", "seed"], columns="cond",
                       values="accuracy_pct", aggfunc="first")
    d = (w[a] - w[b]).groupby("model")
    return d.mean(), d.sem()


def yin_tables(sweep, title, human="faces"):
    df = load(sweep)
    if df is None or df.empty:
        print(f"\n(no results yet for {sweep})")
        return
    df = df.assign(cond=df.study.str[0] + df.test.str[0])
    g = (df.groupby(["model", "category", "cond"])["accuracy_pct"]
           .agg(["mean", "sem", "count"]).reset_index())
    g["cell"] = cell(g)
    n = int(g["count"].max())
    print(f"\n=== {title} (mean ±SEM over {n} seeds) ===")
    tab = by_model(g).pivot_table(index=["model", "category"], columns="cond",
                                  values="cell", aggfunc="first", sort=False)
    hum = YIN_HUMAN[human]
    tab.loc[("HUMAN (Yin 1969)", human), :] = [f"{hum[c]:.2f}" for c in tab.columns]
    print(tab[["UU", "II", "UI", "IU"]].to_string())
    for a, b in CONTRASTS:
        m, s = paired_effect(df, a, b)
        eff = pd.DataFrame({"model": m.index, f"{a} - {b}": m.round(2).values,
                            "sem": s.round(2).values})
        lead = "inversion cost" if (a, b) == ("UU", "UI") else "for reference"
        print(f"\n  {lead}: {a} - {b}, points (paired per seed)")
        print(f"    HUMAN (Yin 1969, {human}): {hum[a] - hum[b]:+.2f}")
        print(by_model(eff).to_string(index=False))


def ratio_tables():
    df = load("ratio")
    if df is None or df.empty:
        print("\n(no results yet for ratio)")
        return
    # "24celeb_16setA" -> 24; sorts the sweep from most to least familiar.
    df = df.assign(n_celeb=df.composition.str.split("celeb").str[0].astype(int))
    g = (df.groupby(["model", "n_celeb", "study"])["accuracy_pct"]
           .agg(["mean", "sem", "count"]).reset_index())
    g["cell"] = cell(g)
    n = int(g["count"].max())
    print(f"\n=== KANWISHER familiar/unfamiliar ratio, 40 identities throughout "
          f"(mean ±SEM over {n} seeds) ===")
    print("    n_celeb = trained-on identities; the other 40 - n_celeb are novel Set_A")
    tab = by_model(g).pivot_table(index=["model", "n_celeb"], columns="study",
                                  values="cell", aggfunc="first", sort=False)
    for label, up_h, inv_h in KANW_HUMAN:
        tab.loc[(label, ""), "Upright"] = f"{up_h:.2f}"
        tab.loc[(label, ""), "Inverted"] = f"{inv_h:.2f}"
    print(tab[["Upright", "Inverted"]].to_string())
    up = g[g.study == "Upright"].set_index(["model", "n_celeb"])["mean"]
    inv = g[g.study == "Inverted"].set_index(["model", "n_celeb"])["mean"]
    print("\n=== inversion effect (upright - inverted), points ===")
    for label, up_h, inv_h in KANW_HUMAN:
        print(f"    {label}: {up_h - inv_h:+.2f}")
    print(by_model((up - inv).round(2).rename("effect").reset_index())
          .pivot(index="model", columns="n_celeb", values="effect")
          .reindex(columns=sorted(df.n_celeb.unique(), reverse=True)).to_string())


def main():
    for sweep in SWEEPS:
        t = noise_table(sweep)
        if t is not None:
            print(f"=== {sweep}: calibrated retrieval noise ===")
            print(t.to_string(index=False))
    yin_tables("celeb24", "YIN — 24 trained-on celeb identities only "
                          "(study 15 / test 9)")
    yin_tables("mixE64", "YIN — 24 celeb + 40 Set_E identities")
    ratio_tables()


if __name__ == "__main__":
    main()
