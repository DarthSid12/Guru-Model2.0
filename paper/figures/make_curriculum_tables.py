"""Build paper/curriculum_results_summary.md: curriculum vs. all-at-once,
against the human (and Dobs et al. CNN) reference data, per category.

Model numbers are means +/- SEM over the multi-seed sweep in
runs/sim_seeds/results_*.csv (run_sim_seeds.py). Re-run this after the sweep
finishes to refresh the tables with the full seed count.

Reference data is transcribed from paper/results_summary.md:
  - Yin (1969) human: Sec. 4 (Tables 1 & 2, accuracy = (24 - mean errors)*100/24)
  - Dobs et al. (2023) human + Face-ID CNN: Sec. 3 (Exp. 5, faces only)
"""

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "paper/curriculum_results_summary.md"

CATS = [("faces", "Faces"), ("objects", "Objects"), ("houses_zubud", "Houses")]
MODELS = [("r7_curriculum", "LP-Net **curriculum** (r7, r18)"),
          ("r5b_allatonce", "LP-Net **all-at-once** (r5b, r18)")]
YIN_CONDS = [("Upright", "Upright", "UU"), ("Inverted", "Inverted", "II"),
             ("Upright", "Inverted", "UI"), ("Inverted", "Upright", "IU")]

# --- reference data, from paper/results_summary.md ---------------------------
YIN_HUMAN = {  # Sec. 4
    "faces":        {"UU": 96.29, "II": 81.88, "UI": 84.13, "IU": 78.58},
    "objects":      {"UU": 84.79, "II": 83.96, "UI": 86.71, "IU": 82.75},
    "houses_zubud": {"UU": 90.71, "II": 85.75, "UI": 88.08, "IU": 85.71},
}
# Dobs et al. (2023) Exp. 5 -- faces only; there is no object/house human or CNN
# reference for this paradigm, which is why those tables are model-only.
KANW_REF_FACES = [
    ("Human (between-subjects, n=1,532/1,219)", 87.5, 76.8),
    ("Human (within-subject, n=364)", 87.5, 75.9),
    ("Dobs et al. Face-ID CNN (Fig. 3B)", 86.9, 66.4),
]


def fmt(mean, sem=None):
    return f"{mean:.2f}%" + (f" ±{sem:.2f}" if sem is not None else "")


def main():
    files = sorted(glob.glob(str(ROOT / "runs/sim_seeds/results_*.csv")))
    if not files:
        raise SystemExit("no runs/sim_seeds/results_*.csv yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = (df.groupby(["sim", "model", "category", "study", "test"])
           .accuracy_pct.agg(["mean", "sem", "count"]))
    noise = (df.groupby(["sim", "model", "category"]).noise.first())
    n_seeds = {(s, m): int(df[(df.sim == s) & (df.model == m)].seed.nunique())
               for s in df.sim.unique() for m in df.model.unique()}

    L = []
    L.append("# Curriculum vs. all-at-once: Yin (1969) and Kanwisher (2023)\n")
    L.append("Model rows are **means ± SEM over independent simulation seeds** "
             "(`run_sim_seeds.py`); each seed redraws the study/test items, so a "
             "single seed is not meaningful on its own — one Yin test pair is 4.2 "
             "points. Retrieval noise `p` is calibrated **once** per model and "
             "category so that upright accuracy matches the human target, then held "
             "fixed across seeds.\n")
    L.append(f"Seeds: curriculum n={n_seeds.get(('yin','r7_curriculum'),0)} (Yin) / "
             f"{n_seeds.get(('kanwisher','r7_curriculum'),0)} (Kanwisher); "
             f"all-at-once n={n_seeds.get(('yin','r5b_allatonce'),0)} / "
             f"{n_seeds.get(('kanwisher','r5b_allatonce'),0)}. "
             "Both models are LP-Net ResNet-18, 393 classes.\n")
    L.append("Reference data transcribed from [results_summary.md](results_summary.md).\n")
    L.append("---\n")

    # ------------------------------- Yin -------------------------------------
    L.append("## 1. Yin (1969) — study/test orientation matching\n")
    L.append("UU/II = same orientation at study and test; UI/IU = mismatched. "
             "The face signature is a large UU→II drop that objects and houses "
             "do not show.\n")
    for cat, pretty in CATS:
        L.append(f"### {pretty}\n")
        L.append("| Source | Noise p | UU | II | UI | IU | Inversion cost (UU−II) |")
        L.append("|---|---|---|---|---|---|---|")
        h = YIN_HUMAN[cat]
        L.append(f"| **Human (Yin 1969)** | — | {h['UU']:.2f}% | {h['II']:.2f}% | "
                 f"{h['UI']:.2f}% | {h['IU']:.2f}% | **{h['UU']-h['II']:+.2f}** |")
        for model, label in MODELS:
            cells, vals = [], {}
            for s, t, name in YIN_CONDS:
                try:
                    r = g.loc[("yin", model, cat, s, t)]
                    vals[name] = r["mean"]
                    cells.append(fmt(r["mean"], r["sem"]))
                except KeyError:
                    cells.append("—")
            p = noise.get(("yin", model, cat), float("nan"))
            cost = (f"**{vals['UU']-vals['II']:+.2f}**"
                    if {"UU", "II"} <= vals.keys() else "—")
            L.append(f"| {label} | {p:.2f} | " + " | ".join(cells) + f" | {cost} |")
        L.append("")

    # ---------------------------- Kanwisher ----------------------------------
    L.append("---\n")
    L.append("## 2. Dobs / Kanwisher (2023) — upright vs. inverted matching\n")
    L.append("A three-image identity-matching task with no memory phase, scored "
             "over ~156k triplets per run — which is why these SEMs are ~5× tighter "
             "than Yin's.\n")
    for cat, pretty in CATS:
        L.append(f"### {pretty}\n")
        L.append("| Source | Noise p | Upright | Inverted | Inversion effect |")
        L.append("|---|---|---|---|---|")
        if cat == "faces":
            for label, up, inv in KANW_REF_FACES:
                L.append(f"| **{label}** | — | {up:.2f}% | {inv:.2f}% | "
                         f"**{up-inv:+.2f}** |")
        for model, label in MODELS:
            try:
                u = g.loc[("kanwisher", model, cat, "Upright", "Upright")]
                i = g.loc[("kanwisher", model, cat, "Inverted", "Inverted")]
            except KeyError:
                continue
            p = noise.get(("kanwisher", model, cat), float("nan"))
            L.append(f"| {label} | {p:.2f} | {fmt(u['mean'], u['sem'])} | "
                     f"{fmt(i['mean'], i['sem'])} | "
                     f"**{u['mean']-i['mean']:+.2f}** |")
        if cat != "faces":
            L.append("")
            L.append(f"*Dobs et al. (2023) Exp. 5 tested faces only, so there is no "
                     f"human or Face-ID CNN reference for {pretty.lower()}; the "
                     f"model rows stand alone.*")
        L.append("")

    # --------------------------- headline contrast ---------------------------
    L.append("---\n")
    L.append("## 3. Face-specific inversion cost\n")
    L.append("The claim in both papers is not that inversion hurts — it is that it "
             "hurts **faces more than other within-category identity tasks**. "
             "Scored per seed as `cost(faces) − mean[cost(objects), cost(houses)]`, "
             "so a model with a large but uniform inversion cost scores zero.\n")
    L.append("| Simulation | Model | Face-specific cost |")
    L.append("|---|---|---|")
    for sim, simlabel in [("yin", "Yin (1969)"), ("kanwisher", "Kanwisher (2023)")]:
        for model, label in MODELS:
            costs = {}
            for cat, _ in CATS:
                try:
                    u = g.loc[(sim, model, cat, "Upright", "Upright")]["mean"]
                    i = g.loc[(sim, model, cat, "Inverted", "Inverted")]["mean"]
                    costs[cat] = u - i
                except KeyError:
                    pass
            if len(costs) == 3:
                fs = costs["faces"] - (costs["objects"] + costs["houses_zubud"]) / 2
                L.append(f"| {simlabel} | {label} | **{fs:+.2f}** |")
    hf = YIN_HUMAN["faces"]; ho = YIN_HUMAN["objects"]; hh = YIN_HUMAN["houses_zubud"]
    human_fs = ((hf["UU"]-hf["II"]) - ((ho["UU"]-ho["II"]) + (hh["UU"]-hh["II"])) / 2)
    L.append(f"| Yin (1969) | **Human** | **{human_fs:+.2f}** |")
    L.append("")
    L.append("Per-seed SEMs and bootstrap CIs for this contrast are in "
             "`figures/sim_seeds_face_specific.csv` "
             "(`figures/aggregate_sim_seeds.py`).\n")

    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
