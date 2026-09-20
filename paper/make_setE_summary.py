"""Build paper/results_all_stimulus_sets.md.

Every model row carries the stimulus set it was scored on. Numbers come from the
CSVs wherever a sweep exists; small spot-checks are embedded as raw per-seed
values and averaged here, never hand-copied means.

  faces store  128 identities, TRAINED on  -> earlier sweep (r7, r5b), 200 seeds
  Set_A         40 novel identities        -> Kanwisher 100 seeds; Yin spot-checks
  Set_E         64 novel identities        -> Yin + Kanwisher 100 seeds
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEW = [("r8_developmental", "LP-Net **developmental** (r8, r18)"),
       ("r9_developmental_weighted", "LP-Net **developmental+weighted** (r9, r18)"),
       ("house_control_r1", "**David's model** (curriculum, r18)\u2020")]

DS_FACE_E = "**Set_E** (64 novel)"
DS_FACE_A = "**Set_A** (40 novel)"
DS_FACE_OLD = "`faces` (128, **trained**)"
DS_OBJ = "objects (64, **trained**)"
DS_HOUSE_HELD = "ZuBuD (161 **held-out**)"
DS_HOUSE_OOD = "ZuBuD (201, never trained)"
DS_HOUSE_TRAINED = "ZuBuD (201, **trained**)"
HOUSE_DS = {"r8_developmental": DS_HOUSE_HELD,
            "r9_developmental_weighted": DS_HOUSE_HELD,
            "house_control_r1": DS_HOUSE_OOD}


def sweep_yin(model, cat, path="runs/sim_seeds/yin_setE/results_{m}.csv"):
    d = pd.read_csv(path.format(m=model))
    d = d[(d.sim == "yin") & (d.category == cat)]
    piv = d.pivot_table(index="seed", columns=["study", "test"], values="accuracy_pct")
    c = lambda s, t: f"{piv[(s, t)].mean():.2f} ±{piv[(s, t)].sem():.2f}"
    diff = piv[("Upright", "Upright")] - piv[("Upright", "Inverted")]
    return (f"{d.noise.iloc[0]:.2f}", c("Upright", "Upright"), c("Inverted", "Inverted"),
            c("Upright", "Inverted"), c("Inverted", "Upright"),
            f"**{diff.mean():+.2f}** ±{diff.sem():.2f}", piv.shape[0])


def sweep_kan(model, cat, path):
    d = pd.read_csv(path.format(m=model))
    d = d[(d.sim == "kanwisher") & (d.category == cat)]
    piv = d.pivot_table(index="seed", columns="study", values="accuracy_pct")
    diff = piv["Upright"] - piv["Inverted"]
    return (f"{d.noise.iloc[0]:.2f}",
            f"{piv['Upright'].mean():.2f} ±{piv['Upright'].sem():.2f}",
            f"{piv['Inverted'].mean():.2f} ±{piv['Inverted'].sem():.2f}",
            f"**{diff.mean():+.2f}** ±{diff.sem():.2f}", piv.shape[0])


# ---- spot checks: raw per-seed values, averaged here ------------------------
# Yin, 3 seeds, all four conditions. {(model, set, p): [(UU,II,UI,IU) per seed]}
SPOT = {
 ("r7_curriculum", "Set_A", "0.00"): [(73.33,60.00,40.00,40.00),(66.67,66.67,33.33,46.67),(66.67,73.33,60.00,40.00)],
 ("r7_curriculum", "Set_A", "0.20"): [(60.00,40.00,40.00,40.00),(46.67,60.00,26.67,40.00),(53.33,60.00,53.33,46.67)],
 ("r7_curriculum", "Set_E", "0.00"): [(100.0,87.50,79.17,75.00),(100.0,95.83,87.50,83.33),(91.67,91.67,75.00,62.50)],
 ("r7_curriculum", "Set_E", "0.20"): [(95.83,83.33,70.83,75.00),(100.0,95.83,70.83,87.50),(87.50,83.33,75.00,70.83)],
 ("r5b_allatonce", "Set_A", "0.00"): [(53.33,40.00,33.33,33.33),(53.33,53.33,33.33,40.00),(40.00,26.67,20.00,20.00)],
 ("r5b_allatonce", "Set_A", "0.20"): [(60.00,40.00,33.33,33.33),(53.33,46.67,20.00,40.00),(46.67,20.00,20.00,20.00)],
 ("r5b_allatonce", "Set_E", "0.00"): [(83.33,87.50,66.67,58.33),(83.33,87.50,70.83,54.17),(87.50,83.33,62.50,58.33)],
 ("r5b_allatonce", "Set_E", "0.20"): [(83.33,87.50,62.50,58.33),(75.00,75.00,62.50,54.17),(70.83,70.83,70.83,45.83)],
 ("r8_developmental", "Set_E", "0.20"): [(91.67,87.50,83.33,83.33),(91.67,95.83,62.50,70.83),(91.67,87.50,75.00,79.17)],
 ("r8_developmental", "Set_E", "0.30"): [(83.33,87.50,75.00,83.33),(87.50,87.50,62.50,75.00),(87.50,79.17,58.33,75.00)],
 ("r9_developmental_weighted", "Set_E", "0.20"): [(87.50,91.67,75.00,54.17),(91.67,91.67,75.00,83.33),(83.33,87.50,62.50,66.67)],
 ("r9_developmental_weighted", "Set_E", "0.30"): [(75.00,79.17,66.67,58.33),(79.17,83.33,62.50,66.67),(79.17,87.50,58.33,62.50)],
}
# Yin on Set_A, same-store, p=0, 8-seed averaged UU only
SETA_UU_8SEED = {"r8_developmental": 61.67, "r9_developmental_weighted": 66.67,
                 "house_control_r1": 68.33}
SPOT_LABEL = {"r7_curriculum": "LP-Net **curriculum** (r7, r18)",
              "r5b_allatonce": "LP-Net **all-at-once** (r5b, r18)",
              "r8_developmental": "LP-Net **developmental** (r8, r18)",
              "r9_developmental_weighted": "LP-Net **developmental+weighted** (r9, r18)"}
DSMAP = {"Set_A": DS_FACE_A, "Set_E": DS_FACE_E}


def spot(key):
    a = np.array(SPOT[key], dtype=float)
    m = a.mean(0)
    d = a[:, 0] - a[:, 2]
    return [f"{v:.2f}" for v in m] + [f"**{d.mean():+.2f}**"]


# ---- earlier sweep (r7, r5b): computed from their CSVs, not transcribed -----
OLD = [("r7_curriculum", "LP-Net **curriculum** (r7, r18)"),
       ("r5b_allatonce", "LP-Net **all-at-once** (r5b, r18)")]
OLD_CSV = "runs/sim_seeds/results_{m}.csv"
OLD_DS = {"faces": DS_FACE_OLD, "objects": DS_OBJ, "houses_zubud": DS_HOUSE_TRAINED}

# Human rows: UU, II, UI, IU -> cost = UU - UI
HUMAN_YIN = {"faces": (96.29, 81.88, 84.13, 78.58),
             "objects": (84.79, 83.96, 86.71, 82.75),
             "houses": (90.71, 85.75, 88.08, 85.71)}
KAN_OLD = [("**Human (between-subjects, n=1,532/1,219)**", "—", "—", "87.50", "76.80", "**+10.70**", ""),
           ("**Human (within-subject, n=364)**", "—", "—", "87.50", "75.90", "**+11.60**", ""),
           ("**Dobs et al. Face-ID CNN (Fig. 3B)**", "—", "—", "86.90", "66.40", "**+20.50**", "")]


def human_row(cat):
    uu, ii, ui, iu = HUMAN_YIN[cat]
    return ("**Human (Yin 1969)**", "—", "—", f"{uu:.2f}", f"{ii:.2f}", f"{ui:.2f}",
            f"{iu:.2f}", f"**{uu - ui:+.2f}**", "")


L = []
w = L.append
YH = "| Source | Stimulus set | p | UU | II | UI | IU | Inversion cost (UU−UI) | seeds |"
YS = "|---|---|---|---|---|---|---|---|---|"

w("# Yin (1969) & Dobs/Kanwisher (2023): all stimulus sets\n")
w("Every model row states **which stimulus set it was scored on**. Rows scored on "
  "different sets are not comparable to each other; compare within a stimulus set.\n")
w("Model rows are means ± SEM over simulation seeds (`run_sim_seeds.py`); each seed "
  "redraws the study/test items. Retrieval noise `p` is calibrated so upright accuracy "
  "matches the human anchor, then held fixed across seeds.\n")

w("## Stimulus sets\n")
w("| Set | Contents | Seen in training? | Used for |")
w("|---|---|---|---|")
w("| `faces` | 128 celebrity identities | **YES — training classes** | earlier sweep (r7, r5b), 200 seeds |")
w("| **Set_A** | 40 female-celebrity identities × 5 photos, 250×250 | no | Kanwisher 100 seeds; Yin spot-checks |")
w("| **Set_E** | 64 identities × 10 photos | no | Yin + Kanwisher 100 seeds (primary) |")
w("| objects | 64 ImageNet classes | **YES — training classes** | all runs |")
w("| ZuBuD | 201 buildings × 5 exterior views | varies by model (see rows) | all runs |")
w("\n**Set_A is unusable** — see §4. It is reported for completeness, not for conclusions.\n")
w("---\n")
w("## 1. Yin (1969) — study/test orientation matching\n")
w("UU/II = same orientation at study and test; UI/IU = mismatched. Inversion cost = "
  "`UU − UI`, paired per seed.\n")

w("### 1a. Faces\n")
w(YH); w(YS)
w("| " + " | ".join(human_row("faces")) + " |")
for m, label in OLD:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "faces", OLD_CSV)
    w(f"| {label} | {DS_FACE_OLD} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
for m, label in NEW:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "faces_setE")
    w(f"| {label} | {DS_FACE_E} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
for m in ["r7_curriculum", "r5b_allatonce", "r8_developmental", "r9_developmental_weighted"]:
    for st in ["Set_A", "Set_E"]:
        for p in ["0.00", "0.20", "0.30"]:
            if (m, st, p) in SPOT:
                uu, ii, ui, iu, d = spot((m, st, p))
                w(f"| {SPOT_LABEL[m]} | {DSMAP[st]} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | 3 |")
for m, label in NEW:
    w(f"| {label} | {DS_FACE_A} | 0.00 | {SETA_UU_8SEED[m]:.2f} | — | — | — | — | 8 |")
w("\n*Set_A rows use a 25 study / 15 test design (only 40 identities available), not the "
  "standard 40/24 — so they are not directly comparable to Set_E rows even within a model.*\n")

w("### 1b. Objects\n")
w(YH); w(YS)
w("| " + " | ".join(human_row("objects")) + " |")
for m, label in OLD:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "objects", OLD_CSV)
    w(f"| {label} | {DS_OBJ} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
for m, label in NEW:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "objects")
    w(f"| {label} | {DS_OBJ} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
w("")

w("### 1c. Houses\n")
w(YH); w(YS)
w("| " + " | ".join(human_row("houses")) + " |")
for m, label in OLD:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "houses_zubud", OLD_CSV)
    w(f"| {label} | {DS_HOUSE_TRAINED} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
for m, label in NEW:
    p, uu, ii, ui, iu, d, n = sweep_yin(m, "houses_zubud")
    w(f"| {label} | {HOUSE_DS[m]} | {p} | {uu} | {ii} | {ui} | {iu} | {d} | {n} |")
w("")

w("---\n")
w("## 2. Dobs / Kanwisher (2023) — upright vs. inverted, faces only\n")
w("Three-image identity matching, no memory phase, ~156k triplets per run — hence SEMs "
  "~5× tighter than Yin's.\n")
w("| Source | Stimulus set | p | Upright | Inverted | Inversion effect | seeds |")
w("|---|---|---|---|---|---|---|")
for r in KAN_OLD:
    w("| " + " | ".join(r) + " |")
for m, label in OLD:
    p, u, i, d, n = sweep_kan(m, "faces", OLD_CSV)
    w(f"| {label} | {DS_FACE_OLD} | {p} | {u} | {i} | {d} | {n} |")
for m, label in NEW:
    p, u, i, d, n = sweep_kan(m, "faces_setE", "runs/sim_seeds/results_{m}.csv")
    w(f"| {label} | {DS_FACE_E} | {p} | {u} | {i} | {d} | {n} |")
for m, label in NEW:
    p, u, i, d, n = sweep_kan(m, "faces_setA", "runs/sim_seeds/kanwisher_setA/results_{m}.csv")
    w(f"| {label} | {DS_FACE_A} | {p} | {u} | {i} | {d} | {n} |")
w("")
w("Kanwisher objects and houses (Set_E sweep, 100 seeds), for the face-specific contrast:\n")
w("| Model | Category | Stimulus set | p | Upright | Inverted | Inversion effect |")
w("|---|---|---|---|---|---|---|")
for m, label in NEW:
    for cat, ds in [("objects", DS_OBJ), ("houses_zubud", HOUSE_DS[m])]:
        p, u, i, d, n = sweep_kan(m, cat, "runs/sim_seeds/results_{m}.csv")
        w(f"| {label} | {cat} | {ds} | {p} | {u} | {i} | {d} |")
w("")

w("---\n")
w("## 3. Caveats\n")
w("**1. Face stimuli differ between blocks — the single biggest confound.** The earlier "
  "r7/r5b sweep scored faces on the 128 identities those models *trained on*; the r8/r9/"
  "David rows use novel Set_E. Trained identities give the model identity-specific, "
  "orientation-tuned features that inversion disrupts, so the earlier Yin costs (+12.10, "
  "+9.12) are probably inflated relative to a held-out test. **Do not read the drop from "
  "r7's +12.10 to r8's +0.83 as a model effect** — the stimuli changed too. The 3-seed "
  "spot-checks of r7/r5b *on Set_E* are the only same-stimulus old-vs-new comparison here, "
  "and they suggest r7 retains a face inversion cost (+5.6) where r8/r9 do not.\n")
w("**2. Inversion cost here is `UU \u2212 UI`** \u2014 study upright, then test in the same vs "
  "the opposite orientation. Note this is a *different* contrast from `UU \u2212 II`: on Set_E "
  "the models show essentially **no** UU\u2212II cost (II tracks UU within a few points, "
  "sometimes above it) because a consistently inverted study+test pair is just as matchable "
  "as an upright one. The orientation cost lands almost entirely on the **mismatched** "
  "conditions, which is what UU\u2212UI measures.\n")
w("**3. Objects are never held out.** All 64 packed object classes were trained on by every "
  "model, so object accuracy is inflated by exposure and cannot reach the 84.79% human "
  "anchor at any usable noise.\n")
w("**4. Set_A is a broken stimulus set.** Same-store Yin at chance (48.9–68.9% across five "
  "models) and Kanwisher ~70% against an 87.5% anchor. Its photos of one person barely "
  "resemble each other (within-identity pixel corr +0.078 vs Set_E's +0.113) while all "
  "identities are demographically similar, so between-identity variability is low too. Not "
  "a resolution effect — Set_E has *more* sub-224px images.\n")
w("**5. \"David's model\" is not a clean control.** It is a curriculum model like r8, but "
  "differs from r8 in two further ways: **128 object classes** (vs 64), and it trained on "
  "the `houses` dataset rather than ZuBuD — so ZuBuD is fully out-of-distribution for it, "
  "not merely held-out. Both models have 40 house classes. Its inversion effects therefore "
  "cannot be attributed to any single design difference. For a clean curriculum vs "
  "no-curriculum contrast use **r7 vs r5b**, which differ only in that.\n")
w("\u2020 **Label discrepancy.** David's model is recorded here as a curriculum model per "
  "the author's description. Its saved run record says otherwise: `config.json` has "
  "`curriculum: False`, `summary.json` has `curriculum: null`, and the run directory "
  "contains no `stage*.pth` checkpoints (r8/r9 have six each). Noted so anyone inspecting "
  "`/home/d1deutsch/Guru-Model2.0/runs/faces_objects_houses_lp_16fix_lr0.001_resnet18_"
  "house_control_r1/` is not confused by the mismatch.\n")
w("**6. `p` differs across rows** and is calibrated per model so upright accuracy matches "
  "the human anchor. On Set_E, Yin pins at p≈0 (models only just reach 96.29% unaided) "
  "while Kanwisher retains headroom (p = 0.09–0.28). Compare inversion **costs**, not raw "
  "accuracies, across rows with different `p`.\n")

w("---\n")
w("## 4. Excluded runs\n")
w("Recorded so they are not silently repeated:\n")
w("- **Yin faces_setA with cross-store distractors** (`runs/sim_seeds/yin_scan/p0.3{2,4,6,8}`, "
  "`calibrated_p_partial/`, `fixed_p03_partial/`). Study items came from Set_A and the "
  "'new' distractors from the `faces` store, so the 2AFC was won on **dataset appearance**, "
  "not memory: studied Set_A vs old-store scored 90%, and *unstudied* Set_A vs old-store "
  "also scored 90%, while the honest same-store test sat at 45% (chance). Symptom was "
  "upright accuracy flat at 95.83% across every noise level. **All faces numbers from these "
  "runs are invalid.** Objects and houses in those same files are same-store and remain "
  "valid.\n")
w("- **Kanwisher faces_setA (100 seeds)** is reported above but should not be used for "
  "conclusions — it is 16–18 points below the human anchor for the reasons in §4 of the "
  "caveats.\n")
w("---\n")
w("Generated by `paper/make_setE_summary.py`. Earlier 200-seed rows transcribed from "
  "[curriculum_results_summary.md](curriculum_results_summary.md).")

out = "paper/results_all_stimulus_sets.md"
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out, f"({len(L)} lines)")
