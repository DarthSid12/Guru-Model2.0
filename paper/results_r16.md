# r16 round — all results

Three from-scratch log-polar models trained 2026-09-01→02, plus the finished
r15_vgg simulation sweeps. Every model shares one 6-stage / 80-epoch r15-matched
curriculum; none of them train on CFD.

| model | diet | wall |
|---|---|---|
| `r16_base` | 128 CelebA + 480 VGGFace2 faces, 64 objects, 40 ZuBuD house identities, 1 generic `houses_gen` class | 18.3 h |
| `r16_acuity` | `r16_base` **exactly**, plus a train-time blur→sharp schedule (σ 8/6/4/2/1/0 per stage, Vogelsang et al. 2018 PNAS). valid/test always at full acuity | 18.7 h |
| `r16_rfw` | `r16_base` plus 1500 RFW identities introduced white-first (1200 Caucasian from stage 1, 300 other-race from stage 4), RFW arm pinned at 20% of the diet split 80/20 white:other | 18.9 h |

`houses_gen` = the 97 leftover ZuBuD buildings (those that are neither the 40
trained identities nor the 64 held out for Yin) plus the 635-building
Houses-dataset, all collapsed into a single `house` label.

## 1. Training-time accuracy (best checkpoint, upright valid / inverted test)

| | faces (CelebA) | faces_vgg | objects | houses_zubud | houses_gen | best valid |
|---|---|---|---|---|---|---|
| r16_base | 93.1 / 68.1 | 89.1 / 63.4 | 74.9 / 68.3 | 87.5 / 37.5 | 97.5 / 88.3 | 82.5% (ep 79) |
| r16_acuity | 92.9 / 57.2 | 89.2 / 54.4 | 73.9 / 66.4 | 90.0 / 45.0 | 98.0 / 87.8 | 82.0% (ep 76) |
| r16_rfw | 90.5 / 63.3 | 85.2 / 52.5 | 74.0 / 67.1 | 87.5 / 40.0 | 99.5 / 88.8 | 74.5% (ep 77) |

`r16_rfw` also scored `faces_rfwW` 9.5% and `faces_rfwO` 18.7%. **That pair is not
a usable other-race measure**: rfwW is 1200-way and rfwO 300-way, so chance
differs 4×, and the top-100-per-race "other" identities are deeper (5.03
photos/id vs 3.23). The reversal is a class-count artefact, not an ORE.

On this readout `r16_acuity` looked like a clean win — faces lost ~10 extra
points to inversion (CelebA 35.7 vs 25.0, VGG 34.8 vs 25.7) while objects and
`houses_gen` moved ~1 point, with upright accuracy unchanged. **Section 2 does
not reproduce it.**

## 2. Yin battery `wm64` — 50 seeds, one calibrated `p` per model

Four stores, each yielding exactly 64 items, all running the standard 40 study /
24 test design with no scaling: `faces_cfdWM64` (64 white male CFD identities,
studio-controlled, one neutral shot each), `faces_rfwWM64` (64 white male RFW
identities, in-the-wild, 4 photos each — built by `scripts/make_rfw_wm64.py`;
RFW ships no sex labels, so the males were read off contact sheets by eye),
`houses_yin64`, and `objects`.

Calibration is fit jointly on the two face stores so all four categories are
scored at one noise level. Compare inversion **costs**, not raw accuracies,
across rows with different `p`.

### faces_cfdWM64

| model | p | UU | II | UI | IU | **UU−UI** | n |
|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | — |
| r16_base | 0.01 | 86.67 | 78.75 | 64.92 | 63.83 | **+21.75** ±2.30 | 50 |
| r16_acuity | 0.0 | 87.50 | 89.33 | 69.50 | 67.25 | **+18.00** ±2.59 | 50 |
| r16_rfw | 0.1 | 95.42 | 81.17 | 65.58 | 67.33 | **+29.83** ±2.62 | 50 |
| r15_vgg | 0.0 | 91.33 | 89.17 | 70.33 | 63.25 | **+21.00** ±2.19 | 50 |

### faces_rfwWM64

| model | p | UU | II | UI | IU | **UU−UI** | n |
|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | — |
| r16_base | 0.01 | 96.67 | 97.08 | 80.33 | 78.08 | **+16.33** ±2.40 | 50 |
| r16_acuity | 0.0 | 97.08 | 96.42 | 75.67 | 74.67 | **+21.42** ±1.97 | 50 |
| r16_rfw | 0.1 | 99.83 | 99.58 | 88.42 | 87.25 | **+11.42** ±1.63 | 50 |
| r15_vgg | 0.0 | 98.17 | 99.08 | 78.08 | 78.75 | **+20.08** ±2.53 | 50 |

### houses_yin64

| model | p | UU | II | UI | IU | **UU−UI** | n |
|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | — |
| r16_base | 0.01 | 33.92 | 36.33 | 32.58 | 33.83 | **+1.33** ±2.27 | 50 |
| r16_acuity | 0.0 | 38.08 | 37.00 | 34.58 | 37.75 | **+3.50** ±1.72 | 50 |
| r16_rfw | 0.1 | 43.08 | 51.17 | 49.75 | 40.92 | **-6.67** ±2.44 | 50 |
| r15_vgg | 0.0 | 90.58 | 85.50 | 71.67 | 69.50 | **+18.92** ±2.32 | 50 |

### objects

| model | p | UU | II | UI | IU | **UU−UI** | n |
|---|---|---|---|---|---|---|---|
| **Human (Yin 1969)** | — | 84.79 | 83.96 | 86.71 | 82.75 | **-1.92** | — |
| r16_base | 0.01 | 98.75 | 99.17 | 96.17 | 97.67 | **+2.58** ±0.90 | 50 |
| r16_acuity | 0.0 | 98.25 | 99.42 | 96.17 | 98.83 | **+2.08** ±1.10 | 50 |
| r16_rfw | 0.1 | 97.83 | 97.92 | 93.50 | 95.50 | **+4.33** ±0.90 | 50 |
| r15_vgg | 0.0 | 99.42 | 100.00 | 96.92 | 98.08 | **+2.50** ±0.81 | 50 |

## 3. r15_vgg — the earlier sweeps (50 seeds each)

### Kanwisher identity matching (UU − II), human 87.50 / 76.80 / **+10.70**

| store | p | UU | II | **effect** |
|---|---|---|---|---|
| `faces_vggHO` (60 novel VGGFace2 ids) | 0.35 | 86.10 | 75.00 | **+11.10** ±0.40 |
| `faces_rfwHO` (500 novel, race-balanced) | 0.31 | 91.50 | 83.20 | **+8.20** ±0.40 |
| `faces_setA` | 0.0 | 83.80 | 76.90 | **+6.90** ±0.10 |

**This is the r15 headline.** On held-out VGGFace2 identities r15 lands within
0.4 points of the human inversion effect and within 1.4 of human upright
accuracy. For scale, Set_A went from +2.90 (r8) / +3.41 (r9) / +2.69 (David's)
to +6.90, with upright ~70 → 83.80 on the identical store — though §4 of
`results_all_stimulus_sets.md` calls Set_A a broken set, so lead with vggHO.
The ordering vggHO > rfwHO > setA is what domain familiarity predicts.

### Yin, per-experiment calibration (UU − UI)

| experiment | p | store | UU | UI | **effect** |
|---|---|---|---|---|---|
| `vggHO_ctrl` | 0.27 | faces_vggHO | 94.8 | 78.3 | **+16.4** ±2.2 |
| | | houses_yin64 | 75.8 | 64.3 | +11.4 ±2.9 |
| | | objects | 98.3 | 97.2 | +1.2 ±0.8 |
| `cfd_h40_ctrl` | 0.0 | faces_cfdWM64 | 91.3 | 70.3 | **+21.0** ±2.2 |
| | | houses_yin64 | 90.6 | 71.7 | +18.9 ±2.3 |
| | | objects | 99.4 | 96.9 | +2.5 ±0.8 |
| `rfwHO` | 0.0 | faces_rfwHO | 95.3 | 76.9 | **+18.3** ±2.2 |

The same `houses_yin64` store reads +11.4 at p=0.27 and +18.9 at p=0.00. Those
two rows are **not** comparable to each other — which is why §2 calibrates once
per model across all four categories.

## 4. Findings

**4.1 `houses_gen` destroys held-out ZuBuD individuation.** Every r16 model
scores `houses_yin64` at 33.9–43.1% upright; r15_vgg scores **90.6%** on the
identical store. The only relevant difference is the generic house class: r15
used `houses` (Houses-dataset only), r16 used `houses_gen`, which folds in 97
leftover **ZuBuD** buildings under a single `house` label. Training 97 ZuBuD
buildings to one label appears to collapse ZuBuD representations together,
destroying the discriminability the Yin 2AFC needs on held-out ZuBuD.

This was the design fork flagged before launch — but the prediction was that
ZuBuD domain familiarity would *raise* house Yin. It did the opposite, and
hard. **Do not read the r16 house rows as "houses show no inversion effect".**
A 2AFC at 34–43% is at or below the 50% chance floor: the arm is degenerate, not
small. A face/house dissociation cannot be claimed from it.

A single-sim diagnostic on `r16_base` with `--shuffle-items` on houses returned
UU 50.00% against 33.92% without, so **two** things are stacked here: the
representation is at chance, and the fixed 40/24 split (the `wm64` preset
shuffles only the face stores, following the `vggHO_ctrl` convention) adds a
systematic below-chance bias on top. Both need fixing before the house arm is
usable.

**4.1b `houses_ho64` was a badly chosen store — it does NOT supersede 4.1.**
`houses_ho64` (64 Houses-dataset buildings, verified disjoint from both
`houses/train` and `houses_gen/train`) run at each model's wm64 `p`, house store
shuffled, returns **chance for all four models**: UU 49.17 (r16_base), 49.25
(r16_acuity), 48.92 (r16_rfw), 50.50 (r15_vgg), against a 50% 2AFC floor.

That is an artefact of the store, not a property of the models. Every model in
the comparison was trained to collapse the Houses-dataset into ONE generic
`house` label -- r15 via `houses` (435 buildings), r16 via `houses_gen` (the same
435 plus 97 ZuBuD). The 64 buildings here were never individuated in training,
but 435 of their dataset-mates were explicitly trained as undifferentiated, so
the model has learned "Houses-dataset-looking thing -> one label" and that
transfers to the held-out 64. This is the one house store guaranteed to read at
chance. **Do not use it.** A held-out house store must come from a dataset the
model was never trained to collapse -- i.e. ZuBuD, as `houses_yin64` does.

Held-out houses have always worked otherwise (`results_all_stimulus_sets.md`):

| model | generic class contains | held-out ZuBuD Yin UU | UU−UI |
|---|---|---|---|
| r8 developmental | *(none)* | 84.83 (161 held out) | +10.96 |
| r9 dev+weighted | *(none)* | 92.00 (161 held out) | +10.75 |
| r15_vgg | Houses-dataset only | 90.58 (64 held out) | +18.92 |
| r16 ×3 | Houses-dataset **+ 97 ZuBuD** | 33.9–43.1 | +1.3 / +3.5 / −6.7 |

ZuBuD individuation survives whenever the generic class does not contain ZuBuD
and collapses when it does, which is exactly 4.1. The oversampling noted below
is a real but separate concern: it did not cause this, since r8/r9/r15 carry
comparable house oversampling and generalise fine.

**4.1c Houses are nonetheless trained in a memorisation regime.** In the final
stage `houses_zubud` supplies 120 images (1,920 unique crops, 0.067% of unique
training content) but takes 10% of every batch, so each unique house crop is
drawn ~149×/epoch and **11,909× over the 80-epoch run**, against 60 for
`faces_vgg` and 54 for `objects` — a 200× oversampling ratio. The heavy weight
is compensating for a dataset two orders of magnitude too small: at natural
frequency houses would be 0.067% of the diet and the 40-way head would not train
at all. 40 identities × 3 photos cannot support the kind of transferable
individuation metric that 608 face identities give. Worth a control that cuts
the face arm to house scale (40 identities × 3 photos, same weight) and re-runs
novel-face Yin: if novel faces also fall to chance, the house arm's behaviour is
data scale rather than anything about houses.

**4.2 The acuity schedule does not reproduce on the Yin measure.** Against
`r16_base` it goes the *wrong* way on CFD (+18.00 vs +21.75) and the right way
on RFW (+21.42 vs +16.33). No consistent direction, and both differences are
within about 2 CIs. The clean +10-point face-specific signal in §1 is most
likely the acuity model simply being worse at inverted *classification*, not a
configural-processing effect. **Vogelsang is not supported by this round.**
The two models are also at slightly different calibrated p (0.01 vs 0.00).

**4.3 RFW training makes a markedly better face model.** `r16_rfw` needs
p=0.10 to be pulled down to the human anchor where `r16_base` needs p=0.01, and
reaches 95.42% upright on `faces_cfdWM64` against `r16_base`'s 86.67% — on a
domain it never trained on. Its CFD inversion effect is the largest in the
round (+29.83). Its `faces_rfwWM64` effect (+11.42) is the smallest, but that
store sits at 99.83% upright for this model, so the effect is ceiling-compressed
rather than genuinely small.

**4.4 The models show an orientation-MISMATCH effect, not an inversion effect.**
This is only visible with all four conditions; the UU−UI summary hides it.
Almost everywhere UU ≈ II — studying *and* testing inverted costs the models
nothing (`r16_base/rfwWM64` 96.67 vs 97.08; `r15_vgg/rfwWM64` 98.17 vs 99.08).
Humans lose 14.4 points there (96.29 → 81.88). The models' entire UU−UI effect
comes from UI and IU dropping together, i.e. from study/test orientation
disagreeing. The human ordering differs in kind too: for humans II (81.88) sits
*below* UI (84.13) — inverted encoding hurts more than a mismatch — and every
model reverses it, with II ≈ UU >> UI ≈ IU.

One exception: **`r16_rfw / faces_cfdWM64`, UU 95.42 / II 81.17, UU−II =
+14.25 ±2.0 against the human +14.41** — the only cell in the battery
reproducing the Yin signature. It is also the only model calibrated at p=0.10,
and higher noise degrades the harder condition faster, so part of that gap may
be the noise level. `r16_base/cfdWM64` gives +7.92 at p=0.01, which suggests a
real effect that p amplifies rather than manufactures. A matched-p run is
needed to separate the two.

**4.4b Correction — `faces_rfwWM64` is NOT the better stimulus set.** An
earlier reading of this round called it well-calibrated because its UU sits near
the 96.29% human anchor. The four-condition view contradicts that: UU ≈ II ≈
97–99% means inverting the faces barely matters, which is what you expect if the
2AFC is won on non-facial image cues — background, lighting, pose and clothing
all differ between in-the-wild photos and survive inversion. High UU there is not
evidence of face processing. `faces_cfdWM64`, studio-controlled so identities
differ mainly in facial structure, is the store that behaves like a face task,
and it is where every UU−II effect in the battery appears.

**4.5 Objects behave correctly everywhere** — +2.08 to +4.33 against a human
−1.92, i.e. near zero in every model, which is the contrast the face rows need.

## 5. Caveats

1. **`faces_cfdWM64` is a novel domain for the r16 models but not for r15_vgg**,
   which trained on 119 CFD identities with only these 64 held out. The r15 CFD
   row is domain-familiar; the r16 rows are not. r16-vs-r16 is clean;
   r16-vs-r15 on CFD is not.
2. **`faces_rfwWM64`'s 64 males were identified by eye** from contact sheets of
   the 125 Caucasian held-out identities — RFW ships no sex labels. Visual
   judgements, not ground-truth metadata. Index list is inline in
   `scripts/make_rfw_wm64.py`; two identities were dropped for stimulus quality
   (109 is a Renaissance painting, 79 a heavily blurred profile).
3. **`p` differs across models** (0.00–0.10). Compare inversion costs, not raw
   accuracies.
4. **No other-race effect was measured.** `faces_rfwWM64` is white males only —
   what was asked for — so it has no non-white counterpart to contrast against.
   Testing the ORE needs a matched non-white held-out store scored at the same p.
