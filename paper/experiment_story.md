# What we did, in order, and why

Last updated 2026-08-21.

## Where we are

We are asking whether a network that sees the world through a log-polar,
fixation-driven front end shows the face inversion effect the way people do. Two
human experiments set the bar: Yin (1969), a memory task, and Dobs/Kanwisher
(2023), a matching task. A model is doing well if faces cost a lot when you turn
them upside down and objects and houses cost almost nothing.

Four things now work:

- **The matching Kanwisher task is matched by all four models.** On our best face set, r7,
  r8, r9 and David's model all land on the human 87.50 upright accuracy with an
  inversion effect of +10.8 to +12.2, against a human +10.70 to +11.60.
- **The memory Yin task is matched too**, on a face set that is three-quarters
  familiar identities: 95.75 upright and a +12.17 cost, where the humans got
  96.29 and +12.16.
- **Objects behave like the control they should be.** Their inversion cost is
  near zero in every run, which is what people show.
- **We have a lead on the houses, in one model.** Swapping known buildings for unknown ones in the test set steadily raises David's model's house inversion cost, from +0.4 all-known to +10.0 all-unknown, passing the human +2.63 on the way. Run on the same sets, r8 and r9 stay flat at +9 to +11 — so this is something that model's training gives it, not something the architecture does.
- **Ordering still stands.** The Yin inversion cost goes faces > houses >
  objects in every model, which is the order people show.

All four orientation cells, at the one noise level per model from Step 13, so
the three categories are comparable within a table. 100 seeds each. UU−UI is the
cost Yin reports; UU−II is there because it tells a different story.

**David's model, p = 0.26**

| | UU | II | UI | IU | UU−UI | UU−II |
| --- | --- | --- | --- | --- | --- | --- |
| Faces | 89.25 ±0.62 | 82.33 ±0.82 | 75.25 ±0.82 | 76.12 ±0.82 | **+14.00** | +6.92 |
| Houses | 90.04 ±0.52 | 88.96 ±0.57 | 81.67 ±0.70 | 78.92 ±0.70 | **+8.38** | +1.08 |
| Objects | 98.83 ±0.21 | 98.42 ±0.24 | 95.83 ±0.39 | 96.25 ±0.29 | **+3.00** | +0.42 |

**r7, p = 0.30**

| | UU | II | UI | IU | UU−UI | UU−II |
| --- | --- | --- | --- | --- | --- | --- |
| Faces | 86.25 ±0.61 | 80.54 ±0.81 | 74.21 ±0.90 | 74.96 ±0.78 | **+12.04** | +5.71 |
| Houses | 91.38 ±0.50 | 83.96 ±0.59 | 81.00 ±0.62 | 82.96 ±0.62 | **+10.38** | +7.42 |
| Objects | 98.04 ±0.25 | 97.71 ±0.28 | 95.58 ±0.34 | 96.21 ±0.34 | **+2.46** | +0.33 |

**r8, p = 0.00**

| | UU | II | UI | IU | UU−UI | UU−II |
| --- | --- | --- | --- | --- | --- | --- |
| Faces | 92.12 ±0.53 | 90.96 ±0.53 | 79.50 ±0.82 | 79.13 ±0.82 | **+12.62** | +1.17 |
| Houses | 84.83 ±0.60 | 81.46 ±0.64 | 73.87 ±0.71 | 74.33 ±0.75 | **+10.96** | +3.38 |
| Objects | 98.29 ±0.25 | 99.17 ±0.18 | 97.12 ±0.28 | 97.00 ±0.32 | **+1.17** | −0.88 |

**Humans (Yin 1969)**

| | UU | II | UI | IU | UU−UI | UU−II |
| --- | --- | --- | --- | --- | --- | --- |
| Faces | 96.29 | 81.88 | 84.13 | 78.58 | **+12.16** | +14.41 |
| Houses | 90.71 | 85.75 | 88.08 | 85.71 | **+2.63** | +4.96 |
| Objects | 84.79 | 83.96 | 86.71 | 82.75 | **−1.92** | +0.83 |

The two mismatch cells, UI and IU, land within about a point of each other in
every row, so what the models pay is a cost for study and test orientation
disagreeing, not a cost for seeing something upside down. Measuring with UU−II
instead would say something quite different — David's houses would read +1.08
against a human +4.96, and r8's faces +1.17 against +14.41 — which is why we use
UU−UI throughout, as Yin does.

The one thing still open: no house set matches human accuracy and human
inversion cost at the same time. The rungs that give the right cost are ones
where the model is at 98% accuracy and the humans were at 91%.

The rest of this document is the story of how we got here, with the numbers.

## The yardstick

"Good" means **the model behaves like a human on the inversion effect** — not
that it is accurate. Two human experiments:

| | Faces | Objects | Houses |
| --- | --- | --- | --- |
| **Yin (1969)** upright-study/upright-test accuracy | 96.29 | 84.79 | 90.71 |
| **Yin** inversion cost (upright-upright − upright-inverted) | **+12.16** | −1.92 | +2.63 |
| **Dobs/Kanwisher (2023)** upright matching accuracy | 87.50 | — | — |
| **Kanwisher** inversion effect | **+10.70** (between-subj) / **+11.60** (within-subj) | — | — |

So the target is a **big face cost and near-zero object/house cost**, at
accuracies close to the human ones. Two knobs let us line a model up with a
human: which face pictures we test on, and one retrieval-noise parameter `p`
(bit-flip noise on the 256-d code) that is tuned until upright accuracy sits on
the human anchor. Everything below is a fight over those two knobs.

---

## Step 1 — The models we use

| Model | Faces | Objects | Houses | Total classes |
| --- | --- | --- | --- | --- |
| **r7** | 128 | 64 | **201 ZuBuD — all of them** | 393 |
| **r8** | 128 | 64 | **40 ZuBuD** | 232 |
| **r9** | 128 | 64 | 40 ZuBuD | 232 |
| **David's** (`house_control_r1`) | 128 | **128** | **40 ZuBuD** | 296 |

All four models train on ZuBuD buildings; three of them on a 40-building subset.
David's house store is `object0101.view02`-style ZuBuD files — 40 buildings × 3
views = 120 training images — i.e. **the same dataset and the same class count as
r8 and r9**.

**r7** grows all three categories on one shared schedule (4 → 8 → 16 → 32 → 64 →
128 → all classes, 60 epochs).

**r8** grows each category on *its own* schedule — objects saturate early, faces
keep growing, houses arrive late and stay few — and caps houses at 40:

| Stage | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Faces | 4 | 8 | 16 | 32 | 64 | 128 |
| Objects | 4 | 16 | 32 | 64 | 64 | 64 |
| Houses | 0 | 0 | 4 | 8 | 16 | 40 |
| Epochs | 6 | 6 | 8 | 8 | 10 | 22 |

**r9** is r8 with one thing changed: a balanced diet. Objects supply ~1,000
images per class and houses only 3, so the natural diet is ~90% object images.
r9 forces every batch to be faces 45% / objects 45% / houses 10%. 

**David's model** (`house_control_r1`) **is a curriculum model**, on a
two-stage schedule rather than r8's six-stage one. Its training history records:

| Stage | `stage_spec` | Active classes | Epochs |
| --- | --- | --- | --- |
| 1 | `all-prehouse` | 256 (128 faces + 128 objects) | 2 |
| 2 | `all` | 296 (houses added) | 29 |

So houses are withheld and then introduced, which is the developmental move; what
it does *not* do is grow the face and object class counts (all 128 of each are
present from epoch 1). The `curriculum: false` in its `config.json` refers only
to that class-count ladder — the house delay runs off a separate
`houses_delay_epochs: 2` setting, which is why there are no `stage*.pth`
checkpoints. Alongside it, category proportions are held fixed at faces 40% /
objects 40% / houses 20% for the whole run.

Two things still differ from r8 and matter when reading its rows:

1. **128 object classes**, not 64.
2. **No learning-rate schedule**, where r7/r8/r9 use one global cosine decay.

So r8 vs David's model is a *curriculum-shape + object-count + LR-schedule*
contrast, not a single-variable one. For a clean single-variable house contrast
use r8 vs r9.

Accuracy on held-out upright images:

| Model | Overall | Faces | Objects | Houses |
| --- | --- | --- | --- | --- |
| r7 | 79.5 | 92.1 | 75.8 | 96.0 |
| r8 | 79.3 | 91.0 | 76.3 | 90.0 |
| r9 | 78.8 | 91.1 | 75.5 | 92.5 |
| David's | **57.8** | 91.5 | **53.4** | 97.5 |

David's overall number is low because it carries 128 object classes and trained
without an LR schedule. Its *faces* are as good as everyone else's, which is why
it stays usable for the face simulations.

---

## Step 2 — first simulations, on stimuli the models trained on

We ran both simulations on the 128 CelebA identities, the 64 object classes and
the ZuBuD houses — all things the model had seen — with 200 seeds, calibrating
`p` separately per category.

**Yin, r7, 200 seeds** (`runs/sim_seeds/results_r7_curriculum.csv`):

| Category | p | UU | II | UI | IU | Cost (UU−UI) | Human cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faces (128, trained) | 0.38 | 93.10 ±0.29 | 81.00 ±0.51 | 81.54 ±0.49 | 81.90 ±0.49 | **+11.56 ±0.55** | +12.16 |
| Houses (201 ZuBuD, trained) | 0.32 | 90.29 ±0.40 | 82.00 ±0.46 | 79.08 ±0.50 | 81.33 ±0.51 | **+11.21 ±0.47** | +2.63 |
| Objects (64, trained) | 0.43 | 72.62 ±0.59 | 69.29 ±0.68 | 71.21 ±0.65 | 70.48 ±0.71 | **+1.42 ±0.81** | −1.92 |

**Kanwisher, r7, 200 seeds.** Dobs et al. Exp. 5 tested faces only, so faces are
the only row with a human reference:

| Category | p | Upright | Inverted | Effect | Human |
| --- | --- | --- | --- | --- | --- |
| Faces (128, trained) | 0.43 | 85.44 ±0.10 | 70.81 ±0.13 | **+14.63 ±0.13** | 87.50 / +10.70–11.60 |

Two things to notice, one of which we did not notice at the time:

- **Faces and objects looked right.** The Yin face cost (+11.56) sits within one
  SEM of the human +12.16, and objects are near-flat in both simulations
  (+1.42 under Yin).
- **Houses were already wrong here.** The Yin house cost is **+11.21 against a
  human +2.63** — the failure Step 12 is about was present in the very first
  sweep, on trained buildings, and it never went away. It is also already
  *specific to Yin*: the same model on the same buildings shows almost no house
  effect under Kanwisher. The Yin-vs-Kanwisher dissociation that is the
  project's open question was there from the first result.

**Also: this stage was r7 only.** r8, r9 and David's model did not exist yet, and
every face in it was an identity the network had been trained to classify.

## Step 3 — the problem: our test faces were the wrong faces

Yin's participants recognised **unfamiliar adult male faces**. Dobs/Kanwisher's
participants matched **female celebrity faces** they may or may not have known.
We were testing on 128 identities the network had been explicitly trained to
classify — so the model could win by looking up a classifier unit it already
owned, not by remembering or perceiving anything. Same problem for houses: the
model was being tested on buildings it had trained on.

## Step 4 — the fix: bring in the real stimulus sets

We packed the two Dobs/Kanwisher stimulus sets as held-out probes. Neither
contributes a single gradient step to any model.

- **Set A** — 40 identities × 5 photos = 200 photographs. Every identity is a
  **young white female celebrity**; the set is deliberately homogeneous.
- **Set E** — 64 identities × 10 photos = **640 photographs**, anonymised
  `id001`–`id064`. Verified by eye: **both sexes, a wide age range, varied
  ethnicity**, and unconstrained in-the-wild photography (press shots, sports,
  studio, snapshots). It is a far more *heterogeneous* pool than Set A in every
  dimension, not just race — that is what makes identities in it easier to tell
  apart.

For houses we started holding buildings out: r8, r9 and David's model are each
scored on the 161 ZuBuD buildings they never trained on, and later we built
`houses_common` — 94 buildings from a different house dataset that none of the
models ever saw.

---

## Step 5 — the first Yin design was broken, and how we found out

Before any of the results below, there is a design failure worth putting on the
record, because it produced the best-looking face numbers in the project and all
of them were false.

**The design.** Yin's task needs 40 study identities plus 24 *never-studied*
distractors — 64 identities in total. Set A has exactly 40. So the first version
studied **all 40 Set A identities** and drew the **24 distractors from the trained
`faces` store**, i.e. from celebrities the network had been trained to classify.

**It worked beautifully.** Upright-upright accuracy across the three models and
every noise level we tried:

| Model | p=0.00–0.10 | p=0.32 | p=0.34 | p=0.36 | p=0.38 |
| --- | --- | --- | --- | --- | --- |
| r8 | 95.42 ±0.50 | 91.25 | 90.21 | 89.38 | 87.71 |
| r9 | 93.75 ±0.68 | 89.79 | 89.58 | 86.88 | 85.83 |
| David's | 97.08 ±0.50 | 95.42 | 94.37 | 92.92 | 91.04 |

Near the 96.29 human anchor, on genuinely novel faces. It was, briefly, the
result we wanted.

**Three things gave it away.**

1. **Noise barely moved it.** Upright accuracy sat flat near 95.8% across the
   whole noise range before finally sagging — a calibration knob that does not
   bite means the task is not being solved the way we think it is.
2. **Unstudied items scored as well as studied ones.** Set A items the model had
   *never been shown in the study phase* still beat old-store distractors ~90% of
   the time. Nothing about memory can explain that.
3. **The same faces scored at chance in an honest test.** Set A against Set A
   distractors: **53.33%**.

**The diagnosis.** The two answer options came from two different photo
collections, so the model could win by spotting **which photo set a picture came
from** — "which of these two looks like it came from the pile I was just shown?"
— with no memory involved. The 40-point gap between the two ways of scoring is
how big that shortcut was.

We threw all of those runs away. The rule we took from it: the studied items
and the distractors have to come from the same set of photos.

### The fix: build mixed stores by ratio

The rule creates a problem — Set A alone cannot fill a 64-identity design. The
answer was to stop treating "familiar" and "unfamiliar" as separate *stores* and
start treating them as a *composition within one store*: `make_faces_mix64.py`
packs a single store from any mix of groups, tagging each identity `celeb_`,
`known_`, `setE_` or `setA_`, so the study and distractor pools are drawn from
one pool while the known/unknown ratio is whatever we choose. Everything after
this point — `faces_mixed64`, the `mixu` ladder, the familiarity sweeps — is
built that way.

It also lets us **test for the shortcut directly**, by building the same design
twice with familiarity pointing opposite ways (r8, p=0.36, 100 seeds):

| Store | Studied | Distractors | UU | Cost |
| --- | --- | --- | --- | --- |
| `faces_k40u24` | 40 **known** | 24 unknown | 96.50 ±0.34 | +4.17 ±0.61 |
| `faces_u40k24` | 40 **unknown** | 24 known | **63.92 ±0.88** | +25.04 ±1.08 |

Same identities, same design, same noise — only which side familiarity favours is
flipped, and accuracy swings **33 points**. Note that `faces_u40k24` is the honest
single-store version of exactly the broken design above (unknown studied, known
distractors): it reads **63.92**, where the cross-store version read 95.42. That
gap *is* the shortcut, measured.

Neither of these is a real result — a two-choice test that can be won by
spotting a face the model was trained on is not measuring memory, in either
direction. They are a check, and we now run that check every time familiarity
lines up with the right answer.

---

## Step 6 — Set A does not work. At all.

This is the big negative result and it cost a lot of runs.

**Kanwisher on Set A** (design is sound — one store throughout — the numbers are
simply bad):

| Model | p | Upright (human 87.50) | Effect (human +10.70) |
| --- | --- | --- | --- |
| r8 | 0.00 | 71.25 ±0.02 | +2.90 ±0.03 |
| r9 | 0.00 | 69.86 ±0.02 | +3.41 ±0.03 |
| David's | 0.00 | 71.18 ±0.03 | +2.69 ±0.03 |

Upright tops out ~16 points below the human anchor **at zero noise**, so `p` has
nowhere to go — the calibration knob is dead. The inversion effect is a quarter
of the human one. Adding noise only makes it worse (~60% at p≈0.4).

**Yin on Set A** is at chance in an honest same-store test: 48.9–68.9%
upright-upright across five models, against a 96.29 anchor.

**Why Set A fails:** its photographs of one person barely resemble each other
(within-identity pixel correlation **+0.078** vs Set E's **+0.113**), while every
identity is a young white female celebrity, so between-identity variability is
low too. Both directions hurt an identity-matching task. It is *not* a resolution
problem — Set E actually has more sub-224px images.

That 53.33% is also what condemned the cross-store design in Step 5: the same
identities read 95.42 when their distractors came from another photo collection.

**Set E, by contrast, worked**: r8 96.29 / +23.67, r9 92.58 / +24.00, David's
96.83 / +18.46. Accuracy lands on the anchor, but the cost is 1.5–2× the human
+12.16 — unfamiliar-only faces *overshoot*.

### Objects: accuracy too high, inversion effect right

Worth stating once here, because it holds in every run from this point on and is
easy to misread as a failure. Objects are **always** far above their human
accuracy anchor — r8 reads 98.29 upright-upright against Yin's 84.79, and the
other models 98–99 — for an unavoidable reason: **every object class was trained
on**, no object item is ever held out, and no noise level brings them down
without collapsing faces at the same `p`.

But the number that matters is right. The object inversion cost is **+1.17 (r8),
+1.67 (David's), +1.42 (r7)** against a human −1.92, and `UU−II` is +0.42 against
a human +0.83 — i.e. **near-zero and flat, which is exactly the human pattern.**
So objects do the job we need them to do — the category that should *not* show
an inversion effect — they just do it from a ceiling. When I quote objects
anywhere below, the number that means something is the cost, not the accuracy.

---

## Step 7 — the familiar half: 24 celebs the models actually trained on

Reasoning: Yin's and Kanwisher's participants are not blank slates. Some test
identities should be familiar. We picked 24 similar-looking trained CelebA
identities (Adele … Tilda Swinton), using photos from the held-out `valid` split
so the *identity* was seen but the *image* was not.

24 items cannot fill a 40/24 design, so Yin scales down to 15 study / 9 test (the
same 5:3 ratio). Nine test pairs = 11.1 points per pair, so read the SEM.

**The noise-calibration bug :** `p` was fitted on one set of item
draws (base seed 42) and reported on a *different* set (seeds 101–120) —
picking the noise level on one set of draws and reporting it on another, which
was worth 1–2.6 points on a 24-identity set. After the
fix (`--calib-on-eval-seeds`), 100 seeds at p=0.30:

| Model | Faces UU (96.29) | Faces cost (+12.16) | Houses at same p | Objects at same p |
| --- | --- | --- | --- | --- |
| r8 | 96.33 ±0.61 | +7.00 ±0.85 | 75.54 / +8.46 | 95.75 / +1.54 |
| David's | 98.56 ±0.41 | +10.55 ±1.10 | **86.96 / +8.04** | 98.29 / +3.04 |

"Houses work well in David's model" means relative **accuracy** (86.96 vs 75.54),
is better but its house inversion **cost** is just as wrong as everyone else's (+8 against
a human +2.63). Accuracy fixed, cost not fixed.

## Step 8 — mixing 24 celebs with 40 Set A faces: results fall again

`faces_mix64` = 24 trained celebs + all 40 Set A identities, 64 items, 20 seeds:

| Model | p | UU (96.29) | Cost (+12.16) |
| --- | --- | --- | --- |
| r8 | 0.00 | 83.96 ±1.29 | +11.46 ±1.28 |
| David's | 0.03 | 84.17 ±1.44 | +13.96 ±2.01 |

`p` pinned at the floor for every model: 40 of the 64 items carry no signal, so
no noise level reaches 96.29. The cost looks reasonable but it is 24 working
items diluted by 40 dead ones. **Not reportable as a calibrated Yin result.**

## Step 9 — Kanwisher on the same mix: pretty good

This was much better than Yin, Kanwisher lands
on its 87.5 anchor exactly, because the task has no memory phase and the anchor
is reachable. Identity pool grown 24 → 49 by adding Set A identities:

| Identity pool | 24 | 29 | 34 | 39 | 44 | 49 |
| --- | --- | --- | --- | --- | --- | --- |
| r8 upright | 95.81 | 93.59 | 91.37 | 89.49 | 87.99 | 86.77 |
| r8 effect | +6.08 | +5.34 | +5.06 | +4.76 | +4.59 | +4.44 |
| David's upright | 96.01 | 94.25 | 92.03 | 90.29 | 88.63 | 87.34 |
| David's effect | +6.26 | +5.76 | +5.45 | +5.14 | +5.03 | +4.90 |

Accuracy passes through the human anchor, but the **inversion effect is only
about half the human +10.70** everywhere. A second sweep holding the pool at 40
and swapping celebs for Set A one at a time gave the same steady fall
(+4.9 → +2.7 as the last celeb leaves).

Both sweeps ran near p≈0. Refitted where the anchor actually binds (p ≈ 0.30–
0.33) the effects grow by about a third and the models finally separate:
r8 **+6.11 ±0.23**, David's **+7.84 ±0.28**. Still below human, but now outside
the error bars, and in the same order as the Yin celeb-24 run.

**Caveat on this whole sweep:** it cannot test *familiarity*, because all models
trained on the same 24 celebs. What it varies is *which stimulus set* the pool is
drawn from, and Set A is simply harder.

---

## Step 10 — the mixes that work: `faces_mixed64` and the familiarity ladder

Rather than keep guessing compositions, we did two things at once: settled on one
64-identity mix for the headline runs, and built a **ladder** — one 64-identity
store per known/unknown ratio, so the Yin design is *identical* at every rung and
only the ratio moves. The two belong together: the headline store is one rung of
the ladder, and the ladder is what tells you whether that rung was lucky.

> **`faces_mixed64` = 24 named celebs + 16 other trained identities + 12 Set E +
> 12 Set A** — 64 identities × 5 photos: 40 known, 24 unknown.

### Kanwisher on `faces_mixed64` — all four models replicate both human numbers

| Model | p | Upright (87.50) | Effect (+10.70 / +11.60) |
| --- | --- | --- | --- |
| **r7** | 0.40 | **87.33 ±0.30** | **+11.09 ±0.30** |
| **r8** | 0.40 | **87.13 ±0.32** | **+10.79 ±0.38** |
| **r9** | 0.40 | **87.08 ±0.36** | **+11.15 ±0.38** |
| **David's** | 0.39 | **87.33 ±0.35** | **+12.17 ±0.26** |

All four land on the 87.50 upright anchor and inside (or within half a point of)
the two human effects. **This is the strongest face result in the project.**

### Yin on the ladder — and where `faces_mixed64` sits on it

r8, one fixed p=0.23, 100 seeds per rung:

| Unknown identities | 8 | 16 | **24 (= `mixed64`)** | 32 | 40 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| Upright-upright | 95.96 | **95.75** | 89.63 | 87.83 | 85.67 | 81.75 |
| Inversion cost | +9.04 | **+12.17** | **+12.21** | +10.83 | +11.79 | +10.33 |

Two things fall out:

- **`faces_mixed64` gets the cost right (+12.21 vs human +12.16) but its accuracy
  is 7 points low.** It is not a lucky rung — the whole 16–40 region gives a
  human-sized cost — but it is not the accuracy optimum either.
- **The best Yin faces result in the project is one rung over:** `faces_mixu16`
  (48 known + 16 unknown), r8, p=0.23 → **95.75 ±0.42 upright-upright (human
  96.29) and +12.17 ±0.65 cost (human +12.16)**. Both anchors at once, which
  nothing else in the project manages.

So the honest summary is: **Kanwisher is matched on `faces_mixed64`; Yin is
matched one rung more familiar, on `faces_mixu16`.** Accuracy falls steadily as the faces get less
familiar, as you would expect; the cost does not — it peaks around 16–24
unknown and flattens.

For completeness, the same four compositions at a matched p=0.30 across all
three models:

| Store | r8 UU / cost | r9 UU / cost | David's UU / cost |
| --- | --- | --- | --- |
| 64 known | 95.00 / +6.92 | 95.46 / +10.46 | 97.83 / +9.12 |
| 32 known + 32 unknown | 86.37 / +12.00 | 85.17 / +12.83 | 90.42 / +16.00 |
| 40 known + 24 unknown (`mixed64`) | 87.17 / +12.00 | 86.96 / +12.87 | 87.08 / +14.21 |
| 64 unknown | 88.29 / +15.92 | 83.58 / +14.50 | 87.12 / +17.50 |

### The two simulations disagree about familiarity

Read the last table against Step 9 and the contradiction is plain:

- **Under Yin, the inversion cost grows as faces get less familiar** (r8: +6.9
  all-known → +15.9 all-unknown).
- **Under Kanwisher, the inversion effect grows as faces get *more* familiar**
  (r8: +6.1 at 24 celebs → +2.8 at zero celebs).

One thing to note about every set on this ladder: none of them lets familiarity
sit on the same side as the right answer. We shuffle the identities before
splitting them into studied and not-studied, so known and unknown faces land on
both sides. Step 5 is what taught us to do that.

---

## Step 11 — what r9 (the balanced diet) adds

r9 is r8 with the diet rebalanced to faces 45 / objects 45 / houses 10, and
nothing else. It is the only place where a single change can be credited with a
single effect, and it does three things:

| Measure (Set E stimuli, 100 seeds) | r8 | r9 | Human |
| --- | --- | --- | --- |
| Yin houses, held-out ZuBuD, upright-upright | 84.83 | **92.00** | 90.71 |
| Yin houses, cost | +10.96 | +10.75 | +2.63 |
| Yin faces, upright-upright | **96.29** | 92.58 | 96.29 |
| Yin faces, cost | +23.67 | +24.00 | +12.16 |
| Yin objects, cost | **+1.17** | +4.58 | −1.92 |
| Kanwisher faces (Set E), upright / effect | 87.76 / +6.45 | 88.98 / +5.92 | 87.50 / +10.70 |
| Kanwisher faces (`faces_mixed64`), upright / effect | 87.13 / +10.79 | **87.08 / +11.15** | 87.50 / +10.70–11.60 |

1. **The diet fixes house accuracy.** r9 is the *only ZuBuD-trained model that
   reaches the human house accuracy anchor* (92.00 vs 90.71) — r8 with the same
   40 house classes and the natural diet sits 6 points lower. Giving houses a
   guaranteed 10% of each batch is worth ~7 points on buildings it never trained
   on.
2. **It does not touch the house inversion cost** (+10.75 vs +10.96). More house
   exposure buys accuracy, not human-like orientation tolerance. That is direct
   evidence the house problem is not a data-starvation problem.
3. **It costs objects.** r9's object inversion cost triples (+1.17 → +4.58),
   moving away from the human −1.92, and face accuracy drops ~4 points.

4. **On the headline face store it is indistinguishable from r8.** Run on
   `faces_mixed64` at the same p=0.40, r9 reads **87.08 ±0.36 upright and
   +11.15 ±0.38** against r8's 87.13 / +10.79 — both on the human anchors, well
   inside each other's error bars. The balanced diet changes houses and objects;
   it does not change the face result.

r9 also gives the best Yin cost match on `faces_mixed64` at p=0.30 (+12.87 vs
human +12.16, against r8's +12.00 and David's +14.21).


---

## Step 12 — houses: still not solved (at this point in the story)

Every model, on every set of houses, has an inversion cost 3–10× the human +2.63.

| Model | House set | p | Upright (90.71) | Cost (+2.63) |
| --- | --- | --- | --- | --- |
| David's* | ZuBuD, tested on all 201, 40 of them trained on | 0.04 | 97.08 ±0.27 | +10.58 ±0.61 |
| r9 | ZuBuD, 161 held out | 0.00 | **92.00 ±0.43** | +10.75 ±0.61 |
| r8 | ZuBuD, 161 held out | 0.00 | 84.83 ±0.60 | +10.96 ±0.64 |
| r7 | ZuBuD, all 201 trained on | 0.32 | 90.29 ±0.40 | +11.21 ±0.47 |
| r8 | `houses_common` (94, unseen by all) | 0.00 | 86.33 ±0.57 | **+22.13 ±0.78** |
| r7 | `houses_common` | 0.01 | 84.92 ±0.55 | **+26.75 ±0.83** |
| David's | `houses_common` | 0.00 | 75.75 ±0.63 | +13.13 ±0.89 |

*The starred row: we test each model only on the buildings it never trained on,
and for David's model that filter was quietly doing nothing — his houses are
recorded under a different category name and with shorter building names than the
test set uses, so it looked for matches and found none. He was tested on all 201
buildings including the 40 he had learned. We fixed the name matching and made
the filter stop the run if it ever matches nothing again.

Two things to take from the table:

1. **`houses_common`, which we built so all three models would be on equal
   footing, made things worse rather than better** — costs of +13 to +27.
2. **No noise level fixes any of it.** Turning the noise up drops accuracy and
   cost together.

We also used to argue from this table that *training on a set of houses makes a
model worse on the rest of that same set*. That rested on David's model, and it
was backwards — he had trained on those buildings. Only r8 still points that way
(89.2 on a set it never saw, 83.7 on held-out buildings of the set it trained
on), which is one model and one comparison, so we are not claiming it.

The part I find most interesting: the *same* models, on the *same* buildings, are
badly wrong on the memory task and essentially flat on the matching task. So the
problem is specific to the study-then-test memory design, not to how the models
represent houses — which also rules out the idea that the log-polar front end is
simply orientation-sensitive for buildings, since that would break both tasks.

Steps 14 and 15 take this further, so read the above as where the house arm stood
before we ran the familiarity ladders.

---

## Step 13 — one noise level for all three categories

Until now each category got its own `p`, which makes the three rows of a table
non-comparable. So we fitted a **single `p` per model** to minimise the summed
distance from *both* the faces anchor (96.29) and the houses anchor (90.71) at
once. Objects ride along without a vote — every object class was trained on, so
objects cannot reach 84.79 at any usable noise.

**Fit on `houses_common`** — pinned at the floor (p = 0.00 / 0.01 / 0.00) for all
three models, i.e. it cannot be calibrated at all. Dead end.

**Fit on ZuBuD** (100 seeds each):

| Model | p | Faces UU / cost | Houses UU / cost | Objects UU / cost |
| --- | --- | --- | --- | --- |
| David's* | 0.26 | 89.25 / +14.00 | 90.04 / +8.38 | 98.83 / +3.00 |
| r7 | 0.30 | 86.25 / +12.04 | 91.38 / +10.38 *(trained-on)* | 98.04 / +2.46 |
| r8 | 0.00 | 92.12 / +12.62 | 84.83 / +10.96 | 98.29 / +1.17 |

r8 pins at zero — no headroom, so it cannot be jointly calibrated. r7's houses
are its own training buildings. **David's model is the only one that takes a real
joint `p`**, which is why the last round focuses on it.

*His house column has the problem described in Step 12 — 40 of those buildings
were ones he trained on — and because the noise level is chosen from faces and
houses together, the 0.26 itself is affected. We have rerun this with the houses
properly held out and are still working out which noise level to report, so his
corrected numbers are not in this document yet.

### Which noise level to use

Before we found the house-filter problem, we compared two candidate noise levels
on this model, 0.24 and 0.26, at 100 seeds each on the same draws. Neither
inversion cost differed in a way you could call real (faces −0.12 ±0.48, houses
+0.83 ±0.48), and the total distance from the human numbers was 15.21 against
15.29 — a dead heat.

Both of those runs scored the houses wrongly, so that comparison is moot. We are
now rerunning this model at several fixed noise levels with the houses properly
held out, and will pick one from those.

The matching task is a separate fit, since it is a different task with its own
human number, and it is unaffected by any of this. David's model on
`faces_mixed64` at p = 0.39 reads **87.33 ±0.35 upright** against the human
87.50, with an inversion effect of **+12.17 ±0.26** against +10.70 to +11.60.

---

## Step 14 — in David's model, the house cost tracks familiarity

The house arm had been stuck since Step 2 — every model, every set of houses, an
inversion cost 3–10× the human +2.63, and no noise level fixing it. A familiarity
ladder gives us the first real handle on it.

**Everything in this step is David's model only.** The ladder has been run on that
one model, and as the last part of this step shows, it does not carry over to the
others.

**The design.** Eleven 40-building sets, identical in every way except
composition: *k* of the buildings are ones **David's model** trained on
("familiar"), the other 40−*k* are ZuBuD buildings **it** never saw. Only the
familiar/unfamiliar ratio moves. One fixed p=0.26 throughout, 100 seeds per rung
(`runs/sim_seeds_hladder`).

| Familiar houses (of 40) | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unfamiliar | 40 | 36 | 32 | 28 | 24 | 20 | 16 | 12 | 8 | 4 | 0 |
| Upright-upright | 92.80 | 92.87 | 93.87 | 92.27 | 92.87 | 96.07 | 96.13 | 97.13 | 97.67 | 98.47 | 100.00 |
| **Inversion cost** | **+10.00** | +8.60 | +7.93 | +6.67 | +5.53 | +5.33 | +4.60 | +4.00 | **+3.00** | **+2.00** | **+0.40** |

(SEMs ±0.16 to ±0.86. Human: 90.71 upright, **+2.63** cost.)

**For David's model the cost falls steadily as the houses get more familiar, from
+10.00 to +0.40** — a 25-fold range produced by nothing but which buildings are in
the set. The human +2.63 sits between k=32 and k=36, i.e. around **85% of the
buildings familiar**.

### This is not just David's model running out of room

The obvious objection: its upright accuracy also climbs across the ladder (92.8 →
100), so the cost may be shrinking simply because there is less room to fall.
The first four rungs answer it. Paired by seed, k=0 vs k=16:

| | k=0 | k=16 | Paired difference |
| --- | --- | --- | --- |
| Upright-upright | 92.80 | 92.87 | −0.07 ±0.81 (**no real difference**) |
| Inversion cost | +10.00 | +5.53 | **+4.47 ±1.22 (t=3.66)** |

**Accuracy is identical and the cost halves.** Over that stretch familiarity is
doing the work on its own, with no headroom change to hide behind. (Higher up the
ladder the two do move together — k=16→32 gains 4.8 points of accuracy alongside
its 2.5-point cost drop — so the top rungs are partly compressed. The bottom half
is the clean evidence.)

### Faces run the same way

The face ladder points the same direction — more unfamiliar, more inversion cost:

| | All familiar | All unfamiliar |
| --- | --- | --- |
| Faces (r8, Step 10) | +6.92 | +15.92 |
| Houses (David's model, this step) | +0.40 | +10.00 |

So familiarity moves the cost in both categories, though these are two different
models and the house line is the only one measured rung by rung.

### It is David's model that does this, not the models in general

Inside David's model, familiarity clearly drives the house inversion cost — that
is what the ladder shows, and the flat-accuracy rungs make it hard to explain
away.

What it does **not** explain is why the house cost stays large in the other
models.
r7 trained on all 201 ZuBuD buildings and was tested on those same buildings, so
every house in its test was as familiar as a house can be — and run on the *same
40-building sets at the same noise* as the ladder above, it reads **+5.47**, an
order of magnitude above David's +0.40. Familiarity alone does not put a model at
the bottom of the ladder.

(r7's headline house row is +11.21, but that is measured on all 201 buildings
with Yin's full 40-study/24-test design. The ladder sets hold 40 items, too few
for that design, so the simulation scales down to 25 and 15 and is an easier
task. Comparing the two directly would confuse the test pool with familiarity —
+5.47 is the number to set against David's +0.40.)

---

## Step 15 — David's model improves with familiarity, r8 and r9 do not

Step 14 showed David's house inversion cost sliding from +10.00 to +0.40 as
familiar buildings replace unfamiliar ones. The next question is whether that is
how our models behave, or how *that* model behaves. r8 and r9 trained on the same
40 ZuBuD buildings as each other, so all three models can be run on the same sets
and the only thing that changes is the model.

Same 40-building sets, same 25 study / 15 test design, same noise (0.26),
100 seeds per rung:

| Familiar houses (of 40) | 0 | 8 | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| **David's model** | +10.00 | +7.93 | +5.53 | +4.60 | +3.00 | **+0.40** |
| **r8** | +9.07 ±0.97 | +9.47 ±0.90 | +9.53 ±0.90 | +8.67 ±0.93 | +10.47 ±0.87 | **+11.00 ±0.82** |
| **r9** | +9.60 ±0.98 | +10.80 ±0.93 | +6.47 ±0.91 | +7.00 ±0.79 | +9.27 ±0.80 | **+8.80 ±0.77** |

Fitted slope per familiar building:

| Model | slope | R² | 0 known → 40 known |
| --- | --- | --- | --- |
| **David's model** | **−0.2276** | **0.984** | +10.00 → +0.40 (**−9.60**) |
| r8 | +0.0421 | 0.519 | +9.07 → +11.00 (+1.93) |
| r9 | −0.0288 | 0.069 | +9.60 → +8.80 (−0.80) |

**All three start in the same place and then only one of them moves.** With every
building unfamiliar they are indistinguishable — +9.07, +9.60, +10.00. By the top
of the ladder David's model is at +0.40 while the other two are still at +11.00
and +8.80. Across the ladder David's cost drops 9.60 points; r8's moves 1.93 the
wrong way and r9's 0.80 the right way, both inside the noise.

For David's model the fall is close to a straight line — `cost = 9.62 − 0.217 ×
(familiar houses)`, R² = 0.984 across all eleven rungs, every rung within 0.61
points of it. Yin's +2.63 falls at **32.2 of 40 familiar buildings, 81% of the
set**. For r8 and r9 there is no line to fit; R² of 0.52 and 0.07 on a flat trend
is another way of saying the cost does not depend on familiarity at all.

**It is not that r8 and r9 fail to recognise the buildings.** Their upright
accuracy climbs across the ladder just as David's does — r8 73.27 → 89.53,
r9 79.93 → 98.53, David's 92.80 → 100.00. All three plainly know the buildings
they trained on. In David's model that knowledge also buys orientation tolerance;
in r8 and r9 it buys accuracy and nothing else.

**What this means.** The familiarity effect is a property of **one model**, not of
the log-polar architecture and not of the memory task. That makes it a fact about
training. r8 and r9 differ only in diet and behave the same here, so the diet is
not it either. What is left is the thing David's model does that the others do
not: houses withheld for the first two epochs and then trained at a fixed 20% of
every batch, rather than grown through a six-stage class ladder. r7 fits the same
picture — trained on all 201 buildings, so fully familiar with every house on
these sets, and it reads +5.47, between David's +0.40 and r8's +11.00 rather than
at the bottom.

The accuracy problem from Step 14 is unchanged. Where David's cost matches the
human +2.63, its upright accuracy is 97.67 against a human 90.71.

---

## Where it stands

| Arm | Status | Best result |
| --- | --- | --- |
| **Kanwisher, faces** | matched, all four models | r7 87.33/+11.09 · r8 87.13/+10.79 · r9 87.08/+11.15 · David's 87.33/+12.17 |
| **Yin, faces** | matched | r8 `faces_mixu16`, p=0.23: 95.75/+12.17 vs human 96.29/+12.16 |
| **Yin, objects** | cost matched, accuracy too high | cost +1.2 to +4.6 (human −1.92); accuracy 11–14 pts high because every object class was trained on |
| **Yin, houses** | not matched; one model behaves | David's model's cost falls with familiarity, +10.00 all-new to +0.40 all-known, passing the human +2.63 on the way (Steps 14–15). r8 and r9 on the same sets stay flat at +9 to +11, so this is one model's behaviour, not the architecture's |
