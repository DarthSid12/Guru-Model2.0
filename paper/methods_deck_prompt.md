# Brief: Methods-section slide deck

You are building a **methods presentation** for a computational-neuroscience project
that trains foveated, log-polar CNNs and then simulates Yin (1969) face-inversion
experiments on them. The deck covers exactly two things: **how the six identical
probe models were trained**, and **how the two-noise Yin simulation works**. It does
not cover results.

---

## Audience and register

Lab meeting / conference methods section. The room knows deep learning and knows the
face-inversion literature, but does **not** know this codebase. Every design choice
should read as a decision with a reason, not a hyperparameter dump. Aim for aw
specialist reader who will ask "why that number?" — so where a number is unusual,
one clause of justification beats three of description.

Nine slides. Dense but not crowded: one idea per slide, the numbers in a visual
wherever a visual can carry them, prose only where it cannot.

---

## Hard rules

1. **Do not invent, round, or extrapolate any number.** Everything quantitative you
   need is in this brief. If you want a number that is not here, leave a clearly
   marked `[TK]` placeholder instead of guessing.
2. **No results.** The simulation battery is still running. Slides 7–9 describe
   procedure and the *shape* of the outputs, never their values. Any curve you draw
   for calibration is a schematic and must be labelled as one.
3. Do not describe the 2AFC decision rule as "voting". It is a kernel-density
   familiarity model. Classification-time inference does use logit voting — that is
   a different mechanism on a different slide; keep them distinct.
4. Do not present UU−II as an effect size anywhere. See slide 7.
5. Where this brief says a feature was **removed** or is **off**, do not describe it
   as part of the pipeline.

---

## Global design direction

- **Consistent category colour throughout the deck.** Pick three hues for
  faces / objects / houses on slide 1 and never reuse them for anything else. On the
  simulation slides, add one accent hue for "upright" and one for "inverted", and
  keep those two consistent from slide 6 to slide 9.
- **Trained vs held-out is the deck's second visual language.** Solid fill = trained
  on; hatched or outlined = held out. Establish this on slide 1 and reuse it on
  slide 6 without re-explaining.
- Diagrams over bullet lists wherever a diagram is possible. Slides 3, 4, 5, 7 and 8
  should each be dominated by one figure.
- Annotate tensor shapes and units on every pipeline diagram.
- Every figure must survive greyscale printing and be legible from the back of a
  room: no 10pt axis labels, no hue-only encoding.

---

# Slide 1 — Datasets

**Headline idea:** one network, one softmax, three categories that differ in what
"a class" means — and a strict wall between what is trained and what is tested.

### Content

A single unified label space of **2,310 classes** across 7 data sources. Faces and
houses are *identity* tasks (fine-level discrimination); objects is a *basic-level
categorization* task. This asymmetry is deliberate — it mirrors Yin's own design,
where faces were the expertise category and houses the mono-oriented control.

**Faces — four sources, 2,108 identities**

| Source | Identities | Train images | ≈ images/identity |
|---|---|---|---|
| CelebA (manually cleaned) | 128 | 16,625 | 130 |
| VGGFace2 (Kaggle mirror) | 480 | 95,480 | 199 |
| RFW — Caucasian | 1,200 (of 1,800 packed) | 3,880 | 3.2 |
| RFW — African/Asian/Indian | 300 (of 900 packed) | 1,510 | 5.0 |

RFW = Racial Faces in the Wild. The two RFW arms are separate categories so the
curriculum can introduce white faces first and the rest later — the other-race-effect
manipulation. The identity caps (1,200 / 300) keep the *deepest* identities: the
store is written in depth-descending order and the trainer takes the first N.

**Objects — 64 ImageNet categories**, 65,689 train / 8,208 valid / 8,218 test images
(769–1,040 per category, mean ≈1,026). Categories chosen to be non-mono-oriented.
The label is the category, not the instance.

**Houses — two roles**
- `houses_zubud137`: **137 ZuBuD buildings**, each photographed from 5 exterior
  viewpoints, split 3 train / 1 valid / 1 test = 411 train images. Label is the
  *building identity*. 137 = all 201 ZuBuD buildings minus the 64 reserved for
  testing; zero overlap, asserted in code rather than assumed.
- `houses`: **one generic `house` class**, 435 train images of Houses-dataset frontal
  exteriors. Buildings the model sees but never individuates — basic-level houseness.

**Held out — four test stores, each exactly 64 items**

| Store | What it is | Disjointness |
|---|---|---|
| `faces_cfdWM64` | 64 Chicago Face Database white-male identities, 1 studio photo each | Never trained — CFD is absent from r19 entirely |
| `faces_rfwWM64` | 64 RFW Caucasian-male identities, 4 in-the-wild photos each | Sliced from the 500-identity held-out RFW pool |
| `houses_yin64` | ZuBuD buildings 41–104, 1 view each | Disjoint from the 137 trained |
| `objects` (valid split) | 64 ImageNet categories | **Categories are trained; only the images are held out** |

The two face stores are matched on N, race and sex and differ only in photographic
control (studio vs in-the-wild), so running both isolates natural image variation.

Flag the objects caveat explicitly on the slide — it is the one store that is not
identity-disjoint, and that is by design, because objects is a categorization control
rather than a memory-for-individuals task.

### Visual

Three columns, one per category, colour-coded. Each column: a strip of 3–4 real
example thumbnails, the identity count set large, images-per-identity secondary, and
a one-line note on what the label means.

Beneath the columns, a full-width **horizontal stacked bar of the 2,310-class label
space**, segmented and coloured by source, with segment widths true to class counts
(1200 / 480 / 300 / 128 faces, 64 objects, 137 + 1 houses). This single bar makes the
face-dominance of the label space obvious at a glance.

Below that, separated by a rule and rendered in the hatched "held-out" style, a row
of **four small cards — 64 items each** — for the test stores. Draw the 64 as a
small 8×8 dot grid inside each card so the 40/24 split on slide 6 can reuse the
same grid.

---

# Slide 2 — Fixation sampling and augmentation

**Headline idea:** the network never sees a whole image. It sees crops at
saliency-chosen fixation points, and each crop is an independent training example.

### Content

**Where fixations come from (computed once, at preprocessing).** Each raw image is
resized and centre-cropped to 224×224, then:
1. A Gaussian centre prior (σ = W/4, α = 2) is multiplied in — the photographer /
   centre bias.
2. The result is **log-polarized before** the saliency operator runs, because
   cortical saliency maps are computed on an already-retinotopic input.
3. Grayscale, contrast-normalized, then convolved with a 31×31 Gabor bank:
   λ ∈ {4, 8}, σ = 0.5λ, θ ∈ {0, π/4, π/2, 3π/4}, ψ ∈ {0, π/2}, γ = 0.5 — 16 filters.
4. Quadrature phase pairs combine into **8 magnitude maps**; their **per-pixel
   variance across orientation channels** is the saliency map.
5. A 10-pixel border is zeroed, then **32 points are drawn without replacement,
   with probability proportional to saliency** (`torch.multinomial`), and mapped back
   through the inverse log-polar map to image (x, y).

These 32 coordinates are stored once per image as int16 and never recomputed.

**What training uses.** The **first 16 of the 32** stored points, deterministically.
One (image, fixation) pair = one training sample, so a base image contributes 16
independent examples. Performance saturates at 16. In the final curriculum stage this
is **2,944,480 training crops per epoch**.

> Be precise here, because it is easy to state backwards: the *sampling* of 32 points
> is stochastic and happens once, offline. The *selection* of 16 from those 32 is
> deterministic. Fixations do not resample across epochs — the augmentation does.

**What is augmented, freshly every epoch, on GPU:**
- **Scale jitter** — batched random-resized-crop, scale ∈ (0.6, 1.0), per-sample zoom
  and shift via an affine grid with reflection padding. Applied to the raw crop
  *before* rotation and foveation, so the lp and cnn variants get identical treatment.
- **Rotation** — uniform in **[−15°, +15°]**, resampled every epoch.

**What was removed (2026-09-04), and must not appear in the deck:** horizontal flip,
colour jitter, random erasing. Models up to r16/r17 trained with all three; the six
probe models here did not.

**Never applied:** a vertical flip or any 180° rotation during training. That would
counterfeit the exact manipulation the experiment measures. Training exposure to
inverted views is 0% (`invert_p = 0.0`).

**Held-out images receive no augmentation at all** — they are presented either
upright (identity) or rotated by exactly 180°.

### Visual

A left-to-right three-panel figure:

1. **One base image with its saliency map overlaid** as a translucent heat layer, all
   32 sampled points marked as small dots, and the 16 actually used filled in solid
   while the unused 16 stay hollow. Caption the multinomial draw.
2. **One 180×180 crop lifted out** of the base image with a leader line from its
   fixation dot — make the crop-from-fixation relationship unmistakable.
3. **The same crop repeated 4–5 times** across a row showing epoch-to-epoch
   augmentation variation: different rotation angle (draw a small dial or arc showing
   the ±15° range) and different scale-jitter framing.

Add a small footnote strip listing the three removed augmentations struck through,
so a viewer who knows the earlier models sees the change. Keep it visually quiet.

---

# Slide 3 — Transformation pipeline

**Headline idea:** every crop passes through a retina before it reaches a
convolution. Foveation and the log-polar map are *architecture*, not augmentation.

### Content

Per crop, on GPU, in this exact order:

```
uint8 crop (3, 180, 180)  [cropped at the fixation point from the 224×224 image]
  → /255 → float
  → scale jitter                      (train only)
  → rotation                          (train: U[−15°,15°] · valid: 0° · inverted: exactly 180°)
  → FOVEATION                         (Jiang et al. 2015)
  → LOG-POLAR MAP                     (Polimeni et al. 2006)
  → float (3, 180, 180) → ResNet-18
```

**Foveation.** A 6-level Gaussian pyramid (blur kernel 5, σ = 0.248) is built and
blended as a function of distance from the crop centre — which *is* the fixation
point, since the crop was taken centred on it. Blend parameters p = 7.5, k = 3,
α = 2.5. Resolution is maximal at the fixation and falls off with eccentricity,
standing in for the cone-density gradient of the human retina.

**Log-polarization.** The foveated crop is remapped from Euclidean (x, y) to
(log r, θ) about the fixation point and resampled to 180×180, approximating the
retinotopic map into V1. Circumscribed positioning, log-polar distance 2.

**This stage is the whole experimental manipulation of the front end.** Three
variants exist and differ only here:
- `lp` — foveation + log-polar. **This is what all six probe models use.**
- `cnn` — foveation only, no log-polar.
- `plain` — neither; the raw crop goes straight to the backbone.

Worth saying out loud: the log-polar map is *not* rotation-invariant about the image
centre — it is computed about the fixation, and a 180° rotation of the input moves
every fixation to a new part of the object. That is why inversion is a real
perturbation for this architecture rather than a relabelling.

Only the saliency coordinates and the resized images are precomputed and stored as
memory-mapped arrays. Cropping, rotation, foveation and the log-polar map are all
applied on the fly, freshly randomized each epoch. One epoch does zero per-sample
file opens after the page cache warms.

### Visual

The centrepiece of the slide: a **single horizontal filmstrip of five real rendered
images**, each labelled with its tensor shape underneath:

`224×224 source` → `180×180 crop @ fixation` → `rotated` → `foveated` → `log-polar`

Use a real face from one of the training sets so the log-polar output is
interpretable rather than abstract. Between the foveated and log-polar panels, inset
a small diagram of the **log-polar sampling grid** — concentric rings and radial
spokes over the crop — which is the clearest way to show what the map does.

Directly beneath, a compact **three-row variant comparison** (`lp` / `cnn` / `plain`)
showing the same crop under each, with `lp` highlighted as the one in use. This is
the ablation structure of the whole project, and it costs three thumbnails to show.

---

# Slide 4 — Developmental curriculum

**Headline idea:** the network's visual world grows the way a child's does. Classes
are introduced in nested stages rather than all at once, and the three categories
grow on separate schedules.

### Content

Six stages, 80 epochs total. Each category has a single seeded random ordering of its
classes; stage *k* activates the first *n_k* of that ordering. Class sets are strictly
**nested** — nothing is ever removed.

| Stage | Epochs | CelebA | VGGFace2 | RFW-W | RFW-O | Objects | ZuBuD | generic | **Total classes** | Train crops |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 4 | 16 | 50 | 0 | 4 | 0 | 1 | **75** | 136,896 |
| 2 | 6 | 8 | 32 | 150 | 0 | 16 | 0 | 1 | **207** | 396,480 |
| 3 | 8 | 16 | 64 | 400 | 0 | 32 | 24 | 1 | **537** | 790,640 |
| 4 | 8 | 32 | 160 | 700 | 75 | 64 | 48 | 1 | **1,080** | 1,677,424 |
| 5 | 12 | 64 | 320 | 1000 | 180 | 64 | 88 | 1 | **1,717** | 2,278,000 |
| 6 | 40 | 128 | 480 | 1200 | 300 | 64 | 137 | 1 | **2,310** | 2,944,480 |

The shape of the schedule is the argument:
- **Objects saturate early** (all 64 by stage 4) — a child learns "cup", "dog",
  "chair" quickly and then stops adding basic-level categories.
- **Faces keep growing throughout** — expertise accrues over years.
- **Buildings arrive late** (stage 3) and stay comparatively few.
- **Other-race faces arrive last** (stage 4), after 700 white identities are already
  in place — this is the other-race-effect manipulation built into the schedule.
- **Stage 6 is half the run** (40 of 80 epochs) — the full-class regime gets the bulk
  of the training.

**Mechanics that make it honest.** Within a stage only active classes are loaded, and
logits of classes not yet introduced are masked out — to −10⁴ in the loss, to −∞ at
evaluation — so an unseen class can neither be predicted nor have its output weights
trained before its stage arrives.

**Sampling diet.** A `WeightedRandomSampler` gives each category a fixed share of
every batch, per-sample weight `share_c / n_c`, renormalized over the categories
present in the current stage. Target shares: faces 0.14, VGGFace2 0.26, RFW-W 0.16,
RFW-O 0.04, objects 0.19, ZuBuD 0.16, generic houses 0.05 — a **0.60 face arm**.
Without this the natural diet is dominated by objects and VGGFace2: in stage 6 the
natural shares are 35.7% objects and 51.9% VGGFace2 against 0.2% ZuBuD. Draws per
epoch equal the subset size, so weighting changes the diet and not the compute.

**Learning rate.** A **single global cosine spanning all 80 epochs**, not a per-stage
cosine — a restart per stage would drive the rate to zero six times over and freeze
the features before the hard many-class stages began. Multiplied by a linear
**200-step warm-up at the start of each new stage** (stages 2–6) to absorb the shock
of newly added classes.

Early stopping (patience 20) and best-model tracking apply **only within the final
stage**, since accuracy is not comparable across stages while the class set grows.

### Visual

This slide should be one large figure — the best chart in the deck.

**A step-area chart.** X-axis: epoch, 0 → 80. Y-axis: number of active classes,
0 → 2,310. Draw it as a **stacked step area**, one band per category in the deck's
category colours, so each band's own staircase is visible and the total envelope is
the class count. Mark the six stage boundaries with light vertical rules and label
each stage with its epoch count.

**Overlay the learning rate** as a single line on a secondary right-hand axis: the
global cosine, with the five warm-up notches visible as small dips at the stage
boundaries. The pairing of these two curves on one pair of axes *is* the slide —
it shows the class set growing while the LR decays monotonically, and makes the
"why not a per-stage cosine" point without a sentence.

Annotate two or three moments directly on the plot rather than in a caption:
"objects saturate", "other-race faces enter", "stage 6: 40 epochs at full class set".

Keep the numeric table off this slide if the chart carries it; if you include it,
shrink it to a footer strip and drop the crop-count column.

---

# Slide 5 — Learning setup

**Headline idea:** a stock ResNet-18 from random init, with one non-standard piece —
a 256-unit stochastic binary bottleneck that the memory model will later read.

### Content

**Backbone.** torchvision **ResNet-18, trained from scratch** — no ImageNet
initialization, so everything the network knows was learned through the log-polar
foveated front end. **11.9M parameters.** Final avgpool and fc are removed, leaving
the convolutional stages; the 512-channel feature map is adaptively average-pooled to
a 512-d vector.

**The bottleneck — this is the only architectural modification.**

```
512-d pooled feature
  → fc1 (512 → 256)
  → h = σ(z / T),  T = 2.0        temperature-scaled logistic, h ∈ [0,1]^256
  → dropout 0.3                   ON THE PATH TO fc2 ONLY — never on the returned h
  → fc2 (256 → 2,310)             unified softmax
```

`h` is used two ways:
- **During training**, deterministically: `h = probs`, the expectation.
- **During the Yin simulation**, the same 256 units are treated as Bernoulli
  variables and *sampled* to a binary code: `h ~ Bernoulli(probs)`. That binary code
  is the representation the memory model reads.

Naming, since this trips people up: the **NIMBLE** part of this project is the
kernel-density *memory model* applied at simulation time (slide 8), not a change to
the backbone. The backbone's only deviation from stock ResNet-18 is this 256-unit
sigmoid bottleneck that makes a samplable binary code available. Say this explicitly
on the slide — it is a question the room will ask.

**Optimization**

| | |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3, global cosine to 0 over 80 epochs |
| Weight decay | 0.05, **applied to `ndim > 1` parameters only** — biases and norm parameters excluded |
| Batch size | 256 |
| Loss | Cross-entropy, no label smoothing |
| Precision | bfloat16 mixed precision, channels-last memory format |
| Epochs | 80 (6 stages) |
| Early stopping | patience 20, final stage only |
| Dataloader | 8 workers, file_system sharing |

**Classification-time inference (distinct from the simulation).** The 16 fixations of
a base image **vote**: their logits are summed and the argmax taken over active
classes. Voting is inference-only — during training every fixation crop is an
independent example. This is the one place the word "voting" is correct.

**The six probe models.** Six runs of this exact recipe, `r19_rfwh_s42` … `s47`.
`--seed` sets `torch.manual_seed` and `np.random.seed`, which controls **weight
initialization and batch order**. `--curriculum-seed` is pinned at **0 in all six**,
so which identities enter at which stage is byte-identical across replicates and
initialization is the only variable.

Final validation accuracy across the six: **69.48 / 69.32 / 69.69 / 69.66 / 69.39 /
69.35 %** — a spread of **0.37 points**. This is the variance control the project
has never had: every cross-model comparison in this line of work has been a single
training run per condition, so a gap between two conditions had never been checked
against the spread you get from simply re-rolling the same recipe.

### Visual

**Left two-thirds: an architecture block diagram**, left to right, with every tensor
shape annotated:

`(3,180,180)` → `[ResNet-18 conv stages, from scratch, 11.9M]` → `(512,6,6)` →
`[AdaptiveAvgPool]` → `(512)` → `[fc1]` → `(256)` → `[σ(z/2.0)]` → `h (256)` →
`[dropout 0.3]` → `[fc2]` → `(2310)`

Branch `h` downward with a distinct arrow labelled **"→ Yin simulation (slide 8)"**
and draw the two modes as a small fork: a solid arrow `h = probs` marked *train*, and
a dashed arrow `h ~ Bernoulli(probs)` marked *simulate*. Render `h` as a short strip
of 256 cells so the binary-code idea is planted here and can be picked up on slide 8.
Show the dropout sitting on the fc2 path only, visually off to the side of the
branch — that placement is a real design decision and a diagram makes it free.

**Right third:** the hyperparameter table, quietly styled.

**Bottom strip:** the six seeds as a small horizontal dot plot of final validation
accuracy on a tight x-axis (say 69.0–70.0), six dots, mean line. The tightness of
that cluster is the point — give it room to read as tight.

---

# Slide 6 — Held-out batteries and the Yin (1969) design

*(Not in the original outline — inserted because slides 7 and 8 are not readable
without it.)*

**Headline idea:** four held-out stores, each exactly 64 items, run through a
2×2 crossing of study and test orientation.

### Content

**The 40 / 24 split.** Each store's 64 items are shuffled (seeded) and split:
**first 40 → study set**, **next 24 → never-studied distractors**. Every store yields
exactly 64 items, so the design is identical across categories with nothing rescaled.

How an "item" is defined depends on the store: for multi-class stores (face
identities, object categories, ZuBuD buildings) one item = one class, represented by
that class's first packed image. For a single-class store one item = one photograph.

Face stores are shuffled before splitting; houses and objects are packed in class
order and are not.

**The four stores, and what each one is for:**

| Store | Role | Held out how |
|---|---|---|
| `faces_cfdWM64` | Faces, studio-controlled | CFD absent from training entirely |
| `faces_rfwWM64` | Faces, in-the-wild — matched to cfdWM64 on N, race and sex | Identity-disjoint from training |
| `houses_yin64` | **Yin's own non-face control** — mono-oriented, fine-level | Building-disjoint from the 137 trained |
| `objects` | Basic-level categorization control | Images held out; **categories are trained** |

Faces are expected to show the inversion cost. Houses and objects are the controls
that should not — the dissociation, not the face effect alone, is the claim.

**The 2×2.** Orientation is crossed between study and test and applied on the fly:
upright = identity transform, inverted = exactly 180°. There is no separate inverted
data split; both orientations are rendered from the same packed images, which makes
this a fully matched-image contrast.

| | Test upright | Test inverted |
|---|---|---|
| **Study upright** | **UU** | UI |
| **Study inverted** | IU | **II** |

UU and II are the anchored diagonal (slide 7). UI and IU are the study–test
congruence conditions.

**The invariant everything rests on:** no model in this set ever saw an inverted view
during training. Inverted exposure was 0%. Any inversion effect is therefore emergent
from upright-only experience, not learned from inverted examples.

### Visual

**Top half:** a 4-across row of store cards, each in the hatched "held-out" style
established on slide 1. Each card: one representative thumbnail, the store name, and
the **8×8 = 64 dot grid with the first 40 dots filled (study) and the last 24 hollow
(distractors)**. Reusing the dot grid from slide 1 makes the 40/24 split legible
without a legend. Give the `objects` card a small warning tick or footnote marker for
the trained-categories caveat.

**Bottom half:** the 2×2 condition matrix rendered as actual images, not a table —
four cells, each showing a small studied-face thumbnail and a test thumbnail in the
correct orientation, so an inverted cell literally shows an upside-down face. Tint
each cell with the upright/inverted accent hues. Mark UU and II as "anchored" with a
small badge; this sets up slide 7 directly.

---

# Slide 7 — Two-noise calibration

**Headline idea:** rather than comparing raw model accuracy to human accuracy, we ask
*how much memory noise* it takes to bring the model to the human level — separately
for upright and inverted study. The noise levels are the measurement.

### Content

**The noise.** Retrieval noise is a **bit flip** on the binary code: each of the 256
bits is independently flipped with probability *p* (XOR with a Bernoulli mask).

**Two parameters, fit per model, per category, per seed.**

| | Fit against | Human anchor (Yin 1969) | Target |
|---|---|---|---|
| **p₁ (upright)** | Upright-study / upright-test accuracy | UU = 96.29% | 0.96 |
| **p₂ (inverted)** | Inverted-study / inverted-test accuracy | II = 81.88% | 0.82 |

Constraint: **p₂ ≥ p₁ + 0.01**.

**How the fit runs.** Binary search over *p* (`utils.search`), initial bracket
[0, 0.40], bracket extension permitted upward only, domain hard-clamped to **[0, 0.5]**,
convergence precision 0.01, assuming a monotonically decreasing accuracy-vs-*p* curve.
Each probe of the search is a full run of that condition.

**Why the domain stops at 0.5** — worth a sentence, because it is the kind of detail
that reveals whether the method was actually validated. Under the default
`--noise-phase both`, accuracy-vs-*p* is **U-shaped**, not monotonic: near *p* = 1
almost every bit flips *consistently*, and Hamming distance is invariant under
complement, so discriminability returns to near its *p* = 0 value. A real measured
trace from this pipeline on objects-II: *p* = 0.33 → 95.83%, *p* = 0.65 → 83.33%,
*p* = 0.99 → 100.00%. Any *p* fitted over the full range is ambiguous with its mirror
at 1 − *p*.

**Which phase the battery actually uses: `--noise-phase study`.** Noise is applied to
the **stored memory bank only**; test probes are encoded clean. The justification is
both mechanistic and practical — the memory trace decays, the probe in front of you
does not; and the resulting curve is monotonic and single-valued over the whole
range, passing through chance at *p* = 0.5 and continuing below chance as the
familiarity judgement inverts, which the `both` curve can never do.

**Then the four conditions run, and the noise level is selected by the STUDY
orientation** — because the memory bank is what the noise was fitted to:

- UU → p₁  ·  UI → p₁  ·  II → p₂  ·  IU → p₂

### ⚠ The point the slide must land

Pinning UU to 0.96 and II to 0.82 makes UU − II ≈ +14 **for any model, by
construction**. It is not an effect size and must never be reported as one. The
measured quantity is **the gap p₂ − p₁**: how much *extra* memory corruption an
upright-studied representation tolerates relative to an inverted-studied one. A model
with a strong inversion effect needs a much lower p₂ than p₁ to fall to the human
inverted level; a model with no inversion effect needs nearly the same *p* for both.

Give this its own visual weight — a callout box, not a bullet.

**Scale and bookkeeping.** 6 models × 4 categories × **10 simulation seeds** (101–110)
= 240 work units; each seed refits **both** noises independently, so the per-seed
fitted p's are themselves the distribution of interest.

Why 10 seeds and not 1: with 24 test pairs a single seed quantises accuracy to
4.17-point steps, so it cannot resolve a 96.29% target at all; and measured
seed-to-seed SD is 2.96 points on one face store and 6.62 on houses. A one-seed
number would be noise.

Targets that no *p* in the domain can reach are recorded as **UNREACHABLE** and never
retried — that is a property of the model and category, not a transient failure.

### Visual

**Main figure — a schematic calibration plot. Label it "schematic" explicitly.**
X-axis: bit-flip probability *p*, 0 → 0.5. Y-axis: 2AFC accuracy, 50% (chance floor,
drawn as a dashed rule) → 100%.

Two monotonically decreasing curves in the upright and inverted accent hues. Draw two
horizontal target lines at **0.96** and **0.82**, and drop verticals from where each
curve crosses its target down to the x-axis, labelled **p₁** and **p₂**. Then
annotate the horizontal distance between those two verticals with a bold bracket
labelled **"p₂ − p₁ — this is the effect size"**. Make that bracket the most visually
prominent element on the slide.

**Inset, small, upper right:** the U-shaped `both` curve over the full [0, 1] range
with the ambiguity at *p* and 1 − *p* marked, and a strike or "not used" tag. Caption
it with the measured objects-II trace (0.33 → 95.83, 0.65 → 83.33, 0.99 → 100.00).
This inset justifies the phase choice in about two square inches.

**Optional, along the x-axis:** a few tick marks showing binary-search bracket
narrowing, to convey that each fitted *p* costs several full condition runs.

**Callout box, bottom:** the UU − II warning, in a visually distinct treatment.

---

# Slide 8 — The Yin run

**Headline idea:** study builds a noisy memory of 40 items; test asks, 24 times,
which of two images feels more familiar.

### Content

**Study phase — 40 items.** For each studied item:
- Load the **first 10** of its 32 stored fixation crops.
- Render at the study orientation, push through the network, take `probs` at the
  256-unit bottleneck, **sample Bernoulli** → a 10 × 256 binary matrix.
- **Apply the calibrated bit-flip noise** at p₁ or p₂ (selected by study orientation).
- Store as that item's memory bank entry `M_c`.

Result: a memory bank of 40 entries, each 10 × 256 binary.

**Test phase — 24 two-alternative forced-choice trials.** Each trial pairs one
**old** item (studied) against one **new** item (from the 24 never-studied
distractors). Each is encoded with **all 32 fixations** at the test orientation,
giving a 32 × 256 binary probe `F`. Under `--noise-phase study` these probes are
**clean** — no bit flips on the test side.

Note that the 10 study fixations are the first 10 of the same 32 used at test, so the
study code is a subset of the probe's fixation set, not an independent sample.

**The decision rule — Barrington/NIMBLE kernel-density familiarity.** Not voting.

For a probe `F` (32 fixations) against memory entry `M_c` (10 stored codes):

```
score(F, M_c) = Σ over the 32 probe fixations i of
                  log ( mean over the 10 stored codes j of
                          exp( − ‖f_i − m_cj‖² / (2σ²) ) )

familiarity(F) = max over all 40 memory entries c of score(F, M_c)

respond OLD if familiarity(F_old) > familiarity(F_new)
```

with bandwidth **σ = 2.0**. On binary codes ‖f − m‖² is exactly **Hamming distance**,
so the kernel is a decaying function of how many of the 256 bits differ.

Read in words for the slide: each probe fixation is scored as a Parzen-window
likelihood under the cloud of 10 stored fixation-codes for a given remembered item;
the 32 fixation log-likelihoods are summed into one score for that item; the probe's
familiarity is its **best match anywhere in the memory bank**; and the trial is won
by whichever of the two images has the higher best-match. Accuracy = correct / 24.

Two properties worth naming: the max over the bank means a probe does not need to
match the *right* memory to feel familiar, only *some* memory — which is what makes
this a familiarity model rather than an identification model. And summing logs across
fixations means a probe must be consistently plausible across all 32 looks, not
plausible on one.

This whole procedure runs four times per calibration, once per condition in the 2×2.

### Visual

A **three-stage flow diagram** running left to right, and this should fill the slide.

**Stage 1 — Study (left).** A column of 40 stacked item cards, compressed with an
ellipsis after the first three. Expand one: show its 10 fixation crops as a small
strip → an arrow through a box labelled `ResNet-18 → h`, then a `Bernoulli` box →
a **10 × 256 grid of black/white cells**. Then a `⊕ noise (p₁ or p₂)` operation with
a scatter of **cells flipping, marked in red**. This red-flip motif is the clearest
way to render "bit-flip noise" and should be visually memorable. Label the output
`M_c : 10 × 256`.

**Stage 2 — Test (middle).** One trial. Two rows side by side, **OLD** and **NEW**,
each showing 32 fixation crops collapsed into a strip → `ResNet-18 → h` →
`Bernoulli` → a `32 × 256` grid. Mark both rows **"clean — no noise"** with a small
badge, since that asymmetry against stage 1 is the single most misreadable part of
the method.

**Stage 3 — Decision (right).** A box containing the scoring formula, set large
enough to read. Beneath it, a small **bar chart of the 40 per-memory-item scores**
for the OLD probe and for the NEW probe, with the maximum bar highlighted in each —
this renders the `max over c` step, which prose handles badly. Then a comparator
symbol and the verdict, and `× 24 trials → accuracy`.

Along the top of the whole diagram, a thin persistent banner showing which of the
four conditions is being illustrated (e.g. UU highlighted in the 2×2 from slide 6),
with a note that the diagram repeats for all four.

---

# Slide 9 — What the six replicates buy *(optional closing)*

**Headline idea:** this is the first time the recipe's own run-to-run variance has
been measured, so condition differences can finally be read against a baseline.

### Content

Six trainings of a byte-identical recipe, varying only weight initialization and
batch order. Curriculum content is pinned identical. Final validation accuracy spans
**69.32 – 69.69%**, a 0.37-point spread.

Every one of the six is then put through the full two-noise battery on all four
held-out stores at 10 simulation seeds each, and **each seed refits both noise
parameters independently**. The deliverable is therefore not a point estimate but a
distribution of fitted p₁, p₂ and (p₂ − p₁) with two nested sources of variance
separated:

- **Between-model** — how much of an effect is initialization luck.
- **Between-seed, within-model** — how much is item-sampling and Bernoulli noise.

The battery runs as 240 independent work units, claimed atomically so workers can be
added or killed without coordination and without redoing finished units.

**Results are pending. State this plainly and show no values.**

### Visual

A **2 × 4 grid of empty axes**, rows = the two face stores and the two control
stores (or 4 panels, one per store), each panel set up to eventually hold six
clusters of ten points — one cluster per model, one point per seed — on a y-axis
labelled **p₂ − p₁**. Draw a horizontal zero line, since "is the gap above zero" is
the question each panel will answer. Leave the panels **visibly empty** with a
"pending" watermark rather than mocking up plausible data.

Beside it, a small schematic of the work matrix: **6 models × 4 categories × 10 seeds
= 240 units**, rendered as a grid so the scale of the battery is immediate.

---

## Source of truth

If you need to verify or extend anything, these are the files the numbers came from.
Prefer reading them over guessing.

| What | Where |
|---|---|
| Training recipe, curriculum, weights | `scripts/train_r21_vgg2k.sh` |
| Resolved config as run | `runs/*_r19_rfwh_s42/config.json` |
| Stage table, class counts, crop counts | header of `runs/logs_r19_rfwh_s42.log` |
| Optimizer, LR schedule, sampler, masking | `train.py` |
| Architecture and the 256-d bottleneck | `model.py` |
| Fixation sampling, Gabor bank, augmentation | `salience_trans.py` |
| Foveation and log-polar parameters | `trans.py` |
| Packed-store layout and the first-16 rule | `datasets.py` |
| Two-noise simulation, KDE, conditions | `simulate_yin1969_bothnoise.py` |
| Binary-search fitting | `utils.py` (`search`) |
| Battery orchestration, seeds, stores | `scripts/run_bothnoise.sh` |
| Dataset counts per store | `fixation_data/<category>/<split>/meta.json` |

Note that `paper_methods.md` in the repo root describes an **older** model round and
disagrees with this brief on augmentation, dataset composition and curriculum shape.
Where they conflict, **this brief is correct for the six probe models**; that file is
not.
