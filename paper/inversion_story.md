# The face inversion effect in log-polar models

## 1. The data

Established clear line - Test and training identities never mix

**Training categories.**

| category | what it is | size |
| --- | --- | --- |
| `faces` | CelebA identities | 128 |
| `faces_vgg` | VGGFace2 identities — natural photo variation | 480 |
| `faces_cfd` | CFD identities — studio, neutral background | 119 |
| `faces_rfwW` / `faces_rfwO` | RFW identities, white / other-race | 1200 / 300 |
| `objects` | object classes | 64 |
| `houses_zubud` | ZuBuD buildings, each its own identity | 40 or 137 |
| `houses_gen` / `houses` | buildings lumped into one generic `house` label | 635–732 |

**Test stores.** Held out of training, so the model has never seen these
identities:

| store | what it is |
| --- | --- |
| `faces_cfdWM64` | 64 white-male CFD faces. Studio-lit, plain background, so identities differ mainly in facial structure. **This is where we read face effects.** |
| `faces_rfwWM64` | 64 white-male RFW faces, in the wild. Used to calibrate, not to measure — see §4. |
| `faces_vggHO` | 60 novel VGGFace2 identities |
| `houses_yin64` | 64 ZuBuD buildings |
| `objects` | the object set |

**The two tasks.** Yin (1969) is a memory task: study 40 items, then 24
old-vs-new 2AFC pairs. Dobs/Kanwisher (2023) is a matching task. One noise
parameter `p` per model — bit-flip noise on the 256-unit code — is tuned so
upright accuracy sits on the human anchor. Because `p` differs between models,
**compare effects, not raw accuracies.**

**Human targets.**

| Yin (1969) | upright accuracy | inversion effect |
| --- | --- | --- |
| Faces | 96.29 | **+14.41** |
| Houses | 90.71 | +4.96 |
| Objects | 84.79 | +0.83 |

---

## 2. The models, before we touched orientation

All ResNet-18, trained from scratch, log-polar crops around 16 fixation points.
Six-stage curriculum, 80 epochs. **No inverted images in training** — a
deliberate rule, so the inversion test stays clean.

| model | face diet | face ids | objects | house ids | generic house class | total classes |
| --- | --- | --- | --- | --- | --- | --- |
| `r15_vgg` | CelebA + VGGFace2 + CFD | 727 | 64 | 40 ZuBuD | 635 HD | 832 |
| `r16_base` | CelebA + VGGFace2 | 608 | 64 | 40 ZuBuD | 635 HD **+ 97 ZuBuD** | 713 |
| `r16_rfw` | `r16_base` **+ 1500 RFW** | **2108** | 64 | 40 ZuBuD | 635 HD **+ 97 ZuBuD** | 2213 |
| `r17_base` | CelebA + VGGFace2 | 608 | 64 | 40 ZuBuD | 635 HD | 713 |
| `r17_houses` | CelebA + VGGFace2 | 608 | 64 | **137 ZuBuD** | 635 HD | 810 |
| `r16rfwftH` | fine-tune of `r16_rfw` | **2108** | 64 | **137 ZuBuD** | 635 HD | 2310 |

Each face and house identity is its own class, and each object class is one
class. The generic house class is **one label no matter how many buildings it
holds** — 635 or 732 — which is what §3 turns out to be about. Note how
lopsided the diet is: even the smallest face arm has 608 identities against 40
houses.

**The faces got better.** Adding VGGFace2 gave `r15` a Kanwisher result within
half a point of humans, on identities it had never seen:

| | upright | inversion effect |
| --- | --- | --- |
| Human | 87.50 | **+10.70** |
| `r15_vgg` on `faces_vggHO` | 86.10 | **+11.10** |

**Adding RFW made it better still.** On held-out CFD faces, `r16_rfw` shows an
upright-over-inverted advantage of **+16.25** against a human +14.41. The same
model without RFW reads +10.75. 1500 in-the-wild identities seem to be what
builds a face representation that cares about orientation.

---

## 3. The houses fell over, and we fixed them

While the faces improved, the house control collapsed. Both `r16` models scored
held-out ZuBuD buildings **below the 50% chance floor**:

| model | generic house class | house ids | `houses_yin64` upright |
| --- | --- | --- | --- |
| Human | — | — | 90.71 |
| `r16_base` | HD **+ 97 ZuBuD** | 40 | **33.92** |
| `r16_rfw` | HD **+ 97 ZuBuD** | 40 | **42.17** |
| `r16rfwftH` | HD only | 137 | 59.92 |
| `r17_base` | HD only | 40 | 85.17 |
| `r15_vgg` | HD only | 40 | 87.75 |
| `r17_houses` | HD only | 137 | **96.50** |

**The cause.** `r16` folded 97 leftover ZuBuD buildings into the single generic
`house` label. Training the network to map 97 ZuBuD buildings onto one label
destroys the representation the other ZuBuD identities depend on.

**The fix.** Take ZuBuD out of the generic class (`r17_base`: 85.17). Add more
house identities on top, 40 → 137, and it goes to ceiling (`r17_houses`: 96.50).
A 20-epoch fine-tune recovers most of it without retraining from scratch
(`r16rfwftH`: 59.92).

Two things this leaves behind:

- **More house training makes houses behave more like faces.** `r17_houses`
  (137 ids) shows a *bigger* inversion cost than `r17_base` (40 ids). The cost
  tracks how much individuation training a category got, not what kind of
  category it is. That weakens any face-specificity claim resting on houses.
- **Houses are memorised, not learned.** 120 house images take 10% of every
  batch, so each crop is seen ~11,909 times over the run against 60 for
  `faces_vgg`. The weight is compensating for a dataset far too small.

---

## 4. The main finding: the model has good memory, bad orientation sense

Yin's design has four cells — study orientation, then test orientation.
`UU`, `UI`, `IU`, `II`. Earlier work here only reported `UU` minus `UI`. Running
all four shows that was the wrong number.

**`faces_cfdWM64`, one calibrated `p` per model, 50 seeds:**

| model | p | UU | II | UI | IU |
| --- | --- | --- | --- | --- | --- |
| **Human** | — | 96.29 | 81.88 | 84.13 | 78.58 |
| `r15_vgg` | 0.09 | 89.17 | 85.17 | 69.50 | 63.25 |
| `r16_base` | 0.08 | 85.25 | 74.50 | 63.83 | 63.67 |
| `r16_rfw` | 0.33 | 81.42 | 65.17 | 59.58 | 59.42 |
| `r16rfwftH` | 0.37 | 73.25 | 57.00 | 56.33 | 55.92 |
| `r17_base` | 0.20 | 70.17 | 77.25 | 61.67 | 56.50 |
| `r17_houses` | 0.06 | 84.50 | 88.58 | 69.58 | 65.00 |

**The pattern is the same everywhere: UU and II are high, UI and IU are low.**

Humans do not do this. For humans `II` (81.88) sits *below* `UI` (84.13).
**Every one of our six models reverses it**, on both face stores — all twelve
cells.

**What this means.** `II` is the hardest condition for a person: the face is
upside down when you learn it and upside down when you are tested. Our models
find it nearly as easy as upright. Studying and testing inverted costs
`r16_base` 0.17 points on RFW faces. What actually hurts the model is the two
orientations *disagreeing*.

So the model is doing something different from a person:

- **Its memory is fine.** It encodes an image and recognises it again through
  the retrieval noise, upside down or not. Orientation does not make the image
  harder to store.
- **It does not know the two views are the same face.** Show it upright, test it
  inverted, and it treats them as different things. A person sees a face they
  already know, just rotated.

Splitting the 2×2 into main effects plus an interaction makes the gap explicit.
The interaction — the extra cost of study and test disagreeing, beyond either
main effect — is the term that is wrong:

| model | study effect | test effect | **mismatch** |
| --- | --- | --- | --- |
| **Human** | +9.98 | +4.43 | **+7.73** |
| `r15_vgg` | +5.13 | −1.12 | +20.79 |
| `r16_base` | +5.46 | +5.29 | +16.12 |
| `r16_rfw` | +8.21 | +8.04 | +13.79 |
| `r16rfwftH` | +8.33 | +7.92 | **+9.00** |
| `r17_base` | −0.96 | −6.12 | +14.62 |

The main effects are roughly human-sized. The mismatch term is two to three
times too big.

**And it comes from the architecture, not from training.** Reading the same
battery out of each layer:

| readout | study | test | **mismatch** |
| --- | --- | --- | --- |
| layer1 | −0.71 | −0.29 | **+4.87** |
| layer2 | −0.50 | −0.58 | +4.00 |
| layer3 | −0.71 | −1.38 | +4.46 |
| layer4 | +1.75 | +4.25 | +26.17 |
| `h` (bottleneck) | +7.17 | +4.75 | +16.50 |

(`r16rfwftH` on `faces_rfwWM64`.) The mismatch cost is **already there at
layer1**, where the orientation main effects are exactly zero. Nothing has been
learned yet at that depth. It is the log-polar sampling grid: rotate the image
180° and the representation shifts, so matching across a mismatched pair is
harder no matter what the image contains. The main effects, by contrast, appear
only at layer4 and the bottleneck — those are learned.

**One caveat on the stores.** `faces_rfwWM64` photos are in the wild, so
background, lighting, pose and clothing all differ between identities and all
survive inversion. The 2AFC can be won without looking at the face — which is
why we calibrate on it but measure on `faces_cfdWM64`.

---

## 5. Putting inverted images into the diet

I thought exposing the network to some inverted images might help orientation sensing but that's obviously a dumb idea you can skip this portion but I still wrote it up.
The `r18` arms are `r16_rfw`'s recipe with one thing changed: a fraction of
training images are flipped. `r16_rfw` itself is the 0% control.

**I expected** the mismatch term to fall toward the human +7.73 while the upright advantage survived — most of the diet is still upright.

**What happened is that everything went at once, and fast.**

Training accuracy first. The gap between upright and inverted classification:

| arm | faces upright / inverted | gap |
| --- | --- | --- |
| `r16_rfw` (0%) | 90.47 / 63.31 | **27.16** |
| `r18_inv02` (2%) | 86.26 / 85.40 | **+0.86** |
| `r18_inv05` (5%) | 86.35 / 86.07 | +0.28 |
| `r18_inv20` (20%) | 86.68 / 86.26 | +0.42 |

**Two percent of the diet erases a 27-point gap.** 5% and 20% add nothing. The
effect saturates below the smallest dose we have tried.

The memory task says the same thing, on `faces_cfdWM64`:

| model | UU | II | UI | IU | study | test | **mismatch** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Human** | 96.29 | 81.88 | 84.13 | 78.58 | +9.98 | +4.43 | **+7.73** |
| `r16_rfw` (0%) | 81.42 | 65.17 | 59.58 | 59.42 | +8.21 | +8.04 | +13.79 |
| `r18_inv02` (2%) | 67.00 | 72.75 | 62.33 | 64.42 | −3.92 | −1.83 | **+6.50** |
| `r18_inv05` (5%) | 72.50 | 78.00 | 70.58 | 71.83 | −3.37 | −2.12 | **+4.04** |
| `r18_inv20` (20%) | 76.67 | 78.67 | 74.92 | 74.08 | −0.58 | −1.42 | **+3.17** |

**The four cells collapse together.** In the 0% control the two mismatch cells
sit about 22 points below `UU` (59.58 and 59.42 against 81.42). With inverted
images in the diet that gap closes to about 4 points at 2% and 2 points at 20%.
`UI` and `IU` climb — 59.58 → 62.33 → 70.58 → 74.92 — which is exactly what we
were trying to achieve. The problem is that `UU` **falls** at the same time,
81.42 → 76.67, so the four conditions meet in the middle rather than the
mismatch cells rising to meet upright. The model ends up equally good at
everything, which is not what a person does.

(`UU` is not monotonic across the arms — 67.00 at 2%, 72.50 at 5%, 76.67 at 20%,
all below the 0% control's 81.42. All four are calibrated to nearly the same
noise level, p = 0.33–0.36, so this is not a calibration artefact. Unexplained.)

**The idea did not work the way we wanted.** The mismatch term did collapse, and
cleanly with dose — +13.79 → +6.50 → +4.04 → +3.17, straight past the human
+7.73. But the study and test effects collapsed with it, from +8.21 and +8.04
down to negative at every dose. `II` is now *higher* than `UU`.

The 2% arm is the awkward one. Its mismatch term, **+6.50, is the closest to the
human +7.73 of any model in this document** — better than `r16rfwftH`'s +9.00.
It gets there with study −3.92 and test −1.83, so the number is right for the
wrong reason: the model is not more human-like, it has simply stopped caring
about orientation at all.

A few percent of inverted exposure does not buy partial orientation tolerance.
It makes the model close to orientation-blind, which removes every orientation
signature at once, including the human-like ones we wanted to keep. There is no
dose here where the architectural term drops and the learned advantage survives.

---

## 6. Where it stands

| | status |
| --- | --- |
| **Kanwisher, faces** | matched — `r15_vgg` 86.10 / +11.10 vs human 87.50 / +10.70 |
| **Yin, faces, upright advantage** | close — `r16_rfw` and `r16rfwftH` +16.25 vs human +14.41 |
| **Yin, faces, mismatch cost** | 2–3× too big in every model; comes from the log-polar encoding, not training |
| **Yin, objects** | correct — near zero everywhere, but stuck at ceiling |
| **Yin, houses** | repaired, but the cost grows with house training, so it is a confounded control |
| **Inverted diet** | every dose from 2% up erases the classification gap and the memory effects together; no dose separates them |
