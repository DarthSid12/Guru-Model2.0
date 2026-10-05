# Methods

## Data

We train a single network jointly on three categories — **faces**, **objects**, and
**houses** — in one unified label space, so that "expertise" (fine-level identity
discrimination) and "basic-level categorization" are learned by the same weights from
the same front end.

**Faces.** 128 celebrity identities (a manually cleaned face set), 16,625 training,
2,110 validation and 2,096 test images (~130 images per identity), split 80/10/10 by
image. Identities appear in a variety of poses, backgrounds and lighting; the label is
the identity, so the face task is fine-level discrimination.

**Objects.** 64 ImageNet (Deng et al., 2009) categories, 65,689 training, 8,208
validation and 8,218 test images (~1,027 training images per category), split 80/10/10.
Categories were chosen to be non-mono-oriented (objects seen in many orientations) and
are labeled at the basic level, so the object task is categorization rather than
identification.

**Houses.** The ZuBuD building database: 201 buildings photographed from 5 different
exterior viewpoints each. The label is the *building identity*, not a generic "house"
class, making houses a mono-oriented, fine-level task — the condition Yin (1969) used as
his non-face control. Of the 5 views per building, 3 are used for training and 1 each for
validation and test. During training the house set is capped at 40 identities
(`--max-classes-per-category houses_zubud=40`), reflecting that people know far fewer
buildings than faces, and giving 120 training photographs.

**Held-out face stimuli for the behavioral simulations.** The recognition-memory and
matching simulations (Experiments 3 and 4) must not be run on the identities the network
was trained to classify: a model asked to tell an old face from a new one should be
drawing on a memory trace, not on a classifier unit it already owns for that person. We
therefore took the test stimuli directly from the stimulus sets used by Dobs, Yuan,
Martinez and Kanwisher (2023), whose inversion experiments we simulate. Their **Set A**
comprises 200 photographs of young female celebrities — 40 identities × 5 photographs
each — and their **Set E** comprises 640 photographs, 64 identities × 10 photographs each.
Neither identity appears anywhere in the 128-identity training set, so every face in these
simulations is genuinely novel to the network; both are used purely as held-out probes and
neither contributes a single gradient step. Set E is the set actually used for the reported
simulations, for two reasons. First, its 64 identities supply, from one homogeneous pool,
both the 40 study faces and the 24 unfamiliar distractors that Yin's (1969) design
requires; drawing distractors from a different photographic source lets the model solve the
task by dataset appearance rather than by memory (studied and unstudied Set A items both
scored ~90% against training-set distractors, while the honest within-set test sat at
chance). Second, Set A's within-identity photographs resemble each other unusually little
(mean within-identity pixel correlation +0.078, versus +0.113 for Set E) while every
identity is a young female celebrity, so between-identity variability is also low — both
directions penalize a matching task, and the same model scored 69.3% on the Dobs et al.
matching paradigm with Set A but 88.0% with Set E under matched conditions (40 identities ×
5 photographs, identical noise parameter). This is not a resolution artifact: Set E in fact
contains more sub-224-pixel images than Set A.

**Unified label space.** 128 face identities + 64 object categories + 40 house identities
= **232 classes** under a single softmax.

**Fixation expansion.** Each base image is expanded into 16 fixation crops (see below),
giving **1,318,944 training crops** (266,000 face, 1,051,024 object, 1,920 house). At
evaluation time the 16 crops of a base image are kept together as one bundle, so the
validation and test sets contain 10,358 and 10,354 *base images* respectively.

## Transformation pipeline

All raw images are first resized and center-cropped to 224×224. Everything downstream is
computed on the GPU: only the (expensive) saliency-derived fixation coordinates and the
resized images are precomputed and stored as memory-mapped arrays; cropping, rotation,
foveation and the log-polar map are applied on the fly, freshly randomized each epoch.

**Saliency and fixations.** A Gaussian center prior (σ = W/4) is applied to the image to
capture the photographer/center bias, and the result is log-polarized *before* the
saliency operator is applied, because cortical saliency maps (e.g. monkey LIP) are
computed on an already retinotopic, log-polar input. We then use the orientation-energy
operator of Yamada and Cottrell (1995): the image is converted to grayscale, contrast-normalized, and
passed through a bank of 31×31 Gabor filters at two scales (λ = 4, 8, with σ = 0.5λ), four
orientations (θ = 0, π/4, π/2, 3π/4), and two phases (ψ = 0, π/2), with aspect ratio
γ = 0.5. Each quadrature phase pair is combined into a magnitude, giving 8 magnitude
maps, and their variance at each pixel is the saliency map: near zero in flat parts of the
image, high where it is "busy". We sample 32 fixation points from this map (excluding a
10-pixel border) and map them back to (x, y) coordinates in the original image; the first
16 are used, a number at which performance saturates.

**Cropping and augmentation.** Centered on each fixation, a 180×180 crop is taken from
the 224×224 image. At training time each crop receives a horizontal flip with p = 0.5
(faces and objects are approximately bilaterally symmetric; a *vertical* flip is never
applied, as it would counterfeit the inversion manipulation the experiment tests), a mild
random-resized-crop (scale 0.6–1.0), color jitter (brightness/contrast/saturation ±0.4),
random erasing (p = 0.25, 2–15% of the area), and a random rotation in ±15°. Holdout
images receive no augmentation: they are presented either upright or rotated by exactly
180° (the inverted condition).

**Foveation.** The retina is highly foveated (Curcio et al., 1990). We simulate this with
the algorithm of Jiang et al. (2015): a 6-level Gaussian pyramid (blur kernel 5, σ =
0.248) is built and blended as a function of distance from the fixation point
(p = 7.5, k = 3, α = 2.5), so resolution is highest at the fixation point and falls off
with eccentricity.

**Log-polarization.** The foveated crop is finally mapped from Euclidean (x, y) to
(log r, θ) about the fixation point, approximating the retinotopic map into V1 (Polimeni
et al., 2006), and resampled to 180×180. This is the only stage that distinguishes the
model (`lp`) from its controls: the `cnn` control applies foveation but no log-polar map,
and the `plain` control applies neither.

## Model

The backbone is a **ResNet-18** (He et al., 2016) trained **from scratch** (11.4M
parameters; no ImageNet initialization, so that everything the network knows is learned
through the log-polar, foveated front end). Its final average-pooling and fully connected
layers are removed, leaving the convolutional stages; the resulting 512-channel feature
map is adaptively average-pooled to a 512-d vector.

That vector passes through `fc1` (512 → 256) and a temperature-scaled logistic
nonlinearity, h = σ(z / T) with T = 2.0, producing a 256-unit code in [0, 1]. During
training h is used deterministically (its expectation); for the memory simulations the
same units are treated as Bernoulli variables and sampled to give a binary code, which is
the representation the Yin/NIMBLE kernel-density memory model reads. A dropout of 0.3 is
applied to h on the way into the classifier only (never to the returned code), and `fc2`
(256 → 232) produces the unified logits.

At inference, the 16 fixations of a base image vote: their logits are summed and the
argmax is taken over classes. Voting is used only at inference — during training every
fixation crop is an independent training example.

**Optimization.** AdamW, initial learning rate 1e-3, weight decay 0.05 (excluded from
biases and normalization parameters), batch size 256, cross-entropy loss, bfloat16 mixed
precision, channels-last memory format, seed 42.

## Developmental (curriculum) training

Rather than presenting all 232 classes at once, the network is grown the way a child's
visual world grows: it starts with a handful of familiar faces and a handful of object
categories, and new classes are added in stages. Each category gets a single seeded random
ordering of its classes and stage *k* activates the first *n_k* of that ordering, so the
class sets are strictly nested — nothing is ever forgotten — and the three categories grow
on *separate* schedules. Objects saturate early (a child learns "cup", "dog", "chair"
quickly), faces keep growing throughout (expertise accrues over years), and buildings are
introduced late and stay few.

| Stage | Faces | Objects | Houses | Total classes | Epochs |
|-------|-------|---------|--------|---------------|--------|
| 1 | 4 | 4 | 0 | 8 | 6 |
| 2 | 8 | 16 | 0 | 24 | 6 |
| 3 | 16 | 32 | 4 | 52 | 8 |
| 4 | 32 | 64 | 8 | 104 | 8 |
| 5 | 64 | 64 | 16 | 144 | 10 |
| 6 | 128 | 64 | 40 | 232 | 22 |

Sixty epochs in total. Within a stage, only the active classes are loaded, and the logits
of classes that have not yet been introduced are masked out (to −10⁴ in the loss, to −∞ at
evaluation), so an unseen class can neither be predicted nor have its output weights
trained before its stage arrives. The learning rate follows a *single* cosine decay
spanning all 60 epochs — a per-stage cosine would drive the rate to zero six times over
and freeze the features before the hard, many-class stages began — multiplied by a linear
200-step warm-up at the start of each new stage to absorb the shock of the newly added
classes. Early stopping (patience 10) and best-model tracking apply only within the final
stage, since accuracy is not comparable across stages while the class set is still
growing.

**Balanced-diet variant.** Because objects contribute ~1,000 images per class and houses
only 3, the natural training diet is 80–94% objects. We therefore ran two versions of the
developmental model that are identical in every respect except the sampling distribution:

- **Unweighted (r8):** classes are sampled in their natural proportions.
- **Weighted (r9):** a weighted sampler with per-sample weight `share_c / n_c` gives each
  category a fixed share of every batch — faces 45%, objects 45%, houses 10% (renormalized
  over the categories present in the current stage, since houses are absent from stages 1–2).
  The number of draws per epoch equals the subset size, so weighting changes the *diet*
  and not the amount of compute.

Both runs reach essentially the same accuracy on held-out upright images: 79.2% overall
for the unweighted model (faces 91.0%, houses 90.0%, objects 76.1%) and 78.8% for the
weighted model (faces 91.1%, houses 92.5%, objects 75.5%), with the weighted diet buying a
small gain on the data-starved house category at a small cost on objects.
