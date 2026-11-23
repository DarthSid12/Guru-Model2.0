# Fine alpha sweep for houses

User authorized further alpha exploration; the previous coarse sweep did not
establish an optimum. Preserve both parent experiments and all training jobs.
Frozen latest LP/CNN no-generic ResNet18 epoch124/trainingseed46; unchanged
20 faces+20 houses study, 12 old/new pairs/category, 10 study/32 test fixations,
UU/UI/IU/II and Bernoulli encoding. Existing single-noise settings LP.38 and
CNN.36 stay fixed during study and test. No training, image/cache changes or
noise calibration. Use stable unfloored float64 KDE, explicitly the separate
readout variant from the parent. The historical sigma2 control is retained.

Use the exact parent study-only reference seed42 and mean binary-code distance:
sigma=alpha*d_bar, shared across categories/orientations. Recompute and require
exact reference-record equality before replacing only the candidate grid.
Freeze twenty alpha values .05,.075,.10,.125,.15,.175,.20,.25,.30,.35,.40,.45,
.50,.60,.75,1,1.5,2,3,4, plus the fixed sigma2 control.

Development: same50 seeds101-150. Select ONE candidate from the twenty alpha
values minimizing mean house UU errors on development seeds, breaking ties by
smaller alpha. This is a declared upright-recognition criterion, not a claim
of best overall human fit. Monitor faces and all house orientations; also
report house human RMSE and whether mean UU errors are below II errors.

Validation: fifty new simulation seeds151-200, with new study/probe splits and
random codes from the same64-photo banks. Select before reading validation
scores in analysis; apply the frozen development winner. Evaluate the full
grid on validation only for transparent diagnostics, never reselect the winner
from it. This is a simulation holdout, not an independent stimulus dataset or
training-seed replication. Compare selected candidate vs sigma2 with paired
mean changes, sampleSD and SEM. Replay all800 legacy sigma2 counts and all
previously overlapping stable alpha-grid rows/masks on development seeds.

Expected33,600 category records,672cells of50seeds,168paired orientation
contrasts,2CPU8GiB30minute bounded shared job. Numerical/actual-cache tests must
pass before the sweep. Independently verify hashes, seeds, counts, moments,
paired changes, distance references and development-only selection locally.
