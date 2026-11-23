# Alpha exploration with zero added XOR noise

User requested different bandwidth values at noise0. Both completed no-generic
ResNet18 LP/CNN epoch124, trainingseed46; same64-image banks and frozen caches.
20faces+20houses study,12old/new pairs/category,10study/32testfixations, shared
40-item memory, UU/UI/IU/II. Added XOR probability0 during both study and test.
Bernoulli binary sampling remains active. No training, pixel or noise fitting.
The face-calibrated reference noise metadata is retained for provenance only;
analysis field noise_rate=0 is the rate used by this experiment.

Stable unfloored float64 KDE, matching the previous separate readout variant.
Reproduce parent seed42 zero-noise distance reference exactly. One shared sigma
per model across both categories/orientations. Sigma=alpha*d_bar. Test23alphas:
.005,.01,.025,.05,.075,.10,.125,.15,.175,.20,.25,.30,.35,.40,.45,.50,.60,.75,
1,1.5,2,3,4, plus sigma2 control. Smaller values extend the previous grid toward
the sharp-kernel limit; stable log-space scoring avoids density underflow.

Development50seeds101-150 select the candidate with minimum houseUU errors,
with ties choosing smaller alpha. Validation50seeds151-200 assess that frozen
choice on new random codes and study/probe splits within the same image bank.
Validation is not independent stimuli or training replication. Full grid and
both category outcomes are retained, but do not reselect based on validation.
Selection optimizes upright recognition; report all orientations and human
references so an incorrect inversion effect is visible.

Require numerical tests and8actual-cache score pilots, unchanged development
plans,800exact sigma2 controls and4800overlapping coarse-sweep rows/masks.
Expected38400category rows,768cells of50seeds,192paired orientation contrasts.
Local Mac CPU-only fallback after SDSC password rejection. Two single-threaded model workers,30minute walltime bound. Same torch2.5.1 CPU RNG version as SDSC; binary histograms must match exactly; computed binary/continuous distance moments must agree within1e-10, after which the stored SDSC scale is reused exactly and all4800coarse zero-noise decisions/masks must reproduce. This permits platform libm/reduction differences without changing any scale or code. Preserve all prior runs and inputs; no Slurm submission is involved.

Version2 preserves the failed version1 attempt. Its strict float-equality guard caught only tiny Mac/Linux distance-summary differences (about1e-15), with identical Hamming histograms. No output row had been generated before that guard. This recovery adds moment tolerances while reusing original reference floats; all4800parent correctness masks remain required to match exactly.
