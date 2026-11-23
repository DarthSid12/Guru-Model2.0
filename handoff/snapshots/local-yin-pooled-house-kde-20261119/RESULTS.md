# Pooled house KDE experiment

Accuracy % ± sample SD across 50 simulation seeds 101–150. Fixed sigma 2. LP noise 0.373, CNN noise 0.364 at study and test, unchanged from face UU calibration. 20 faces + 20 houses studied, 12 house pairs percondition. The existing baseline is unchanged.

The pooled KDE averages all 200 studied house kernels before taking the log for each test fixation, then sums over 32 test fixations. No identity maximization remains. Test/novel fixation vectors never enter memory.

| Model / readout | UU | UI | IU | II |
|---|---:|---:|---:|---:|
| Human reference |81.42 ±13.33|76.17 ±15.25|71.42 ±12.25|71.50 ±11.33|
|LP Current: best of 40 memories|57.33 ± 12.67|53.17 ± 12.70|54.00 ± 12.05|58.33 ± 13.36|
|LP Best of 20 house memories|57.33 ± 12.67|53.17 ± 12.70|54.00 ± 12.05|58.33 ± 13.36|
|LP Pooled 200 house fixations|53.83 ± 13.06|51.83 ± 12.96|53.17 ± 12.24|55.50 ± 13.21|
|CNN Current: best of 40 memories|52.50 ± 15.82|50.17 ± 13.09|50.17 ± 14.92|62.00 ± 13.49|
|CNN Best of 20 house memories|52.50 ± 15.82|50.17 ± 13.09|50.33 ± 15.15|62.00 ± 13.49|
|CNN Pooled 200 house fixations|53.50 ± 13.27|51.17 ± 15.15|50.33 ± 14.18|56.17 ± 15.04|

Model SD is across Monte Carlo seeds, human SD across participants. Same fitting/report seeds as the parent experiment, not independent validation. The pooled arm assumes the stimulus category is known to select house memories.
