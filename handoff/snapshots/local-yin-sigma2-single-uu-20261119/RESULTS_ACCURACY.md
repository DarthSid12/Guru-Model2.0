# Fixed sigma 2, one noise calibrated on face UU

Accuracy %, mean ± sample SD. 50 simulation seeds 101–150; same fitting/report seeds, not independent validation. LP noise = 0.373; CNN noise = 0.364. Each rate is used at study and test in all orientations and transferred unchanged from faces/houses to faces/objects.

Each study group contains 40 images: 20 per category. Test: 12 old/new pairs per category, comparing only within categories. Original KDE, sigma 2 for both models, 10 study and 32 test fixations, epoch124/trainingseed46 checkpoints. II is not calibrated.

## 20 faces + 20 houses

### Faces

| System | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---:|---:|---:|---:|---:|---:|
| Human | — | 92.58 ± 9.08 | 68.25 ± 14.25 | 57.17 ± 11.58 | 63.75 ± 11.75 | 29.53 |
| LP | 0.373 | 92.50 ± 6.57 | 63.67 ± 12.68 | 62.50 ± 14.01 | 63.17 ± 13.89 | 29.39 ± 10.44 |
| CNN | 0.364 | 92.67 ± 6.44 | 47.00 ± 12.91 | 50.00 ± 13.47 | 61.83 ± 11.80 | 39.72 ± 10.74 |

### Houses

| System | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---:|---:|---:|---:|---:|---:|
| Human | — | 81.42 ± 13.33 | 76.17 ± 15.25 | 71.42 ± 12.25 | 71.50 ± 11.33 | 8.39 |
| LP | 0.373 | 57.33 ± 12.67 | 53.17 ± 12.70 | 54.00 ± 12.05 | 58.33 ± 13.36 | 2.17 ± 11.53 |
| CNN | 0.364 | 52.50 ± 15.82 | 50.17 ± 13.09 | 50.17 ± 14.92 | 62.00 ± 13.49 | -1.61 ± 17.81 |

## 20 faces + 20 objects

### Faces

| System | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---:|---:|---:|---:|---:|---:|
| Human | — | 92.58 ± 9.08 | 68.25 ± 14.25 | 57.17 ± 11.58 | 63.75 ± 11.75 | 29.53 |
| LP | 0.373 | 91.17 ± 8.15 | 61.50 ± 16.31 | 61.17 ± 17.30 | 64.67 ± 13.74 | 28.72 ± 11.22 |
| CNN | 0.364 | 95.17 ± 6.54 | 50.67 ± 15.23 | 47.00 ± 15.22 | 62.67 ± 15.54 | 41.72 ± 11.63 |

### Objects

| System | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |
|---|---:|---:|---:|---:|---:|---:|
| Human airplanes reference | — | 69.58 ± 14.08 | 73.42 ± 16.17 | 65.50 ± 13.17 | 67.92 ± 16.92 | 0.64 |
| LP | 0.373 | 85.83 ± 10.81 | 75.17 ± 12.20 | 74.83 ± 11.36 | 83.83 ± 8.81 | 7.89 ± 10.28 |
| CNN | 0.364 | 86.33 ± 10.48 | 69.17 ± 12.63 | 70.50 ± 11.44 | 83.17 ± 9.60 | 12.06 ± 10.75 |

Model SD measures variation across simulation seeds; human SD is across participants. Human contrast SD is unavailable. The historical airplane reference uses different stimuli. The object bank contains 16 categories outside the revised training diet.
