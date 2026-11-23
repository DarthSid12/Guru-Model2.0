# Yin with nearest-memory prototypes

Both authorized designs completed locally using the unchanged completed Delta no-generic LP/CNN ResNet18 checkpoints,epoch124/trainingseed46. Each studied image stores the mean of10binary fixation codes;each test image is represented by the mean of32codes. Retrieve the closest ofall40studied image prototypes in256D squaredEuclidean distance. Choose the test picture with the smaller nearest-memory distance. The score function receives no identity labels or old/new flags. This replaces Gaussian KDE pooling with nearest-prototype matching;there is no alpha or sigma in the new readout.

Fit one added XOR probability per model/design using only faceUU on50calibration simulation seeds101–150,aiming at the human92.5833%reference. The same fitted probability applies at study/test and across allorientations/categories. Evaluate50different simulation seeds151–200. Houses,objects and faceUI/IU/II are not fitted. This is a simulation holdout using the same64-photo banks,not an independent stimulus sample or training replication.

## Face calibration

| Model | Study design | New noise | Calibration faceUU | Validation faceUU | Historical target |
|---|---|---:|---:|---:|---:|
| LP | faces_houses | 0.380 | 92.00% | 90.50% | 92.58% |
| LP | faces_houses_objects | 0.370 | 93.50% | 96.00% | 92.58% |
| CNN | faces_houses | 0.370 | 93.83% | 95.00% | 92.58% |
| CNN | faces_houses_objects | 0.380 | 92.25% | 92.00% | 92.58% |

Finite noise-grid resolution and held-out simulation variation mean the displayed validation faceUU values need not equal the fitting target. No calibration was adjusted after inspecting validation. Alltargets lie within the sampled calibration ranges.

## 20faces+20houses studied;12old/new pairs/category

Values are errors on a12-point scale,mean ± sampleSD across50validation seeds. In the three-category design,raw errors out of8are multiplied by1.5;actual counts are retained inCSV. HumanSD reflects participant variation and is not model simulationSD. Human references use different pictures;airplanes are only a historical proxy reference for the diverse object bank.

### faces

| Model / reference | Noise | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| Human faces | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 |
| LP nearest memory | 0.380 | 1.14 ± 0.99 | 4.84 ± 1.62 | 4.90 ± 2.22 | 3.94 ± 2.00 |
| CNN nearest memory | 0.370 | 0.60 ± 0.78 | 5.84 ± 1.71 | 6.08 ± 1.76 | 4.54 ± 1.51 |

### houses

| Model / reference | Noise | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| Human houses | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP nearest memory | 0.380 | 5.34 ± 1.88 | 5.64 ± 1.84 | 5.94 ± 1.63 | 5.18 ± 1.49 |
| CNN nearest memory | 0.370 | 4.98 ± 1.58 | 5.78 ± 1.72 | 5.82 ± 1.72 | 4.42 ± 1.86 |

## 14faces+13houses+13objects studied;8old/new pairs/category

Values are errors on a12-point scale,mean ± sampleSD across50validation seeds. In the three-category design,raw errors out of8are multiplied by1.5;actual counts are retained inCSV. HumanSD reflects participant variation and is not model simulationSD. Human references use different pictures;airplanes are only a historical proxy reference for the diverse object bank.

### faces

| Model / reference | Noise | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| Human faces | — | 0.89 ± 1.09 | 3.81 ± 1.71 | 5.14 ± 1.39 | 4.35 ± 1.41 |
| LP nearest memory | 0.370 | 0.48 ± 0.83 | 5.52 ± 2.28 | 4.53 ± 2.30 | 3.39 ± 1.79 |
| CNN nearest memory | 0.380 | 0.96 ± 1.12 | 5.79 ± 1.82 | 6.18 ± 1.91 | 4.56 ± 1.99 |

### houses

| Model / reference | Noise | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| Human houses | — | 2.23 ± 1.60 | 2.86 ± 1.83 | 3.43 ± 1.47 | 3.42 ± 1.36 |
| LP nearest memory | 0.370 | 4.41 ± 1.83 | 5.43 ± 2.74 | 5.16 ± 1.69 | 4.02 ± 1.67 |
| CNN nearest memory | 0.380 | 4.47 ± 2.09 | 5.73 ± 1.78 | 5.94 ± 2.14 | 4.89 ± 1.91 |

### objects

| Model / reference | Noise | UU | UI | IU | II |
|---|---:|---:|---:|---:|---:|
| Human airplanes reference | — | 3.65 ± 1.69 | 3.19 ± 1.94 | 4.14 ± 1.58 | 3.85 ± 2.03 |
| LP nearest memory | 0.370 | 1.71 ± 1.42 | 3.06 ± 1.89 | 3.33 ± 1.99 | 2.13 ± 1.69 |
| CNN nearest memory | 0.380 | 1.65 ± 1.49 | 4.29 ± 2.27 | 4.23 ± 2.09 | 2.19 ± 1.82 |

## House comparison against KDE

The original KDE control keeps sigma2and its original per-model/design face-calibrated noise. A second control uses the new fitted noise with the original KDE,so comparison with nearest memory isolates the readout at identical encoded samples. Comparisons below retain the same study/test plans and common random draws.

| Model | Design | Method | Noise | UU | UI | IU | II |
|---|---|---|---:|---:|---:|---:|---:|
| LP | faces_houses | nearest_memory | 0.380 | 5.34 ± 1.88 | 5.64 ± 1.84 | 5.94 ± 1.63 | 5.18 ± 1.49 |
| LP | faces_houses | kde_original_face_fit | 0.380 | 5.36 ± 1.51 | 5.58 ± 1.49 | 5.64 ± 1.65 | 4.78 ± 1.52 |
| LP | faces_houses | kde_at_nearest_noise | 0.380 | 5.36 ± 1.51 | 5.58 ± 1.49 | 5.64 ± 1.65 | 4.78 ± 1.52 |
| CNN | faces_houses | nearest_memory | 0.370 | 4.98 ± 1.58 | 5.78 ± 1.72 | 5.82 ± 1.72 | 4.42 ± 1.86 |
| CNN | faces_houses | kde_original_face_fit | 0.360 | 5.60 ± 1.59 | 6.04 ± 1.46 | 6.08 ± 1.78 | 4.46 ± 1.63 |
| CNN | faces_houses | kde_at_nearest_noise | 0.370 | 5.80 ± 1.82 | 6.22 ± 1.64 | 6.02 ± 1.58 | 4.76 ± 1.78 |
| LP | faces_houses_objects | nearest_memory | 0.370 | 4.41 ± 1.83 | 5.43 ± 2.74 | 5.16 ± 1.69 | 4.02 ± 1.67 |
| LP | faces_houses_objects | kde_original_face_fit | 0.380 | 4.80 ± 2.12 | 5.13 ± 2.17 | 5.13 ± 2.41 | 4.05 ± 2.25 |
| LP | faces_houses_objects | kde_at_nearest_noise | 0.370 | 4.83 ± 2.25 | 4.95 ± 2.25 | 5.07 ± 2.25 | 4.11 ± 2.05 |
| CNN | faces_houses_objects | nearest_memory | 0.380 | 4.47 ± 2.09 | 5.73 ± 1.78 | 5.94 ± 2.14 | 4.89 ± 1.91 |
| CNN | faces_houses_objects | kde_original_face_fit | 0.380 | 5.46 ± 1.86 | 5.73 ± 1.86 | 6.03 ± 2.41 | 4.23 ± 2.00 |
| CNN | faces_houses_objects | kde_at_nearest_noise | 0.380 | 5.46 ± 1.86 | 5.73 ± 1.86 | 6.03 ± 2.41 | 4.23 ± 2.00 |

## Paired effects and inversion ordering

Positive improvement means fewer errors with nearest memory than with the named control. These are paired differences per seed;the SD/SEM describes those differences.

| Model | Design | Condition | Control | Mean improvement | Paired SD | Paired SEM |
|---|---|---|---|---:|---:|---:|
| LP | faces_houses | UU | kde_original_face_fit | 0.020 | 2.551 | 0.361 |
| LP | faces_houses | UU | kde_at_nearest_noise | 0.020 | 2.551 | 0.361 |
| LP | faces_houses | II | kde_original_face_fit | -0.400 | 2.000 | 0.283 |
| LP | faces_houses | II | kde_at_nearest_noise | -0.400 | 2.000 | 0.283 |
| LP | faces_houses_objects | UU | kde_original_face_fit | 0.390 | 2.532 | 0.358 |
| LP | faces_houses_objects | UU | kde_at_nearest_noise | 0.420 | 2.536 | 0.359 |
| LP | faces_houses_objects | II | kde_original_face_fit | 0.030 | 2.668 | 0.377 |
| LP | faces_houses_objects | II | kde_at_nearest_noise | 0.090 | 2.579 | 0.365 |
| CNN | faces_houses | UU | kde_original_face_fit | 0.620 | 1.794 | 0.254 |
| CNN | faces_houses | UU | kde_at_nearest_noise | 0.820 | 2.017 | 0.285 |
| CNN | faces_houses | II | kde_original_face_fit | 0.040 | 1.862 | 0.263 |
| CNN | faces_houses | II | kde_at_nearest_noise | 0.340 | 1.902 | 0.269 |
| CNN | faces_houses_objects | UU | kde_original_face_fit | 0.990 | 2.704 | 0.382 |
| CNN | faces_houses_objects | UU | kde_at_nearest_noise | 0.990 | 2.704 | 0.382 |
| CNN | faces_houses_objects | II | kde_original_face_fit | -0.660 | 2.465 | 0.349 |
| CNN | faces_houses_objects | II | kde_at_nearest_noise | -0.660 | 2.465 | 0.349 |

For the three-category CNN,the mean houseUU−II error contrast is−0.42,but its MonteCarloSEM is0.45. Thus the mean ordering reverses relative to original KDE,but the run does not establish a reliable upright advantage. LP retains a mean inverted advantage. The two-category CNNalso retains a mean inverted advantage. Neither model matches the full human pattern across allfour orientations.

## Memory retrieval remains distinct from old/new performance

The new mechanism compares nearest-memory DISTANCES between both test pictures. It does not simply count a trial correct whenever the old picture retrieves its own known identity. That identity check remains an evaluation diagnostic and is not available as an oracle decision rule. Correct retrieval rates bycondition/category are retained inSUMMARY.json;allold/new scores and winning memory IDs are retained inall_trials.csv.

## Stimuli and scope

Both designs reuse the frozen64CFDface,64DalHouses residential-photo,and64Wordbank object banks. No new photographs,features,training,fixation coordinates or identity orders were introduced. The object bank contains16categories absent from the revised trainingdiet;this inherited limitation is pinned inOBJECT_CATEGORY_OVERLAP.json. Object comparisons to historical Yin airplanes are labelled references,not matched human measurements.

Version1 is preserved. Independent verification caught a few exact centroid-distance ties whose winning identity depended on floating mean/reduction order. Version2 calculates sum(ntest*study_bitcounts−nstudy*test_bitcounts)^2/(nstudy*ntest)^2,using integer-valued numerators/sums exactly representable in float64. Equal distances remain equal;first-memory ties are deterministic and old/new ties count incorrect. Category-level accuracy means were unchanged,but version2 is the independently verified final run.

Independent verification:6,000category/seed records,57,600old/new trial rows,120condition cells,500exact original KDE calibration-UU controls,and19,200primary validation decisions independently recomputed using integer NumPy distances. All200selected-fit calibration seed decisions,calibration-grid selection,seeds,noise,hashes,actual counts,12-point normalization,means/sampleSD/SEM and paired comparisons passed.

[Verification receipt](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-nearest-memory-20261113-v2/INDEPENDENT_VERIFICATION.json) · [Summary JSON](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-nearest-memory-20261113-v2/results/SUMMARY.json) · [Category/seed CSV](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-nearest-memory-20261113-v2/results/all_records.csv) · [Raw trial CSV](/Users/darthsid/Documents/ChatGPT/GURU/operations/local-yin-nearest-memory-20261113-v2/results/all_trials.csv)
