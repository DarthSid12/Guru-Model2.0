# VGG16-BN training curves

Four models · 124 epochs · overall and individual training categories. [Behavioral report](vgg16bn_results_report.md).

## What each curve means

- **Overall training (solid blue):** the original per-epoch accuracy on augmented fixation crops while parameters were being updated. It was recorded and has not been reconstructed.
- **Validation (solid green):** recorded upright image accuracy, combining logits across 16 fixation crops.
- **Test (solid orange):** recorded image accuracy on the test split after **180° inversion**, again combining 16 fixations. The inversion is verified in the transform and training code.
- **Category training (dashed blue dots):** a new evaluation of the saved stage-end model on a deterministic sample of up to 16 training photos per active class. Generic houses use all stage-eligible photos. This uses the first 16 saved fixation crops, upright presentation, evaluation mode, and no augmentation. It is **not the missing historical per-category training statistic**.
- **ZuBuD inverted validation (dashed orange squares):** a new inverted evaluation of its one validation view per active building. ZuBuD has **no test split**, so this is not labelled as a recorded test curve.

The saved checkpoints are at epochs **2, 4, 8, 12, 20, 28, 44, 60, 92, and 124**. Dotted vertical lines mark curriculum changes. Dashed segments connect only the measured checkpoint endpoints; intermediate values are not observed. A category appears when its classes enter the curriculum.

Training and evaluation curves use different units and sampling: the overall training curve scores augmented individual crops during learning, whereas validation/test score images after combining crops. Reconstructed category training curves also use a sample of the training images. Their gaps should not be treated as a like-for-like generalization estimate.

These are accuracies on the supervised training task. The training categories differ from the held-out behavioral stores: ZuBuD building identities are `houses_zubud137_41`, generic houses are one classifier class, and the Yin house set is `houses_yin64`. The mixed model’s RFW training stores are also distinct from Yin’s RFW WM64 evaluation set. High supervised house accuracy does not imply high held-out Yin memory accuracy.

## Test inversion and split audit

`train.py` sends the test loader through `OnTheFlyTransform("test")`; that transform selects `Rotate(invert=True)`, which is exactly 180°. It reads a separate test store: the curve is not generally validation accuracy with its orientation changed. VGG2k uses the next 12 photos per identity for validation and the last 12 for test; old VGG/CelebA and objects also have different stored pictures. RFW-W, RFW-O and generic houses have matching validation/test image arrays, but **different fixation coordinates** (all arrays checked in the [split audit](vgg16bn_report_data/split_audit.json)). ZuBuD buildings have only train and validation stores.

Overall validation includes the ZuBuD category, while overall test omits it. Also, the trainer records overall evaluation as the mean of batch accuracies, while per-category evaluation uses image-level correct/total. Therefore neither the overall valid–test gap nor a simple average of category curves is an exact isolated inversion effect. The Yin behavioral tasks above compare orientations on the same images.

The `summary.json` per-category values were computed after reloading the **best** checkpoint. These plots use the per-epoch **history CSV** to avoid substituting best-checkpoint values for the final epoch.

Code evidence: [training evaluation and history](../training/train.py#L1064), [evaluation aggregation](../training/train.py#L276), [test transform](../training/salience_trans.py#L270), [180° rotation](../training/trans.py#L47), [missing-split handling](../training/datasets.py#L258).

## Overall curves

![All four models: recorded overall accuracy](figures/vgg16bn_training/all_models_overall.png)

| Model | Final recorded train | Final upright validation | Final inverted test |
| --- | ---: | ---: | ---: |
| VGG2k / no AA | 85.39% | 83.90% | 78.42% |
| VGG2k / AA4 | 86.56% | 84.30% | 78.34% |
| VGG2k / bin5 | 86.25% | 83.55% | 77.50% |
| Mixed faces / AA4 | 97.19% | 79.32% | 73.37% |

## VGG2k / no AA

![VGG2k / no AA category dashboard](figures/vgg16bn_training/noaa_dashboard.png)

[Dashboard PDF](figures/vgg16bn_training/noaa_dashboard.pdf) · [Overall PNG](figures/vgg16bn_training/noaa_overall.png) · [Original history](../runs/faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_r21vgg2k_vgg16_bn_s42/training_history_20260926_230735.csv)

<details><summary>Individual category images and checkpoint sample sizes</summary>

| Category | Individual PNG | PDF | Photos in final reconstructed train evaluation |
| --- | --- | --- | ---: |
| VGGFace2 · 2,048 identities | [Image](figures/vgg16bn_training/noaa_faces_vgg2k.png) | [PDF](figures/vgg16bn_training/noaa_faces_vgg2k.pdf) | 32,768 |
| Objects | [Image](figures/vgg16bn_training/noaa_objects.png) | [PDF](figures/vgg16bn_training/noaa_objects.pdf) | 1,024 |
| ZuBuD · building identities | [Image](figures/vgg16bn_training/noaa_houses_zubud137_41.png) | [PDF](figures/vgg16bn_training/noaa_houses_zubud137_41.pdf) | 548 |
| Generic houses | [Image](figures/vgg16bn_training/noaa_houses.png) | [PDF](figures/vgg16bn_training/noaa_houses.pdf) | 128 |

</details>

## VGG2k / AA4

![VGG2k / AA4 category dashboard](figures/vgg16bn_training/aa4_dashboard.png)

[Dashboard PDF](figures/vgg16bn_training/aa4_dashboard.pdf) · [Overall PNG](figures/vgg16bn_training/aa4_overall.png) · [Original history](../runs/faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_aa_r21vgg2k_vgg16_bn_aa_s42/training_history_20260925_110304.csv)

<details><summary>Individual category images and checkpoint sample sizes</summary>

| Category | Individual PNG | PDF | Photos in final reconstructed train evaluation |
| --- | --- | --- | ---: |
| VGGFace2 · 2,048 identities | [Image](figures/vgg16bn_training/aa4_faces_vgg2k.png) | [PDF](figures/vgg16bn_training/aa4_faces_vgg2k.pdf) | 32,768 |
| Objects | [Image](figures/vgg16bn_training/aa4_objects.png) | [PDF](figures/vgg16bn_training/aa4_objects.pdf) | 1,024 |
| ZuBuD · building identities | [Image](figures/vgg16bn_training/aa4_houses_zubud137_41.png) | [PDF](figures/vgg16bn_training/aa4_houses_zubud137_41.pdf) | 548 |
| Generic houses | [Image](figures/vgg16bn_training/aa4_houses.png) | [PDF](figures/vgg16bn_training/aa4_houses.pdf) | 128 |

</details>

## VGG2k / bin5

![VGG2k / bin5 category dashboard](figures/vgg16bn_training/bin5_dashboard.png)

[Dashboard PDF](figures/vgg16bn_training/bin5_dashboard.pdf) · [Overall PNG](figures/vgg16bn_training/bin5_overall.png) · [Original history](../runs/faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_aa5_r21vgg2k_vgg16_bn_aa5_s42/training_history_20260927_145006.csv)

<details><summary>Individual category images and checkpoint sample sizes</summary>

| Category | Individual PNG | PDF | Photos in final reconstructed train evaluation |
| --- | --- | --- | ---: |
| VGGFace2 · 2,048 identities | [Image](figures/vgg16bn_training/bin5_faces_vgg2k.png) | [PDF](figures/vgg16bn_training/bin5_faces_vgg2k.pdf) | 32,768 |
| Objects | [Image](figures/vgg16bn_training/bin5_objects.png) | [PDF](figures/vgg16bn_training/bin5_objects.pdf) | 1,024 |
| ZuBuD · building identities | [Image](figures/vgg16bn_training/bin5_houses_zubud137_41.png) | [PDF](figures/vgg16bn_training/bin5_houses_zubud137_41.pdf) | 548 |
| Generic houses | [Image](figures/vgg16bn_training/bin5_houses.png) | [PDF](figures/vgg16bn_training/bin5_houses.pdf) | 128 |

</details>

## Mixed faces / AA4

![Mixed faces / AA4 category dashboard](figures/vgg16bn_training/mixed_dashboard.png)

[Dashboard PDF](figures/vgg16bn_training/mixed_dashboard.pdf) · [Overall PNG](figures/vgg16bn_training/mixed_overall.png) · [Original history](../runs/faces_vgg_faces_faces_rfwW_faces_rfwO_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_aa_r21dev_vgg16_bn_aa_s42/training_history_20260925_064347.csv)

<details><summary>Individual category images and checkpoint sample sizes</summary>

| Category | Individual PNG | PDF | Photos in final reconstructed train evaluation |
| --- | --- | --- | ---: |
| VGGFace2 · old face store | [Image](figures/vgg16bn_training/mixed_faces_vgg.png) | [PDF](figures/vgg16bn_training/mixed_faces_vgg.pdf) | 7,680 |
| CelebA | [Image](figures/vgg16bn_training/mixed_faces.png) | [PDF](figures/vgg16bn_training/mixed_faces.pdf) | 2,048 |
| RFW · white | [Image](figures/vgg16bn_training/mixed_faces_rfwW.png) | [PDF](figures/vgg16bn_training/mixed_faces_rfwW.pdf) | 3,784 |
| RFW · other | [Image](figures/vgg16bn_training/mixed_faces_rfwO.png) | [PDF](figures/vgg16bn_training/mixed_faces_rfwO.pdf) | 1,458 |
| Objects | [Image](figures/vgg16bn_training/mixed_objects.png) | [PDF](figures/vgg16bn_training/mixed_objects.pdf) | 1,024 |
| ZuBuD · building identities | [Image](figures/vgg16bn_training/mixed_houses_zubud137_41.png) | [PDF](figures/vgg16bn_training/mixed_houses_zubud137_41.pdf) | 548 |
| Generic houses | [Image](figures/vgg16bn_training/mixed_houses.png) | [PDF](figures/vgg16bn_training/mixed_houses.pdf) | 128 |
| All face sources pooled | [Image](figures/vgg16bn_training/mixed_all_faces.png) | [PDF](figures/vgg16bn_training/mixed_all_faces.pdf) | 14,970 |

</details>

## Reproducibility

[All curve coordinates](vgg16bn_report_data/training_curves.csv) · [History hashes](vgg16bn_report_data/training_source_manifest.json) · [Stage evaluation records and photo samples](../runs/vgg16bn_training_curve_audit_20260928/) · [Evaluator](../training/evaluate_vgg_training_checkpoints.py) · [Plot generator](../reporting/plot_vgg16bn_training_curves.py)

The deterministic sample seed is 20260928. Training samples respect each saved active-class list and the stage-specific generic-house image cap. Stage weights are loaded strictly; no weights are updated. Photo lists, correct/total counts, checkpoint hashes, and evaluation-code hashes are saved. The same VGG2k class/photo samples are used across its three architecture variants. All stages completed for all four models.
