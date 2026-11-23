# Delta no-generic-house training completed

Both scratch ResNet18 models,training seed46,completed all124 epochs with2240classes and zero generic-house images.

| Model | Final train accuracy | Upright validation | Inverted classification test | Training time |
|---|---:|---:|---:|---:|
| LP | 76.48% | 66.93% | 19.74% | 8.17h |
| CNN | 92.66% | 82.90% | 7.80% | 7.76h |

Best validation checkpoint is epoch124 for both models. These are classifier accuracies; Yin has not yet been run on these checkpoints. Two registered continuation jobs remain queued and are programmed to exit when they see the verified COMPLETE markers.
