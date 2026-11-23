# New no-generic-house LP/CNN batch

User explicitly requested LP/CNN requeued on both clusters, remove generic houses, otherwise same, and show diet before training starts. All10jobs were registered held for preview and have now been released after the diet was shown. See per-site RELEASED_AFTER_DIET_PREVIEW receipts. Do not modify frozen operations,source,inputs or SPEC.

Local DIET.md/DIET.json show10stages/124epochs,2048faceIDs,64revisedobjectcategories,128houseIDs,no generic house label/images. Three house training views(original,left15,right15),left30validation,right30invertedtest. Same identities/orders/objects/optimizer/AMP/settings, fresh starts,final2240classes.

Delta newroot /projects/bjaa/sagrawal5/guru-resnet18-no-generic-20261109: prepare23684223;LP23684224/23684225;CNN23684226/23684227. SSH master /tmp/guru-delta-update.sock.
Anvil newroot /anvil/projects/x-cis260738/x-sagrawal5/guru-resnet18-no-generic-20261109: prepare22111363;LP22111364/22111365;CNN22111366/22111367. Public-key SSHaliasanvil.

Both preparations are pending; dependent training jobs await preparation. Anvil preparation22111363 estimated November15,17:45Pacific at last check; Delta preparation23684223 has no current estimate. No GPU smoke or training started. Real training CLI/curriculum checks passed for both sites without instantiating models. Full data-hash checks and optimizer/resume smokes are prerequisites in preparation jobs after release.

Anvil's full3.4GB shared updated-object archive is staged in incoming and SHA256 verified08056863d51db2a0ddf99e7fee4eee3b78327e2d8be9f77c4e5645ea8645cf44. The old data-transfer blockage is resolved for this batch. Direct authenticated transfer used a temporary local-agent forward; both forward and added agent identity were removed. No private key left the Mac.

All registered jobs have already been released. Do not re-release or submit duplicates. Check actual squeue/sacct/logs. Slurm shows H100 maintenance reservation Nov16,01:00Pacific toNov23,01:00Pacific. A100 request with current AI account was rejected; no GPU credit allocation. A6h walltime per Anvil training chunk was recommended based on prior3.1–3.4h runtimes but has NOT been applied; current24h requests remain unchanged. Host MaxRSS previously about84GiB; keep128GiB host RAM unless separately validated. If user changes diet/settings before release, create a new version and preserve/cancel these held registrations as authorized; never edit registered code in place.

Prior models/checkpoints/logs/results and SDSC Yin were preserved. No new training/evaluation automation was created.
