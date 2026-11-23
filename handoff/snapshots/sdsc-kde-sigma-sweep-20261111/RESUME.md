# Resume sigma diagnostic

Read SPEC.json,JOBS.json,LATEST_STATUS.json and README.md. Preserve frozen source,
parents and all training jobs. Remote root is in status. Use existing SDSC SSH
master in status; never reuse old MFA codes. Check registered CPU job and log.
Do not submit a duplicate. Require SMOKE_PASSED,VALIDATED and 1600 exact baseline
replays. Export all results, specification and log, then independently verify
19200 records/384 cells/50 seeds per cell, hashes, count masks, distance histogram
moments/scale/grid, sample SD/SEM and paired contrasts. Distinguish legacy floor
from stable unfloored scoring and fixed noise from zero-XOR diagnostic. No
retraining, noise refit or selection claimed as held-out human prediction.
