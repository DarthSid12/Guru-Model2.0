# Yin: completed Delta LP/CNN without generic-house training

Both scratch ResNet18 variants,training seed46,final epoch124,2240classifier
classes. Training excludes generic-house images/classes. All128 house identities
and other diet/settings match their frozen no-generic-house registration.

Same prior evaluation bank:64CFD white-male neutral faces,64screened DalHouses
residential identities (House18 replaces duplicate House71),64approved fixed
Wordbank validation images. Same pixels,32fixation coordinates and350 plans.
The object bank deliberately includes16categories removed from the revised
training diet; retain that limitation in reports. Houses remain independent
evaluation identities under the frozen prior audit; training-house inputs unchanged.

Seven40-item study designs:40faces/houses/objects individually;20+20 for every
category pair;14faces+13houses+13objects.24total same-category old/new pairs.
10study/32test fixations,sigma2,strict old>new KDE.50simulations101-150.

Fresh face calibration per model and face-containing design,50seeds,coarse.05
and fine.01 grid,targetsUU92.583333% andII63.75%; preserve range flags.
Face40 calibration transfers to designs without faces. No old fitted noise reused.
single_uu:study/test p1/p1 everywhere.
study_only_fitted:UU/UI p1/p1;II/IU p2/p1.
two_cross:UU p1/p1;UI p1/p2;II p2/p2;IU p2/p1.

One30-minute V100 cache/smoke job gates one4-core CPU production,max4hours.
Seven scientific tests,FP32 inference batch checks and56pilots gate production.
No AMP/TF32; exact frozen probabilities bind to input/checkpoint/source hashes.
Expected14400raw records,288cells,72paired contrasts; mean/sampleSD/SEM across
50MonteCarlo seeds for each fixed training seed46 checkpoint.
Published human participant SDs and airplane-reference limitations retained.
Never alter preceding jobs,source,data,checkpoints or results. No model retraining.
