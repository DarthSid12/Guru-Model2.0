# Yin with nearest-memory prototypes

User requested replacing the KDE mechanism with memory retrieval, then adding
faces/houses/objects with calibration on faces. Run both previously authorized
20faces+20houses and new three-category versions, using unchanged completed
Delta LP/CNN ResNet18 epoch124 trainingseed46 and frozen input caches.

New readout: average the10Bernoulli+XOR study fixation codes into one256D memory
prototype per studied image. Average32test codes into one query vector. Retrieve
the memory with smallest squared Euclidean distance. In each same-category
old/new pair, choose the picture with the smaller nearest-memory distance.
Each picture searches ALL40memories. Both pictures use the identical rule;
true stimulus identity/old-new status is used only for evaluation, not scoring.
Strictly smaller distance defines correct;ties count incorrect as previously.

Fit one shared added XOR rate per model/design to human faceUU92.583333%,using
50calibration seeds101-150. Same noise at study/test for every orientation and
category. Coarse0:.05:.5 then fine.005 in ±.05around best coarse point;minimize
absolute error to target,breaking ties by lower rate. Save allseed counts and
range/boundary flags. Houses/objects and II/UI/IU are not calibration targets.
Evaluate50fresh simulation seeds151-200. Same64-photo banks,new random codes and
study/test selections;not independent image or training-seed replication.

Controls: original legacy KDE sigma2 at its unchanged parent face-calibrated
noise,plus original KDE at the NEW nearest-memory fitted noise to isolate the
readout effect at matched noise. Preserve old fit/source/raw outputs and require
exact replay of500parent calibration-UUcategory/seed records before evaluation.
No calibration on houses or objects,no retraining,new images or GPU extraction.

Two-category design:20faces+20houses study,12old/new pairs/category from32-image
category pools. Three-category:14faces+13houses+13objects study,8pairs/category
from22/21/21image pools. Report both actual errors and12-point normalized errors
for comparison;triple raw counts are out of8,not12. AllconditionsUU/UI/IU/II.

Keep frozen object bank to isolate mechanism. It contains16categories absent
from the revised64-category training diet;the exact overlap ledger is retained.
Historical airplane references are not matched human data for these object
pictures and are not an object calibration target.

Record correct old-probe memory identity,wrong same-category/cross-category
matches and raw decision scores separately from old/new accuracy. Same RNG/
plans/encodings are paired across readouts. Local CPUonly2single-threaded model
workers,30minute bound. Test exact averaging,distance and search behavior before
calibration;independently verify seeds,counts,fit selection,hashes,SD/SEM and
counterpart baseline replays. Preserve all training and registered evaluations.

Version2 numerical recovery:the independent NumPy check found equal integer
centroid distances differing by about1e-15when means were divided before
squaring. Preserve version1 unchanged. Compute squared distance as
sum(ntest*study_bitcounts-nstudy*test_bitcounts)^2/(nstudy*ntest)^2.
Allnumerators and squared sums are exact integers in float64 for these binary
codes. Reuse the declared first-memory tie rule and count old/new ties as
incorrect. Same scientific mechanism,grid,targets,seeds,inputs and controls.
Require independent integer NumPy verification for every validation decision.
