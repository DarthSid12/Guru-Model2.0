# New-model KDE bandwidth with two face anchors

User requested the newer model and selected new sigma with two-noise anchoring,
following the traced old-study-only screenshot. Keep original KDE and completed
Delta no-generic ResNet18 LP/CNN epoch124 trainingseed46. Fixed sigma from the
retrospective bandwidth ranking:LP4.509564323062932,CNN6.753987849071428. Values
remain fixed while fitting noise;alpha is provenance,not recalculated per rate.

Fit study-only p1/p2 on faceUU andII in20faces+20houses study. UU/UI(p1,p1),
IU/II(p2,p1),ordered study/test. p1shared alltest conditions. Fitp1toUU92.5833%,
thenp2toII63.75% withtestp1heldfixed. Use50calibration seeds101-150. Coarse
0:.05:.5,fine.005within±.05of bestcoarse,then.001within±.005of best sampled
point. Choose minimum absolute targeterror;ties choose smaller noise. Do not
imposep2>=p1. Save allcounts,range andboundary flags. No fitting onUI,IU,houses,
objects orvalidation. Gaussian KDE retains originalfloat32+1e-12floor andsigma
normalization behavior;no prototype mechanism or retraining.

Evaluate50new seeds201-250 beyond sigma-ranking seeds101-200. Same frozen
64photo banks,new study/probe selections and random codes. Not independent
stimuli or training replications.20faces+20houses gives12pairs/category.
Also20faces+20objects,12pairs/category,with F/Hfacefit transferred unchanged.
Objects are not fitted. Retain original object bank and overlap caveat16of64
categories absent from revisedtrainingdiet. Historical airplanes are reference
only,not matched human object data.

Four readouts(alloriginalKDE):new_sigma_two_noise; sigma2_two_noise_originalfit
(import original F/Hstudy-only fit);sigma2_two_noise_matchedfit(newp1/p2);
new_sigma_single_p1(usep1forbothstudy/test). Matched fit isolates bandwidth;
singlep1control isolates effect ofinverted-study noise. Same RNG/codes/plans
paired across methods. Each probe searches all40memory images;10study and32test
fixations;strict old>new. Require exact400parent UU/IIcategory-control replays.
Expected6400category rows,76800trial rows,128cells of50seeds. Local CPUonly,
2single-threaded workers,30minutebound. Freeze source,cachehashes,checkpoints,
plans,sigma-selection provenance andoriginal calibration references. Preserve
allpriorruns andactive defaults. Independently verify calibration choices,
counts,seeds,SD/SEM,noise assignments,paircontrasts and rawKDEdecisions.
