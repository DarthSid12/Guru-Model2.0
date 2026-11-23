# Matched anchored tables for fixed versus adjusted KDE sigma

User requested two experiments,each with faceUU/II2noise anchoring and
faces,houses,objects tables. Same newer no-generic LP/CNN ResNet18 checkpoints,
epoch124/trainingseed46,full256D codes andoriginal KDE. Experiment1sigma2forboth;
Experiment2sigmaLP4.509564323062932/CNN6.753987849071428.

Fit p1tofaceUU92.5833%,thenp2tofaceII63.75%withtestp1fixed. UU/UI(p1,p1),
IU/II(p2,p1),study/test. Use same50fitting/report seeds101-150 asolder anchored
screenshot;label facesUU/IIasfitted,notindependentvalidation. Experiment1refits
noise with EXACT same coarse.05/fine.005/ultrafine.001algorithm asExperiment2.
Experiment2reuse exact frozenfits andfull fitting-set records from priorrun.
No parameters adjusted after seeing the tables. Allcoarse/fine curves saved.

Faces/houses:20faces+20houses studied,12testpairs/category. Objects:20faces+
20objects,12pairs/category,noise transferred unchangedfromface/housefit.
These are separate40itemstudyexperiments,not3categoriesstudied simultaneously.
10study/32testfixations perimage. Samebanks,plans,RNG,coordinates,models.
Fixed absolute sigma;no recalculationofalpha during noise fitting. Standard
KDEfloat32 exp/mean/log(+1e-12),doublefixationsum,max40memoryimages;no prototypes.

Fitting-seedtables are primary to matchrequested anchored convention. Prior
fresh201-250results remain separatelyavailable andmustnotbemixedorrelabeled.
Objectbankincludes16categoriesoutsidetherevisedtrainingdiet;historical airplanes
referenceonly. Prioradjustedsigma selectionusedfaceandhousehumanmeans;therefore
housefitisnotfullyheldoutfromallparameterselection,althoughp1/p2fitusesonlyfaces.

Two single-threaded localCPUworkers,30minutebound. Expected3200category records,
64conditioncells of50seeds. Savebotherrorsandaccuracytables,pairedcomposite
contrasts,andindependentverification. No training oractivebaselinechanges.
