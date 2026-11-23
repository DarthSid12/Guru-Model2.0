# Object source continuation — 2026-11-06

The active bank has 64 object classes and 44,800 images, exactly 500 train + 100 validation + 100 test per class. This pass downloaded 4,045 new source photos, superseded 2,445 parent photos and added 1,600 unused reserves (100 for each of the 16 replacement classes). The other 42,355 parent photos retain their class and split assignments.

Class names, Wordbank comprehension numerators/denominators, introduction ages, stage counts and ordering are unchanged. Physical object orientation is not an eligibility criterion. Glass source metadata is corrected from generic Drinkware to human-positive Wine glass with a matching non-depiction object box. This represents glass with wine-glass instances, so it is narrower than the earlier broad vessel source.

Human-positive nonhuman-animal labels across 110 hierarchy entries were screened. This flagged 1,760 parent photos, including false matches to stuffed toys. Six unmistakable manufactured teddy-bear toys were visually confirmed and retained; the other flags were excluded from the updated bank. All 700 parent glass images were superseded because the broad source also contained mugs, scenes and an animal photo. There are zero unresolved animal-annotation flags in the active bank/reserves; this annotation-based screen does not certify every pixel is animal-free.

All 4,045 new PNG files and normalized pixel hashes were checked. Active and reserve source IDs, source-byte hashes and pixel hashes are disjoint. Human-positive source label membership, exact split counts and unchanged inherited assignments passed independent checks. Inherited photos reference the pinned, previously verified parent bank; their full bytes were not all rehashed during this pass.

Source photos can include people/background context. Visual checks cover parent-source review sheets, corrected-glass examples, six toy adjudications and the 64 gallery examples; there was no exhaustive manual review of all 44,800 images. No new training, calibration, fixed Yin stimulus selection or Yin run was performed. Existing model runs and historical results are unchanged.

Files: [gallery](index.html), [contact sheet](object-inventory.jpg), [verification receipt](VERIFIED.json), [source plan](SOURCE_PLAN.json), [toy decisions](PHYSICAL_TOY_OVERRIDES.json).

Authoritative data manifests and code are in ../../operations/wordbank-object-curation-20261106/. PHOTO_SPLITS.json is the active bank; RESERVE_PHOTOS.json stays outside training/evaluation; SUPERSEDED_PARENT_PHOTOS.json preserves the audit trail. raw/objects is the active split tree; raw/objects-unassigned is a download/audit cache and must not be packed directly.
