# Non-animal Wordbank object sources — 2026-11-05

Current word inventory: 64 classes, 48 retained physical-object classes and 16 replacements for actual animal classes. Teddybear is retained as a stuffed toy. No orientation eligibility filter is applied.

See ../../output/wordbank-inanimate-20261105/index.html for the photo gallery and ../../output/wordbank-inanimate-20261105/README.md for the method and class list.

The raw source inventory has 44,800 image entries: 33,600 retained photo references with their frozen original splits and 11,200 newly downloaded/decoded photos. Each class has 500 train, 100 valid and 100 test entries. raw/objects/{train,valid,test}/ contains only the selected classes and exact manifest photos; raw/objects-unassigned/ is an audit/download cache and includes rejected or superseded candidates, so do not pack that cache directly.

PHOTO_MANIFEST_VERIFIED.json validates counts, label membership, prior split preservation, cross-dataset image-ID exclusion and exact source-byte/pixel-hash uniqueness. Retained photos inherit the pinned previously verified manifest; their current bytes were not all rehashed. The original manifest SHA256 also matches the live immutable Anvil source bank. Recheck/materialize retained files when packing on the training host.

The pinned WG Parquet was re-extracted exactly (4,356 word/month rows). Original source reserve preference is 1,000 human-positive images/class; noodles has a documented 919-source exception, and successfully supplies all 700 final photos. The image budget is unchanged. Word rates, age checkpoints and alphabetical tie-breaking match the original method.

Source photo previews and 128 deterministic replacement samples were visually inspected. SOURCE_REVIEW.json records scope and limitations, including people/background context and the broad Drinkware alias for glass. This is not an exhaustive per-image manual audit. No new model training, Yin run, calibration or fixed Yin pool selection was performed.

Reproduction on this Mac uses the existing DuckDB/Pillow environment:

1. extract_wordbank.py
2. index_additional_labels.py
3. build_inventory.py
4. prepare_image_manifest.py
5. fetch_photos.py --workers 32
6. verify_photo_manifest.py
7. build_report.py

Input provenance is under research/, including the exact pinned WG source. Photo provenance is in PHOTO_SPLITS.json. Previous datasets and results are preserved.
