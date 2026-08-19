"""Step 4: Import the staged workshop pool into FiftyOne as a media-only
dataset: no label FIELDS attached (no `ground_truth`, no `human_annotated`).
This is the starting point for every act, a real pool of unlabeled images.

One tag gets applied at import time, though: `eval_holdout`, on the
stratified ~25-per-class slice carved out in step 2's manifest (tier
"eval_holdout:<class>"). Tagging is not the same as labeling -- it makes
these samples excludable from search/prioritization/annotation from the
very first act, without revealing what's actually in them. Their real
`ground_truth` still comes only from heldout_ground_truth.json, and still
only gets loaded in the "evaluate" act, at the end.
"""
import json
from pathlib import Path

import fiftyone as fo

WORK_DIR = Path(__file__).parent / "data"
POOL_DIR = WORK_DIR / "workshop_pool"
MANIFEST_PATH = WORK_DIR / "workshop_pool_manifest.json"
DATASET_NAME = "insplad-workshop-pool"

if DATASET_NAME in fo.list_datasets():
    raise SystemExit(
        f"Dataset '{DATASET_NAME}' already exists. Delete it first with "
        f"fo.delete_dataset('{DATASET_NAME}') if you want to rebuild it."
    )

dataset = fo.Dataset(DATASET_NAME, persistent=True)
dataset.add_images_dir(POOL_DIR)
dataset.info = {
    "source": "https://github.com/andreluizbvs/InsPLAD",
    "note": (
        "Media-only workshop pool sampled from InsPLAD-det. No label "
        "fields attached by design; see 02_build_workshop_pool.py for the "
        "sampling manifest and heldout_ground_truth.json for the real "
        "boxes. `eval_holdout`-tagged samples are excluded from every "
        "interactive act (search, prioritization, annotation, correction); "
        "their `ground_truth` is only loaded in the final 'evaluate' act."
    ),
}
dataset.save()

manifest = json.loads(MANIFEST_PATH.read_text())
fn_to_tier = {s["file_name"]: s["tier"] for s in manifest["samples"]}

eval_holdout_ids = []
for sample in dataset.select_fields([]):
    fn = sample.filepath.split("/")[-1]
    tier = fn_to_tier.get(fn, "")
    if tier.startswith("eval_holdout:"):
        eval_holdout_ids.append(sample.id)

if eval_holdout_ids:
    dataset.select(eval_holdout_ids).tag_samples("eval_holdout")

print(f"Imported {len(dataset)} images into '{DATASET_NAME}' (media-only, no label fields)")
print(f"Tagged {len(eval_holdout_ids)} samples 'eval_holdout' (excluded from every interactive act)")
print("Launch the App to start Act 1: fo.launch_app(dataset)")

session = fo.launch_app(dataset)
