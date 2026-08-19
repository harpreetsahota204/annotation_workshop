# Cold Pool to Hot Queue: Take-Home Pipeline

The exact code that runs at the "Cold Pool to Hot Queue" workshop: same scripts, same seed, same dataset. Run Steps 1-4 in order to build the pool, then switch to the FiftyOne App for the workshop itself. Step 5 is optional cleanup. Step 6 only runs at the end, to evaluate.

All commands below assume you're inside `workshop-notebook/`:

```bash
cd workshop-notebook
```

The 4 target classes, balanced to roughly the same size, none labeled yet in the pool you're about to build:

![What we're annotating for: the workshop's 4 target classes](assets/class_examples_grid.png)

## Fast path: skip straight to the pool

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("harpreetsahota/InsPLAD-workshop-pool")
session = fo.launch_app(dataset)
```

This is the identical 1,754-image, media-only dataset Steps 1-4 below build. Skip to [Work in the App](#work-in-the-app).

## Before you start

- Python environment with `fiftyone`, `requests` installed
- `transformers` and `torch` installed too, for fine-tuning and running the detector in the App phase (not needed for Steps 1-5)
- ~11 GB free disk space at peak, droppable to ~200 MB after Step 5
- No GPU needed for Steps 1-5; a GPU makes fine-tuning and inference in the App phase far faster, but isn't required
- [InsPLAD](https://github.com/andreluizbvs/InsPLAD) is CC BY-NC 3.0, non-commercial use only

## Step 1: Download the source dataset

```bash
python 01_download_insplad_det.py
```

Downloads InsPLAD's outer zip from Mendeley (resumable) and extracts only `InsPLAD-det.zip`.

**Expect:** `Done. 10561 images available under data/InsPLAD-det`

## Step 2: Build the workshop pool manifest

```bash
python 02_build_workshop_pool.py
```

Builds a deterministic, balanced, stratified sample (seed=51):

| Tier | Images |
|---|---|
| 0: Eval holdout (25/class) | 100 |
| 1: Balanced annotation targets (`tower id plate` 214, `polymer insulator`/`glass insulator`/`yoke` 217 each) | 865 |
| 2: Duplicate-wall flights (14 flights, every frame) | 574 |
| 3: Long-tail texture (1/remaining flight) | 215 |
| **Total** | **1,754** |

**Expect:** `Total workshop pool size: 1754`, plus a per-tier breakdown. Writes `data/workshop_pool_manifest.json`.

To change the sample, edit the constants at the top of the script (`N_DUP_WALL_FLIGHTS`, `TARGET_CLASSES`, `EVAL_HOLDOUT_PER_CLASS`, `TARGET_CLASS_QUOTA`, `TARGET_CLASS_PER_FLIGHT_CAP`) and rerun.

## Step 3: Stage the pool and hold out the real labels

```bash
python 03_stage_pool_and_heldout_labels.py
```

Copies the 1,754 manifest images into `data/workshop_pool/`, and writes their COCO boxes (converted to FiftyOne's relative `[x, y, w, h]`) to `data/heldout_ground_truth.json`. That file is not loaded into the FiftyOne dataset.

**Expect:** `Staged 1754 images to data/workshop_pool` and `Held-out ground truth for 1754 images saved to data/heldout_ground_truth.json`

## Step 4: Import into FiftyOne, media-only

```bash
python 04_import_workshop_dataset.py
```

Creates the persistent `insplad-workshop-pool` FiftyOne dataset from the staged images, no label fields attached, and tags the 100 Tier-0 samples `eval_holdout`. Launches the App.

**Expect:** `Imported <N> images into 'insplad-workshop-pool' (media-only, no labels)`, then the App opens in your browser.

To reload this dataset later:

```python
import fiftyone as fo
dataset = fo.load_dataset("insplad-workshop-pool")
session = fo.launch_app(dataset)
```

## Work in the App

1. Browse the pool. Confirm zero label fields in the sidebar.
2. Compute a CLIP embedding for every image (one call, reused below).
3. Find and tag near-duplicates using that embedding.
4. Visualize the pool with UMAP; lasso clusters; compute `uniqueness` + `representativeness`.
5. Try a plain-text search for a class. Then search with a few labeled examples instead of a word, via `sort_by_similarity()` or the [Crop Query](https://github.com/harpreetsahota204/crop_query) panel (`fiftyone plugins download https://github.com/harpreetsahota204/crop_query`).
6. Optionally cross-check with a second embedding backbone (C-RADIO).
7. Blend `uniqueness` and `representativeness` into a triage score to prioritize what search didn't find, then annotate that hot queue into a `human_annotated` field (not `ground_truth`).
8. Fine-tune a detector on `human_annotated` from the App's action menu, via the [hf_fine_tuner_plugin](https://github.com/harpreetsahota204/hf_fine_tuner_plugin) (`fiftyone plugins download https://github.com/harpreetsahota204/hf_fine_tuner_plugin`; saves a HuggingFace Transformers checkpoint to `finetuned_detection_model/` by default).
9. Run that checkpoint over everything not yet annotated and scroll the predictions class by class. There's no `ground_truth` out here, so it's a vibe check, not a metric. Where a class looks weak, annotate a few more examples into `human_annotated` and re-run the fine-tune operator; repeat until every class looks trustworthy.
10. Once satisfied, run `08_reveal_eval_holdout_ground_truth.py` (Step 6 below) and evaluate.

## Step 5: Clean up (optional)

```bash
python 05_cleanup_unused_images_OPTIONAL.py
```

Interactively offers to delete the outer zip and/or the full `InsPLAD-det/` extract. Refuses to run if `data/workshop_pool/` is empty.

## Step 6: Reveal the eval holdout and close the loop

`06_seeded_search_all_classes.py` and `07_tag_potential_matches.py` are helpers you run from inside the App phase above (step 5 of [Work in the App](#work-in-the-app)), not part of this numbered pipeline.

```bash
python 08_reveal_eval_holdout_ground_truth.py
```

Loads the real boxes from `data/heldout_ground_truth.json` into `ground_truth`, scoped to the 100 `eval_holdout` samples and the 4 target classes only. Saves an `eval_holdout_annotated` view.

**Expect:** `ground_truth populated on 100/100 eval_holdout samples` and `Saved view 'eval_holdout_annotated' created.`

Then run inference on that view and evaluate:

```python
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained("finetuned_detection_model")
eval_view = dataset.load_saved_view("eval_holdout_annotated")
eval_view.apply_model(model, label_field="predictions", confidence_thresh=0.5)
results = eval_view.evaluate_detections("predictions", gt_field="ground_truth")
results.print_report()
```

`confidence_thresh` matters here: RF-DETR always emits a fixed number of query predictions per image (300), most of them near-zero-confidence noise. Skip the threshold and `predictions` drowns in that noise instead of holding the handful of real detections.
