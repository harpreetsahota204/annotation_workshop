---
annotations_creators:
- no-annotation
language: en
license: cc-by-nc-3.0
size_categories:
- 1K<n<10K
task_categories:
- object-detection
task_ids: []
pretty_name: InsPLAD Workshop Pool
tags:
- active-learning
- data-curation
- fiftyone
- image
- object-detection
- power-line-inspection
- uav
- unlabeled
description: A 1,754-image, media-only sample of InsPLAD-det (UAV power line inspection
  imagery), built for a hands-on FiftyOne workshop on a complete annotation workflow.
  No label fields are attached by design; this is a genuine cold-start pool for
  practicing compression, embedding, seeded similarity search, and annotation
  prioritization before ever touching a model. A deterministic, balanced stratified
  sample (seed=51) drawn from the full 10,561-image InsPLAD-det: roughly 217 images
  each of `tower id plate`, `polymer insulator`, `glass insulator`, and `yoke`
  (deliberately capped at the same quota so no class dominates the annotation
  budget), 574 images from 14 intact drone-flight sequences (a real near-duplicate
  wall), and 215 long-tail images for texture.
dataset_summary: '




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 1754 samples.


  ## Installation


  If you haven''t already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include ''max_samples'', etc

  dataset = load_from_hub("harpreetsahota/InsPLAD-workshop-pool")


  # Launch the App

  session = fo.launch_app(dataset)

  ```

  '
---

# Dataset Card for InsPLAD Workshop Pool

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 1754 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("harpreetsahota/InsPLAD-workshop-pool")

# Launch the App
session = fo.launch_app(dataset)
```

## Dataset Details

### Dataset Description

InsPLAD Workshop Pool is a 1,754-image, **media-only** sample of
[InsPLAD-det](https://huggingface.co/datasets/harpreetsahota/InsPLAD), built to
teach a complete annotation workflow in FiftyOne: compress a raw image pool,
embed it, search for examples of each target class from a handful of seed
examples, prioritize the rest for annotation, fine-tune a detector, and correct
its mistakes. This dataset ships with **zero label fields by design**. The point
of the exercise is deciding which images deserve human attention before any
labels exist. The images sampled into this pool were deliberately stratified
(not randomly subsampled) so that every step of that workflow has something real
to work with: genuine near-duplicate sequences, and four target classes
deliberately balanced to roughly the same size instead of reproducing the class
imbalance already present in the raw source data. See
[Curation Rationale](#curation-rationale) below for exactly how, and the original
scripts under `workshop-notebook/` in the
[source repository](https://github.com/andreluizbvs/InsPLAD) companion materials
for the full, runnable pipeline.

- **Curated by:** Harpreet Sahota (Voxel51), sampled from InsPLAD (see Dataset
  Sources for the original dataset's curators)
- **Funded by:** N/A (derivative sample; see original InsPLAD for its funding)
- **Shared by:** Harpreet Sahota, via Hugging Face Hub
- **Language(s):** en (asset class names in the source data; not an NLP dataset)
- **License:** cc-by-nc-3.0 (inherited from InsPLAD; non-commercial use only)

### Dataset Sources

- **Repository:** https://github.com/andreluizbvs/InsPLAD
- **Paper:** InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in
  UAV Images, International Journal of Remote Sensing (2023),
  https://arxiv.org/abs/2311.01619
- **Original data download:** https://data.mendeley.com/datasets/5n3fjgvfyz/1
- **Full FiftyOne build (all 3 InsPLAD sub-datasets, with labels):**
  https://huggingface.co/datasets/harpreetsahota/InsPLAD

## Uses

### Direct Use

Practicing (or teaching) a complete annotation loop end to end: near-duplicate
compression, embedding-based visual exploration, seeded similarity search for a
target class, uniqueness/representativeness-based annotation prioritization,
detector fine-tuning on the curated subset, and model-assisted correction. Also
useful as a small, realistic stand-in for InsPLAD-det when testing FiftyOne
workflows without downloading the full 10,561-image, 4.2 GB source dataset.

### Out-of-Scope Use

Not intended as a benchmark dataset for reporting detection accuracy. It is a
deliberately non-random, stratified sample built for a specific teaching workflow,
not an i.i.d. sample of InsPLAD-det. Any commercial use is out of scope; the
source license (CC BY-NC 3.0) is non-commercial only. Not suitable for identifying
individuals; it contains no personal or sensitive data by design (aerial images of
power line hardware only).

## Dataset Structure

This is a flat image dataset (`media_type = "image"`), not grouped or video, with
**1,754 samples** and no splits or saved views. Every sample carries only
FiftyOne's default fields; there is no `ground_truth`, no per-sample split tag,
and no per-sample sampling-tier label. This is intentional: the dataset is meant
to be loaded and explored exactly as if no prior work had been done on it.

### Fields

| Field | FiftyOne type | Description |
|-------|---------------|-------------|
| `filepath` | `StringField` | Path to the image file |
| `tags` | list of `str` | Empty for every sample; no split or tier tags are shipped |
| `metadata` | `ImageMetadata` | Not populated (`None`) until `dataset.compute_metadata()` is run |

### `dataset.info`

```python
{
    "source": "https://github.com/andreluizbvs/InsPLAD",
    "note": (
        "Media-only workshop pool sampled from InsPLAD-det. No labels "
        "attached by design; see 02_build_workshop_pool.py for the "
        "sampling manifest and heldout_ground_truth.json for the real "
        "boxes, held out until the 'close the loop' act."
    ),
}
```

### Parsing decisions

- **Media-only import, on purpose.** The staging step that builds this pool does
  compute real bounding boxes for every sampled image (converted from InsPLAD's
  COCO format to FiftyOne's relative `[x, y, w, h]`), but those boxes are written
  to a separate `heldout_ground_truth.json` file and never attached to the
  FiftyOne dataset. This dataset is the "before" half of a before/after teaching
  exercise.
- **No tier or split metadata shipped.** Which sampling tier (eval holdout,
  balanced target, duplicate-wall, long-tail) or original InsPLAD split
  (`train`/`val`) each image came from is recorded in
  `workshop_pool_manifest.json` at build time, not carried into this Hub
  dataset's fields, including the `eval_holdout` tag itself. That keeps the
  pool looking like a genuine unlabeled pool, not a labeled one with fields
  hidden. If you build the pool locally via the companion
  `01_download_insplad_det.py` through `04_import_workshop_dataset.py`
  scripts instead of loading from the Hub, step 4 re-applies the
  `eval_holdout` tag to the same 100 stratified samples on import.
- **Whole scenes only, no crops.** Unlike the full InsPLAD-fault sub-datasets
  (cropped, near-square asset images), every image in this pool is a full UAV
  scene from InsPLAD-det. A bounding-box task only makes sense on full scenes,
  and every act of the target workflow (including fine-tuning and correcting a
  detector) depends on that.

## Dataset Creation

### Curation Rationale

A naive random subsample of InsPLAD-det's 10,561 images breaks the workflow this
dataset is meant to teach: dedupe before subsampling and there's no duplicate wall
left for the "compress" step to find; subsample without correcting for class
frequency and the resulting pool just reproduces whatever imbalance already
exists in the raw data (`tower id plate` has only 242 images total; some other
classes have thousands). Instead, this pool uses a **deterministic, balanced
stratified sample** (seed=`51`, same result every run) that guarantees every step
of the workflow has something real to work with, at a fraction of the source
data's size, with all 4 target classes capped at the same rough quota.

Two scripts build this pool from the original InsPLAD-det source; both are
included verbatim below for full reproducibility.

#### Step 1: Download InsPLAD-det from source

InsPLAD ships as a single Mendeley Data record containing three inner zips
(`InsPLAD-det.zip`, `supervised_fault_classification.zip`,
`unsupervised_anomaly_detection.zip`). This workshop only uses whole UAV scene
images, so only `InsPLAD-det.zip` is extracted; the other two stay zipped and
untouched.

```python
"""Step 1: Download InsPLAD from source and extract only the detection
(InsPLAD-det) sub-dataset: full UAV scene images, no cropped fault/anomaly
images. This workshop uses whole images only.

Source: Mendeley Data, https://data.mendeley.com/datasets/5n3fjgvfyz/1
The Mendeley record ships one outer zip containing three inner zips
(InsPLAD-det.zip, supervised_fault_classification.zip,
unsupervised_anomaly_detection.zip). We download the outer zip (it's a
single file on Mendeley, can't be split at the API level), but only extract
InsPLAD-det.zip from it; the other two are left zipped and untouched.
"""
import zipfile
from pathlib import Path

import requests

MENDELEY_DATASET_ID = "5n3fjgvfyz"
WORK_DIR = Path(__file__).parent / "data"
OUTER_ZIP = WORK_DIR / "InsPLAD_Dataset.zip"
DET_DIR = WORK_DIR / "InsPLAD-det"


def get_download_url():
    """Query the Mendeley public API for the current file download URL
    (avoids hardcoding a URL that may rotate)."""
    resp = requests.get(
        f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
        "/files?folder_id=root&version=1"
    )
    resp.raise_for_status()
    files = resp.json()
    assert len(files) == 1, f"expected 1 file, got {len(files)}"
    return files[0]["content_details"]["download_url"], files[0]["size"]


def extract_det_only(outer_zip, det_dir):
    with zipfile.ZipFile(outer_zip) as outer:
        names = outer.namelist()
        det_zip_name = next(n for n in names if n.endswith("InsPLAD-det.zip"))
        outer.extract(det_zip_name, WORK_DIR)

    inner_zip_path = WORK_DIR / det_zip_name
    with zipfile.ZipFile(inner_zip_path) as inner:
        inner.extractall(det_dir)
    inner_zip_path.unlink()  # don't need the intermediate inner zip anymore
```

Result: `data/InsPLAD-det/{train,val}/*.jpg` plus COCO annotation JSONs: 10,561
unique images (46 duplicate COCO `image_id` entries for the same file are a known
quirk of the source data, resolved during staging).

#### Step 2: Build the balanced stratified sample

```python
"""Step 2: Build the reproducible, balanced stratified workshop pool
manifest from InsPLAD-det's raw images. Whole scene images only, no
labels attached to the resulting pool: ground truth for the sampled
images is saved separately in step 3, held out for the "close the loop"
act.

Tiers:
  0. Eval holdout: a stratified ~25-per-class slice across all 4 target
     classes, carved out FIRST, before any other tier is built. These
     images are never available to prioritization or annotation in any
     downstream act -- they exist purely so there's a clean, never-touched
     benchmark to evaluate the fine-tuned detector against later. Tagged
     `eval_holdout` at import time (step 4).
  1. Balanced annotation targets: capped per-flight, per-class samples of
     all 4 target classes (`tower id plate`, `polymer insulator`,
     `glass insulator`, `yoke`), excluding whatever tier 0 already claimed
     for eval, each capped at the same quota so the pool is balanced
     across classes instead of skewed toward whichever ones are naturally
     most common. The quota is set by `tower id plate`'s natural ceiling
     (only 242 images exist in all of InsPLAD-det, 25 of which tier 0
     already claimed), so every class gets an equal shot at the
     annotation budget.
  2. Duplicate-wall flights: N whole flights kept 100% intact, giving the
     "compress" act a real wall of near-identical drone frames to find
     (not simulated: these are actual contiguous DJI frame sequences).
  3. Long-tail texture: one image per remaining flight, so the embedding
     plot's messy middle still looks like a messy middle.

Deterministic given SEED: same manifest every run, same code whether
this runs live at the workshop or at home.
"""
import random
import re
from collections import defaultdict

SEED = 51
N_DUP_WALL_FLIGHTS = 14
TARGET_CLASSES = ["tower id plate", "polymer insulator", "glass insulator", "yoke"]
EVAL_HOLDOUT_PER_CLASS = 25
TARGET_CLASS_QUOTA = 217
TARGET_CLASS_PER_FLIGHT_CAP = 4

FLIGHT_PATTERN = re.compile(r"^(.+?)_DJI_(\d+)\.jpg$", re.IGNORECASE)
rng = random.Random(SEED)


def flight_of(filename):
    """Images are named `<flight_id>_DJI_<frame>.jpg`; grouping by
    flight_id recovers each drone's actual, contiguous flight sequence."""
    m = FLIGHT_PATTERN.match(filename)
    return m.group(1) if m else None


# Tier 0: eval holdout, carved out FIRST and stratified per class, before
# any annotation tier gets a chance to claim these images.
for cls in TARGET_CLASSES:
    candidates = [fn for fn, lbls in fn_labels.items() if cls in lbls and fn not in selected]
    rng.shuffle(candidates)
    picked = candidates[:EVAL_HOLDOUT_PER_CLASS]
    for fn in picked:
        selected[fn] = f"eval_holdout:{cls}"

# Tier 1: all 4 target classes, capped at the same per-flight, per-class quota
for cls in TARGET_CLASSES:
    candidates = [fn for fn, lbls in fn_labels.items() if cls in lbls and fn not in selected]
    rng.shuffle(candidates)
    per_flight_count = defaultdict(int)
    picked = []
    for fn in candidates:
        fid = flight_of(fn)
        if per_flight_count[fid] < TARGET_CLASS_PER_FLIGHT_CAP and len(picked) < TARGET_CLASS_QUOTA:
            picked.append(fn)
            per_flight_count[fid] += 1

# Tier 2: N_DUP_WALL_FLIGHTS whole flights, every frame kept
flight_ids_sorted = sorted(flights.keys())
dup_wall_flights = rng.sample(flight_ids_sorted, N_DUP_WALL_FLIGHTS)

# Tier 3: one remaining image per flight, for long-tail texture
for fid, fns in flights.items():
    remaining = [fn for fn in fns if fn not in selected]
    if remaining:
        pick = rng.choice(remaining)
```

Result, with `seed=51`:

| Tier | What it keeps | Images |
|---|---|---|
| 0: Eval holdout (25/class, carved out first) | `tower id plate`, `polymer insulator`, `glass insulator`, `yoke`, 25 each | 100 |
| 1: Balanced annotation targets (capped per class, per flight) | `tower id plate` (214), `polymer insulator` (217), `glass insulator` (217), `yoke` (217) | 865 |
| 2: Duplicate-wall flights (14 flights, 100% intact) | real contiguous drone-frame sequences | 574 |
| 3: Long-tail texture (1/remaining flight) | everything else, thinly | 215 |
| **Total** | | **1,754** |

The 100 eval-holdout images are tagged `eval_holdout` at import time (step 4)
and excluded from every interactive step of the companion workshop; the true
interactive pool at any point is 1,654 images. A later staging step copies
all 1,754 images into a lean pool directory and converts their real COCO
boxes to FiftyOne's relative `[x, y, w, h]` format, but writes them to
`heldout_ground_truth.json` rather than into the FiftyOne dataset, which is
imported strictly media-only.

### Source Data

#### Data Collection and Processing

The underlying images were captured by UAV (drone) during real-world inspections
of operating power lines, at 1920x1080 resolution, under varied environmental
conditions, orientations, and distances. See the original
[InsPLAD dataset card](https://huggingface.co/datasets/harpreetsahota/InsPLAD)
for the full collection and annotation process. This derivative pool applies no
further transformation to the images themselves; it only selects which 1,754 of
the 10,561 to include, per the stratified sampling above.

#### Who are the source data producers?

UAV inspection imagery of real, operating power lines, collected by the Voxar
Labs group at Universidade Federal de Pernambuco (see the original InsPLAD paper
and dataset card).

### Annotations

This dataset ships with no annotations. The real bounding boxes for these same
1,754 images exist (extracted from InsPLAD-det's COCO annotations during
staging) but are deliberately withheld from this Hub dataset, distributed
alongside the workshop's companion code as `heldout_ground_truth.json` instead.

#### Personal and Sensitive Information

Not addressed explicitly in the source paper. Images are aerial captures of
power line hardware and surrounding infrastructure; there is no statement in the
source material regarding incidental capture of people or other personal data.
`[More Information Needed]`

## Citation

**BibTeX:**

```
@article{doi:10.1080/01431161.2023.2283900,
   author    = {André Luiz Buarque Vieira e Silva, Heitor de Castro Felix, Franscisco Paulo Magalhães Simões, Veronica Teichrieb, Michel dos Santos, Hemir Santiago, Virginia Sgotti and Henrique Lott Neto},
   title     = {InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images},
   journal   = {International Journal of Remote Sensing},
   volume    = {44},
   number    = {23},
   pages     = {1-27},
   year      = {2023},
   publisher = {Taylor & Francis},
   doi       = {10.1080/01431161.2023.2283900},
   URL       = {https://doi.org/10.1080/01431161.2023.2283900},
   eprint    = {https://doi.org/10.1080/01431161.2023.2283900},
}
```

**APA:**

Vieira-e-Silva, A. L. B., de Castro Felix, H., Simões, F. P. M., Teichrieb, V., dos Santos, M., Santiago, H., Sgotti, V., & Lott Neto, H. (2023). InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images. *International Journal of Remote Sensing*, 44(23), 1-27.

## More Information

This is a derivative sampling of InsPLAD-det for the "Cold Pool to Hot Queue"
FiftyOne workshop. The full pipeline that builds this pool from scratch (source
download through media-only import) lives in the `workshop-notebook/` directory
of the workshop's companion materials, as five numbered, reproducible scripts.
For the full, labeled InsPLAD dataset (all three official sub-tasks, 49,706
samples), see https://huggingface.co/datasets/harpreetsahota/InsPLAD.

## Dataset Card Authors

Harpreet Sahota (FiftyOne / Voxel51 sampling and card)

## Dataset Card Contact

harpreetsahota
