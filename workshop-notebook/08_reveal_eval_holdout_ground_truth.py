"""
Step 12's "reveal" mechanism, actually implemented.

Loads the real ground truth from heldout_ground_truth.json into the
`ground_truth` field, but ONLY for the 100 samples tagged `eval_holdout`.
Every other sample in the pool keeps zero label fields, exactly as before.

This is safe to run at any point in the workshop, including in dry runs,
because:
  - `ground_truth` never touches the ~1,654 interactive-pool samples.
  - the `eval_holdout` tag is what scoped every act's views all along, so
    nothing upstream needs to change.

Creates a saved view, `eval_holdout_annotated`, scoped to exactly those
100 samples with `ground_truth` populated, for the "Evaluate" act to open
directly in the App.
"""
import json

import fiftyone as fo
from fiftyone import Detection, Detections

DATASET_NAME = "insplad-workshop-pool"
HELDOUT_PATH = "data/heldout_ground_truth.json"

# InsPLAD-det has 11 fine-grained classes; this workshop only searches for,
# annotates, and trains on these 4. Loading the other 7 (shackle variants,
# stockbridge damper, yoke suspension) into `ground_truth` would penalize
# the model for "missing" objects it was never trained to detect, which
# would corrupt precision/recall for no reason.
TARGET_CLASSES = {"tower id plate", "polymer insulator", "glass insulator", "yoke"}


def fname(filepath):
    return filepath.split("/")[-1]


def main():
    dataset = fo.load_dataset(DATASET_NAME)
    heldout = json.load(open(HELDOUT_PATH))

    eval_view = dataset.match_tags("eval_holdout")
    n = len(eval_view)
    print(f"Revealing ground_truth on {n} eval_holdout samples (scoped to {sorted(TARGET_CLASSES)})...")

    dropped = 0
    for sample in eval_view.iter_samples(autosave=True):
        info = heldout.get(fname(sample.filepath))
        if info is None:
            print(f"WARNING: no held-out ground truth found for {sample.filepath}")
            continue
        detections = []
        for d in info["detections"]:
            if d["label"] not in TARGET_CLASSES:
                dropped += 1
                continue
            detections.append(Detection(label=d["label"], bounding_box=d["bounding_box"]))
        sample["ground_truth"] = Detections(detections=detections)

    print(f"Dropped {dropped} out-of-scope detections (classes outside {sorted(TARGET_CLASSES)})")

    if dataset.has_saved_view("eval_holdout_annotated"):
        dataset.delete_saved_view("eval_holdout_annotated")
    dataset.save_view(
        "eval_holdout_annotated",
        dataset.match_tags("eval_holdout"),
        description="The 100-sample stratified evaluation holdout, with real "
        "ground_truth revealed. Never used for annotation, search, or "
        "training -- only for the final evaluate step.",
    )

    with_gt = dataset.match_tags("eval_holdout").exists("ground_truth")
    print(f"ground_truth populated on {len(with_gt)}/{n} eval_holdout samples")
    print("Saved view 'eval_holdout_annotated' created.")
    print("count_sample_field ground_truth total detections:",
          sum(len(s.ground_truth.detections) for s in with_gt))


if __name__ == "__main__":
    main()
