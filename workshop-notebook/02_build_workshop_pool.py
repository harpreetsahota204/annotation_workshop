"""Step 2: Build the reproducible, stratified workshop pool manifest from
InsPLAD-det's raw images. Whole scene images only, no labels attached to
the resulting pool: ground truth for the sampled images is saved
separately in step 3, held out for the "close the loop" act.

This pool is built around one goal: annotate a small, balanced, high-value
set from a large unlabeled pool, then fine-tune a detector on it. All 4
target classes (`tower id plate`, `polymer insulator`, `glass insulator`,
`yoke`) are capped at the same per-class quota, so none of them dominates
the annotation budget and none of them is a special case.

Tiers:
  0. Eval holdout: a stratified ~25-per-class slice across all 4 target
     classes, carved out FIRST, before any other tier is built. These
     images are never available to prioritization or annotation in any
     downstream act -- they exist purely so there's a clean, never-touched
     benchmark to evaluate the fine-tuned detector against later.
  1. Annotation targets: capped per-flight, per-class samples of all 4
     target classes (excluding whatever tier 0 already claimed for eval),
     each capped at the same quota so the pool is balanced across classes
     instead of skewed toward whichever ones are naturally most common.
     The quota is set by `tower id plate`'s natural ceiling (only 242
     images exist in all of InsPLAD-det), so every class gets an equal
     shot at the annotation budget.
  2. Duplicate-wall flights: N whole flights kept 100% intact, giving the
     "compress" act a real wall of near-identical drone frames to find
     (not simulated: these are actual contiguous DJI frame sequences).
  3. Long-tail texture: one image per remaining flight, so the embedding
     plot's messy middle still looks like a messy middle.

Deterministic given SEED: same manifest every run, same code whether
this runs live at the workshop or at home.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 51
N_DUP_WALL_FLIGHTS = 14
TARGET_CLASSES = ["tower id plate", "polymer insulator", "glass insulator", "yoke"]
EVAL_HOLDOUT_PER_CLASS = 25
# tower id plate only has 242 images in the whole 10,561-image dataset;
# 242 - 25 held out = 217 is the hard ceiling for this class after eval
# holdout. Every class is capped at that same number, so the pool stays
# perfectly balanced instead of skewed toward whichever classes are
# naturally more common.
TARGET_CLASS_QUOTA = 217
TARGET_CLASS_PER_FLIGHT_CAP = 4

WORK_DIR = Path(__file__).parent / "data"
DET_DIR = WORK_DIR / "InsPLAD-det"
OUT_MANIFEST = WORK_DIR / "workshop_pool_manifest.json"

FLIGHT_PATTERN = re.compile(r"^(.+?)_DJI_(\d+)\.jpg$", re.IGNORECASE)

rng = random.Random(SEED)


def flight_of(filename):
    m = FLIGHT_PATTERN.match(filename)
    return m.group(1) if m else None


def load_split_labels(jsonf):
    """Returns {file_name: set(labels)}, merging any duplicate image_ids
    that share a file_name (see InsPLAD-det's 46 known duplicate COCO
    entries in the train split)."""
    coco = json.load(open(DET_DIR / "annotations" / jsonf))
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    fn_to_imgids = defaultdict(list)
    for im in coco["images"]:
        fn_to_imgids[im["file_name"]].append(im["id"])
    anns_by_imgid = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_imgid[a["image_id"]].append(a)

    fn_labels = {}
    for fn, imgids in fn_to_imgids.items():
        labels = set()
        for iid in imgids:
            for a in anns_by_imgid[iid]:
                labels.add(cat_map[a["category_id"]])
        fn_labels[fn] = labels
    return fn_labels


def main():
    fn_labels, fn_split = {}, {}
    for split, jsonf in [("train", "instances_train.json"), ("val", "instances_val.json")]:
        labels = load_split_labels(jsonf)
        for fn, lbls in labels.items():
            fn_labels[fn] = lbls
            fn_split[fn] = split

    all_files = list(fn_labels.keys())
    print(f"Total unique InsPLAD-det images: {len(all_files)}")

    flights = defaultdict(list)
    unmatched = []
    for fn in all_files:
        fid = flight_of(fn)
        if fid is None:
            unmatched.append(fn)
        else:
            flights[fid].append(fn)
    print(f"Total flights: {len(flights)} (+{len(unmatched)} filenames with no flight-prefix match)")

    selected = {}

    # Tier 0: eval holdout, carved out FIRST and stratified per class, before
    # any annotation tier gets a chance to claim these images.
    eval_holdout_counts = {}
    for cls in TARGET_CLASSES:
        candidates = [fn for fn, lbls in fn_labels.items() if cls in lbls and fn not in selected]
        rng.shuffle(candidates)
        picked = candidates[:EVAL_HOLDOUT_PER_CLASS]
        for fn in picked:
            selected[fn] = f"eval_holdout:{cls}"
        eval_holdout_counts[cls] = len(picked)
    print(f"Tier 0 (eval holdout, stratified): {eval_holdout_counts}, total {sum(eval_holdout_counts.values())}")

    # Tier 1: all 4 target classes, capped at the same quota, so the pool
    # is balanced across classes instead of skewed toward whichever ones
    # are naturally most common.
    tier1_counts = {}
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
        for fn in picked:
            selected[fn] = f"target:{cls}"
        tier1_counts[cls] = len(picked)
    print(f"Tier 1 (balanced annotation targets, {TARGET_CLASS_QUOTA}/class cap): {tier1_counts}")

    flight_ids_sorted = sorted(flights.keys())
    dup_wall_flights = rng.sample(flight_ids_sorted, N_DUP_WALL_FLIGHTS)
    tier2_count = 0
    for fid in dup_wall_flights:
        for fn in flights[fid]:
            if fn not in selected:
                selected[fn] = "duplicate_wall"
                tier2_count += 1
    print(f"Tier 2 (duplicate-wall flights {dup_wall_flights}): {tier2_count} new images")

    tier3_count = 0
    for _fid, fns in flights.items():
        remaining = [fn for fn in fns if fn not in selected]
        if remaining:
            pick = rng.choice(remaining)
            selected[pick] = "long_tail"
            tier3_count += 1
    print(f"Tier 3 (long-tail, 1/flight): {tier3_count} new images")

    unmatched_remaining = [fn for fn in unmatched if fn not in selected]
    rng.shuffle(unmatched_remaining)
    unmatched_pick = unmatched_remaining[: min(len(unmatched_remaining), 20)]
    for fn in unmatched_pick:
        selected[fn] = "long_tail_unmatched"
    print(f"Tier 3b (unmatched-filename long-tail): {len(unmatched_pick)} new images")

    print(f"\nTotal workshop pool size: {len(selected)}")

    manifest = {
        "seed": SEED,
        "params": {
            "n_dup_wall_flights": N_DUP_WALL_FLIGHTS,
            "target_classes": TARGET_CLASSES,
            "target_class_quota": TARGET_CLASS_QUOTA,
            "target_class_per_flight_cap": TARGET_CLASS_PER_FLIGHT_CAP,
            "eval_holdout_per_class": EVAL_HOLDOUT_PER_CLASS,
            "eval_holdout_counts": eval_holdout_counts,
            "tier1_counts": tier1_counts,
            "dup_wall_flight_ids": dup_wall_flights,
        },
        "total": len(selected),
        "samples": [
            {"file_name": fn, "split": fn_split[fn], "tier": tier}
            for fn, tier in sorted(selected.items())
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
