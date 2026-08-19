"""
Rebuild seeded-similarity search indexes for all 4 target classes on the
interactive pool (eval_holdout excluded). Reports top-100 hit rate vs random
baseline for each class, using the same seeding pattern used for
`tower id plate` earlier in this session (rng.Random(51), 3 seeds/class).
"""
import json
import random

import fiftyone as fo
import fiftyone.brain as fob

dataset = fo.load_dataset("insplad-workshop-pool")
interactive = dataset.match_tags("eval_holdout", bool=False)

heldout = json.load(open("data/heldout_ground_truth.json"))
fn_labels = {
    fn: {x["label"] for x in info["detections"]} for fn, info in heldout.items()
}


def fname(fp):
    return fp.split("/")[-1]


fp_to_id = {s.filepath: str(s.id) for s in interactive.select_fields([])}
base_dir = interactive.first().filepath.rsplit("/", 1)[0]

classes = ["tower id plate", "glass insulator", "yoke", "polymer insulator"]
rng = random.Random(51)

results = {}
for cls in classes:
    brain_key = f"seeded_sim_{cls.replace(' ', '_')}"
    if brain_key in interactive.list_brain_runs():
        interactive.delete_brain_run(brain_key)

    class_fns = [
        fn
        for fn in fn_labels
        if cls in fn_labels[fn] and f"{base_dir}/{fn}" in fp_to_id
    ]
    seed_fns = rng.sample(class_fns, 3)
    seed_ids = [fp_to_id[f"{base_dir}/{fn}"] for fn in seed_fns]

    fob.compute_similarity(interactive, embeddings="clip_embedding", brain_key=brain_key)
    view = interactive.sort_by_similarity(seed_ids, k=100, brain_key=brain_key)
    hits = sum(
        1 for s in view.select_fields([]) if cls in fn_labels.get(fname(s.filepath), set())
    )
    random_baseline = 100 * len(class_fns) / len(interactive)
    results[cls] = {
        "seeds": seed_fns,
        "pool_count": len(class_fns),
        "top100_hits": hits,
        "random_baseline_pct": round(random_baseline, 2),
    }
    print(f"{cls:20s} pool={len(class_fns):4d}  seeds={seed_fns}  top-100 hits={hits}  random baseline/100={random_baseline:.1f}")

with open("data/seeded_search_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("SEEDED_SEARCH_DONE")
