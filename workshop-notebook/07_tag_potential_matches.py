"""
Tag `potential_match` on the union of each class's seeded-similarity top-100
search results. This is the "already found" set that the prioritization
step must exclude, per the exclusion-fix mechanism validated earlier this
session.
"""
import json

import fiftyone as fo

dataset = fo.load_dataset("insplad-workshop-pool")
interactive = dataset.match_tags("eval_holdout", bool=False)

results = json.load(open("data/seeded_search_results.json"))
base_dir = interactive.first().filepath.rsplit("/", 1)[0]

fp_to_id = {s.filepath: str(s.id) for s in interactive.select_fields([])}

# clear any stale potential_match tag first
dataset.untag_samples("potential_match")

all_ids = set()
for cls, info in results.items():
    brain_key = f"seeded_sim_{cls.replace(' ', '_')}"
    seed_ids = [fp_to_id[f"{base_dir}/{fn}"] for fn in info["seeds"]]
    view = interactive.sort_by_similarity(seed_ids, k=100, brain_key=brain_key)
    ids = [s.id for s in view.select_fields([])]
    all_ids.update(ids)
    print(f"{cls}: {len(ids)} found ids (running union size {len(all_ids)})")

dataset.select(list(all_ids)).tag_samples("potential_match")
print("TOTAL potential_match tagged:", len(all_ids))
print("count_sample_tags:", dataset.count_sample_tags())
