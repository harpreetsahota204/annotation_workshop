"""Step 3: Copy the sampled images into a lean workshop_pool/ directory
(so step 4's import doesn't need the full 4.2 GB InsPLAD-det folder around),
and save the REAL ground truth for those images separately, in
heldout_ground_truth.json, never loaded into the working FiftyOne
dataset. This is the cold-start illusion: the workshop pool is imported
media-only in step 4; the held-out file below is only opened in the
"close the loop" act, at the end.
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

WORK_DIR = Path(__file__).parent / "data"
DET_DIR = WORK_DIR / "InsPLAD-det"
MANIFEST_PATH = WORK_DIR / "workshop_pool_manifest.json"
POOL_DIR = WORK_DIR / "workshop_pool"
HELDOUT_PATH = WORK_DIR / "heldout_ground_truth.json"


def load_coco_boxes():
    """Returns {file_name: {"width": w, "height": h, "detections": [...]}}
    with boxes already converted to FiftyOne's relative [x, y, w, h]."""
    result = {}
    for _split, jsonf in [("train", "instances_train.json"), ("val", "instances_val.json")]:
        coco = json.load(open(DET_DIR / "annotations" / jsonf))
        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        imgid_info = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in coco["images"]}
        fn_to_imgids = defaultdict(list)
        for im in coco["images"]:
            fn_to_imgids[im["file_name"]].append(im["id"])
        anns_by_imgid = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_imgid[a["image_id"]].append(a)

        for fn, imgids in fn_to_imgids.items():
            _, w, h = imgid_info[imgids[0]]
            detections = []
            for iid in imgids:
                for a in anns_by_imgid[iid]:
                    x, y, bw, bh = a["bbox"]
                    detections.append({
                        "label": cat_map[a["category_id"]],
                        "bounding_box": [x / w, y / h, bw / w, bh / h],
                    })
            result[fn] = {"width": w, "height": h, "detections": detections}
    return result


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    samples = manifest["samples"]
    print(f"Staging {len(samples)} images from the manifest...")

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for s in samples:
        src = DET_DIR / s["split"] / s["file_name"]
        dst = POOL_DIR / s["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)

    n_copied = len(list(POOL_DIR.glob("*.jpg")))
    print(f"Staged {n_copied} images to {POOL_DIR}")

    print("Extracting held-out ground truth for the sampled images only...")
    all_boxes = load_coco_boxes()
    heldout = {s["file_name"]: all_boxes[s["file_name"]] for s in samples}
    HELDOUT_PATH.write_text(json.dumps(heldout, indent=2))
    print(f"Held-out ground truth for {len(heldout)} images saved to {HELDOUT_PATH}")
    print("(Not loaded into the working dataset; reserved for the 'close the loop' act.)")


if __name__ == "__main__":
    main()
