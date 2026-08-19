"""Optional: reclaim disk space by removing the parts of the download that
the workshop pool doesn't use. Run this only after 03_stage_pool_and_heldout_labels.py
has copied the images you need into data/workshop_pool/; that directory
is untouched by this script.

Three independent things you can clean up, pick what you want:
  1. The full InsPLAD-det/ folder (4.2 GB), only needed if you want to
     rebuild the manifest with different parameters later. Safe to delete
     if you're done experimenting with the sampling step.
  2. The outer InsPLAD_Dataset.zip (6.4 GB), only needed to re-run step 1
     from a cold cache. Safe to delete once InsPLAD-det/ is extracted.
  3. Nothing from the fault/anomaly sub-datasets was ever extracted (see
     01_download_insplad_det.py), so there's nothing to clean up there.
"""
import shutil
from pathlib import Path

WORK_DIR = Path(__file__).parent / "data"
DET_DIR = WORK_DIR / "InsPLAD-det"
OUTER_ZIP = WORK_DIR / "InsPLAD_Dataset.zip"
POOL_DIR = WORK_DIR / "workshop_pool"


def confirm(prompt):
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def main():
    if not POOL_DIR.exists() or not any(POOL_DIR.iterdir()):
        raise SystemExit(
            "workshop_pool/ is empty or missing; run "
            "03_stage_pool_and_heldout_labels.py first. Refusing to "
            "delete the source data before the pool is safely staged."
        )

    if OUTER_ZIP.exists() and confirm(f"Delete {OUTER_ZIP} (6.4 GB)?"):
        OUTER_ZIP.unlink()
        print("Deleted outer zip.")

    if DET_DIR.exists() and confirm(f"Delete {DET_DIR} (4.2 GB, full InsPLAD-det)?"):
        shutil.rmtree(DET_DIR)
        print("Deleted full InsPLAD-det folder.")

    print(f"\nKept: {POOL_DIR} (the {len(list(POOL_DIR.glob('*.jpg')))}-image workshop pool)")


if __name__ == "__main__":
    main()
