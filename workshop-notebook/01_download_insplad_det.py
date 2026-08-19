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


def download_with_resume(url, dest, expected_size):
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing >= expected_size:
        print(f"Already downloaded: {dest} ({existing} bytes)")
        return

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        mode = "ab" if existing else "wb"
        with open(dest, mode) as f:
            downloaded = existing
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 8):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024 * 200) < len(chunk):
                    pct = 100 * downloaded / expected_size
                    print(f"  {downloaded / 1e9:.2f} / {expected_size / 1e9:.2f} GB ({pct:.1f}%)")
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes)")


def extract_det_only(outer_zip, det_dir):
    if det_dir.exists() and any(det_dir.iterdir()):
        print(f"Already extracted: {det_dir}")
        return

    with zipfile.ZipFile(outer_zip) as outer:
        names = outer.namelist()
        det_zip_name = next(n for n in names if n.endswith("InsPLAD-det.zip"))
        print(f"Extracting {det_zip_name} from the outer archive...")
        outer.extract(det_zip_name, WORK_DIR)

    inner_zip_path = WORK_DIR / det_zip_name
    with zipfile.ZipFile(inner_zip_path) as inner:
        inner.extractall(det_dir)
    inner_zip_path.unlink()  # don't need the intermediate inner zip anymore
    print(f"Extracted InsPLAD-det to {det_dir}")


if __name__ == "__main__":
    url, size = get_download_url()
    print(f"Downloading InsPLAD ({size / 1e9:.2f} GB)... this can take a while.")
    download_with_resume(url, OUTER_ZIP, size)
    extract_det_only(OUTER_ZIP, DET_DIR)

    n_images = len(list(DET_DIR.glob("*/*.jpg")))
    print(f"\nDone. {n_images} images available under {DET_DIR}")
    print(
        "Note: supervised_fault_classification.zip and "
        "unsupervised_anomaly_detection.zip remain zipped inside "
        f"{OUTER_ZIP} untouched; this workshop doesn't use them."
    )
