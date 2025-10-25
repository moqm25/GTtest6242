#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# moved import of timed_step until after PROJECT_ROOT is defined so we can fix sys.path
# (see below where the import is added)
URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2025-04-24.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DICT_DIR = DATA_DIR / "dictionaries"

# Ensure the project root is on sys.path so the `src` package can be imported when running this script.
sys.path.insert(0, str(PROJECT_ROOT))
from src.progress import timed_step

WANTED = {
    # keep this aligned with your pipeline
    "food.csv",
    "foundation_food.csv",
    "sr_legacy_food.csv",
    "branded_food.csv",
    "food_nutrient.csv",
    "nutrient.csv",
    "food_portion.csv",
    "measure_unit.csv",
    "food_category.csv",
    "wweia_food_category.csv",
}

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    DICT_DIR.mkdir(exist_ok=True)

def download_zip() -> bytes:
    print(f"==> Downloading USDA dataset:\n    {URL}")
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        buf = resp.read()
    if len(buf) < 10_000_000:
        raise RuntimeError("Downloaded file seems too small; network error?")
    print(f"==> Download size: {len(buf)/1e6:.1f} MB, sha256={sha256(buf)[:16]}…")
    return buf

def extract_needed(zip_bytes: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        # Find csv files at root of the zip or nested; copy only required CSVs
        count = 0
        for name in names:
            if not name.lower().endswith(".csv"):
                continue
            base = Path(name).name
            if base in WANTED:
                target = DATA_DIR / base
                if target.exists():
                    print(f"    [skip] {base} already exists.")
                    continue
                print(f"    [write] {base}")
                with zf.open(name) as src, open(target, "wb") as out:
                    out.write(src.read())
                count += 1
        if count == 0:
            print("[warn] No new files extracted; they may already exist.")

def main():
    parser = argparse.ArgumentParser(description="Download & stage USDA FoodData Central CSVs into DATA/")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite CSVs.")
    args = parser.parse_args()

    ensure_dirs()

    if args.force:
        for f in WANTED:
            p = DATA_DIR / f
            if p.exists():
                print(f"    [rm] {p}")
                p.unlink()

    # Download and extract
    try:
        with timed_step("Downloading USDA zip"):
            blob = download_zip()
        with timed_step("Extracting CSVs"):
            extract_needed(blob)
    except Exception as e:
        print(f"[ERROR] Failed to set up data: {e}", file=sys.stderr)
        sys.exit(2)

    # Create subfolders expected by pipeline
    (DATA_DIR / "raw").mkdir(exist_ok=True)
    (DATA_DIR / "processed").mkdir(exist_ok=True)

    # Seed dictionaries if missing
    with timed_step("Seeding dictionaries"):
        (DATA_DIR / "dictionaries").mkdir(exist_ok=True)
        fg = DATA_DIR / "dictionaries" / "food_groups.csv"
        if not fg.exists():
            fg.write_text("keyword,group\n" "yogurt,Dairy\n" "chicken,Poultry\n")
            print("    [seed] dictionaries/food_groups.csv")

    print("==> Data setup complete.")
    print("Next: run `make ingest` then `make build`.")

if __name__ == "__main__":
    main()
