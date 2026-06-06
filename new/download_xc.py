"""
BirdCLEF+ 2026 — Download external audio from Xeno-canto API.

Queries all 209 named species (skips 25 insect sonotypes).
Filters: quality A-D (drop E), cap MAX_PER_SPECIES clips per species.
Downloads to c:\birdCLEF\external\xeno-canto\{primary_label}\*.mp3

Usage:
    C:\birdCLEF\.venv-tf\Scripts\python.exe C:\birdCLEF\new\download_xc.py

References:
    - TheoViel/kaggle_birdclef2024: quality filter + 500 cap
    - VSydorskyy/BirdCLEF_2025_2nd_place: xenocanto lib wrapper
    - myso1987/BirdCLEF-2025-5th-place-solution: max 500/class
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAXONOMY_CSV = Path(r"C:\birdCLEF\birdclef-2026\taxonomy.csv")
OUT_DIR = Path(r"C:\birdCLEF\external\xeno-canto")
META_CSV = OUT_DIR / "xc_metadata.csv"

MAX_PER_SPECIES = 200
MIN_QUALITY = "D"  # keep A, B, C, D; drop E
DELAY_BETWEEN_QUERIES = 1.0  # seconds, be respectful to XC API
DELAY_BETWEEN_DOWNLOADS = 0.3

QUALITY_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "no score": 0}
MIN_QUALITY_RANK = QUALITY_RANK[MIN_QUALITY]

XC_API = "https://xeno-canto.org/api/3/recordings"

# *** API KEY REQUIRED ***
# XC API v3 requires an API key. Get yours at: https://xeno-canto.org/account
# Set the environment variable XC_API_KEY before running, or paste it below.
import os
XC_API_KEY = os.environ.get("XC_API_KEY", "")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_xc(scientific_name: str) -> list[dict]:
    """Query XC API v3 for a species, paginating through all results."""
    if not XC_API_KEY:
        raise RuntimeError(
            "XC_API_KEY not set. Get your key at https://xeno-canto.org/account "
            "then run: set XC_API_KEY=your_key_here"
        )
    import requests as _requests
    all_recordings = []
    page = 1
    while True:
        params = {
            "query": scientific_name,
            "page": page,
            "key": XC_API_KEY,
        }
        try:
            resp = _requests.get(XC_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [ERROR] API query failed page={page}: {e}")
            break

        recordings = data.get("recordings", [])
        all_recordings.extend(recordings)

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(DELAY_BETWEEN_QUERIES * 0.5)

    return all_recordings


def filter_and_cap(recordings: list[dict], max_n: int) -> list[dict]:
    """Filter by quality, sort by ID desc (newest first), cap at max_n."""
    filtered = []
    for r in recordings:
        q = r.get("q", "no score")
        rank = QUALITY_RANK.get(q, 0)
        if rank >= MIN_QUALITY_RANK:
            r["_quality_rank"] = rank
            filtered.append(r)

    # Sort: highest quality first, then newest (highest ID) first
    filtered.sort(key=lambda r: (r["_quality_rank"], int(r.get("id", 0))), reverse=True)
    return filtered[:max_n]


def download_file(url: str, dest: Path, retries: int = 2) -> bool:
    """Download a file with retries."""
    for attempt in range(retries + 1):
        try:
            if url.startswith("//"):
                url = "https:" + url
            elif not url.startswith("http"):
                url = "https://" + url
            urlretrieve(url, str(dest))
            return True
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"    [FAIL] {url} -> {e}")
                return False
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tax = pd.read_csv(TAXONOMY_CSV)
    print(f"[taxonomy] {len(tax)} species total")

    # Skip sonotypes (scientific_name starts with "Insect son")
    queryable = tax[~tax["scientific_name"].str.startswith("Insect son")]
    print(f"[queryable] {len(queryable)} species (skipped {len(tax) - len(queryable)} insect sonotypes)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = []
    total_downloaded = 0
    total_skipped = 0

    for idx, row in queryable.iterrows():
        plabel = row["primary_label"]
        sci_name = row["scientific_name"]
        class_name = row["class_name"]

        sp_dir = OUT_DIR / str(plabel)
        sp_dir.mkdir(parents=True, exist_ok=True)

        # Check existing downloads
        existing = list(sp_dir.glob("*.mp3")) + list(sp_dir.glob("*.ogg")) + list(sp_dir.glob("*.wav"))
        if len(existing) >= MAX_PER_SPECIES:
            print(f"[{idx+1}/{len(queryable)}] {plabel} ({sci_name}) — already have {len(existing)} files, skipping")
            total_skipped += 1
            continue

        print(f"[{idx+1}/{len(queryable)}] {plabel} ({sci_name}, {class_name}) — querying XC...")
        time.sleep(DELAY_BETWEEN_QUERIES)

        recordings = query_xc(sci_name)
        if not recordings:
            print(f"    No recordings found on Xeno-canto")
            continue

        # How many more do we need?
        need = MAX_PER_SPECIES - len(existing)
        selected = filter_and_cap(recordings, need)
        print(f"    Found {len(recordings)} total, {len(selected)} selected (quality>={MIN_QUALITY}, cap={MAX_PER_SPECIES})")

        for rec in selected:
            xc_id = rec.get("id", "unknown")
            file_url = rec.get("file", "")
            file_name = rec.get("file-name", f"XC{xc_id}.mp3")
            dest = sp_dir / f"XC{xc_id}_{file_name}"

            if dest.exists():
                continue

            ok = download_file(file_url, dest)
            if ok:
                total_downloaded += 1
                all_meta.append({
                    "primary_label": plabel,
                    "scientific_name": sci_name,
                    "class_name": class_name,
                    "xc_id": xc_id,
                    "quality": rec.get("q", ""),
                    "recordist": rec.get("rec", ""),
                    "country": rec.get("cnt", ""),
                    "length": rec.get("length", ""),
                    "also": rec.get("also", ""),
                    "file_path": str(dest.relative_to(OUT_DIR)),
                    "url": file_url,
                })
            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        # Save metadata incrementally (in case of crash)
        if all_meta:
            pd.DataFrame(all_meta).to_csv(META_CSV, index=False, encoding="utf-8")

    # Final save
    if all_meta:
        meta_df = pd.DataFrame(all_meta)
        meta_df.to_csv(META_CSV, index=False, encoding="utf-8")
        print(f"\n[DONE] Downloaded {total_downloaded} files, skipped {total_skipped} species")
        print(f"[META] Saved to {META_CSV}")
        print(f"[STATS] Species with downloads: {meta_df['primary_label'].nunique()}")
        print(f"[STATS] Per-species count: min={meta_df.groupby('primary_label').size().min()}, "
              f"max={meta_df.groupby('primary_label').size().max()}, "
              f"median={meta_df.groupby('primary_label').size().median():.0f}")
    else:
        print(f"\n[DONE] No new downloads. Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
