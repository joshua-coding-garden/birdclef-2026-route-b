"""
BirdCLEF+ 2026 — Download external audio from iNaturalist API.

Queries all 209 named species via iNaturalist API (free, no API key).
Filters: research-grade observations with sounds, CC0/CC-BY/CC-BY-NC licenses.
Downloads to c:\birdCLEF\external\inaturalist\{primary_label}\*.{mp3,wav,m4a}

Usage:
    C:\birdCLEF\.venv-tf\Scripts\python.exe C:\birdCLEF\new\download_inat.py

References:
    - sergheibrinza (BirdCLEF 2026 discussion): iNaturalist research-grade audio
    - API docs: https://api.inaturalist.org/v1/docs/
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAXONOMY_CSV = Path(r"C:\birdCLEF\birdclef-2026\taxonomy.csv")
OUT_DIR = Path(r"C:\birdCLEF\external\inaturalist")
META_CSV = OUT_DIR / "inat_metadata.csv"

MAX_PER_SPECIES = 200
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-nc"}
DELAY_BETWEEN_QUERIES = 1.2  # iNat recommends max 60 req/min
DELAY_BETWEEN_DOWNLOADS = 0.3

INAT_API = "https://api.inaturalist.org/v1/observations"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_inat(taxon_id: int | str) -> list[dict]:
    """Query iNat API for audio observations of a taxon."""
    all_obs = []
    page = 1
    per_page = 200  # max allowed

    while True:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "sounds": "true",
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
            "license": "cc0,cc-by,cc-by-nc",
        }
        try:
            resp = requests.get(INAT_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [ERROR] API page={page}: {e}")
            break

        results = data.get("results", [])
        all_obs.extend(results)

        total = data.get("total_results", 0)
        if len(all_obs) >= total or len(all_obs) >= MAX_PER_SPECIES or not results:
            break
        page += 1
        time.sleep(DELAY_BETWEEN_QUERIES * 0.5)

    return all_obs


def extract_sounds(observations: list[dict]) -> list[dict]:
    """Extract downloadable sound entries from observations."""
    sounds = []
    for obs in observations:
        obs_id = obs.get("id")
        obs_license = obs.get("license_code", "")
        for s in obs.get("sounds", []):
            sound_license = s.get("license_code") or obs_license or ""
            if sound_license.lower() not in ALLOWED_LICENSES:
                continue
            file_url = s.get("file_url", "")
            if not file_url:
                continue
            sounds.append({
                "obs_id": obs_id,
                "sound_id": s.get("id"),
                "file_url": file_url,
                "license": sound_license,
                "attribution": s.get("attribution", ""),
            })
            if len(sounds) >= MAX_PER_SPECIES:
                return sounds
    return sounds


def download_file(url: str, dest: Path, retries: int = 2) -> bool:
    """Download a file with retries."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"    [FAIL] {url}: {e}")
                return False
    return False


def url_extension(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
        return ext
    return ".mp3"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tax = pd.read_csv(TAXONOMY_CSV)
    print(f"[taxonomy] {len(tax)} species total")

    # Skip sonotypes
    queryable = tax[~tax["scientific_name"].str.startswith("Insect son")].copy()
    print(f"[queryable] {len(queryable)} species (skipped {len(tax) - len(queryable)} insect sonotypes)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_meta = []
    total_downloaded = 0
    species_with_audio = 0
    species_skipped = 0

    for idx, row in queryable.iterrows():
        plabel = str(row["primary_label"])
        taxon_id = row["inat_taxon_id"]
        sci_name = row["scientific_name"]
        class_name = row["class_name"]

        sp_dir = OUT_DIR / plabel
        sp_dir.mkdir(parents=True, exist_ok=True)

        # Check existing downloads
        existing = list(sp_dir.glob("*.*"))
        existing = [f for f in existing if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac")]
        if len(existing) >= MAX_PER_SPECIES:
            print(f"[{idx+1}] {plabel} ({sci_name}) — already have {len(existing)} files, skipping")
            species_skipped += 1
            continue

        print(f"[{idx+1}/{len(queryable)}] {plabel} ({sci_name}, {class_name}, taxon={taxon_id})")
        time.sleep(DELAY_BETWEEN_QUERIES)

        observations = query_inat(taxon_id)
        if not observations:
            print(f"    No research-grade audio observations")
            continue

        sounds = extract_sounds(observations)
        need = MAX_PER_SPECIES - len(existing)
        sounds = sounds[:need]

        if not sounds:
            print(f"    {len(observations)} observations but 0 licensed audio")
            continue

        print(f"    Found {len(sounds)} downloadable sounds")
        species_with_audio += 1
        dl_count = 0

        for s in sounds:
            ext = url_extension(s["file_url"])
            dest = sp_dir / f"inat_{s['sound_id']}{ext}"
            if dest.exists():
                continue

            ok = download_file(s["file_url"], dest)
            if ok:
                dl_count += 1
                total_downloaded += 1
                all_meta.append({
                    "primary_label": plabel,
                    "scientific_name": sci_name,
                    "class_name": class_name,
                    "inat_taxon_id": taxon_id,
                    "obs_id": s["obs_id"],
                    "sound_id": s["sound_id"],
                    "license": s["license"],
                    "attribution": s["attribution"],
                    "file_path": str(dest.relative_to(OUT_DIR)),
                    "url": s["file_url"],
                })
            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        print(f"    Downloaded {dl_count} files")

        # Save metadata incrementally
        if all_meta:
            pd.DataFrame(all_meta).to_csv(META_CSV, index=False, encoding="utf-8")

    # Final save
    if all_meta:
        meta_df = pd.DataFrame(all_meta)
        meta_df.to_csv(META_CSV, index=False, encoding="utf-8")
        print(f"\n[DONE] Downloaded {total_downloaded} files from {species_with_audio} species")
        print(f"[META] Saved to {META_CSV}")
        per_sp = meta_df.groupby("primary_label").size()
        print(f"[STATS] Per-species count: min={per_sp.min()}, max={per_sp.max()}, "
              f"median={per_sp.median():.0f}, total_species={len(per_sp)}")

        # Show class breakdown
        cls_counts = meta_df.groupby("class_name").size()
        print(f"[CLASS] {cls_counts.to_dict()}")
    else:
        print(f"\n[DONE] No audio downloaded. Skipped {species_skipped} species.")


if __name__ == "__main__":
    main()
