"""
BirdCLEF+ 2026 — Process AnuraSet for external training data.

AnuraSet: 42 anuran species, 93k 3-second WAV samples at 22.05 kHz.
17 species overlap with BirdCLEF 2026 target amphibians.

This script:
1. Unzips anuraset.zip (if not already done)
2. Reads weak_labels.csv to identify recordings with target species
3. Extracts and organizes audio by primary_label
4. Pads 3s samples to 5s for Perch v2 input

Output: c:\birdCLEF\external\anuraset\{primary_label}\*.wav

Usage:
    C:\birdCLEF\.venv-tf\Scripts\python.exe C:\birdCLEF\new\process_anuraset.py
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANURASET_DIR = Path(r"C:\birdCLEF\external\anuraset")
ZIP_PATH = ANURASET_DIR / "anuraset.zip"
EXTRACT_DIR = ANURASET_DIR / "extracted"
OUT_DIR = Path(r"C:\birdCLEF\external\anuraset")  # output organized by primary_label

TAXONOMY_CSV = Path(r"C:\birdCLEF\birdclef-2026\taxonomy.csv")
WEAK_LABELS_CSV = ANURASET_DIR / "weak_labels.csv"
SPECIES_CSV = ANURASET_DIR / "species.csv"

TARGET_SR = 32000  # Perch v2 sample rate
TARGET_DURATION = 5.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

MAX_PER_SPECIES = 200

# AnuraSet code -> BirdCLEF primary_label mapping
ANURASET_TO_BIRDCLEF = {
    "PITAZU": "517063",
    "PHYSAU": "23724",
    "PHYNAT": "476521",
    "BOALUN": "555123",
    "AMEPIC": "64898",
    "LEPPOD": "22961",
    "LEPELE": "22967",
    "LEPLAB": "22983",
    "SCIFUS": "24287",
    "BOARAN": "555146",
    "SCIFUV": "24285",
    "DENNAN": "65380",
    "PHYALB": "23158",
    "ELABIC": "25092",
    "SCINAS": "24279",
    "DENMIN": "65377",
    "LEPFUS": "22973",
}


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    n_samples = int(duration * target_sr)
    if n_samples == 0:
        return np.zeros(1, dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def pad_to_5s(audio: np.ndarray) -> np.ndarray:
    """Pad or trim audio to exactly 5 seconds at TARGET_SR."""
    if len(audio) >= TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES]
    return np.pad(audio, (0, TARGET_SAMPLES - len(audio))).astype(np.float32)


def main():
    # Step 1: Unzip if needed
    if not EXTRACT_DIR.exists():
        if not ZIP_PATH.exists():
            print(f"[ERROR] {ZIP_PATH} not found. Download first.")
            return
        print(f"[unzip] Extracting {ZIP_PATH} ({ZIP_PATH.stat().st_size // 1024 // 1024} MB)...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print(f"[unzip] Done")
    else:
        print(f"[unzip] Already extracted at {EXTRACT_DIR}")

    # Step 2: Find WAV files in extracted directory
    wav_files = list(EXTRACT_DIR.rglob("*.wav"))
    print(f"[scan] Found {len(wav_files)} WAV files in extracted directory")

    # Step 3: Read weak labels to identify which recordings have target species
    wl = pd.read_csv(WEAK_LABELS_CSV)
    print(f"[labels] {len(wl)} recordings in weak_labels.csv")

    # Build a mapping: AUDIO_FILE_ID -> list of (anuraset_code, primary_label) for target species
    file_to_species = {}
    for _, row in wl.iterrows():
        file_id = row["AUDIO_FILE_ID"]
        species_present = []
        for anura_code, birdclef_label in ANURASET_TO_BIRDCLEF.items():
            col = f"SPECIES_{anura_code}"
            if col in wl.columns and row[col] > 0:
                species_present.append((anura_code, birdclef_label))
        if species_present:
            file_to_species[file_id] = species_present

    print(f"[labels] {len(file_to_species)} recordings contain at least one target species")

    # Step 4: Process WAV files
    # WAV filenames: SITE_DATE_TIME_START_END.wav
    # AUDIO_FILE_ID: SITE_DATE_TIME
    # Extract prefix (first 3 underscore-separated parts) for O(1) lookup
    counts = {v: 0 for v in ANURASET_TO_BIRDCLEF.values()}
    total_saved = 0
    matched_count = 0

    for i, wav_path in enumerate(wav_files):
        if i % 10000 == 0:
            print(f"  [{i}/{len(wav_files)}] matched={matched_count} saved={total_saved}", flush=True)

        parts = wav_path.stem.split("_")
        if len(parts) >= 3:
            file_id = "_".join(parts[:3])
        else:
            file_id = wav_path.stem

        matched_species = file_to_species.get(file_id)
        if matched_species is None:
            continue
        matched_count += 1

        # Load audio
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            continue

        # Resample and pad
        audio = resample_linear(audio, sr, TARGET_SR)
        audio = pad_to_5s(audio)

        # Save for each target species present
        for anura_code, birdclef_label in matched_species:
            if counts[birdclef_label] >= MAX_PER_SPECIES:
                continue

            sp_dir = OUT_DIR / birdclef_label
            sp_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"anuraset_{anura_code}_{wav_path.stem}.wav"
            out_path = sp_dir / out_name

            if out_path.exists():
                continue

            sf.write(str(out_path), audio, TARGET_SR, subtype="FLOAT")
            counts[birdclef_label] += 1
            total_saved += 1

    print(f"\n[DONE] Saved {total_saved} audio files")
    print("\nPer-species counts:")
    for birdclef_label, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 0:
            anura_code = [k for k, v in ANURASET_TO_BIRDCLEF.items() if v == birdclef_label][0]
            print(f"  {birdclef_label} ({anura_code}): {count}")


if __name__ == "__main__":
    main()
