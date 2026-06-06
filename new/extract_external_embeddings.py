"""
BirdCLEF+ 2026 — Extract Perch v2 embeddings from external audio (local CPU).

Reads audio files from c:\birdCLEF\external\{source}\{primary_label}\*.{mp3,wav,ogg,m4a,flac}
Runs perch_v2_cpu on each 5-sec frame, outputs embeddings to c:\birdCLEF\embedding_external\

Usage (must use .venv-tf which has TensorFlow 2.20):
    set KAGGLEHUB_CACHE=C:\birdCLEF\cache\kagglehub
    C:\birdCLEF\.venv-tf\Scripts\python.exe C:\birdCLEF\new\extract_external_embeddings.py

Benchmark: batch=32 → 0.11s/clip on CPU. 10,000 clips → ~18 min.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KAGGLEHUB_CACHE"] = r"C:\birdCLEF\cache\kagglehub"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAXONOMY_CSV = Path(r"C:\birdCLEF\birdclef-2026\taxonomy.csv")
EXTERNAL_DIRS = [
    Path(r"C:\birdCLEF\external\inaturalist"),
    Path(r"C:\birdCLEF\external\xeno-canto"),
    Path(r"C:\birdCLEF\external\anuraset"),
]
OUT_DIR = Path(r"C:\birdCLEF\embedding_external")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
SAMPLE_RATE = 32000
FRAME_LEN = SAMPLE_RATE * 5  # 160000 samples = 5 seconds
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}

N_CLASSES = 234


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Load audio file, resample to target_sr, mono."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            # Simple resampling via linear interpolation
            duration = len(audio) / sr
            n_samples = int(duration * target_sr)
            if n_samples == 0:
                return None
            indices = np.linspace(0, len(audio) - 1, n_samples)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        return audio
    except Exception:
        try:
            # Fallback: try with pydub for mp3/m4a
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(path))
            seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
            samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
            return samples
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[extract] Loading taxonomy...")
    tax = pd.read_csv(TAXONOMY_CSV)
    species_list = tax["primary_label"].tolist()
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    assert len(species_list) == N_CLASSES

    # Collect all external audio files
    print("[extract] Scanning external audio directories...")
    file_list = []  # (path, primary_label, source)
    for ext_dir in EXTERNAL_DIRS:
        if not ext_dir.exists():
            print(f"  {ext_dir} — not found, skipping")
            continue
        for sp_dir in sorted(ext_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            plabel = sp_dir.name
            if plabel not in species_to_idx:
                continue
            for f in sorted(sp_dir.iterdir()):
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    file_list.append((f, plabel, ext_dir.name))

    print(f"[extract] Found {len(file_list)} audio files across {len(set(f[1] for f in file_list))} species")
    if not file_list:
        print("[extract] No files to process. Run download scripts first.")
        return

    # Load Perch model
    print("[extract] Loading perch_v2_cpu...")
    from perch_hoplite.zoo import model_configs
    t0 = time.time()
    model = model_configs.load_model_by_name("perch_v2_cpu")
    print(f"[extract] Model loaded in {time.time()-t0:.1f}s (sr={model.sample_rate})")
    assert model.sample_rate == SAMPLE_RATE

    # Warm up with batch=BATCH_SIZE
    _ = model.batch_embed(np.zeros((BATCH_SIZE, FRAME_LEN), dtype="float32"))
    print(f"[extract] Warm-up done")

    # Process files
    all_embs = []
    all_labels = []
    all_meta = []
    buf_audio = []
    buf_info = []  # (primary_label, filename, frame_idx)
    fail_count = 0
    t_start = time.time()

    def flush_buffer():
        if not buf_audio:
            return
        batch = np.stack(buf_audio, axis=0).astype("float32")
        out = model.batch_embed(batch)
        embs = out.embeddings[:, 0, 0, :].astype("float32")  # (B, 1536)
        # Build labels
        for i, (plabel, fname, fidx) in enumerate(buf_info):
            label = np.zeros(N_CLASSES, dtype="float32")
            label[species_to_idx[plabel]] = 1.0
            all_labels.append(label)
            all_meta.append((plabel, fname, fidx))
        all_embs.append(embs)
        buf_audio.clear()
        buf_info.clear()

    for file_idx, (fpath, plabel, source) in enumerate(file_list):
        if file_idx % 100 == 0 and file_idx > 0:
            elapsed = time.time() - t_start
            n_frames = sum(e.shape[0] for e in all_embs)
            fps = n_frames / elapsed if elapsed > 0 else 0
            eta_min = (len(file_list) - file_idx) / max(file_idx / elapsed, 0.01) / 60
            print(f"  [{file_idx}/{len(file_list)}] frames={n_frames}, "
                  f"speed={fps:.1f} frames/s, ETA={eta_min:.1f} min, fails={fail_count}")

        audio = load_audio(fpath, SAMPLE_RATE)
        if audio is None:
            fail_count += 1
            continue

        # Frame to 5-sec chunks
        if len(audio) < FRAME_LEN:
            audio = np.pad(audio, (0, FRAME_LEN - len(audio)))
            n_frames = 1
        else:
            n_frames = len(audio) // FRAME_LEN

        fname = fpath.name
        for f_idx in range(n_frames):
            frame = audio[f_idx * FRAME_LEN:(f_idx + 1) * FRAME_LEN]
            buf_audio.append(frame)
            buf_info.append((plabel, fname, f_idx))
            if len(buf_audio) >= BATCH_SIZE:
                flush_buffer()

    # Flush remaining
    flush_buffer()

    if not all_embs:
        print("[extract] No embeddings produced!")
        return

    # Concatenate and save
    X_ext = np.concatenate(all_embs, axis=0)
    Y_ext = np.stack(all_labels, axis=0)
    meta_df = pd.DataFrame(all_meta, columns=["primary_label", "filename", "frame_idx"])

    print(f"\n[extract] Done!")
    print(f"  X_ext shape: {X_ext.shape}")
    print(f"  Y_ext shape: {Y_ext.shape}")
    print(f"  Total files: {len(file_list)}, failed: {fail_count}")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"  Species covered: {meta_df['primary_label'].nunique()}")

    np.save(OUT_DIR / "external_embeddings.npy", X_ext)
    np.save(OUT_DIR / "external_labels.npy", Y_ext)
    meta_df.to_parquet(OUT_DIR / "external_meta.parquet")

    print(f"\n[saved]")
    print(f"  {OUT_DIR / 'external_embeddings.npy'} ({X_ext.nbytes/1e6:.1f} MB)")
    print(f"  {OUT_DIR / 'external_labels.npy'} ({Y_ext.nbytes/1e6:.1f} MB)")
    print(f"  {OUT_DIR / 'external_meta.parquet'}")

    # Per-class summary
    per_sp = meta_df.groupby("primary_label").size().sort_values(ascending=False)
    print(f"\n[per-species] top 10:")
    for sp, cnt in per_sp.head(10).items():
        print(f"  {sp}: {cnt} frames")
    print(f"[per-species] bottom 10:")
    for sp, cnt in per_sp.tail(10).items():
        print(f"  {sp}: {cnt} frames")


if __name__ == "__main__":
    main()
