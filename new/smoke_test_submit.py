"""
BirdCLEF+ 2026 — Phase 3 submission notebook smoke test (no Perch).

Perch v2 can't run locally on Windows (CLAUDE.md Constraint 1), so this script
mocks the encoder with a deterministic random linear projection and verifies
EVERY OTHER piece of the submission pipeline:
  - MLPHead loads mlp_best.pt state_dict cleanly
  - taxonomy species order matches sample_submission columns
  - framing: 60-sec audio -> 12 frames of 160000 samples
  - batched buffer logic: correct n_valid tracking at the partial-batch tail
  - row_id format: f"{stem}_{end_sec}" with end_sec in {5,10,...,60}
  - final submission.csv shape, column order, row count, prob range

Uses `train_soundscapes/*.ogg` as stand-ins for test_soundscapes (same format,
1-min files). Writes an output CSV to new/_smoke_submission.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

# --- paths --------------------------------------------------------------------
ROOT = Path(r"C:\birdCLEF")
DATA = ROOT / "birdclef-2026"
NEW  = ROOT / "new"
HEAD_DIR  = NEW              # fold weights live in new/
OUT_CSV   = NEW / "_smoke_submission.csv"
N_FOLDS   = 5

# --- constants (must match kaggle_02_submit.ipynb) ----------------------------
SR        = 32000
FRAME_SEC = 5
FRAME_LEN = SR * FRAME_SEC
BATCH     = 32
N_CLASSES = 234
EMB_DIM   = 1536
TEST_LEN_SEC  = 60
SEGS_PER_FILE = TEST_LEN_SEC // FRAME_SEC  # 12


# --- MLPHead (copied verbatim from train_mlp_local.py) -----------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim=EMB_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --- Perch stub: deterministic linear projection ------------------------------
class PerchStub:
    """Stands in for perch_v2_cpu. Maps (B, 160000) float32 audio -> (B, 1536)."""
    def __init__(self, emb_dim=EMB_DIM, frame_len=FRAME_LEN, seed=0):
        rng = np.random.default_rng(seed)
        # Small deterministic projection; values won't match real Perch but
        # shape and dtype are what the submission pipeline cares about.
        self.W = rng.standard_normal((frame_len, emb_dim)).astype("float32") * 1e-3

    def batch_embed(self, batch):
        # Match Perch API: returns object with .embeddings of shape (B, 1, 1, D)
        B = batch.shape[0]
        emb = batch @ self.W  # (B, D)
        out = type("Out", (), {})()
        out.embeddings = emb.reshape(B, 1, 1, EMB_DIM).astype("float32")
        return out


# --- framing helpers (copied from notebook) ----------------------------------
def load_audio(fp):
    audio, sr = sf.read(str(fp), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"expected SR={SR} got {sr} for {fp}"
    return audio


def frame_audio(audio):
    if len(audio) < FRAME_LEN:
        return [np.pad(audio, (0, FRAME_LEN - len(audio)))]
    n = len(audio) // FRAME_LEN
    return [audio[i * FRAME_LEN:(i + 1) * FRAME_LEN] for i in range(n)]


# --- main ---------------------------------------------------------------------
def main():
    print("[1/6] loading taxonomy + sample_submission")
    taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
    species_cols = taxonomy_df["primary_label"].tolist()
    assert len(species_cols) == N_CLASSES, len(species_cols)

    sample = pd.read_csv(DATA / "sample_submission.csv", nrows=1)
    sample_species = [c for c in sample.columns if c != "row_id"]
    assert set(sample_species) == set(species_cols), (
        "species_cols and sample_submission.csv columns disagree"
    )
    print(f"       species classes = {len(species_cols)}")
    print(f"       sample_submission col ordering differs from taxonomy? "
          f"{sample_species != species_cols}")

    print(f"[2/6] loading {N_FOLDS}-fold MLPHead ensemble")
    heads = []
    for i in range(N_FOLDS):
        h = MLPHead()
        state = torch.load(HEAD_DIR / f"mlp_fold{i}.pt", map_location="cpu")
        h.load_state_dict(state)
        h.eval()
        heads.append(h)
    n_params = sum(p.numel() for p in heads[0].parameters())
    print(f"       {N_FOLDS} heads loaded, {n_params:,} params each")
    assert 1_500_000 < n_params < 1_600_000, f"unexpected head size: {n_params}"

    print("[3/6] picking 3 train_soundscape files as fake test set")
    train_ss_dir = DATA / "train_soundscapes"
    test_files = sorted(train_ss_dir.glob("*.ogg"))[:3]
    assert len(test_files) == 3, f"only found {len(test_files)} ogg files"
    print(f"       using {[p.name for p in test_files]}")

    encoder = PerchStub(seed=0)

    print("[4/6] running framing + batched inference")
    rows = []
    all_probs = []
    buf_audio, buf_rows = [], []

    def flush():
        if not buf_audio:
            return
        n_valid = len(buf_audio)
        if n_valid < BATCH:
            pad = [np.zeros(FRAME_LEN, dtype="float32")] * (BATCH - n_valid)
            batch = np.stack(buf_audio + pad, axis=0).astype("float32")
        else:
            batch = np.stack(buf_audio, axis=0).astype("float32")
        out = encoder.batch_embed(batch)
        embs = out.embeddings[:n_valid, 0, 0, :].astype("float32")
        with torch.no_grad():
            embs_t = torch.from_numpy(embs)
            probs = sum(torch.sigmoid(h(embs_t)) for h in heads) / N_FOLDS
            probs = probs.numpy()
        all_probs.append(probs)
        rows.extend(buf_rows)
        buf_audio.clear()
        buf_rows.clear()

    t0 = time.time()
    for fp in test_files:
        stem = fp.stem
        audio = load_audio(fp)
        frames = frame_audio(audio)
        print(f"       {stem}: audio_sec={len(audio)/SR:.1f} n_frames={len(frames)}")
        for k, f in enumerate(frames[:SEGS_PER_FILE]):
            end_sec = (k + 1) * FRAME_SEC
            buf_audio.append(f)
            buf_rows.append((stem, end_sec))
            if len(buf_audio) >= BATCH:
                flush()
        if len(frames) < SEGS_PER_FILE:
            for k in range(len(frames), SEGS_PER_FILE):
                end_sec = (k + 1) * FRAME_SEC
                buf_audio.append(np.zeros(FRAME_LEN, dtype="float32"))
                buf_rows.append((stem, end_sec))
                if len(buf_audio) >= BATCH:
                    flush()
    flush()
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, N_CLASSES))
    print(f"       inference took {time.time()-t0:.2f}s  probs={probs_arr.shape}")

    print("[5/6] writing CSV")
    expected_rows = len(test_files) * SEGS_PER_FILE
    assert len(rows) == probs_arr.shape[0] == expected_rows, (
        f"row count mismatch: rows={len(rows)} probs={probs_arr.shape[0]} "
        f"expected={expected_rows}"
    )
    row_ids = [f"{stem}_{end_sec}" for stem, end_sec in rows]
    df = pd.DataFrame(probs_arr, columns=species_cols)
    df.insert(0, "row_id", row_ids)
    df = df[["row_id"] + sample_species]  # align with sample_submission order
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"       wrote {OUT_CSV}  shape={df.shape}")

    print("[6/6] sanity checks")
    check = pd.read_csv(OUT_CSV)
    prob_cols = [c for c in check.columns if c != "row_id"]
    vals = check[prob_cols].to_numpy()
    assert check.shape == (expected_rows, 1 + N_CLASSES), check.shape
    assert vals.min() >= 0.0 and vals.max() <= 1.0, (vals.min(), vals.max())
    # row_ids follow stem_N pattern with N in {5,10,...,60}
    end_secs = [int(rid.rsplit("_", 1)[1]) for rid in check["row_id"]]
    assert set(end_secs) == set(range(5, 65, 5)), sorted(set(end_secs))
    # each file contributes exactly 12 rows
    stems = [rid.rsplit("_", 1)[0] for rid in check["row_id"]]
    stem_counts = pd.Series(stems).value_counts()
    assert (stem_counts == SEGS_PER_FILE).all(), stem_counts.to_dict()
    # prob cols match sample_submission exactly
    assert prob_cols == sample_species, "column order mismatch"

    print(f"       shape           : {check.shape}")
    print(f"       prob min/max    : {vals.min():.6f} / {vals.max():.6f}")
    print(f"       prob mean       : {vals.mean():.6f}")
    print(f"       row_id examples : {check['row_id'].head(3).tolist()}")
    print(f"       per-file rows   : {stem_counts.iloc[0]} (expected {SEGS_PER_FILE})")
    print("\n[OK] smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n[FAIL] assertion: {e}", file=sys.stderr)
        sys.exit(1)
