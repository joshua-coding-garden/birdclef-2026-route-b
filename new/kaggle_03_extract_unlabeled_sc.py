"""
Kaggle GPU notebook 程式碼 — 提取未標記 train_soundscapes 的 Perch v2 embedding。

使用方法：
  1. 在 Kaggle 開一個 GPU notebook（Internet ON）
  2. Cell 1: pip install
  3. 重啟 kernel
  4. Cell 2: 貼下面的程式碼，跑完
  5. Save Version → 從 Output 建立 Dataset 下載

預計 GPU 時間：~30 分鐘（10,592 檔 × 12 frames = ~127k embeddings）
"""

# ======================================================================
# 以下是要貼到 Kaggle notebook 的程式碼
# ======================================================================

# ---- Cell 1 ----
# !pip install -q -U "tensorflow==2.20.0" perch-hoplite
# ↑ 跑完後 Kernel → Restart，再從 Cell 2 開始

# ---- Cell 2 ----
CELL_2_CODE = """
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm
import tensorflow as tf

print("TF:", tf.__version__)
assert tf.__version__.startswith("2.20")

from perch_hoplite.zoo import model_configs

# === 路徑 ===
DATA_ROOT = Path("/kaggle/input/birdclef-2026")
SC_DIR    = DATA_ROOT / "train_soundscapes"
LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
OUT_DIR   = Path("/kaggle/working")

# === 常數 ===
SR = 32000
FRAME_LEN = SR * 5   # 160000
BATCH = 32

# === 找出所有 soundscape 檔案 ===
all_ogg = sorted(SC_DIR.glob("*.ogg"))
print(f"total soundscape files: {len(all_ogg)}")

# === 載入 Perch v2 GPU ===
t0 = time.time()
model = model_configs.load_model_by_name("perch_v2_gpu")
print(f"Perch loaded in {time.time()-t0:.1f}s, SR={model.sample_rate}")
assert model.sample_rate == SR

# warm-up
_ = model.batch_embed(np.zeros((BATCH, FRAME_LEN), dtype="float32"))
print("warm-up done")

# === 輔助函式 ===
def load_audio(fp):
    audio, sr = sf.read(str(fp), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        print(f"WARNING: {fp.name} sr={sr}, expected {SR}")
    return audio

def frame_audio(audio):
    if len(audio) < FRAME_LEN:
        return [np.pad(audio, (0, FRAME_LEN - len(audio)))]
    n = len(audio) // FRAME_LEN
    return [audio[i * FRAME_LEN:(i + 1) * FRAME_LEN] for i in range(n)]

# === 批次提取 embedding ===
all_embs = []      # list of (n_frames, 1536)
meta_rows = []     # list of dict

buf = []           # audio frames buffer
buf_meta = []      # (filename, frame_idx) buffer

def flush():
    if not buf:
        return
    n_valid = len(buf)
    if n_valid < BATCH:
        pad = [np.zeros(FRAME_LEN, dtype="float32")] * (BATCH - n_valid)
        batch_arr = np.stack(buf + pad, axis=0).astype("float32")
    else:
        batch_arr = np.stack(buf, axis=0).astype("float32")
    out = model.batch_embed(batch_arr)
    embs = out.embeddings[:n_valid, 0, 0, :].astype("float32")
    all_embs.append(embs)
    meta_rows.extend(buf_meta)
    buf.clear()
    buf_meta.clear()

t0 = time.time()
skipped = 0
for fi, fp in enumerate(tqdm(all_ogg, desc="extracting")):
    try:
        audio = load_audio(fp)
    except Exception as e:
        print(f"[skip] {fp.name}: {e}")
        skipped += 1
        continue
    frames = frame_audio(audio)
    for k, f in enumerate(frames):
        buf.append(f)
        buf_meta.append({"filename": fp.name, "frame_idx": k})
        if len(buf) >= BATCH:
            flush()
flush()

elapsed = time.time() - t0
embeddings = np.concatenate(all_embs, axis=0)
meta_df = pd.DataFrame(meta_rows)

print(f"\\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"embeddings: {embeddings.shape}")
print(f"meta rows: {len(meta_df)}")
print(f"skipped: {skipped}")

# === 儲存 ===
np.save(OUT_DIR / "sc_all_embeddings.npy", embeddings)
meta_df.to_parquet(OUT_DIR / "sc_all_meta.parquet", index=False)
print(f"saved {OUT_DIR / 'sc_all_embeddings.npy'} ({embeddings.nbytes / 1e9:.2f} GB)")
print(f"saved {OUT_DIR / 'sc_all_meta.parquet'}")

# === 驗證 ===
assert embeddings.shape[1] == 1536
assert len(meta_df) == embeddings.shape[0]
print(f"unique files: {meta_df['filename'].nunique()}")
print(f"frames per file (first 5): {meta_df.groupby('filename').size().head().tolist()}")
print("\\n✓ all done")
"""

print("="*60)
print("把上面 CELL_2_CODE 的內容貼到 Kaggle GPU notebook")
print("="*60)
print(CELL_2_CODE)
