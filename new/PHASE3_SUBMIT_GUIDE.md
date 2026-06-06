# Phase 3 上傳 & 提交操作指南

> 這份指南是針對本機已有 `new/kaggle_02_submit.ipynb` 與 `new/mlp_best.pt`,準備第一次上 Kaggle 提交的逐步流程。第二次之後只要重跑 Step 6–7 即可。

---

## 整體流程總覽

```
[一次性備料]
  Step 1. 在 Kaggle 開一個「備料 notebook」(有網路的 GPU / CPU notebook)
  Step 2. 下載 perch-hoplite wheels + Perch v2 CPU 模型 → 存成 Kaggle Dataset
  Step 3. 上傳 mlp_best.pt → 存成 Kaggle Dataset

[真正提交]
  Step 4. 在 Kaggle 新建「提交 notebook」,貼上 kaggle_02_submit.ipynb 內容
  Step 5. 四個 Input 掛上去 (Competition + 3 個自己的 Dataset/Model)
  Step 6. Save & Run All (本地 dry-run 確認能跑完)
  Step 7. Submit to Competition → 等評分
```

---

## Step 1 — 開一個「備料 notebook」

1. 登入 <https://www.kaggle.com/>
2. 右上角 `+ Create` → **New Notebook**
3. 右側 **Notebook settings**:
   - **Accelerator**: `None` 就好 (這個 notebook 只是下載東西,不跑模型)
   - **Internet**: **ON**
   - **Persistence**: `Files only`
4. 命名(例:`birdclef-prep-deps`)

---

## Step 2 — 下載 Perch + perch-hoplite,打包成 Dataset

### 2.1 先檢查 Perch v2 是否已經在 Kaggle 官方 Model 上

**重要**: Perch v2 是 Google 的官方模型,**應該**已經可以在 Kaggle 當 Kaggle Model 直接掛載,不需要自己重傳。做法:

1. 在備料 notebook 右側 `Add Input` → 切到 **Models** 頁籤
2. 搜尋 `perch` 或 `bird-vocalization-classifier`
3. 若能找到 Google 官方的 `google/bird-vocalization-classifier` → 直接選 **TensorFlow2 / perch_v2_cpu** → `Add`
4. 掛載後路徑會長像 `/kaggle/input/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1/` 之類

**如果找得到官方 Model**:可以跳過 2.2 的 Perch 下載步驟,只要做 2.3 (wheels)。在真正的提交 notebook 裡再把 `KAGGLEHUB_CACHE` 指到正確的路徑結構,或直接改成從 SavedModel 路徑載入 (我到時再告訴你怎麼改)。

**如果找不到**:繼續 2.2 自己下載上傳。

### 2.2 (Fallback) 在備料 notebook 下載 Perch v2 CPU

貼進 Cell 1:

```python
!pip install -q -U "tensorflow==2.20.0" perch-hoplite
```

跑完後 **Kernel → Restart**。

Cell 2:

```python
import os
os.environ["KAGGLEHUB_CACHE"] = "/kaggle/working/kagglehub"

from perch_hoplite.zoo import model_configs

# 兩個變體都下載(我們要的是 cpu 版,但兩個都抓起來備用)
_ = model_configs.load_model_by_name("perch_v2_cpu")

# 看一下檔案結構
!find /kaggle/working/kagglehub -maxdepth 6 -type d
```

這會把 SavedModel 下載到 `/kaggle/working/kagglehub/...` 底下,大約 400 MB–1 GB。

### 2.3 下載 perch-hoplite wheels (離線安裝用)

Cell 3:

```python
!mkdir -p /kaggle/working/wheels
!pip download perch-hoplite tensorflow==2.20.0 \
    -d /kaggle/working/wheels \
    --only-binary=:all: \
    --python-version 310 \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --abi cp310

!ls -lh /kaggle/working/wheels/ | head -20
```

> ⚠️ Kaggle 本身是 Python 3.10 + Linux x86_64,所以上面那組參數一般能抓到對的 wheel。若報錯說找不到某套件,把 `--platform` 那行拿掉再試一次(讓 pip 自動偵測)。

### 2.4 把下載結果存成 Kaggle Dataset

1. 在備料 notebook 右上角 **Save Version** → **Save & Run All (Commit)**
2. 等 commit 跑完 (應該只要 2–5 分鐘,因為只是下載東西)
3. Commit 完後,點 notebook 頁面的 **Output** 頁籤 → 右上 **New Dataset**
4. 命名(例:`birdclef-phase3-deps`),按 Create

現在你在 Kaggle 上有一個 Dataset,裡面包含:
- `/kagglehub/...` (Perch model,若 Step 2.1 找不到官方 Model 才有)
- `/wheels/` (perch-hoplite + TF 的離線 wheel 檔)

---

## Step 3 — 上傳 mlp_best.pt

這個最簡單,檔案只有 6 MB。

1. 右上 `+ Create` → **New Dataset**
2. 拖 `C:\birdCLEF\new\mlp_best.pt` 進去
3. 命名(例:`birdclef-mlp-head`)
4. Visibility: Private(不想公開的話)
5. Create

---

## Step 4 — 新建「提交 notebook」

1. 進到比賽頁面 <https://www.kaggle.com/competitions/birdclef-2026>
2. 左側選單 **Code** → **+ New Notebook**(這樣比賽資料會自動被當 Input 掛上)
3. 打開 notebook 後,刪掉預設的 Cell,改成「File → Import Notebook」上傳 `C:\birdCLEF\new\kaggle_02_submit.ipynb`
   - 或直接把每個 cell 的內容手動貼進去
4. **Notebook settings** (右側):
   - **Accelerator**: `None` (必須是 CPU,這是比賽規則)
   - **Internet**: **OFF** (提交時一定要關)
   - **Persistence**: `Files only`

---

## Step 5 — 掛 4 個 Input

在提交 notebook 的右側 **Add Input**,按順序加入:

1. **Competitions** → `BirdCLEF+ 2026` (一般已自動加了)
   - 掛載路徑: `/kaggle/input/birdclef-2026/`
2. **Models** 或 **Datasets** → 你的 Perch v2 CPU 來源
   - 若用 Step 2.1 官方 Model: `/kaggle/input/bird-vocalization-classifier/...`
   - 若用 Step 2.2 自己打包的 Dataset: `/kaggle/input/birdclef-phase3-deps/kagglehub/...`
3. **Datasets** → 你的 wheels Dataset
   - 掛載路徑: `/kaggle/input/birdclef-phase3-deps/wheels/` (若跟 Perch 打包在一起) 或自己獨立的路徑
4. **Datasets** → `birdclef-mlp-head`
   - 掛載路徑: `/kaggle/input/birdclef-mlp-head/mlp_best.pt`

### 若路徑與 notebook 預設不同 → 改 Cell 2 的常數

在 Cell 2 找到這幾行,照掛載後的實際路徑改:

```python
DATA_ROOT  = Path("/kaggle/input/birdclef-2026")
PERCH_DIR  = Path("/kaggle/input/perch-v2-cpu")      # ← 你的 Perch 真實路徑
HEAD_CKPT  = Path("/kaggle/input/birdclef-mlp-head/mlp_best.pt")
```

Cell 1 的 pip install 那行也要指向正確的 wheels 路徑:

```python
!pip install -q --no-index --find-links /kaggle/input/perch-hoplite-wheels perch-hoplite
#                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 改成你的實際路徑
```

---

## Step 6 — Save & Run All(本地 dry-run)

**提交前一定要跑一次本地版本確認沒錯,不然直接 Submit 如果出錯會浪費額度。**

1. 右上 **Save Version** → 選 **Save & Run All (Commit)**
2. 等 commit 跑完 (應該 5–15 分鐘)
3. 跑完後去看 **Output**:
   - 應該有 `submission.csv`
   - shape 會是 `(0, 235)` 或 `(N, 235)` — 因為測試階段 `test_soundscapes/` 只有 `readme.txt`,所以不會有實際預測資料
   - 但只要 cell 1–10 都跑完沒報錯,邏輯就是對的

4. **會看到 Cell 10 的 assert 失敗**(因為 0 筆 test file × 12 ≠ len(rows) 通常沒問題,但 `vals.min()` 對空 array 會出錯) — 這個我在 notebook 裡會額外處理。若失敗,告訴我看到的錯誤訊息。

---

## Step 7 — Submit to Competition

1. commit 成功後,點 notebook 頁面右上的 **Submit to Competition** 按鈕
2. 選剛剛的 Version
3. 填一個描述(例: `variant B mlp_best.pt, first submission`)
4. **Submit**

現在 Kaggle 後台會:
- 把真實的 ~600 筆測試音檔倒進 `/kaggle/input/birdclef-2026/test_soundscapes/`
- 重跑你的 notebook(CPU、90 分鐘內)
- 讀 `/kaggle/working/submission.csv`
- 跟他們的答案算 macro ROC-AUC → 給你 Public LB 分數

等 5–10 分鐘後刷新,就能看到分數出現在 <https://www.kaggle.com/competitions/birdclef-2026/leaderboard>。

---

## 之後每次提交的流程(簡化版)

**90% 的情況**(只是重跑、或小改 notebook 邏輯):

1. 打開提交 notebook
2. 改想改的 cell
3. Save Version (Save & Run All)
4. Submit to Competition

**10% 的情況**(本機重訓練出新 `mlp_best.pt`):

1. 打開 `birdclef-mlp-head` Dataset
2. **New Version** → 拖新的 `mlp_best.pt` → Save
3. 回到提交 notebook,等 Dataset 更新(通常幾秒)
4. Save Version → Submit

---

## 常見問題

**Q: Kaggle 找不到我上傳的 Dataset?**
A: Dataset 上傳後要等它處理幾分鐘。在 <https://www.kaggle.com/你的帳號/datasets> 確認狀態是 `Ready`。

**Q: 提交後分數是 0 或 NaN?**
A: 大概率是 `submission.csv` 的 row_id 或欄位順序跟 sample_submission 對不上。重新跑一次 Cell 10 的 sanity check。

**Q: Run All 在 Cell 1 (pip install) 就卡住?**
A: 檢查 `--find-links` 路徑是否存在,`!ls /kaggle/input/perch-hoplite-wheels/` 看看是否真的有 .whl 檔。

**Q: Cell 6 載入 Perch 模型時報 `NotFoundError: ... platforms: [CPU]`?**
A: 你載到 `perch_v2_gpu` 不是 `perch_v2_cpu`。重下載一次(Step 2.2),確認名稱對。

**Q: 90 分鐘跑不完?**
A: Phase 1 GPU 跑 35k 檔花 47 分;CPU 對 600 檔估算下來大約 10–20 分,應該有 5–9 倍餘裕。若意外超時,告訴我實際秒數我再看哪裡可以優化。
