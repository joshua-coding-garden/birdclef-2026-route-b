# BirdCLEF+ 2026 — Route B

**Frozen Perch v2 Encoder + Trainable MLP Head** 在巴西 Pantanal 濕地多物種聲音辨識的解法。

Kaggle 比賽: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

## 最終成績

| 指標 | 數值 |
|---|---|
| **Private LB (macro ROC-AUC)** | **0.90112** |
| 排名 | 2621 / ~2800 |
| Local Val AUC (5-fold OOF) | 0.9732 ± 0.0017 |
| 訓練時間 | ~12 分鐘 (RTX 5070) |

## 方法摘要

```
Audio (.ogg) → Perch v2 (frozen) → 1536-dim embeddings → 5-fold MLP Head → 234 species probs
               (Phase 1, Kaggle GPU)                      (Phase 2, local GPU)
```

### 主要優化

1. **5-Fold GroupKFold Ensemble** — 防止同檔案跨 fold 洩漏
2. **Embedding-level MixUp** — Beta(0.4, 0.4)，正則化
3. **Soundscape 20x Upweighting** — 解決 0.3% 的類別不平衡
4. **Pseudo-labeling** — 127k 筆未標記 soundscape soft labels
5. **Temporal Smoothing** — 推論時相鄰 segment 平均
6. **外部資料擴增** — iNaturalist、Xeno-canto、AnuraSet

### LB 提升歷程

| Version | LB Score | 改動 |
|---|---|---|
| V1 | 0.872 | Baseline 單模型 |
| V2 | 0.873 | +5-fold +MixUp +SC 加權 |
| V6 | 0.875 | +Pseudo-labels |
| Final | **0.901** | Private LB |

## 核心檔案

```
new/
├── train_5fold.py                 # 5-fold 訓練（MixUp + SC 加權 + pseudo-labels）
├── generate_pseudo_labels.py      # Pseudo-label 生成
├── kaggle_01_extract_embeddings.ipynb  # Phase 1: Perch 嵌入提取 (Kaggle GPU)
├── kaggle_02_submit.ipynb              # Phase 3: 提交 notebook (Kaggle CPU)
├── kaggle_03_extract_unlabeled_sc.py   # 未標記 soundscape 嵌入提取
├── smoke_test_submit.py           # 本地煙霧測試
├── mlp_fold0.pt ~ fold4.pt       # 5-fold 模型權重
├── fold_results.json              # 訓練結果
├── REPORT.md                      # 進度報告
├── EXPERIMENT_LOG.md              # 實驗紀錄（含失敗嘗試）
└── FINAL_PROJECT.md               # 期末專題報告
```

補充性的實驗腳本（SED、HGNet 蒸餾、ProtoSSM 等）見 [birdclef-2026-supplementary](https://github.com/joshua-coding-garden/birdclef-2026-supplementary)。

## 環境部署

### 前置需求

- Python 3.10+
- NVIDIA GPU (CUDA 12.8+)
- Kaggle 帳號

### Step 1: 下載比賽資料

```bash
pip install kaggle
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d birdclef-2026/
```

### Step 2: 建立 PyTorch 環境

```bash
python -m venv .venv-torch
.venv-torch\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn matplotlib tqdm soundfile pyarrow
```

### Step 3: Phase 1 — 提取 Perch v2 Embeddings (Kaggle GPU)

> Perch v2 需要 TF 2.20，無法在 Windows 本地跑，必須在 Kaggle GPU 執行。

1. Kaggle 新建 **GPU Notebook**，開 Internet
2. 第一個 cell: `!pip install -q -U "tensorflow==2.20.0" perch-hoplite`
3. **重啟 Kernel**
4. 跑 `kaggle_01_extract_embeddings.ipynb` 的內容
5. 從 Output 下載 5 個檔案到 `embedding/`

或直接用已快取的 output:
```bash
kaggle kernels output ejoshfu/google-bird -p ./embedding/
```

### Step 4: Phase 2 — 訓練 MLP Head (本地 GPU)

```bash
python new/train_5fold.py
```

### Step 5: Phase 3 — Kaggle 提交

1. 上傳 `mlp_fold0~4.pt` 到 Kaggle Dataset
2. Kaggle 新建 **CPU Notebook** (Accelerator=None, Internet OFF)
3. 用 `kaggle_02_submit.ipynb` 的內容
4. Submit

## License

Academic use. Competition data subject to [Kaggle rules](https://www.kaggle.com/competitions/birdclef-2026/rules).
