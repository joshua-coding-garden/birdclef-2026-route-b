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
| 推論時間 | ~15 分鐘 (Kaggle CPU) |

## 方法摘要

```
train_audio/*.ogg ──┐
                    ├─→ Perch v2 (frozen) ─→ 1536-dim embeddings ─→ MLP Head ─→ 234 species probs
train_soundscapes/ ─┘                        (Phase 1, Kaggle GPU)              (Phase 2, local GPU)
```

**核心思路**: 用 Google Perch v2 預訓練模型作為 frozen encoder，只訓練最後的 MLP 分類頭（~1.57M 參數）。這是一個 transfer learning / linear probing 的方法。

### 主要優化

1. **5-Fold GroupKFold Ensemble** — 防止同檔案跨 fold 洩漏
2. **Embedding-level MixUp** — Beta(0.4, 0.4) 分布，提升泛化
3. **Soundscape 20x Upweighting** — 解決 soundscape 僅佔 0.3% 的類別不平衡
4. **Pseudo-labeling** — 用 5-fold teacher 對 ~127k 未標記 soundscape 生成 soft labels
5. **Temporal Smoothing** — 推論時相鄰 segment 預測平均
6. **Soundscape Dedup** — 修正官方標籤的 2x 重複問題 (1478 → 739)

### LB 提升歷程

| Version | LB Score | 改動 |
|---|---|---|
| V1 | 0.872 | Baseline 單模型 |
| V2 | 0.873 | 5-fold + MixUp + SC upweight |
| V6 | 0.875 | + Pseudo-labels |
| Final | **0.901** | Private LB |

## 專案結構

```
birdCLEF/
├── new/                          # 所有程式碼、模型、報告
│   ├── train_mlp_local.py        # Phase 2: 3-variant ablation (Linear/MLP/Focal)
│   ├── train_5fold.py            # Phase 2: 5-fold 訓練 + MixUp + pseudo-labels
│   ├── eval_domain_auc.py        # 分析 train_audio vs soundscape 各自的 AUC
│   ├── generate_pseudo_labels.py # 用 5-fold ensemble 生成 pseudo-labels
│   ├── smoke_test_submit.py      # 提交 notebook 本地煙霧測試
│   ├── train_sed.py              # 實驗: SED (Sound Event Detection) 架構
│   ├── train_hgnet_distill.py    # 實驗: HGNet 蒸餾
│   ├── train_proto_ssm.py        # 實驗: Prototypical + SSM
│   ├── download_inat.py          # 外部資料: iNaturalist 下載
│   ├── download_xc.py            # 外部資料: Xeno-canto 下載
│   ├── process_anuraset.py       # 外部資料: AnuraSet 處理
│   ├── extract_external_embeddings.py  # 外部資料 embedding 提取
│   ├── kaggle_01_extract_embeddings.ipynb  # Phase 1: Perch 嵌入提取 (Kaggle GPU)
│   ├── kaggle_02_submit.ipynb              # Phase 3: 提交 notebook (Kaggle CPU)
│   ├── kaggle_03_ensemble_submit.ipynb     # Phase 3: ensemble 提交
│   ├── kaggle_03_extract_unlabeled_sc.py   # Kaggle GPU: 未標記 soundscape 嵌入
│   ├── mlp_best.pt               # Ablation 最佳模型 (Variant B)
│   ├── mlp_fold0.pt ~ fold4.pt  # 5-fold ensemble 模型權重
│   ├── REPORT.md                 # 完整進度報告
│   ├── EXPERIMENT_LOG.md         # 所有實驗紀錄 (含死路)
│   ├── SPEAKER_SCRIPT.md         # 簡報講稿
│   ├── PHASE3_SUBMIT_GUIDE.md    # Phase 3 提交指南
│   ├── kaggle_討論整理.md         # Kaggle 論壇優化方向整理
│   ├── architecture*.svg         # 架構圖 (多種詳細度)
│   ├── *_results.json            # 實驗結果
│   └── *_curves.png              # 訓練曲線圖
├── CLAUDE.md                     # Claude Code 專案指引
└── .gitignore
```

## 環境部署

### 前置需求

- **Python 3.10+**
- **NVIDIA GPU** (CUDA 12.8+) — 訓練用，RTX 30/40/50 系列皆可
- **Kaggle 帳號** — Phase 1 embedding 提取 & 提交需要

### Step 1: 下載比賽資料

```bash
# 安裝 Kaggle CLI
pip install kaggle

# 設定 API key (從 kaggle.com → Settings → API → Create New Token)
# 把 kaggle.json 放到 ~/.kaggle/

# 下載比賽資料 (~16 GB)
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d birdclef-2026/
```

### Step 2: 建立 PyTorch 環境

```bash
# 建立 venv
python -m venv .venv-torch

# 啟動 (Windows)
.venv-torch\Scripts\activate

# 安裝 PyTorch + CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 安裝其他依賴
pip install numpy pandas scikit-learn matplotlib tqdm soundfile pyarrow
```

### Step 3: Phase 1 — 提取 Perch v2 Embeddings (Kaggle GPU)

> **注意**: Perch v2 需要 TensorFlow 2.20，無法在 Windows 本地跑。必須在 Kaggle GPU 上執行。

1. 到 Kaggle 新建一個 **GPU Notebook**
2. 打開 Internet
3. 上傳 `new/kaggle_01_extract_embeddings.ipynb` 的內容
4. 第一個 cell 跑 `!pip install -q -U "tensorflow==2.20.0" perch-hoplite`
5. **重啟 Kernel** (Kernel → Restart & Clear All Outputs)
6. 跑剩下的 cells
7. Save Version，從 Output 建立 Dataset
8. 下載 5 個檔案到 `embedding/` 目錄:
   - `train_audio_embeddings.npy` (233101, 1536)
   - `train_audio_labels.npy` (233101, 234)
   - `train_audio_meta.parquet`
   - `soundscape_embeddings.npy` (1478, 1536)
   - `soundscape_labels.npy` (1478, 234)

或者直接用已快取的 output:
```bash
kaggle kernels output ejoshfu/google-bird -p ./embedding/
```

### Step 4: Phase 2 — 訓練 MLP Head (本地 GPU)

```bash
# 基本 3-variant ablation (~2 分鐘)
python new/train_mlp_local.py

# 5-fold 訓練 + MixUp + pseudo-labels (~12 分鐘)
python new/train_5fold.py
```

輸出:
- `new/mlp_fold0.pt` ~ `mlp_fold4.pt` — 5-fold 模型權重
- `new/fold_results.json` — 訓練結果
- `new/fold_curves.png` — 訓練曲線

### Step 5: Phase 3 — Kaggle 提交

1. 上傳 5 個 `.pt` 檔到 Kaggle Dataset
2. 在 Kaggle 新建 **CPU Notebook** (Accelerator = None, Internet OFF)
3. 掛載: 比賽資料、Perch v2 CPU model、MLP head dataset、TF 2.20 wheels dataset
4. 用 `new/kaggle_02_submit.ipynb` 的內容
5. Submit

詳細步驟見 [PHASE3_SUBMIT_GUIDE.md](new/PHASE3_SUBMIT_GUIDE.md)

### (可選) Pseudo-labeling

如果想複現 pseudo-label 流程:

1. 在 Kaggle GPU 跑 `kaggle_03_extract_unlabeled_sc.py` 提取未標記 soundscape embeddings
2. 下載 `sc_all_embeddings.npy` + `sc_all_meta.parquet` 到 `embedding/`
3. 本地跑:
```bash
python new/generate_pseudo_labels.py
```
4. 重新跑 `train_5fold.py`（會自動偵測 pseudo 資料）

## 關鍵設計決策

| 決策 | 原因 |
|---|---|
| Frozen encoder | Perch v2 在 Windows 跑不動 (TF 2.20 + Blackwell GPU 不相容)，所以只在 Kaggle GPU 跑一次 embedding 提取 |
| GroupKFold | 同一個音檔的多個 frame 必須在同一個 fold，否則會 data leakage |
| Soundscape 20x upweighting | Train_audio 有 233k rows，soundscape 只有 739 rows，不加權模型會忽略 soundscape domain |
| Embedding-level MixUp | 在 embedding 空間做 MixUp 比在 audio 空間更高效，且不需要重跑 encoder |
| Soft pseudo-labels | 保留 teacher 的不確定性，比 hard threshold 更穩定 |

## 實驗紀錄

所有嘗試過的方向（包含失敗的）都記錄在 [EXPERIMENT_LOG.md](new/EXPERIMENT_LOG.md)，包含:

- EXP-001: 5-fold + MixUp + SC upweighting + pseudo-labels
- EXP-002: ProtoSSM chain (dead end — 太複雜且不適合 frozen embedding)
- EXP-003: HGNetV2 distillation 研究
- EXP-004: SED, distillation, external data 實驗

## License

This project is for academic purposes (final-year project). Competition data is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/birdclef-2026/rules).
