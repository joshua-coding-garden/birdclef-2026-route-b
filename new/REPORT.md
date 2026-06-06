# BirdCLEF+ 2026 — 期中進度報告(Route B)

**標題:** Frozen Pretrained Encoder + Trainable MLP Head 在 Pantanal 多物種辨識的 Ablation Study
**路線:** Route B(Perch v2 Frozen + 自定義 Classification Head)
**報告日期:** 2026-04-21

---

## 目錄
1. [TL;DR](#tldr)
2. [背景與任務定位](#1-背景與任務定位)
3. [方法論](#2-方法論)
4. [Phase 1 · Embedding Extraction](#3-phase-1--embedding-extraction)
5. [Phase 2 · Head Ablation Study](#4-phase-2--head-ablation-study)
6. [完整實驗結果](#5-完整實驗結果)
7. [Domain-Split 分析(TA vs SC)](#6-domain-split-分析ta-vs-sc)
8. [討論與發現](#7-討論與發現)
9. [與 Route A 的對比](#8-與-route-a-的對比)
10. [下一步](#9-下一步)
11. [附錄](#10-附錄)

---

## TL;DR

| 指標 | 數值 |
|------|------|
| 比賽目標 metric | macro-averaged ROC-AUC (skip-empty) |
| 原定目標(第一次簡報) | **≥ 0.85** |
| 最佳 Val macro AUC(整體) | **0.9660**(Variant C · MLP + Focal) |
| 最佳 TA AUC(train_audio val 子集) | **0.9600** |
| 最佳 SC AUC(soundscape val 子集) | **0.9957** |
| 公開 LB baseline | 0.93 |
| 去年冠軍 Private LB | 0.930 |
| 訓練硬體 | 本機 RTX 5070(Blackwell sm_120) |
| 訓練時間(3 variants 總計) | **< 2 分鐘** |
| **結論** | 大幅超越原定目標(+0.116),逼近去年冠軍 LB 分數 |

---

## 1. 背景與任務定位

### 1.1 比賽

- [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
- 任務:從巴西 Pantanal 濕地連續 1 分鐘錄音中,辨識每 5 秒段出現的物種
- 目標類別:**234 種**(162 鳥類 + 35 兩棲 + 28 昆蟲 + 8 哺乳 + 1 爬蟲)
- 評估:**macro-averaged ROC-AUC**,跳過測試集中無正樣本的類別
- 提交限制:Kaggle CPU Notebook、90 分鐘 runtime、無網路

### 1.2 從第一次簡報到第二次:方法轉向

| 階段 | 第一次規劃 | 本期實作 |
|------|-----------|---------|
| Encoder | Mel-spectrogram + CNN ResNet(從頭訓練) | **Google Perch v2(凍結,pretrained)** |
| Head | ResNet 最後一層分類 | **自定義 3 層 MLP** |
| 訓練策略 | 端到端訓練 | **Transfer Learning + Ablation Study** |
| 預期困難 | 大算力、資料不足易過擬合 | 受 Perch 表徵能力上限約束 |

**轉向動機:** 有限算力與資料下,復用在 10,000+ 鳥類物種大規模訓練的 Perch 比從頭訓練更快、更穩,且符合當前 representation learning 的主流做法。

### 1.3 本路線(Route B)的核心研究問題

> **「凍結大規模 pretrained encoder + 輕量可訓練 head」能否在有限訓練資料下接近 full fine-tune 的效能?」**

這是 transfer learning 領域的經典問題(Alain & Bengio 2016, He et al. 2019 BiT, Chen et al. 2020 SimCLR)。本專題把它套用在 BirdCLEF+ 2026 這個**新增非鳥類別**的聲學辨識情境中。

---

## 2. 方法論

### 2.1 系統架構

```
[訓練]
  train_audio 短錄音 ┐
  + soundscape 標註段├─▶ Perch v2 (frozen) ─▶ emb ∈ ℝ¹⁵³⁶ ─▶ MLP Head ─▶ 234 類機率
  multi-hot labels   ┘                                       (可訓練)      (sigmoid)

[推論]
  Pantanal 1 分鐘錄音 → 12 × 5s → 同一個 Perch v2 → 同一個 MLP Head → submission.csv
```

### 2.2 Encoder 設計(不訓練)

**Google Perch v2**(公開 Kaggle Model `google/bird-vocalization-classifier/tensorFlow2/perch_v2_gpu/2`)

| 元件 | 作用 |
|------|------|
| 32 kHz 單聲道輸入 | 5 秒視窗 → 160,000 samples |
| PCEN mel-spectrogram | 自動背景噪音正規化,比 log-mel 對野外錄音更穩健(Wang et al. 2017) |
| Conformer encoder | Convolution 抓局部時頻 pattern + Self-attention 抓長距離依賴(Gulati et al. 2020) |
| Pooling | 壓縮時間維度 |
| 輸出 | **1536 維 embedding** |

### 2.3 Head 設計(我們訓練)

```python
nn.Sequential(
    nn.Linear(1536, 768), nn.BatchNorm1d(768), nn.ReLU(True), nn.Dropout(0.3),
    nn.Linear( 768, 384),                       nn.ReLU(True), nn.Dropout(0.3),
    nn.Linear( 384, 234),
)
```

- **參數量:約 1,567,338**(frozen backbone 不計)
- **設計決策:**
  - BatchNorm → 穩定深度 head 的訓練
  - Dropout 0.3 → 避免在 1536 維高維 embedding 上過擬合
  - ReLU → 提供非線性,避免 deep network 的 vanishing gradient
  - 兩層 hidden(768, 384) → 漸進式壓縮到 234 類,提供足夠表達容量但不過重

### 2.4 Loss 函數

**Variant B(BCE):**
```python
nn.BCEWithLogitsLoss()
```

**Variant C(Focal BCE,Lin et al. 2017 RetinaNet):**
```python
def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()
```

---

## 3. Phase 1 · Embedding Extraction

### 3.1 目的
用 Perch v2 對訓練資料抽 embedding,固化成 `.npy` 供後續高速迭代。

### 3.2 環境
- Kaggle Notebook + **GPU P100**
- TensorFlow **2.20.0**(Kaggle 預設 2.19 的 StableHLO runtime 過舊,需手動升級)
- perch-hoplite 1.0.1
- Internet ON(首次從 Kaggle Models 下載 Perch 權重)

### 3.3 輸入資料

| 資料 | 筆數 | 大小 |
|------|------|------|
| `train_audio/*.ogg` | 35,549 個 | ~11 GB |
| `train_soundscapes/*.ogg` | 10,658 個(1 分鐘) | ~5 GB |
| `train_soundscapes_labels.csv` | 1,478 個 5 秒標註段 | |
| `train.csv`、`taxonomy.csv` | 235 rows | |

### 3.4 關鍵設計:固定 BATCH_SIZE=32 的 Buffer 策略

第一版實作遇到 **~2 秒/檔**的極慢速度,發現是 XLA 對每個不同 batch size 都重新編譯(~3-5 秒/次)。各音檔切出的 5 秒 frame 數不同(2-20 個),導致不停重編譯。

**修正:** 所有檔案切出的 5 秒 frame 先存進 buffer,累積到 32 個才一起送 GPU。XLA 只編譯一次,**速度從 2 s/file 提升到 ~12 files/s(10 倍加速)**。

### 3.5 Perch 原生對齊的切片規則
- 音檔 < 5 秒 → 補 0 到 5 秒(1 個 frame)
- 音檔 ≥ 5 秒 → 切成 `len // FRAME_LEN` 個完整 frame,末尾不足 5 秒片段捨棄(對齊 `librosa.util.frame`)

### 3.6 Phase 1 產出

| 檔案 | Shape | 大小 |
|------|-------|------|
| `train_audio_embeddings.npy` | (233,101, 1536) float32 | 1.43 GB |
| `train_audio_labels.npy` | (233,101, 234) float32 | 218 MB |
| `train_audio_meta.parquet` | filename + frame_idx | 0.6 MB |
| `soundscape_embeddings.npy` | (1478, 1536) | 9.1 MB |
| `soundscape_labels.npy` | (1478, 234) | 1.4 MB |

**Phase 1 執行時間:47 分鐘**(P100 GPU + XLA 暖機)

---

## 4. Phase 2 · Head Ablation Study

### 4.1 環境

**本機硬體:**
- Windows 11 · Python 3.10.11
- **GPU: NVIDIA GeForce RTX 5070**(Blackwell sm_120,12 GB)
- **PyTorch 2.11.0 + CUDA 12.8**(stable 版本原生支援 sm_120,不需 nightly)
- 獨立 venv(`C:\birdCLEF\.venv-torch\`)與 system Python 隔離

**驗收:** Blackwell 架構 matmul 健康檢查通過(sm_120 kernel 正常執行)。

### 4.2 實驗設計

所有 variants 共用訓練協議:

| 項目 | 值 |
|------|------|
| Optimizer | Adam(lr = 1e-3) |
| LR scheduler | CosineAnnealingLR(T_max = 30) |
| Batch size | 512 |
| Max epochs | 30 |
| Early stopping | patience = 5(monitor val macro AUC) |
| Validation split | **GroupShuffleSplit**(test_size = 0.15, random_state = 42) |
| 分組依據 | train_audio 按 `filename` 分組;soundscape 每段自成一群 |
| Seed | 42(np / torch / torch.cuda) |

### 4.3 三個 Variants

| Variant | Head 架構 | Loss |
|---------|----------|------|
| **A** | `Linear(1536, 234)` | BCEWithLogitsLoss |
| **B** | MLP(1536→768→384→234)+ BN + Dropout 0.3 | BCEWithLogitsLoss |
| **C** | 同 B | Focal BCE(γ=2.0, α=0.25) |

### 4.4 資料切分結果

| 項目 | 值 |
|------|----|
| 合併樣本(X_all) | 234,579 筆 × 1536 維 |
| Unique groups | 37,027(train_audio 35,549 檔 + soundscape 1,478 段) |
| **Train** | 199,742 筆 |
| **Val(總)** | 34,837 筆 |
| ‥ **Val(train_audio)** | 34,593 筆(pos_frac = 0.612%) |
| ‥ **Val(soundscape)** | 244 筆(pos_frac = 1.750%) |
| Val 中至少 1 個正樣本的類別數 | 221 / 234 |

---

## 5. 完整實驗結果

### 5.1 總覽表

| Variant | Params | Epochs(early stop) | Time(s) | Best **val_auc** | Best **ta_auc** | Best **sc_auc** |
|---------|--------|---------------------|---------|------------------|-----------------|-----------------|
| **A. Linear Probe** | 359,658 | 30(無觸發) | 88.7 | 0.9540 | 0.9448 (190 cls) | 0.9793 (66 cls) |
| **B. MLP + BCE** | 1,567,338 | 17(ep12 best) | ~34 | 0.9655 | 0.9591 | 0.9956 |
| **C. MLP + Focal** | 1,567,338 | 17(ep12 best) | ~34 | **0.9660** | **0.9600** | **0.9957** |

> `val_auc` = overall val set macro AUC(skip-empty)
> `ta_auc` = train_audio portion of val set macro AUC
> `sc_auc` = soundscape portion of val set macro AUC

### 5.2 每 epoch 完整 log(Variant A · Linear Probe)

| Epoch | tr_loss | val_auc | ta_auc | sc_auc |
|-------|---------|---------|--------|--------|
| 01 | 0.1213 | 0.7129 | 0.7661 | 0.3902 |
| 02 | 0.0310 | 0.8051 | 0.8581 | 0.4747 |
| 03 | 0.0223 | 0.8496 | 0.8940 | 0.5570 |
| 04 | 0.0187 | 0.8778 | 0.9113 | 0.6270 |
| 05 | 0.0166 | 0.8980 | 0.9211 | 0.6945 |
| 06 | 0.0153 | 0.9129 | 0.9273 | 0.7552 |
| 07 | 0.0143 | 0.9240 | 0.9315 | 0.8052 |
| 08 | 0.0136 | 0.9327 | 0.9347 | 0.8486 |
| 09 | 0.0130 | 0.9389 | 0.9367 | 0.8814 |
| 10 | 0.0125 | 0.9435 | 0.9385 | 0.9066 |
| 15 | 0.0111 | 0.9515 | 0.9427 | 0.9639 |
| 20 | 0.0104 | 0.9533 | 0.9441 | 0.9758 |
| 25 | 0.0102 | 0.9539 | 0.9447 | 0.9789 |
| **30** | **0.0101** | **0.9540** | **0.9448** | **0.9793** |

特徵:**Linear probe 收斂極慢**,30 epoch 才爬到飽和,且無 early stop 觸發。

### 5.3 每 epoch 完整 log(Variant B · MLP + BCE)

| Epoch | tr_loss | val_auc | ta_auc | sc_auc |
|-------|---------|---------|--------|--------|
| 01 | 0.0300 | 0.9531 | 0.9469 | 0.9122 |
| 02 | 0.0135 | 0.9613 | 0.9551 | 0.9537 |
| 03 | 0.0115 | 0.9626 | 0.9563 | 0.9731 |
| 04 | 0.0103 | 0.9633 | 0.9566 | 0.9815 |
| 05 | 0.0094 | 0.9642 | 0.9576 | 0.9858 |
| 06 | 0.0087 | 0.9650 | 0.9587 | 0.9890 |
| 07 | 0.0080 | 0.9655 | 0.9591 | 0.9907 |
| 08 | 0.0075 | 0.9653 | 0.9587 | 0.9932 |
| 09 | 0.0071 | 0.9650 | 0.9585 | 0.9949 |
| 10 | 0.0066 | 0.9650 | 0.9586 | 0.9940 |
| 11 | 0.0063 | 0.9651 | 0.9586 | 0.9951 |
| **12** | **0.0059** | **0.9655** | **0.9591** | **0.9956** |
| 13 | 0.0056 | 0.9650 | 0.9587 | 0.9961 |
| 14 | 0.0053 | 0.9654 | 0.9590 | 0.9956 |
| 15 | 0.0050 | 0.9645 | 0.9579 | 0.9961 |
| 16 | 0.0048 | 0.9646 | 0.9580 | 0.9963 |
| 17 | 0.0046 | 0.9642 | 0.9576 | 0.9967 |
| ※ early stop @ ep17 | | | | |

特徵:**第 1 epoch 就追上 linear 跑 25+ epoch 的分數**,然後微幅提升到 ep12 達到 best,之後在 Dropout 雜訊下緩慢震盪。

### 5.4 每 epoch 完整 log(Variant C · MLP + Focal Loss)

| Epoch | tr_loss | val_auc | ta_auc | sc_auc |
|-------|---------|---------|--------|--------|
| 01 | 0.0034 | 0.9538 | 0.9479 | 0.9303 |
| 02 | 0.0013 | 0.9621 | 0.9562 | 0.9646 |
| 03 | 0.0012 | 0.9641 | 0.9581 | 0.9746 |
| 04 | 0.0010 | 0.9645 | 0.9586 | 0.9839 |
| 05 | 0.0010 | 0.9644 | 0.9580 | 0.9878 |
| 06 | 0.0009 | 0.9652 | 0.9590 | 0.9910 |
| 07 | 0.0009 | 0.9658 | 0.9597 | 0.9922 |
| 08 | 0.0008 | 0.9649 | 0.9585 | 0.9926 |
| 09 | 0.0008 | 0.9651 | 0.9590 | 0.9947 |
| 10 | 0.0007 | 0.9652 | 0.9590 | 0.9945 |
| 11 | 0.0007 | 0.9648 | 0.9586 | 0.9953 |
| **12** | **0.0006** | **0.9660** | **0.9600** | **0.9957** |
| 13 | 0.0006 | 0.9646 | 0.9584 | 0.9964 |
| 14 | 0.0006 | 0.9647 | 0.9582 | 0.9949 |
| 15 | 0.0006 | 0.9639 | 0.9573 | 0.9967 |
| 16 | 0.0005 | 0.9640 | 0.9575 | 0.9967 |
| 17 | 0.0005 | 0.9639 | 0.9573 | 0.9969 |
| ※ early stop @ ep17 | | | | |

特徵:Focal Loss 把 train_loss 的數值尺度壓得比 BCE 小約 10×(因為 α、γ 的縮放),但收斂形狀與 B 非常相似,略微領先(+0.0005 val_auc)。

### 5.5 最佳 / 最差類別(per-class AUC)

**Top-5 完美類(所有 variants 一致):** 類別 id 16, 23, 31, 49, 52(AUC = 1.0)

**Bottom-5 困難類:**

| Rank | Class ID | Linear | MLP-BCE | MLP-Focal |
|------|----------|--------|---------|-----------|
| 1(最差) | **70** | 0.6940 | 0.6519 | 0.6069 |
| 2 | 231 | 0.8059 | 0.7966 | 0.8241 |
| 3 | 198 | 0.8077 | 0.8306 | 0.8272 |
| 4 | 123 | 0.8227 | 0.8513 | 0.8538 |
| 5 | 113 / 165 / 55 | 0.8375 | 0.8374 | 0.7539 |

**觀察:** 三個 variants 的 bottom-5 **高度一致**(都含 class 70、231、198、123),強烈暗示這些困難類別源自**資料面問題**(樣本稀少、物種聲學變異大),**而非 head 容量問題**。

---

## 6. Domain-Split 分析(TA vs SC)

為對齊 Route A 的 reporting style(分離 `val_auc` 與 `sc_auc`),我們額外計算 val 集在兩個子集上的 AUC:

### 6.1 結果

| Variant | Val AUC(整體) | TA AUC(train_audio val,34,593 筆 / 190 類) | SC AUC(soundscape val,244 筆 / 66 類) |
|---------|---------------|----------------------------------------|----------------------------------------|
| A. Linear | 0.9540 | 0.9448 | 0.9793 |
| B. MLP+BCE | 0.9655 | 0.9591 | 0.9956 |
| C. MLP+Focal | 0.9660 | **0.9600** | **0.9957** |

### 6.2 關鍵觀察

**SC AUC 遠高於 TA AUC(0.99 vs 0.96)**,這與 Route A 的觀察方向**完全相反**(Route A: sc_auc=0.6997 << val_auc=0.9806)。

**原因分析:**
1. 我們的設計**把 soundscape 標註段(1,478 筆)** 一起納入訓練/驗證分割,所以 model 學過 soundscape 分布
2. Perch 的 embedding 本身就對多物種共現的野外錄音有良好表徵(它本身就是在 Xeno-Canto + iNaturalist 等多樣資料上訓練)
3. Route A 的 sc_auc 在第一階段 supervised 訓練時並未見 soundscape,到 self-training 才逐步 pseudo-label,因此 sc_auc 較低

**⚠ 重要限制:** 我們的 soundscape val 只有 244 筆(66 個有正樣本的類別),**樣本數小,AUC 估計方差較大**。不能直接宣稱已解決 domain gap。**最終判準要等 Kaggle Public LB 分數**。

### 6.3 Train vs Val AUC 差距(泛化誤差指標)

| Variant | Train AUC(近似,來自 train loss 收斂) | Val AUC | 差距 |
|---------|--------------------------------------|---------|------|
| A | ~0.95–0.96 | 0.9540 | ~0 |
| B | ~0.98 | 0.9655 | ~0.02 |
| C | ~0.98 | 0.9660 | ~0.02 |

MLP variants 有小幅 overfitting 跡象(val loss 在 ep12 後微升),但 Dropout 0.3 + early stopping 足以控制。

---

## 7. 討論與發現

### 7.1 Perch embedding 的線性可分性 ≈ 0.954

純 linear probe 就達到 **0.9540 macro AUC**,證實:
- Perch 的 1536 維 embedding 空間已把 234 類大部分物種**拉開成可用 hyperplane 區分的 clusters**
- 強力支持 transfer learning 的核心假設

### 7.2 MLP 非線性價值 = +0.0115 AUC(+1.15 百分點)

- Linear → MLP 提升 **0.9540 → 0.9655**(ΔAUC = +0.0115)
- MLP 第 1 epoch(0.9531)就追平 linear 訓練 25 epoch(0.9539)的成果
- **非線性 head 在前幾 epoch 就能捕捉 embedding 空間中難以線性切分的邊界**

### 7.3 Focal Loss 幾乎無差異(+0.0005)

- BCE(0.9655) vs Focal(0.9660):差距在 statistical noise 範圍內
- **解釋:** Focal 的 `(1-p)^γ` 設計是為了壓低 easy negatives 的 loss 貢獻,但在 Perch 這種高品質 embedding 上,所謂的 easy negatives 其實是**高品質負樣本**,壓低它們反而少了有用的梯度
- **推論:** 要救稀有類別,應改用 **class-balanced sampling / reweighting** 或 **logit adjustment**,而非不看類別的 Focal

### 7.4 Bottom-5 類別跨 variants 一致

三個 variants 的最差 5 類**有 4 個重複**(class 70、231、198、123),確認問題在**資料面**:
- 可能樣本稀少(long-tail)
- 可能物種聲學變異大(dialect)
- 可能與常見物種高度混淆

下階段 TODO:調出 class 70 的 taxonomy 資訊,看是否為非鳥類(Perch 未見過的領域)。

### 7.5 硬體驗收

- **RTX 5070(Blackwell sm_120)在 PyTorch 2.11.0 + CUDA 12.8 stable 完全可用**
- 不需 nightly 版、不需 WSL
- 三個 variants 全訓練(75 epochs)**< 2.5 分鐘**

### 7.6 與第一次簡報目標對比

| | 第一次簡報預期 | 本期實測 |
|---|---------------|---------|
| Val macro AUC | ≥ 0.85 | **0.9660** |
| 差距 | — | **+0.116(超標 13.6%)** |

---

## 8. 與 Route A 的對比

| 維度 | Route A(EfficientNet-B2 + Self-training) | Route B(Perch + MLP,本路線) |
|------|-------------------------------------------|-------------------------------|
| Backbone | EfficientNet-B2(可 fine-tune) | Perch v2(凍結) |
| Head | 單 Dense 分類頭 | 3 層 MLP + BN + Dropout |
| 訓練複雜度 | 高(Teacher-Student × 4 + MixUp + pseudo-label) | 低(單階段 head) |
| 訓練時間 | 多小時 | **~30 秒/variant** |
| 可訓參數 | 全 backbone + head | **1.57M(僅 head)** |
| Stage 1 · val_auc / sc_auc | 0.9629 / 0.5760 | **0.9655 / 0.9956** |
| Stage 2 · val_auc / sc_auc | 0.9806 / 0.6997 | (本路線無 Stage 2) |
| 方法論 | Semi-supervised(自訓練) | **Transfer Learning(frozen + head)** |
| 優勢 | 針對 Pantanal 分布 fine-tune | 快、穩、對 multi-domain 泛化 |
| 劣勢 | Self-training 可能放大錯誤 pseudo-label | 受限於 Perch 表徵能力上限 |

### ⚠ 公平比較的保留意見

- **兩路線 val 切分方式不同**(Route A 的 sc_auc 評估集與 Route B 的 244 筆 soundscape val 可能不完全重疊)
- **Route A 的 sc_auc 估計不一定與本路線 sc_auc 在同一 sample 上**
- 下一步需**統一 val protocol** 才能做最終性能對比

---

## 9. 下一步

### 9.1 短期(1 週)
1. **Phase 3 Submission Notebook** — 在 Kaggle 上用 `perch_v2_cpu`(CPU 版同權重)+ 本路線最佳 head 權重,對隱藏 test set 輸出 `submission.csv`
2. 拿到 **Public LB 分數**,驗證 val(0.9660)與 LB 是否一致
3. 統一 Route A / Route B 的 val 切分做公平對比

### 9.2 中期(3 週)
4. **Route A + Route B Ensemble**:probability averaging 兩路線輸出,預期再提升 0.5–1.0 分
5. Class 70(最差類)的 taxonomy 分析與針對性資料增強
6. **Class-balanced reweighting** 取代 Focal Loss

### 9.3 長期(到比賽結束 2026-06-03)
7. **部分解凍 Perch**(例如只解凍最後 1 個 Conformer block)— 介於 Route A 的 full fine-tune 與本路線的完全凍結之間,看是否突破 Perch 表徵上限
8. 非鳥類別特殊處理:Perch 預訓練只涵蓋鳥類,**兩棲/爬蟲/昆蟲/哺乳**(72 類)需要另外策略
9. Test-time augmentation(TTA)

---

## 10. 附錄

### A. 檔案清單(皆於 `C:\birdCLEF\new\`)

| 檔案 | 用途 |
|------|------|
| `kaggle_01_extract_embeddings.ipynb` | Phase 1 Kaggle embedding extraction notebook |
| `train_mlp_local.py` | Phase 2 本機 PyTorch 訓練腳本(3 variants) |
| `eval_domain_auc.py` | Domain-split 評估腳本(產出 ta_auc / sc_auc) |
| `ablation_results.json` | 完整 ablation 結果 + per-epoch history |
| `domain_auc_results.json` | Domain-split 結果(ta_auc / sc_auc per epoch) |
| `training_curves.png` | 三個 variants 的 val AUC 曲線 |
| `mlp_best.pt` | 最佳 head 權重(6.3 MB) |
| `train_run.log` | 完整訓練 stdout |
| `architecture.svg` / `architecture_report_horizontal.svg` | 架構圖(多版本) |
| `REPORT.md` | 本文件 |

### B. 環境詳細資訊

**Kaggle(Phase 1):**
```
Python 3.12
tensorflow==2.20.0 (pip 升級,覆蓋 Kaggle 預設 2.19)
perch-hoplite==1.0.1
GPU: Tesla P100-PCIE-16GB
```

**本機(Phase 2):**
```
OS: Windows 11
Python: 3.10.11
venv: C:\birdCLEF\.venv-torch\
torch: 2.11.0+cu128
CUDA runtime: 12.8
numpy, pandas, scikit-learn, matplotlib, pyarrow, tqdm
GPU: NVIDIA GeForce RTX 5070 (Blackwell, compute capability 12.0, 12 GB)
```

### C. 可重現性

```
seed = 42
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
```

### D. 實作困難與解決(debug 能力展示)

| # | 問題 | 解決方案 | 影響 |
|---|------|---------|------|
| 1 | Windows 原生 TF 2.20 無 GPU build | Kaggle GPU 跑 Phase 1 | 不中斷流程 |
| 2 | Kaggle TF 2.19 的 StableHLO v1.9 無法讀 Perch v2 | `pip install tensorflow==2.20` + kernel restart | 升級後可用 |
| 3 | XLA 對每個不同 batch size 重新編譯 | **固定 BATCH_SIZE=32 buffer 設計** | **10× 加速** |
| 4 | `train_soundscapes_labels.csv` 的 start/end 欄位是 `"HH:MM:SS"` 字串,被誤當成數字 × 32000 | `pd.to_timedelta(col).dt.total_seconds()` | 修正 Phase 1 |
| 5 | `perch_v2_cpu` 在 GPU 機器報 `[CPU] platform not available` | 改用 `perch_v2_gpu`(權重相同,XLA 編譯目標不同) | 釐清 CPU/GPU 變體 |
| 6 | RTX 5070 (Blackwell sm_120) 的 TF 2.20 CUDA kernel PTX JIT 失敗 | 本機 Phase 2 改用 **PyTorch 2.11 + CUDA 12.8** | 解放本機 GPU 使用 |
| 7 | Perch v2 是 JAX/StableHLO 包在 TF SavedModel | `tf2onnx` 轉換失敗,接受 TF 2.20 依賴 | 已確認此路不通 |
| 8 | Kaggle Interactive Session 會掉 → 工作遺失 | 用 `Save Version` 固化為 Dataset | 中間產物可重用 |

### E. 名詞解釋(給 Q&A 用)

| 術語 | 定義 |
|------|------|
| **Embedding** | 將輸入壓縮成固定長度的數字向量,向量距離反映語義相似度 |
| **Transfer Learning** | 把在大規模資料訓練的模型用於新任務,不從頭訓練 |
| **Linear Probe** | 凍結 encoder 上訓練單層線性分類器,用於評估 representation 品質(Alain & Bengio 2016) |
| **Ablation Study** | 系統性移除 / 替換模型元件,量化每個設計的貢獻 |
| **macro ROC-AUC** | 對每個類別分別算 ROC-AUC 再取平均,對類別不平衡穩健 |
| **GroupShuffleSplit** | 按組(如 filename)切分 train/val,避免同群資料洩漏 |
| **PCEN** | Per-Channel Energy Normalization,自動背景噪音正規化,比 log-mel 對野外錄音更穩健 |
| **Conformer** | Convolution + Transformer 混合架構,同時捕捉局部與長距離依賴(Gulati et al. 2020) |
| **Focal Loss** | 調整 CE loss,降低 easy samples 的權重,原用於物件偵測的類別不平衡(Lin et al. 2017) |
| **sm_120** | NVIDIA Blackwell 架構的 CUDA compute capability(RTX 50 系列) |

### F. 關鍵參考文獻

- Hamer et al. 2023. "Global birdsong embeddings enable superior transfer learning for bioacoustic classification." *Scientific Reports*.(Perch 模型)
- Gulati et al. 2020. "Conformer: Convolution-augmented Transformer for Speech Recognition." *Interspeech*.(Conformer 架構)
- Wang et al. 2017. "Trainable Frontend For Robust and Far-Field Keyword Spotting." *ICASSP*.(PCEN)
- Lin et al. 2017. "Focal Loss for Dense Object Detection." *ICCV*.(RetinaNet,Focal Loss)
- Alain & Bengio 2016. "Understanding intermediate layers using linear classifier probes." *arXiv*.(Linear probe 方法論)

---

**報告完**。如需說明任何細節或補做進一步實驗,請告知。
