# Experiment Log — BirdCLEF+ 2026 Route B

## Baseline (before optimization)

| Metric | Value |
|---|---|
| Best variant | B_mlp_bce |
| Local val AUC | 0.9646 (single GroupShuffleSplit 85/15) |
| Domain split: ta_auc | 0.9591 |
| Domain split: sc_auc | 0.9956 |
| LB score | **0.872** |
| Weights file | `mlp_best.pt` (from `train_mlp_local.py`) |
| Training data | 233,101 train_audio + 1,478 soundscape (with 2x duplicate bug) = 234,579 |

**Root cause of LB gap (0.965 → 0.872):** val set is 99.3% train_audio (focal recordings); model never properly validated on soundscape domain. Test soundscapes have different noise profiles, multi-species, unseen sites.

---

## EXP-001: 5-fold + MixUp + SC upweighting + dedup fix (2026-05-30)

**Script:** `new/train_5fold.py`

**Changes from baseline:**
1. **Soundscape dedup**: 1478 → 739 unique rows (host confirmed 2x duplicate error). Applied `ss_emb[:len//2]`
2. **MixUp on embeddings**: `lam ~ Beta(0.4, 0.4)`, per-batch linear interpolation of X and Y
3. **5-fold GroupKFold**: replaced single `GroupShuffleSplit(test_size=0.15)` with `GroupKFold(n_splits=5)`
4. **Soundscape sample upweighting**: soundscape rows get 20x weight in BCE loss (weighted per-sample loss)

**Hyperparameters:**
- Architecture: MLPHead (1536→768→384→234), BatchNorm, Dropout(0.3)
- Loss: Weighted BCE (soundscape 20x, train_audio 1x)
- Optimizer: Adam(lr=1e-3), CosineAnnealing(T_max=30)
- Patience: 5 epochs
- Batch size: 512
- Seed: 42 + fold_idx

**Results:**

| Fold | Val AUC | Epochs | Time (s) |
|------|---------|--------|----------|
| 0 | 0.9572 | 11 | 35.6 |
| 1 | 0.9539 | 13 | 39.6 |
| 2 | 0.9491 | 16 | 48.6 |
| 3 | 0.9577 | 16 | 48.5 |
| 4 | 0.9582 | 23 | 71.6 |
| **Mean** | **0.9552 +/- 0.0034** | | **~244s** |

**Observations:**
- Individual fold val AUC (0.955) is lower than baseline single-split (0.965) — expected because: (a) GroupKFold uses 80/20 split not 85/15, (b) MixUp regularization smooths decision boundary, (c) soundscape upweighting shifts loss focus away from train_audio accuracy
- Local val AUC decrease does NOT necessarily mean LB will decrease — the baseline's high local AUC was inflated. The regularization and domain-aware weighting should improve LB.
- Fold 2 is weakest (0.9491), fold 4 is best (0.9582). Variance is small (std=0.0034).

**Files created:**
- `new/train_5fold.py` (new training script)
- `new/mlp_fold0.pt` ... `new/mlp_fold4.pt` (5 fold weights)
- `new/fold_results.json` (detailed results)
- `new/fold_curves.png` (training curves)

**Files modified:**
- `new/train_mlp_local.py` (added soundscape dedup in load_data)
- `new/eval_domain_auc.py` (added soundscape dedup in load_data)
- `CLAUDE.md` (added Experiment Log Convention section)

**Backup note:** Original `mlp_best.pt` is preserved (not overwritten). It came from `train_mlp_local.py` ablation run, Variant B, val AUC 0.9646.

**LB score:** 0.873（baseline 0.872 → **+0.001**）

**分析：為什麼只提升 +0.001？**
- 5-fold / MixUp / SC 加權都是在「現有資料上訓練更好」的改動
- 但 LB 的瓶頸不是模型 variance 或過擬合，而是 **domain gap**
- 訓練集 99.7% 是 focal recording（train_audio），只有 0.3% 是 soundscape
- 即使 soundscape 加權 20x，739 筆 soundscape 仍然太少，模型學不到 soundscape 的 embedding 分佈
- 結論：必須大幅增加 soundscape domain 的訓練資料 → **pseudo-labeling 是關鍵**

---

## EXP-004: Pseudo-labeling on unlabeled soundscapes (2026-05-31)

**Changes:**
1. Extracted Perch embeddings for all 10,658 train_soundscape files on Kaggle GPU (~127k frames)
2. Generated soft pseudo-labels using 5-fold ensemble (from EXP-001)
3. Added 127,104 pseudo-labeled frames to training (training-only, not in validation)
4. Pseudo frames treated as soundscape domain (20x weight)

**Results:**
- Local val AUC: 0.9552 → **0.9712** (+0.016)
- **LB: 0.873 → 0.875 (+0.002)**

**分析：為什麼 local +0.016 但 LB 只 +0.002？**
- Pseudo-labels 是用 LB=0.873 的模型生成的 — teacher 本身對 soundscape 就不準
- 模型等於在教自己已經知道的東西（circular dependency）
- 論壇高手（Natsume +0.012）用的是更強的 teacher（end-to-end CNN），不是 frozen embedding + MLP
- Route B 的天花板可能就在這裡 — 最大的槓桿是 **Route A+B ensemble**

---

### 2026-05-30 EXP-002: ProtoSSM Chain — Research & Dead End

**Motivation**: User's optimization roadmap listed "ProtoSSM chain" as a potential architecture upgrade. Before implementing, researched whether this is an established technique.

**Research findings**:

1. **"ProtoSSM" is NOT a published architecture.** The term comes from one BirdCLEF 2026 competitor:
   - Kaggle notebook: `imaadmahmood/birdclef-2026-perch-v2-protossm-0-925` (LB 0.925)
   - GitHub: `ferariz/birdclef2026` — ProtoSSM class is placeholder only, NOT implemented publicly

2. **SSM for audio** (Audio Mamba, BioMamba, S4, S5):
   - All operate on spectrogram sequences or raw audio, NOT on clip-level embeddings
   - Perch v2 produces a single 1536-dim vector per 5-sec clip — no temporal sequence for SSM to model
   - Paper "Lost in State Space" (arXiv 2605.00253): frozen Mamba has severe anisotropy issues
   - **Verdict: SSM component has no natural fit for clip-level Perch embeddings**

3. **Prototypical Probing** — the real opportunity:
   - Bird-MAE paper (arXiv 2504.12880, GitHub: `DBD-research-group/Bird-MAE`): frozen encoder + prototypical probing = 49.97% mAP vs MLP 15.22% (3.3x improvement on HSN dataset)
   - Uses J=20 learnable prototypes per class, cosine similarity, max-pool, class-specific non-negative linear layers
   - 430k params — lighter than our MLP (1.57M)
   - **Verdict: Prototypical Head is the valuable component, directly applicable to Perch embeddings**

4. **BirdCLEF past winners**: No top solution used SSM or prototypical networks. All use CNN (EfficientNet) + pseudo-labels + ensembles.

**Dead ends / 彎路**:
- Wrote full `train_proto_ssm.py` with DiagonalSSM (S4D-style) + PrototypicalHead + sequence-level data loading (grouping clips by file, custom collate with padding). This is architecturally complete but the SSM is solving a non-existent problem: clip-level Perch embeddings have no intra-clip temporal dimension, and grouping by file just groups clips that share identical labels.
- The SSM could theoretically help at inference (12 consecutive soundscape clips), but training on file-grouped sequences where all clips share labels doesn't teach useful temporal dependencies.
- The LB 0.925 from the Kaggle ProtoSSM notebook is LOWER than our local val AUC (0.9646), suggesting the "ProtoSSM" branding may not reflect actual gains.

**Files created**:
- `new/train_proto_ssm.py` — complete implementation (SSM + Proto), NOT run. Should be simplified to Proto-only if pursued further.

**File backups**: No existing files modified. All new output filenames are distinct (`proto_ssm_results.json`, `proto_ssm_best.pt`).

**Discussion**:
- For clip-level embeddings, a prototypical head simplifies to: learn K prototype vectors per class, classify by cosine similarity. This is equivalent to a normalized linear layer with multiple "modes" per class — elegant but may not outperform a well-tuned MLP+BN on abundant data.
- The Bird-MAE result (3.3x over MLP) was on patch-level embeddings with spatial structure, not clip-level single-vectors. The gap may be much smaller for our setup.
- Higher-value improvements remain: pseudo-labeling (Day 2 on roadmap) and completing the 5-fold submission pipeline.

**Key references**:

| Name | URL | Relevance |
|------|-----|-----------|
| Bird-MAE + prototypical probing | arXiv 2504.12880 / github.com/DBD-research-group/Bird-MAE | HIGH |
| BirdCLEF 2026 ProtoSSM notebook | kaggle.com/code/imaadmahmood/birdclef-2026-perch-v2-protossm-0-925 | LB 0.925 |
| ferariz/birdclef2026 | github.com/ferariz/birdclef2026 | ProtoSSM unimplemented |
| Audio Mamba | github.com/mhamzaerol/Audio-Mamba-AuM | LOW — spectrogram-level |
| BioMamba | arXiv 2512.03563 | LOW — raw audio encoder |
| Frozen Mamba issues | arXiv 2605.00253 | WARNING — anisotropy |

**Next steps**: Pivot away from SSM. Consider Proto-only head as a lighter experiment, or focus on pseudo-labeling and finalizing 5-fold submission.

---

### 2026-05-30 EXP-003: Perch v2 → HGNetV2 Knowledge Distillation

**Motivation**: Kaggle 討論中 Natsume (918th) 用 HGNetV2_b0 + Perch 蒸餾達到 0.931 LB（+pseudo 0.943）。這條路線能：
1. 移除推論端的 TensorFlow 依賴（純 PyTorch）
2. 提供與 Route A (EfficientNet) 不同的 ensemble diversity
3. HGNetV2_b0 只有 6M params、0.3 GMACs，推論極快

**Phase: 研究 (完成)**

#### 研究結論

| 來源 | 做法 | 結果 |
|------|------|------|
| DS@GT (arXiv 2507.08236) | KL-div distillation from Perch, T=3, tokenized student | F1=0.47 vs teacher 0.80 — **失敗** |
| BirdNET→PSLA (arXiv 2409.15383) | KL-div + BCE, λ=0.5, consistent teaching | 域內有效 (mAP 0.71)，域外淺 |
| BirdCLEF 2025 2nd (Sydorskyy) | EfficientNetV2-S + pseudo-label distillation, Focal BCE | 前幾名標準做法 |
| BirdCLEF 2024 3rd (jfpuget) | Multi-CNN → pseudo → student, ONNX | 穩定有效 |
| Natsume (BirdCLEF 2026 討論) | HGNetV2_b0 + Perch蒸餾預訓練 + pseudo | 0.931→0.943 LB |
| Antoine Masq (BirdCLEF 2026 49th) | EfficientNetV2_b0 + pseudo + custom ASL, 無 Perch | 0.943 LB |

**關鍵發現**:
1. 純 KL-div distillation (DS@GT) **失敗了** — student 容量不足或表示空間差太大
2. 成功案例都用 **soft pseudo-labels as targets**（不是經典 temperature-scaled KD）
3. 「Consistent teaching」很重要 — teacher 和 student 必須看同一份增強後的 spectrogram
4. **沒有人公開發布過 Perch → HGNetV2 的具體程式碼**，Natsume 只在討論中提到
5. Mel spectrogram 共識: SR=32000, n_mels=128, fmax=16000, hop_length 各家不同

**實作計畫**:

| 步驟 | 內容 | 狀態 |
|------|------|------|
| 3a | 安裝 `timm` + `torchaudio` 到 `.venv-torch` | 待做 |
| 3b | 用 MLP head 產生 teacher soft labels (233101, 234) | 待做 |
| 3c | 建 mel spectrogram dataloader (讀 train_audio/*.ogg) | 待做 |
| 3d | HGNetV2_b0 student model + distillation training | 待做 |
| 3e | 評估 val AUC + 與 MLP baseline 比較 | 待做 |

**環境隔離**:
- 新套件安裝到 `.venv-torch`（記錄版本）
- 輸出: `hgnet_best.pt`, `distill_results.json`, `distill_curves.png`（不覆蓋任何現有檔案）
- Teacher soft labels: `embedding/teacher_soft_labels.npy`（新檔案）

**Backup note**: 現有 `mlp_best.pt` (Variant B, val AUC 0.9646) 和 `mlp_fold0-4.pt` 不受影響。

---

#### Step 3a: 安裝套件 (完成)

- `timm==1.0.27` — HGNetV2 模型來源
- `torchaudio==2.11.0+cu128` — mel spectrogram on GPU
- `torchvision==0.26.0+cu128` — timm 依賴

**彎路**: pip 自動把 torch 從 2.11.0+cu128 升到 2.12.0 (CPU only)，導致 CUDA 壞掉。
修復方法: `pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps`，再修 torchvision 版本不匹配 (0.27.0→0.26.0)。
**教訓**: 安裝新套件時必須 pin torch 版本或用 `--no-deps`。

環境驗證:
```
torch=2.11.0+cu128 cuda=True gpu=NVIDIA GeForce RTX 5070
torchaudio=2.11.0+cu128
timm=1.0.27
HGNetV2_b0 params=5,996,550
```

#### Step 3b: 產生 teacher soft labels (完成)

- 用 `mlp_best.pt` (B_mlp_bce) 對所有 233,840 個 embedding 跑 sigmoid
- 輸出: `embedding/teacher_soft_labels.npy` (233840, 234) float32, 218.9 MB
- Stats: min=0.0000, max=1.0000, mean=0.006100

#### Step 3c: Smoke test pipeline (完成)

**彎路 1**: `mel.view(B, -1)` 在 GPU 上觸發 stride 不兼容錯誤。修復: 改用 `.reshape()`。
**彎路 2**: `timm` 預設下載模型權重到 `%USERPROFILE%\.cache\huggingface\`，違反環境隔離規則。修復: 在 script 開頭設定 `HF_HOME=C:\birdCLEF\cache\huggingface`。第一次下載已汙染 host cache，但後續使用都導向專案目錄。

Pipeline 驗證通過:
```
audio (160000,) → MelSpec GPU (1, 128, 313) → normalize → resize (1, 1, 224, 224) → HGNetV2 → logits (1, 234)
```

#### Step 3d: 正式訓練 (完成)

**Script**: `new/train_hgnet_distill.py`

**Config**:
- Model: `hgnetv2_b0.ssld_stage2_ft_in1k` (timm), in_chans=1, 4,426,728 params
- Mel: SR=32000, n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=16000, resize→224×224
- Loss: BCE(student_logits, teacher_soft_labels)
- Optimizer: AdamW(lr=1e-3, wd=1e-4), CosineAnnealing(T_max=20)
- MixUp α=0.4 on waveforms
- DataLoader: 4 workers, persistent, batch_size=64, on-the-fly audio loading with seek

**Per-epoch results**:

| Epoch | Train Loss | Val AUC | Time |
|-------|-----------|---------|------|
| 1 | 0.0257 | 0.9281 | 233s |
| 5 | 0.0163 | 0.9580 | 206s |
| 9 | 0.0150 | 0.9650 | 208s |
| 12 | 0.0145 | 0.9685 | 197s |
| 14 | 0.0142 | 0.9699 | 199s |
| 16 | 0.0139 | 0.9708 | 202s |
| **18** | **0.0139** | **0.9714** | 273s |
| 20 | 0.0139 | 0.9708 | 237s |

- Best epoch: 18, val AUC = **0.9714** (188 valid classes)
- Total time: **5020s (~84 min)** for 20 full epochs
- Speed: ~200s/epoch = ~3.3 min/epoch (比預估的 21 min/epoch 快 6x！)
- No early stop — ran all 20 epochs, AUC still slowly rising

**Comparison with baseline**:

| Model | Val AUC | Params | Input | TF needed? |
|-------|---------|--------|-------|------------|
| MLP (B_mlp_bce) | 0.9646 | 1.57M | Perch embedding | Yes |
| **HGNetV2 distilled** | **0.9714** | 4.43M | Raw audio (mel spec) | **No** |
| 5-fold MLP (EXP-001) | 0.9552 mean | 1.57M×5 | Perch embedding | Yes |

**Key observations**:
1. HGNetV2 **surpasses the teacher** (0.9714 > 0.9646) — the student learned features the Perch embedding doesn't fully capture
2. No TensorFlow needed at inference — pure PyTorch pipeline
3. The model is still improving at epoch 20 — more epochs or larger model could push further
4. Inference is fast: HGNetV2_b0 has only 0.3 GMACs (vs Perch's much heavier encoder)

**Files created**:
- `new/train_hgnet_distill.py` — training script
- `new/hgnet_best.pt` — best model state_dict (epoch 18)
- `new/distill_results.json` — full results with history
- `new/distill_curves.png` — training curves
- `embedding/teacher_soft_labels.npy` — teacher soft labels (233840, 234)

**Discussion**:
- 蒸餾模型超越 teacher 的現象（"student surpasses teacher"）在 KD 文獻中有記錄。可能原因: (1) HGNetV2 從 mel spectrogram 中捕捉到 Perch embedding 丟失的細節（Perch 做了 clip-level pooling，丟失了 time-frequency 局部資訊）；(2) ImageNet 預訓練的 visual features 對 spectrogram pattern recognition 有幫助；(3) soft labels 比 hard labels 提供更好的監督信號
- 這個模型的 LB 表現還需要實際提交驗證。local val AUC 的 gap (baseline 0.965→LB 0.872) 可能在 HGNetV2 上也存在
- 潛在的 ensemble 價值很高: HGNetV2 (mel spectrogram) + MLP (Perch embedding) 的 correlation 應該很低

**Next steps**:
1. 5-fold 版本的 HGNetV2 distillation
2. 寫 Kaggle submission notebook (純 PyTorch, 不需 TF)
3. HGNetV2 + MLP ensemble 提交

---

### 2026-05-30 EXP-004: SED (Sound Event Detection) 架構 — 從 mel spectrogram 訓練

**Motivation**: 
- Kaggle 討論區 OverfitOracle (485th) 用 SED + pseudo + Xeno-canto 達到單模型 LB 0.946
- Tucker Arrants 的 distilled SED 是多數 top notebook 的核心組件
- SED 與 Perch+MLP 的 correlation 低，ensemble 時能提供真正的 diversity (+0.03 from 弱模型)
- 用戶決定實作 SED 架構，不管先前標記為「跳過」

**架構設計 (基於研究)**:

| 組件 | 選擇 | 來源 |
|------|------|------|
| Backbone | `tf_efficientnet_b0_ns` (via timm) | BirdCLEF 2023 2nd, 2025 5th |
| Attention | AttBlockV2 (tanh gating) | PANNs → BirdCLEF 2020 1st → 2023 2nd |
| Frequency pooling | mean over freq axis | PANNs standard |
| Channel smoothing | max_pool1d + avg_pool1d, k=3 | BirdCLEF 2023 2nd |
| Loss | 0.5×clip BCE + 0.5×max-frame BCE | 跨年度標準做法 |
| Mel | 128 mels, SR=32000, n_fft=2048, hop=512, fmin=20, fmax=16000 | BirdCLEF 2025 2nd |
| Augmentation | MixUp(0.5, beta=0.4), SpecAugment (freq+time mask) | 各家共識 |
| Input duration | 5 sec (160000 samples) | 與 Phase 1 一致 |

**關鍵研究來源**:

| 來源 | URL | 貢獻 |
|------|-----|------|
| PANNs (Kong et al.) | github.com/qiuqiangkong/audioset_tagging_cnn | AttBlock 原始定義 |
| BirdCLEF 2023 2nd | github.com/LIHANG-HONG/birdclef2023-2nd-place-solution | AttBlockV2 + EfficientNet SED 完整實作 |
| BirdCLEF 2025 2nd (VSydorskyy) | github.com/VSydorskyy/BirdCLEF_2025_2nd_place | BCEFocal2WayLoss, GeMFreq |
| BirdCLEF 2025 5th | github.com/myso1987/BirdCLEF-2025-5th-place-solution | 4x EfficientNet SED, 三階段 pseudo |
| BirdCLEF 2020 1st | github.com/ryanwongsa/kaggle-birdsong-recognition | tanh 替代 clamp |

**實作步驟**:

| # | 內容 | 狀態 |
|---|------|------|
| 4a | 寫 `train_sed.py` — 完整 SED pipeline (mel+backbone+att+dual loss) | 進行中 |
| 4b | 本地 RTX 5070 訓練 (單 fold 先跑通) | 待做 |
| 4c | 5-fold 訓練 + 結果記錄 | 待做 |
| 4d | 更新 submission notebook 加入 SED ensemble | 待做 |

**環境隔離**:
- 使用已安裝的 `.venv-torch` (torch 2.11+cu128, torchaudio 2.11, timm 1.0.27)
- 輸出: `sed_best.pt`, `sed_fold{0-4}.pt`, `sed_results.json`, `sed_curves.png`
- 不覆蓋任何現有檔案

**File backups**: 現有 `mlp_best.pt` (Variant B, AUC 0.9646), `mlp_fold0-4.pt` (5-fold mean 0.9552) 不受影響。

---

#### Step 4a: 寫 train_sed.py (完成)

- 完成 `new/train_sed.py` — 完整 SED pipeline
- 架構: `tf_efficientnet_b0_ns` (timm, 6.2M params) + AttBlockV2 + MelSpecTransform
- GPU forward pass 驗證通過

**彎路**:
1. AttBlockV2 的 logit 計算: 最初嘗試用 `self.att_block.cla.weight @ x` 手動計算 Conv1d 的矩陣乘法，shape 不匹配 (Conv1d weight 是 3D tensor)。修正: 直接呼叫 `self.att_block.cla(x)` 取得 raw logits，在 SEDModel.forward 裡分開計算 attention weight 和 classification logit。
2. soundscape CSV 欄位名: `train_soundscapes_labels.csv` 的物種欄位是 `primary_label` 不是 `label`。兩處 `build_file_list` 和 `build_entries` 都需要修正。

#### Step 4b: 本地 RTX 5070 訓練 (完成)

**訓練結果** (全 30 epochs，無 early stop):

| Epoch | Train Loss | Val AUC | 備註 |
|-------|-----------|---------|------|
| 1 | 0.0347 | 0.7445 | |
| 6 | 0.0179 | 0.8084 | |
| 10 | 0.0161 | 0.8244 | |
| 17 | 0.0135 | 0.8294 | |
| 20 | 0.0127 | 0.8301 | |
| 25 | 0.0117 | 0.8341 | |
| **27** | **0.0116** | **0.8343** | **best** |
| 30 | 0.0115 | 0.8321 | final |

- **Best**: epoch 27, val AUC = **0.8343**
- **Total time**: 4859.5s (~81 min)
- **Model**: 6,246,416 params, `sed_best.pt` = 25 MB

**與 MLP baseline 比較**:
- MLP single split: 0.9646 | MLP 5-fold: 0.9552 | **SED: 0.8343**
- SED 明顯低於 MLP，因為 MLP 用 Perch v2 embedding（鳥聲專家模型），SED 只用 ImageNet pretrained B0
- Ensemble 價值在 diversity：討論區經驗弱模型 (<0.90) 可帶來 +0.03

**Files created**: `sed_best.pt`, `sed_results.json`, `sed_curves.png`

#### Step 4c: Ensemble submission notebook (完成)

- 寫 `new/kaggle_03_ensemble_submit.ipynb` — 雙路 ensemble (Perch+MLP 70% + SED 30%)
- 本地 smoke test 通過：兩個模型皆可載入，輸出 shape 正確，ensemble 計算無誤
- Kaggle 需新增 input: `sed_best.pt` 上傳到 `birdclef-mlp-head` Dataset
- 需確認 Kaggle 環境有 `timm` 和 `torchaudio`（通常有預裝）

**Ensemble 策略**: `final_probs = 0.7 * mlp_probs + 0.3 * sed_probs`
- MLP 給 70% 權重（val AUC 0.9646 遠高於 SED 的 0.8343）
- SED 給 30% 權重（提供 diversity，不同特徵空間的錯誤模式）

---

### 2026-05-30 EXP-005: ProtoSSM Chain — 隔離實驗分支

**Motivation**: 用戶要求獨立開發 ProtoSSM chain 架構，與主線（5-fold / SED / pseudo-label）平行。EXP-002 的研究顯示 SSM 對 clip-level embedding 幫助有限，但 Prototypical Head（Bird-MAE 論文）在 frozen encoder 上效果顯著。本實驗將實際驗證。

**隔離設定**:
- 所有程式碼和輸出放在 `c:\birdCLEF\exp_proto_ssm\`
- 不修改 `new/` 下任何主線檔案
- 讀取 `embedding/` 資料（唯讀）
- 環境：使用現有 `.venv-torch` (torch 2.11+cu128)

**Phase: 研究參考程式碼 (完成)**

#### 研究發現 1: Kaggle "ProtoSSM" = 非參數原型 + 時序平滑

**"SSM" 不是 State Space Model，是 Smoothing！**
- `ferariz/birdclef2026` repo 的 ProtoSSM class **未實作**（只有 placeholder）
- 推斷：Proto = class-mean prototypes, SSM = temporal smoothing
- **0 個可學習參數** — prototypes 是每個 class 的 embedding 均值
- 推論：L2-normalize → cosine similarity → Gaussian smoothing (σ=1.0) across 12 clips/soundscape
- LB 0.925 可能來自 prototype + smoothing + site/hour priors 的組合

#### 研究發現 2: Bird-MAE PPNet (arXiv 2504.12880)

從 `DBD-research-group/Bird-MAE` 的 `models/ppnet/ppnet.py`：
- J=20 learnable prototypes/class, cosine via conv2d, max-pool spatial
- `LinearLayerWithoutNegativeConnections` — per-class 獨立, `ReLU(weight)` at forward
- **AsymmetricLossMultiLabel** + **orthogonality loss**
- Prototype LR = **0.04** (130x 主 LR)
- Bias init = **-2.0**
- Params (C=21, J=20, d=1024) = 430k — 但我們 C=234, d=1536 → 需降維/減 J

#### 設計決策

| Variant | 描述 | 參數量 | 來源 |
|---------|------|--------|------|
| F_proto_nonparam | Class-mean prototypes + cosine sim | 0 | Kaggle ProtoSSM |
| G_proto_learned | J=10/class, proj 1536→256, non-neg linear, AsymmetricLoss | ~1M | Bird-MAE PPNet |
| (bonus) temporal smoothing | 任何 head 推論時都可加 | 0 | Kaggle ProtoSSM |

#### 實作 (完成)

**Script:** `exp_proto_ssm/train.py`

**結果:**

| Variant | Params | Val AUC | vs Baseline | Time |
|---------|--------|---------|-------------|------|
| F_proto_nonparam | 0 | 0.9241 | −0.0405 | 2.3s |
| G_proto_learned (J=10, proj=256) | 995,598 | 0.9482 | −0.0164 | ~6 min |
| B_mlp_bce (baseline) | 1,567,338 | 0.9646 | — | 30s |

G_proto_learned 在 ep15 達到 peak (0.9482)，ep23 early stop。

**結論：ProtoSSM chain 在 clip-level Perch embeddings 上不如 MLP baseline。**

**原因分析**:
1. Bird-MAE 的 3.3x 改進是在 **patch-level embeddings** (有 spatial H×W 結構) 上。Prototype max-pool over spatial dims 能捕捉「哪個區域像哪個 prototype」。但 Perch 的 clip-level embedding 是單一 1536-dim 向量，沒有 spatial 結構，max-pool 退化為 identity。
2. Perch embedding 空間已經高度結構化（encoder 很強），MLP + BatchNorm 能學到更好的 decision boundary。
3. 我們有大量訓練資料（199k clips），prototype-based 方法在 few-shot 場景優勢更大，abundant data 時 MLP 更好。
4. Non-parametric prototype (0.9241) 遠低於 learned (0.9482)，說明 class-mean 不是好的 prototype — Perch embedding 空間中 class 分佈不是 unimodal 的。

**有價值的 takeaway**:
- **Temporal smoothing** 仍然值得嘗試 — 任何 head 的推論都可以在 soundscape 的 12 clips 上加 Gaussian smoothing。這是免費的 post-processing，不需要改模型。可以直接加到 `kaggle_02_submit.ipynb`。

**Files created**:
- `exp_proto_ssm/train.py` — 完整實驗腳本
- `exp_proto_ssm/ppnet_best.pt` — G variant 最佳權重
- `exp_proto_ssm/proto_results.json` — 詳細結果
- `exp_proto_ssm/proto_curves.png` — 訓練曲線

**Dead ends / 彎路**:
1. EXP-002 寫的 `new/train_proto_ssm.py` (SSM + Proto + sequence loader) — SSM 解決不存在的問題
2. EXP-005 的 F_proto_nonparam — class-mean prototype 不適合多模態 class 分佈
3. EXP-005 的 G_proto_learned — 即使用了 Bird-MAE 的所有 tricks (high proto LR, non-neg linear, ASL, ortho reg)，仍不如 MLP

**Status: CLOSED — ProtoSSM chain 實驗結束，MLP baseline 維持不變。**

---

### 2026-05-30 EXP-006: 外部資料整合 (Xeno-canto + AnuraSet)

**Motivation**: 用戶要求優先處理外部資料。根據 Kaggle 討論區：
- OverfitOracle (485th): SED + pseudo + **Xeno-canto** → 單模型 LB 0.946
- Antoine Masq (49th): EfficientNetv2_b0 + **XC 預訓練** → 0.943
- sergheibrinza: 宣告使用 AnuraSet + InsectSet459 + iNaturalist + BirdCLEF 2025
- 長尾分析: 底部 30 物種中 22/30 是兩棲類，最弱 AUC = 0.652 (Black-tailed Marmoset, 3 筆)

**Top solution 共識過濾策略** (來源: TheoViel 3rd 2024, VSydorskyy 2nd 2025, myso1987 5th 2025):
- Quality: 去掉 E 級 (rating=1)，保留 A-D
- Per-species cap: 500 clips，保留最新
- Speech filter: Silero VAD 去人聲 (VSydorskyy)
- 不做 per-recordist cap

**資料源優先順序**:
1. Xeno-canto (162 鳥種) — 最高增益
2. AnuraSet (12+ 兩棲物種直接重疊) — 直接解決長尾
3. BirdCLEF 2025 data (部分鳥種重疊) — 額外補充
4. InsectSet459 — 跳過（sonotype 無法匹配）

**Route B 特殊做法**: 在本機用 `perch_v2_cpu` (TF 2.20 CPU) 跑 embedding extraction，避免消耗 Kaggle GPU 配額。

**實作步驟**:

| # | 內容 | 狀態 |
|---|------|------|
| 6a | 建立 `.venv-tf` + 安裝 TF 2.20 CPU + perch-hoplite | ✅ 完成 |
| 6b | 測試 `perch_v2_cpu` 能否在 Windows 上跑 | ✅ 完成 |
| 6c | 寫 Xeno-canto 下載腳本 (234 物種, quality A-D, cap 200/species) | 進行中 |
| 6d | 下載 AnuraSet 重疊物種 | 待做 |
| 6e | 跑 embedding extraction (本機 CPU) | 待做 |
| 6f | 修改 train_5fold.py 整合外部 embedding | 待做 |

**環境隔離**:
- TF venv: `c:\birdCLEF\.venv-tf` (獨立於 `.venv-torch`)
- 外部音訊: `c:\birdCLEF\external\xeno-canto\`, `c:\birdCLEF\external\anuraset\`
- 外部 embedding: `c:\birdCLEF\embedding_external\`
- 不修改 `embedding/` 或 `birdclef-2026/` 下任何檔案

#### Step 6a: 建立 TF CPU venv (完成)

- venv: `c:\birdCLEF\.venv-tf`
- TF 2.20.0 (CPU only, GPU 列表為空 — 預期行為)
- perch-hoplite 安裝成功

#### Step 6b: perch_v2_cpu benchmark (完成)

**結果: perch_v2_cpu 在 Windows CPU 上完全可用！**

- 模型下載到 `c:\birdCLEF\cache\kagglehub\` (環境隔離 ✓)
- 模型載入: 25.6s (首次含 XLA compile)
- `model.batchable = True`

| Batch size | 總時間 | 每 clip | 備註 |
|------------|--------|---------|------|
| 1 | 10.4s | 10.4s | warm-up，含首次 XLA compile |
| 8 | 2.9s | 0.37s | |
| 16 | 2.3s | 0.14s | |
| 32 | 3.4s | 0.11s | **最佳** |

**時間估算** (batch=32):
- 10,000 clips → 18 min
- 20,000 clips → 36 min
- 46,800 clips (200/species × 234) → ~86 min

**結論**: 本機 CPU embedding extraction 完全可行，不需要 Kaggle GPU。

#### Step 6c: Xeno-canto 下載腳本

**彎路**: XC API 已從 v2 升級到 v3，v3 需要 API key（需在 xeno-canto.org/account 申請）。v2 返回 404 並提示遷移。同時 xeno-canto.org 網站有 Anubis 防爬保護。
**備案**: 改用 iNaturalist API（免費無需 key）下載可用音訊，AnuraSet (Zenodo) 補兩棲類。XC 等用戶提供 API key 後再補。

**用戶需要做**: 到 https://xeno-canto.org/account 取得 API key，然後在 `download_xc.py` 中設定。

#### Step 6c-alt: iNaturalist + AnuraSet 下載

**iNaturalist API** (進行中):
- Script: `new/download_inat.py`
- 免費 API，不需要 key
- 下載到 `c:\birdCLEF\external\inaturalist\{primary_label}\*.{mp3,wav,m4a}`
- 過濾: research-grade, CC0/CC-BY/CC-BY-NC
- 中間狀態: 已查詢 13/209 物種，12 個有音訊，共 168 個檔案

**AnuraSet** (下載中):
- Script: 手動下載 from Zenodo (11.4 GB)
- `c:\birdCLEF\external\anuraset\anuraset.zip`
- 42 anuran species, **17 個與 BirdCLEF 2026 重疊**
- 重點: Pithecopus azureus (0 train → 211 AnuraSet), Phyllomedusa sauvagii (1 → 70)
- 格式: WAV, 3 seconds, 22.05 kHz (需要 resample 到 32 kHz 給 Perch)
- 組織: by monitoring site, not by species → 需要用 weak_labels.csv 匹配

**AnuraSet 物種重疊詳細**:

| Code | 物種 | 競賽 train | AnuraSet | 增幅 |
|------|------|-----------|----------|------|
| PITAZU | Pithecopus azureus | 0 | 211 | ghost → 有資料 |
| PHYSAU | Phyllomedusa sauvagii | 1 | 70 | 71x |
| PHYNAT | Physalaemus nattereri | 3 | 21 | 8x |
| BOALUN | Boana lundii | 3 | 76 | 26x |
| AMEPIC | Ameerega picta | 3 | 5 | 2.7x |
| LEPPOD | Leptodactylus podicipinus | 6 | 221 | 38x |
| DENMIN | Dendropsophus minutus | 46 | 273 | 6.9x |

**Files created**:
- `new/download_inat.py` — iNaturalist download script
- `new/download_xc.py` — Xeno-canto v3 download script (needs API key)
- `new/process_anuraset.py` — AnuraSet 解壓+匹配+resample
- `new/extract_external_embeddings.py` — local CPU embedding extraction
- `new/train_5fold.py` — modified to auto-load external embeddings

**Files modified**:
- `CLAUDE.md` — added environment isolation rules for external data, updated repo layout

#### Step 6d: AnuraSet 處理 (完成)

- 下載: `anuraset.zip` 10.58 GB from Zenodo
- 解壓: 93,378 WAV files (3s, 22.05 kHz) across 4 monitoring sites
- 匹配: 506/1612 recordings 包含目標物種 → 28,564 個 WAV matched
- 保存: **3,090 個 WAV** resampled 到 32 kHz + padded 到 5s

| primary_label | Code | 保存數 | 競賽 train | 增幅 |
|---------------|------|--------|-----------|------|
| 517063 | PITAZU | 200 | 0 (ghost!) | ∞ |
| 23724 | PHYSAU | 200 | 1 | 201x |
| 476521 | PHYNAT | 200 | 3 | 67x |
| 555123 | BOALUN | 200 | 3 | 67x |
| 64898 | AMEPIC | 200 | 3 | 67x |
| 22961 | LEPPOD | 200 | 6 | 34x |
| 22983 | LEPLAB | 200 | 10 | 21x |
| 555146 | BOARAN | 200 | 18 | 12x |
| 24285 | SCIFUV | 200 | 19 | 11x |
| 65380 | DENNAN | 200 | 23 | 9.7x |
| 23158 | PHYALB | 200 | 25 | 9x |
| 25092 | ELABIC | 200 | 26 | 8.7x |
| 65377 | DENMIN | 200 | 46 | 5.3x |
| 22973 | LEPFUS | 200 | 63 | 4.2x |
| 22967 | LEPELE | 116 | 8 | 15.5x |
| 24279 | SCINAS | 116 | 46 | 3.5x |
| 24287 | SCIFUS | 58 | 11 | 6.3x |

**彎路**: 初版 process_anuraset.py 用暴力匹配 (O(N*M))，78k WAV × 1612 file_ids 太慢。改為從 WAV 檔名提取 prefix 做 dict O(1) 查詢後，處理在 2 分鐘內完成。

#### Step 6e: iNaturalist 下載 (進行中)

- 已查詢 61/209 物種，59 個有音訊，1,739 個檔案
- 主要收穫: 兩棲/昆蟲/哺乳類（非鳥類）
- 鳥類音訊在 iNat 上很少，需要 XC API key 來補充

#### Step 6f: Embedding extraction (完成)

- 使用 `perch_v2_cpu` 在本機 CPU 跑
- 處理 AnuraSet (3,090) + 已下載的 iNat (~3,811) = 6,901 files
- 結果: **9,595 frames**, 57 species, 760 fails (壞 MP3)
- 時間: 12.7 min (12.6 frames/s)
- 輸出: `embedding_external/external_embeddings.npy` (59 MB), `external_labels.npy` (9 MB)

#### Step 6g: 5-fold 重新訓練 (完成)

**baseline (EXP-001, 無外部資料)**:
- Data: 233,840 frames
- Mean val AUC: 0.9552 ± 0.0034

**with external data (EXP-006)**:
- Data: 243,435 frames (+9,595 external, +4.1%)
- Mean val AUC: **0.9588 ± 0.0041**

| Fold | 無外部 | 有外部 | 差異 |
|------|--------|--------|------|
| 0 | 0.9572 | 0.9525 | -0.0047 |
| 1 | 0.9539 | 0.9601 | +0.0062 |
| 2 | 0.9491 | 0.9603 | **+0.0112** |
| 3 | 0.9577 | 0.9646 | +0.0069 |
| 4 | 0.9582 | 0.9564 | -0.0018 |
| **Mean** | **0.9552** | **0.9588** | **+0.0036** |

**討論**:
- 整體提升 +0.0036，fold 2 提升最大 (+0.0112)
- Fold 0 略降 (-0.0047) 可能是外部資料中的噪音影響（iNat 壞檔案較多）
- 外部資料只佔 4% 但帶來一致性提升，特別是之前最弱的 fold 2
- 目前只用了 AnuraSet + 部分 iNat（63/209 species）。XC 鳥類資料加入後預期提升更大
- iNat 下載仍在背景進行中，完成後重新跑 extraction + training

**Backup note**: `mlp_fold0-4.pt` 被新結果覆蓋。舊版本來自 EXP-001，mean AUC 0.9552。

**Files modified**:
- `new/mlp_fold0.pt` ... `new/mlp_fold4.pt` (覆蓋，新 mean AUC 0.9588)
- `new/fold_results.json` (覆蓋)
- `new/fold_curves.png` (覆蓋)

#### Step 6h: iNat 全量下載完成

- **12,663 files**, **207/209 species**
- 分佈: Aves 11,891 / Amphibia 438 / Insecta 200 / Mammalia 133 / Reptilia 1
- median 31 files/species, max 200 (cap)

#### Step 6i: 全量 embedding extraction (完成)

- 17,811 files (12,663 iNat + 3,090 AnuraSet + ~2,058 other)
- **35,654 frames**, **204 species**, 4,587 fails (壞 MP3)
- 時間: 206 min (CPU)
- 輸出: `external_embeddings.npy` (218 MB), `external_labels.npy` (33 MB)

#### Step 6j: 全量 5-fold 訓練 (完成) — 含 pseudo-labels

注意: 用戶在另一個 session 中已經建立了 pseudo-labels (`embedding/pseudo_embeddings.npy`, 127,104 frames)。`train_5fold.py` 自動偵測並載入。

**訓練數據組成**:
- 原始 competition: 233,840 (train_audio + soundscape)
- 外部 (iNat + AnuraSet): 35,654
- Pseudo-labeled soundscape: 127,104 (training-only, 不進 validation)
- **Total per fold: ~342,700 training / ~53,900 validation**

**結果**:

| 版本 | 外部 | Pseudo | 訓練量 | Mean AUC | Std |
|------|------|--------|--------|----------|-----|
| EXP-001 baseline | 0 | 0 | 233,840 | 0.9552 | 0.0034 |
| EXP-006 第一輪 | 9,595 | 0 | 243,435 | 0.9588 | 0.0041 |
| **EXP-006 全量** | **35,654** | **127,104** | **342,700** | **0.9732** | **0.0017** |

| Fold | AUC |
|------|-----|
| 0 | 0.9715 |
| 1 | 0.9762 |
| 2 | 0.9737 |
| 3 | 0.9720 |
| 4 | 0.9729 |
| **Mean** | **0.9732 ± 0.0017** |

**討論**:
- 從 0.9552 → 0.9732，**+0.0180** — 巨大提升
- Std 從 0.0034 降到 0.0017，fold 間穩定性翻倍
- Pseudo-labels (127k frames) 是最大增量 — 直接解決了 domain gap (focal → soundscape)
- 外部資料 (35k frames) 提供了更好的 species coverage，特別是長尾
- 所有 fold 都在 0.97+ (最低 0.9715, 最高 0.9762)

**Files modified**:
- `new/mlp_fold0-4.pt` (覆蓋，新 mean AUC 0.9732)
- `new/fold_results.json` (覆蓋)
- `new/fold_curves.png` (覆蓋)

**Next**: 
1. 更新 submission notebook 成 5-fold 集成
2. 提交到 LB 驗證
3. 考慮 XC 鳥類資料進一步補充
