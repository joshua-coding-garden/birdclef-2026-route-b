# BirdCLEF+ 2026 Kaggle 討論區完整整理

> **來源**: `討論資料.docx`（Kaggle 討論區多篇帖子）
> **整理日期**: 2026-05-23
> **用途**: 彙整所有競賽經驗、方法、建議與細節，供 AI 複查

---

## 目錄

1. [外部預訓練模型授權問題](#1-外部預訓練模型授權問題)
2. [單模型 0.946 — SED + Pseudo Labels + Xeno-canto](#2-單模型-0946--sed--pseudo-labels--xeno-canto)
3. [外部資料宣告（sergheibrinza）](#3-外部資料宣告sergheibrinza)
4. [Ghost Species 處理策略](#4-ghost-species-處理策略)
5. [無 Perch 方案的上限探討](#5-無-perch-方案的上限探討)
6. [Pseudo-labeling 效益與方法](#6-pseudo-labeling-效益與方法)
7. [失敗方法經驗分享](#7-失敗方法經驗分享)
8. [多樣性與集成策略](#8-多樣性與集成策略)
9. [Distilled SED 運行時間優化](#9-distilled-sed-運行時間優化)
10. [Training Recipe 全面討論](#10-training-recipe-全面討論)
11. [train_soundscapes_labels.csv 標註方法論（官方回覆）](#11-train_soundscapes_labelscsv-標註方法論官方回覆)
12. [高分 Notebook 發布倫理](#12-高分-notebook-發布倫理)

---

## 1. 外部預訓練模型授權問題

**發文者**: 匿名（未標示）

### 問題

能否在 BirdCLEF 2026 中使用 **CC-BY-NC-SA-4.0** 授權的外部預訓練模型？

### 具體模型

- **EarthSpeciesProject/esp-aves2-sl-beats-all**（Hugging Face）
- 基於 BEATs 的生物聲學骨幹網路（bioacoustic backbone）
- 在動物聲音分類上表現強勁
- 授權: CC-BY-NC-SA-4.0（非商業性）

### 模型特點

- 公開於 Hugging Face Hub
- 任何參賽者均可免費下載
- 附有文件說明與預訓練權重

### 潛在衝突

- 授權為非商業性，可能與競賽的獎金性質（prize-eligible）衝突

### 相關連結

- Model card: `huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-all`
- License: `creativecommons.org/licenses/by-nc-sa/4.0/`

---

## 2. 單模型 0.946 — SED + Pseudo Labels + Xeno-canto

**發文者**: OverfitOracle（第 485 名）

### 核心成績

- **單模型 LB = 0.946**，無任何集成

### 方法細節

| 要素 | 內容 |
|------|------|
| 模型架構 | SED（Sound Event Detection）風格 |
| 訓練資料 | 競賽官方資料 + 大量 Xeno-canto 額外音訊 |
| 外部資料來源 | 透過 **Xeno-canto API** 拉取（非網頁爬取），比 scraping 簡單得多 |
| 資料篩選 | 按目標物種 + 品質（decent quality）過濾 |
| 去偏差處理 | 限制每個物種/錄音者的片段數上限（cap clips per species/recordist），以減少偏差（bias） |
| 標籤生成 | 使用 **teacher model** 為額外錄音生成 **pseudo frame/event labels** |
| 驗證策略 | 使用 holdout split，根據 holdout metric（val loss，前提是它能追蹤良好）checkpoint 最佳 epoch |

### 相關問答

- **Q (MarkDjadchenko, 74th)**: 是否使用 Perch v2？用什麼 augmentation？是否過濾 pseudo-tags？如何做 pseudo-blending？
  - **A**: 未直接回覆所有細節（僅描述了 SED + pseudo labels 的整體流程）

- **Q (Nick Pellegrin, 932nd)**: 是否有現成的 Xeno-canto 資料集在 Kaggle 上？
  - **A**: 沒有看到完善、全面的 XC 資料集在 Kaggle 上，所以自行透過 XC API 拉取

- **Q (Nick Pellegrin)**: 沒用集成的情況下是否還用 holdout validation set？還是根據 loss 來 checkpoint？
  - **A**: 用了 holdout split，根據 holdout metric checkpoint 最佳 epoch（val loss only if it tracked well）

- **Q (Nick Pellegrin)**: 驗證集用了什麼？只用 focal 還是也用 labeled soundscapes？用什麼 backbone？什麼 loss？
  - **A**: 未直接回覆

---

## 3. 外部資料宣告（sergheibrinza）

**發文者**: sergheibrinza
**用途聲明**: non-commercial academic research

### 3.1 外部資料集

#### 3.1.1 AnuraSet (Cañas et al., 2023)

| 欄位 | 內容 |
|------|------|
| DOI | 10.5281/zenodo.8342596 |
| 來源 | Zenodo: `zenodo.org/records/8342596` |
| 授權 | CC BY 1.0（依 Zenodo 記錄卡驗證，2026-05-21 UTC） |
| 備註 | 隨附論文 (Sci Data 10:771, 2023) 聲明 CC0；此處採用更保守的 Zenodo metadata |
| 作者 | Cañas, J.S., Toro-Gómez, M.P., Sugai, L.S.M. et al.（19 位作者） |
| 用途 | **兩棲類 (Amphibia) 訓練增強** |

#### 3.1.2 InsectSet459 (Faiß, Ghani & Stowell, 2025)

| 欄位 | 內容 |
|------|------|
| DOI | 10.5281/zenodo.14056458（version 0.1, Train+Val） |
| 來源 | Zenodo: `zenodo.org/records/14056458` |
| 外層授權 | CC BY 4.0（透過 Zenodo API 驗證，2026-05-21 UTC） |
| 內部逐檔授權 | CC-BY-NC, CC-BY-SA, CC-BY, CC-BY-NC-SA, CC0（無 ND、無 All Rights Reserved） |
| Zenodo 建立者 | Faiß, M. & Stowell, D. |
| 論文 | Faiß, M., Ghani, B. & Stowell, D. (2025) arXiv:2503.15074, Scientific Data |
| 用途 | **昆蟲 (Insecta) 預訓練** |

#### 3.1.3 iNaturalist Research-Grade Audio Observations

| 欄位 | 內容 |
|------|------|
| 來源 | `api.inaturalist.org/v1/observations` |
| 檢索日期 | 2026-05-21 UTC |
| API 過濾條件 | `license=cc0,cc-by,cc-by-nc; quality_grade=research; sounds=true` |
| 逐檔授權 | 僅 CC0、CC-BY、CC-BY-NC |
| 歸屬標記 | 完整保留 TASL attribution metadata |
| 用途 | **稀有物種補充資料** |

#### 3.1.4 BirdCLEF 2025 Competition Data

| 欄位 | 內容 |
|------|------|
| 來源 | `kaggle.com/competitions/birdclef-2025` |
| 授權 | CC BY-NC-SA 4.0（依競賽規則，2026-05-21 UTC 驗證） |
| 允許用途 | 非商業學術研究及競賽參與（依已接受的規則） |
| 用途 | **鳥類 (Aves) 預訓練** |

### 3.2 Kaggle Datasets（全部 CC0: Public Domain，2026-05-21 透過 Kaggle API 驗證）

| 編號 | Dataset | 作者 | 備註 |
|------|---------|------|------|
| 2.1 | `vladimirsydor/bird-clef-2025-models-v2` | Volodymyr (@vladimirsydor) | BirdCLEF+ 2025 第 2 名 |
| 2.2 | `vladimirsydor/bird-clef-2025-all-pretrained-models` | Volodymyr (@vladimirsydor) | |
| 2.3 | `vladimirsydor/bird-clef-2025-code-final` | Volodymyr (@vladimirsydor) | |
| 2.4 | `rishikeshjani/perch-onnx-for-birdclef-2026` | Rishikesh Jani (@rishikeshjani) | |
| 2.5 | `backtracking/birdclef2026-pseudo-cache-v1` | Le Trong Hieu (@backtracking) | |
| 2.6 | `chaneyma/birdclef-2026-cv9245-moe-artifacts` | Suda (@chaneyma) | |

### 3.3 預訓練模型

| 模型 | 來源 | 授權 | 論文 |
|------|------|------|------|
| Google Perch 2.0 | `huggingface.co/cgeorgiaw/Perch` | Apache 2.0 | van Merriënboer B. et al. (2025), arXiv:2508.04665 |
| EfficientNet (timm) | `rwightman/pytorch-image-models` | Apache 2.0 | 骨幹: `tf_efficientnetv2_s_in21k`, `eca_nfnet_l0` |

### 3.4 合規聲明

#### ShareAlike 處理

- Creative Commons 指引指出：ShareAlike 及 Attribution 條件通常在作品或改編版本**公開分享**時觸發
  - 來源: "Using CC-licensed Works for AI Training", May 2025, `creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf`
- 提交物及訓練模型權重**不由參賽者公開再分發**；而是依據競賽規則的 WINNER LICENSE 條款交付給 Competition Sponsor
- 來源 CC-BY-SA / CC-BY-NC-SA 錄音**不包含在提交物中**

#### 無再分發

- 參賽者不對原始受著作權保護的音訊檔案進行任何再分發

#### 可及性

- 上述所有外部資源均可從其原始提供者處公開取得，受各自授權與條款約束

#### WINNER LICENSE 合規

- 提交物（程式碼 + 訓練權重）將依 WINNER LICENSE 條款以 **Apache 2.0** 授權
- Apache 2.0 授權**不延伸至**：
  - (a) 非參賽者所有的一般商業可得軟體
  - (b) 競賽資料及外部 CC-BY-NC、CC-BY-NC-SA、CC-BY-SA 資料集（僅作為訓練輸入，保留其原始授權；不作為提交物的一部分再分發）

### 3.5 引用文獻

- Cañas et al. AnuraSet. Sci Data 10:771 (2023). DOI: 10.1038/s41597-023-02666-2
- Faiß, Ghani & Stowell. InsectSet459. arXiv:2503.15074 (2025)
- van Merriënboer B. et al. Perch 2.0. arXiv:2508.04665 (2025)
- Sydorskyi V. & Gonçalves F. (2025). Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation: A Case Study on BirdCLEF+ 2025. CEUR-WS Vol. 4038

---

## 4. Ghost Species 處理策略

**發文者**: 匿名（未標示排名）

### 定義

Ghost species = 沒有 focal clip（train_audio 中無對應錄音）但出現在 labeled soundscapes 中的物種。

### 結論

#### 是否重要？

**是，絕對重要。**

#### 該怎麼處理？

- 如果你使用 **perch-proto chain**（Perch + ProtoSSM + MLP-probe + ResidualSSM），**幾乎不用做什麼**，直到你嘗試了更大的改進手段
- 測試了公開 0.946 notebook 對本地驗證集的表現：該 chain 對 ghost species 的辨識能力相當好
- 幾乎所有 **28 個** ghost species 都被正確賦予高機率
- **唯一例外**: `517063`（Southern Orange-legged Leaf Frog，兩棲類 ghost），平均機率 mean_p = **0.747**，可以稍作提升
- **建議**: ghost species 不應該是你的主要擔憂，除非你已經解決了更大的問題

### 相關回覆

- **Dhruv Pai Dukle (1040th)**: 下載了額外資料，甚至訓練了專門的 B0 模型（specialist B0 model），但未帶來改善。注入 0.945 SED+ProtoSSM pipeline 後：
  - 作為「rescue」注入：無改善
  - 全域混合（globally blended）：反而造成退步（regression）
  - 本來打算分享該資料集，但因為導致退步所以沒有分享

---

## 5. 無 Perch 方案的上限探討

**發文者**: sghwr（第 769 名）

### 觀察

- 排行榜上幾乎所有頂尖 notebook 都重度依賴 Perch（提取 audio embeddings 或 Perch-based knowledge distillation）
- 這導致模型方法**嚴重缺乏多樣性**
- 在集成中再加一個 Perch-based model，LB 改善最多只有 **~0.001**，基本在隨機波動範圍內
- 嘗試用純 CNN + mel-spectrogram + EfficientNet-B0（完全不引入任何 Perch 權重/embeddings/蒸餾策略），但**很難突破 public LB 0.88**

### 提出的問題

1. 純 CNN pipeline（零 Perch 參與）能達到的 public LB 上限約是多少？
2. 在當前主流集成框架下，加入訓練良好但單模型分數低於 0.9 的 non-Perch 模型，能否帶來有效的 LB 增益？

### 社群回覆彙總

| 回覆者 | 排名 | 內容 |
|--------|------|------|
| **Arunodhayan** | **7th** | 單模型 LB = **0.946** |
| **tennogh** | 33rd | 最佳 LB score 討論串中有人做出了不用 Perch 的強模型。即使用 Perch，不同方法仍可互補（如用 Perch embeddings 的 notebook 與用 SED 蒸餾的 notebook 相關性不高） |
| **Antoine Masq** | 49th | 僅使用自定義 CNN pipeline（只用 EfficientNetv2_b0，不用 Perch），最佳單模型 LB = **0.936**（後續更新至 **0.943**），集成不同 run 可達 **0.948** |
| **Antoine Masq** | 49th | 使用了 ImageNet pretrained weights，並在 **Xeno-Canto 資料上預訓練** |
| **Antoine Masq** | 49th | 0.943 是用 EfficientNetv2_b0，推論時間 **10 分鐘** |
| **Antoine Masq** | 49th | 使用 **pseudo-labels** + 經典 **TTA**（sliding window + smoothing patterns） |
| **Antoine Masq** | 49th | Loss 函數: **custom ASL**（細節保密） |
| **Antoine Masq** | 49th | 0.936 就已經使用 pseudo labels（改善 teacher model 後提升到 0.943） |
| **MengYe** | 586th | 使用 **asymmetric loss** + 公開 hgnet notebook + 調參可達 **0.90+ LB**，加上一些公開的 post-processing 方法可達 **0.91+ LB** |

### 相關討論（HGNet 細節）

- **BNBU Anpeng Yuan (520th)**: 在公開 HGNet notebook 中將 output chunk 改為 10 秒後，推論程式碼應如何調整？因為這不是 SED 模型，無法產生 frame-level outputs，如果輸入變成 10 秒，輸出也變成 10 秒
- **To be happy (1795th)**: 問 asymmetric loss 是否就是某個特定版本？為什麼基於公開 hgnet notebook 做修改後 LB 下降了 0.03？
- **Dhruv Pai Dukle (1040th)**: 表示也遇到同樣問題（ASL 改動後 LB 下降）

### Jack vs Antoine Masq 的對話（pseudo labels 細節）

- **Jack (21st)**: 問是否用了 pseudo labels 或 TTA 或後處理
  - **Antoine Masq**: 是，用了 pseudo-labels 和經典 TTA（sliding window + smoothing patterns）
- **Jack**: 問他的 0.936 是否也用了 pseudo labels 和 TTA
  - **Antoine Masq**: 0.936 確實用了 pseudo labels（主要是改善了 teacher model）
- **Jack**: 問用什麼 loss
  - **Antoine Masq**: custom ASL，細節保密
- **Jack**: 是不是唯一一個沒能從 pseudo labels 獲益的人？他從頭訓練（from scratch），試過今年討論的所有方法加上去年前兩名的方法，都沒看到別人說的那種提升。唯一做到的是把集成蒸餾到單一模型，但分數略低於集成
  - **Antoine Masq**: 他只是把未標記資料和已標記資料一樣使用，但標籤是由 teacher model 預測的

---

## 6. Pseudo-labeling 效益與方法

**主帖發文者**: MarkDjadchenko（第 74 名）

### 問題背景

- 單模型 0.926 分（solo mode）
- 加入 pseudo-labeling 後分數幾乎相同（only ever so slightly better）
- 嘗試了在 MixUp 中使用 pseudo labels，以及作為補充資料加入訓練集，結果一致差不多
- **只訓練 head**（訓練 backbone 會顯著降低 LB 分數）

### 社群回覆彙總

#### Natsume (918th) — 最詳盡的回覆

| 要素 | 內容 |
|------|------|
| 模型 | hgnetv2_b0 |
| 無 pseudo | 單模型 LB = **0.931** |
| 有 pseudo | 單模型 LB = **0.943**（提升 +0.012） |
| Pseudo 方法 | 嘗試了很多方法，但**簡單地把 pseudo-labels 和訓練樣本串接（concatenate）效果最好** |
| Epoch 數 | **20 epochs** |
| 是否過濾 | **否**。嘗試過按最大標籤機率過濾和按樣本 cross-entropy 過濾，但兩者都損害 LB |
| 推論時間 | 約 **9 分鐘** |
| 訓練視窗 | **隨機 5 秒 clip** |
| 推論視窗 | 仍然 **5 秒** |
| 標籤類型 | **soft labels** |
| 集成嘗試 | 嘗試了各種集成（不同 folds、不同 backbone、不同 mel-spectrogram 參數），**全部沒用** |
| HGNetV2 vs EfficientNet-B0 | 未用 pseudo 前 EfficientNet-B0 的 LB 約低 **0.02**；用 pseudo 後差異不顯著 |
| Perch 蒸餾 | **是**，使用 Perch 和 train_soundscapes 預訓練了 HGNetV2 backbone |
| Pseudo-blending 時是否微調 backbone | **是** |
| 本地 CV | 無 pseudo CV = **0.989**；有 pseudo CV = **0.986↓**（用完全相同的驗證集）|

#### CV 分數討論

- **Jack (21st)**: 0.986-0.989 聽起來非常高。5-fold 平均 val AUC 約 0.955-0.96，最終 LB 0.932。是否在驗證集中包含了 audio？
- **Ochir Dorzhiev (942nd)**: 懷疑高 CV 是因為用了 train_audio 在驗證集中
- **Ochir Dorzhiev**: 有趣的是用 train_audio 做驗證仍能達到不錯的 LB 分數 (0.931)
- **Zejun_ (11th)**: CV 0.98+ 難以置信（同意可能用了 train_audio 驗證）

#### 五折 (5-fold) 預期增益

- **!!!!! (126th)**: 如果是 one fold，五折可能達到 **0.95+**，因為五折通常能提升約 **0.006 - 0.01 LB**

#### Pseudo label 迭代輪數

- **Donghui Zhang (1562nd)**: 好奇大家做幾輪 pseudo-tag 迭代才失效
- **Tucker Arrants (203rd)**: 目前在**第 5 輪**，0.919 → 0.936 的增長。預期效果即將 plateau

#### Ali Ozan Memetoglu (3rd) 的 Pseudo 替換方法

- AUC 在各 fold 間差異顯著
- Labeled SC 資料保持為 **gold labels**，pseudo labels 作為**替代品（substitutes）**使用
- 具體做法：對每個 XC clip，以機率 x 將其 audio+label **替換**為一個 soundscape pseudo chunk（該 chunk 的 teacher 預測 top species 與該 XC clip 的真正 primary label 匹配）
- 這使 public LB 提升約 **+0.01**
- 此方法類似**去年第 2 名的方法**

#### Pseudo Labels 的洩漏問題

- **Nick Pellegrin (932nd)**: 如果 teacher model 是在 labeled soundscape + focal data 上訓練的，然後用它生成 unlabeled soundscape 的 pseudo labels，再用所有 labeled soundscapes 作為新 student model 的驗證集，這是否造成洩漏？（因為 student model 是對一個 teacher model 訓練過的集合做驗證）
- **Tucker Arrants (203rd)**:
  - 是的，需要小心洩漏
  - **最佳做法**: 生成 **OOF (Out-Of-Fold) pseudo labels**（每個 fold 有自己的 pseudo label 包），但這意味著每次 fold 改變都需要重新生成 pseudo labels
  - 如果算力有限，可以忽略洩漏繼續進行，但要意識到 **CV 會被膨脹**
  - 對此競賽，如果主要依賴排行榜評估（如很多人那樣），洩漏不是那麼嚴重。前幾屆比賽的解決方案中，有人**故意引入洩漏**因為他們更依賴排行榜而非 CV

#### 驗證策略討論

- **Zejun_ (11th)**: 直接用 LB 作為 CV（"Just take LB as CV."）
- **Nick Pellegrin**: 確實用 LB 做最終驗證（因為本地 val-auc 與 LB 相關性不太好），但只靠 LB 驗證很難知道何時保存模型。是否每次都調 lr/num_epochs/etc. 然後提交到 LB？
  - **Zejun_**: 用 **loss** 來決定何時停止訓練。仍然需要多次提交到 LB
- **Nick Pellegrin**: 用那個策略（基於 loss checkpoint，不用驗證集），無 pseudo labels 的最高單模型 LB 是多少？是用 folds 還是在全部資料上訓練單一模型？
  - **Zejun_**: **No fold, single model, LB = 0.921**

#### Nick Pellegrin 的 Pseudo Labels 失敗案例

- 使用 perch+protossm 和自己的集成（兩者都在全部或部分 labeled soundscapes 上訓練）生成 pseudo labels
- 訓練 5-fold EfficientNet，用所有 labeled soundscapes 作為驗證集
- Training AUC 達到約 **0.95**，但 **LB 約 0.5**
- 懷疑是洩漏造成的

#### 28 個 no-signal species

- **PRASHANT SHUKLA91 (467th)**: 能否解決 28 個 no-signal species？這個數字在 private set 中可能更大
- **BNBU Anpeng Yuan (520th)**: 問是否完全凍結 backbone（如 ConvNeXt、HGNet），只直接訓練 head

---

## 7. 失敗方法經驗分享

**發文者**: BNBU Anpeng Yuan（第 520 名）

### 明確失敗的方法（有限時間內請三思再試）

| # | 失敗方法 | 詳情 |
|---|---------|------|
| 1 | **ASL Loss、Focal BCE、CE** 在 CNN/SED 模型上表現不好 | 在訓練 CNN 模型或 SED 模型時，ASL Loss、Focal BCE 和 CE **都不如預期** |
| 2 | **SuMix + CutMix + MixUp 組合** 比單獨用 MixUp **更差** | 單獨使用 MixUp 的效果與單獨使用 SuMix 相同 |
| 3 | **更長的 chunk 時長**（如 15s）在 clip-level 模型上**更差** | 當模型只能輸出 clip-level 而非 frame-level 預測時，嘗試更長的 chunk duration 會導致效果變差 |

### 相關討論

- **Nick Pellegrin (932nd)**:
  - 發現 **CE + hard labels** 比 BCE-with-logits-loss（使用 pos_weight=0.3）效果更好：單 fold SED 模型從 ~0.88 LB 提升到 ~0.89 LB
  - 看到很多常見 SED 方法對 **0.5×clip + 0.5×frame** 施加 loss，但推論時只用 **frame max logits**
  - 去年冠軍用 20 秒 clip 效果更好。他目前的做法是取 20 秒 clip 中**中間 5 秒的標籤**（嘗試加入 20 秒完整標籤反而讓模型偏向預測中心 5 秒之外的物種）
- **BNBU Anpeng Yuan**: BCE 對他的模型可能是最優的
- **BNBU Anpeng Yuan**: MixUp 似乎與 CE loss **不兼容**

---

## 8. 多樣性與集成策略

**發文者**: Zejun_（第 11 名）

### 核心觀點

**"Diversity is all you need."**（多樣性就是你需要的一切）

- 單模型的分數不能決定一切
- 重點在於集成和後處理帶來了多少改善

### 社群回覆彙總

| 回覆者 | 排名 | 內容 |
|--------|------|------|
| **cmasch** | **16th** | 完全同意。一些最佳集成提升（**+0.03**）來自單獨看起來很弱的模型（**<0.90**）。現階段更關注多樣性而非獨立分數，特別是跨 taxa 和 species 的互補行為。難點在於找出什麼真正增加了新的錯誤模式，而非只是複製相同信號。後處理也有幫助，但主要是小型定向修正。**下一步嘗試: higher-resolution mels** |
| **Zejun_** | 11th | 多樣性 = lift / {overfit} |
| **noobanot** | 123rd | 加了三個 ~0.930 的 SED 模型到 0.939 的 CNN 集成中，只提升到 **0.940**。如何從低分模型獲得那麼多增益？ |
| **Zejun_** | 11th | 認為 **0.02+ 的集成提升是可能的**。可以計算模型間的 **correlation** 來設定權重 |
| **OpPrime** | 679th | 在 perch→protossm 設定中，從 0.922 到 0.930+，再加上 Tucker 的 5-fold SED head 到 0.946。集成仍然是關鍵 |
| **Zejun_** | 11th | **一個額外的模型 > overlap/TTA** |
| **shanzhong8** | 36th | 問 Zejun_ 目前的單模型分數範圍 |
| **Zejun_** | 11th | 保密，但「不高」 |
| **Zejun_** | 11th | **任何 >0.925 的單模型在集成結構中都會是強力幫手** |

### Tucker Arrants (203rd) 的重要觀察

- 大部分增益**不能**只歸因於蒸餾 SED 的集成，因為那些 notebook 是數千輪後處理微調和**過擬合排行榜**的產物
- 當他拿一個 SSM 公開 notebook 加上自己最好的蒸餾模型，幾乎**沒有集成增益**，遠少於某些 notebook 分享的 ~0.1 提升
- 他用**不同 backbone 的另一個蒸餾模型**集成，反而獲得更多提升
- 非序列模型的加入只是允許了更多**過擬合**
- 有超過 **200 個公開 notebook** 使用他的資料集，每個都有多個版本 → 數千次提交，最高分的那個獲得關注但**不可重現**，他對它們在 private leaderboard 上的表現持懷疑態度
- **蒸餾 SED 和 SSM 模型共享 Perch 特徵空間**，所以它們不可能有太大差異
- **真正的多樣性**將來自用**完全不同的架構**在**不同的 mel 上從頭訓練**的模型

---

## 9. Distilled SED 運行時間優化

**發文者**: 匿名

### 問題

在 Colab（G4 high RAM）上嘗試重現 Tucker Arrants 的 `bc2026-distilled-sed` notebook。使用完整 trainaudio 和 trainsoundscape 訓練蒸餾模型時，每個 epoch 約需 **20 分鐘**。唯一能讓推論在 90 秒內完成的方式是**關閉 perchv2**。

### Tucker Arrants (203rd) 的回覆

- 問題是 **Drive IO 瓶頸**
- **解決方案**: 避免直接從 Drive 讀取檔案，先將資料**解壓到 Colab SSD**
- 這樣做可以把訓練時間降到每 epoch **40 秒**（在 G4 上）

---

## 10. Training Recipe 全面討論

**發文者**: Nick Pellegrin（第 932 名）

### 背景

- 正在訓練簡單的 5-fold SED 模型（使用去年冠軍解決方案的架構）
- 單 fold 無法突破 LB 0.9（單 fold val-auc 約 0.93-0.95）
- 看到別人不用 pseudo labels 就能達到單模型 0.93
- 受限於每週 GPU 配額，無法做詳盡的消融實驗

### 10.1 BCE vs CE

#### Nick 的經驗

- 從 BCE (with logits loss) 切換到 CE 後模型效果**下降**（單 fold LB 從 0.85-0.88 降到 0.78-0.79）
- 不確定在討論中人們說的「CE」是否其實是指「BCE」

#### Tucker Arrants (203rd) 的回覆

- **CE**: 收斂更快
- **BCE**: CV 和 LB 分數更好
- 總結: BCE 在最終效果上優於 CE

#### Tucker Arrants 對 CE vs BCE 原理的解釋

- CE 是典型的多類別（multi-class）問題 loss，用於 one-hot（單標籤）目標，機率和為 1
- 本競賽是**多標籤（multi-label）**問題，多個物種可同時出現在一個 clip 中
- 但因為競賽指標是 **AUC**，calibration 不重要，**只有排名重要**
- CE 搭配未正規化的 soft targets 不再是機率 loss，而變成 **ranking loss**，恰好與 AUC 獎勵的東西一致
- CE 處理**類別不平衡**也比 BCE 好：稀有類別只在出現時才獲得有意義的梯度，而不是被 99% 不存在的行的「be negative everywhere」信號淹沒
- 這是此特定問題的一個巧妙 loss 技巧

#### 去年冠軍（Nikita Babich）的 CE 觀點

- CE 和 BCE/Focal 在 learning rate 和 epoch 數調好時能給出類似結果
- 但 CE 結果稍好，原因推測：
  - CE 中每個標籤的更新幅度取決於正面標籤的分類好壞。如果稀有正面標籤 A 得到低機率，則有較高機率的過度代表負面標籤 B 會被**更強的更新推向零**
  - CE 更好地處理不平衡標籤，通過在 Softmax 無法為 A 給出更高分數時**懲罰過度代表的類別 B**（因為 B 在之前的不平衡更新中已獲得過高的 logits）
  - 他**沒有將樣本標籤正規化為和為一**，動機是讓更困難的樣本（有更多正面標籤的）對 loss 有更大影響

#### To be happy (1796th) 的困惑

- 問 CE 不是二分類的 loss 嗎？BCE 才是多類別的 loss？無法想像如何在此競賽中使用 CE
  - Tucker 的解釋見上方

### 10.2 Soft vs Hard Secondary Labels

- **Tucker Arrants**: 他使用 **hard labels**，但看到有人用 **0.5** 來反映 secondary labels 較低的「品質」。認為差別不大
- **Natsume (918th)**: 使用 **soft labels**

### 10.3 Background Noise 增強

- **Nick**: 嘗試從 `honglihang/background-noise` 資料集引入背景噪音作為訓練增強，但聽到有些音訊中明確有叫聲
- **Tucker**: 「no call」更準確地說是「no bird call」— 很多這些 clip 中有昆蟲和兩棲動物的聲音。是否影響訓練是經驗問題。理論上，你可能在懲罰模型正確預測「no call」clip 背景中的東西。他**跳過了 no call augmentation**

### 10.4 Label Smoothing

- **Nick**: 目前使用 label smoothing（約 **0.02**）
- **Tucker**: Label smoothing 在他的訓練中**沒有太大差異**，他**關閉了**它

### 10.5 Validation Set

- **Nick**: 目前只使用 OOF labeled soundscapes 做驗證集（因為更接近測試域）。但每個 fold 的物種覆蓋很薄（fold 0 只有 **63 / 234** 個物種）。是否在 OOF focal audio 中加入 train_audio 能產生更好的與 LB 相關的 OOF-AUC？
- **Tucker**:
  - 驗證集太薄但確實更能反映測試集
  - 前幾屆主要靠**提交到排行榜**作為驗證策略
  - **技巧**: 線上找到與競賽物種有合理重疊的**新熱帶（Neotropical）soundscapes**，用它們做驗證。這些很適合驗證模型對類似生態區但不同地點的泛化能力。他已找到很多這樣的資料集
- **Ali Ozan Memetoglu (3rd)**: AUC 在各 fold 間差異顯著
- **Jack (21st)**: 想知道有沒有人嘗試用 **S22** 作為驗證和/或用 S22 以外的所有站點作為驗證。想像這是人們做單模型時的做法
- **Jack**: 完整 labeled soundscapes 對 pseudo-label 訓練的模型似乎**不是最優的**驗證集

### 10.6 EMA Checkpoints

- **Tucker**: 他**不使用** EMA，但值得快速測試：用和不用 EMA 各跨 2-3 個 seed 提交到 LB

### 10.7 Freezing the Backbone

- **Nick**: 目前不凍結 backbone（與大多數人相同），但好奇是否有人只訓練 attention head
- **MarkDjadchenko (原帖 74th)**: 問 pseudo-labeling 主帖的發文者是否在 pseudo-blending 時微調 backbone
- **Natsume (918th)**: 是，微調 backbone
- **提醒（從主帖 MarkDjadchenko 的帖子）**: 訓練 backbone 會**顯著降低 LB 分數**（在某些設定下）

### 10.8 訓練視窗大小

- **MR.h (74th)**: 問 5 秒還是 20 秒訓練視窗？他的實驗中 5 秒總是最好，但別人用 20 秒
- **Natsume**: 使用隨機 **5 秒** clip
- **Nick Pellegrin**: 去年冠軍用 20 秒效果更好。他目前做法是取 20 秒 clip 中**中間 5 秒的標籤**

### 10.9 Loss 進階討論

- **Nick**: CE + hard labels 比 BCE (pos_weight=0.3) 好：fold-0 SED 從 ~0.88 提升到 ~0.89
- **Nick**: 0.5×clip + 0.5×frame loss，推論只用 frame max logits
- **BNBU Anpeng Yuan**: BCE 對他的模型可能最優
- **BNBU Anpeng Yuan**: MixUp 似乎與 CE loss 不兼容

---

## 11. train_soundscapes_labels.csv 標註方法論（官方回覆）

**發文者**: kta_jpn（第 482 名）

### 發問動機

在逐行檢查 `train_soundscapes_labels.csv` 時發現：

- **蛙類/昆蟲標籤**: 在標記的 5 秒窗口中有清楚的聲學和頻譜證據，模型一致同意
- **部分目標鳥類標籤**: 在音訊和頻譜圖中都**無法偵測到**該鳥類，但 Y=1。模型嚴重不一致：
  - **Perch v2** 讀數高（如 0.93-0.98）
  - **Tucker Distilled SED** 讀數接近零（0.02-0.05）
- 此模式在**多個窗口**中重複，尤其是 **S22 夜間合唱密集檔案**
- 標籤檔案有**精確 2 倍重複**（1,478 行 / 739 唯一行）
- **45%** 的行同時標記 **≥5 個物種** — 這不太像純「人工專家標註」

### 提出的問題

1. **時間粒度**: 5 秒窗口標籤是否逐物種在每個 5 秒窗口中獨立驗證？還是 (a) 從較長窗口的專家觀察擴展到所有 12 個窗口，(b) 其他生成方式？
2. **AI 輔助**: 是否有任何模型（Perch v2、BirdNET 等）用作人類標註者的建議或預篩選？
3. **不可見物種**（昆蟲聲型 + 一些在 train_audio 中有 0 片段的兩棲類，如 517063 Pithecopus azureus）：專家如何在 soundscapes 中識別它們？
4. **2× 重複行**: 精確 2× 重複的來源是什麼？
5. **標註者間一致性**: 每個檔案是否有多個標註者？

### 為什麼這很重要（建模影響）

- 公開 0.946 LB 幾乎都是 Perch + ProtoSSM + Distilled-SED 堆疊
- 這些共享一個特徵空間
- **Perch vs SED 在合唱窗口中對目標鳥類的分歧**正是標籤語義決定正確混合策略的地方：
  - 如果標籤是**窗口精確的**且專家能捕捉微弱 TP → **信任 Perch** 的高響應
  - 如果標籤是**窗口粗糙的**（檔案級擴展）→ **信任 SED** 的低響應，混合應平滑分歧而非放大

### 官方回覆 — Stefan Kahl (Competition Host)

| 問題 | 回覆 |
|------|------|
| 時間粒度 | **是的，將專家標籤匯聚到 5 秒窗口中。專家標籤可能更短或更長**（若更長則拆分到所有 5 秒窗口） |
| AI 輔助 | **否** |
| 不可見物種 | 有些物種只出現在 train soundscapes 中，沒有 focal examples。詳見 "Data" tab |
| 2× 重複行 | **是我的失誤，忽略重複行即可** |
| 標註者間一致性 | **通常沒有**多個標註者，所以預期標註品質會有差異 — 也預期會有錯誤；手動標註 soundscapes 很困難 |

### 關鍵結論

- 標籤是**窗口粗糙的**（較長觀察擴展到 5 秒窗口），這意味著在混合策略中應傾向平滑 Perch 和 SED 的分歧
- 1,478 行中有精確的 2× 重複是主辦方的失誤，應去重（739 唯一行）
- 標註品質參差不齊，因為通常只有一個標註者，且手動標註 soundscapes 很困難

---

## 12. 高分 Notebook 發布倫理

**發文者**: Nick Pellegrin（第 932 名）

### 背景

過去幾天很多高分 notebook 被發布（0.940-0.945 LB），大幅動搖了排行榜

### 社群觀點彙總

| 回覆者 | 排名 | 觀點 |
|--------|------|------|
| **Antoine Masq** | 49th | 矛盾心情：幫助新手入門很好，但讓人能零努力 fork 拿金牌令人沮喪。他從 top-50 被拖到 top-600。Kaggle 應考慮更早禁止 notebook/model 分享。通常有用的分享在比賽早期 |
| **Zejun_** | 11th | 高分程式碼只是公開模型的消融實驗，**不穩定**（重複運行幾乎無法得到相同的高分） |
| **FOYSAL** | 324th | 同樣觀察（不穩定） |
| **BNBU Anpeng Yuan** | 520th | 倫理上對大多數選手不公平，但 Kaggle 允許，所以認為發布高分程式碼是合理的 |
| **tennogh** | 33rd | 比賽結束前 **7 天**有截止期，新 notebook 無法發布。除此之外沒有真正的規則。之前比賽有人到最後一刻才分享（有時只分享模型權重） |
| **Tucker Arrants** | 203rd | 沒有真正的規則，但大多數人嘗試在比賽早期分享。最近的高分 notebook 部分是由他的 notebook 促成的（他沒預料到會被這樣使用）。如果發現了強大但不想大幅影響排行榜的新方法，可以在討論帖中用偽碼/圖表分享，或等比賽結束 |
| **Jack** | 18th/21st | 如果 Tucker 把 notebook 設為私有會造成完全混亂 |
| **Mattia Angeli** | 44th | 或者只把訓練好的模型權重資料集設為私有 — 因為重新訓練/蒸餾他的模型很耗時，設為私有基本上會使 top 10-15 個公開 notebook 立即失效 |
| **Mattia Angeli** | 44th | 承認高分 notebook cascade 部分是他造成的。他只是想測試在 Tucker 的蒸餾 SED 上加一些時間上下文（temporal context）是否有幫助。結果是基本上是兩個公開模型的 10 行推論側集成，且分數略低於當時最好的公開 notebook，所以發布感覺相對無害。結果發現一兩個參數的小改動就能大幅移動 LB，引發了目前的高 LB notebook cascade |
| **Mattia Angeli** | 44th | 高於 ~0.94 的分數似乎**很難穩定重現**，懷疑大部分是 public-LB 過擬合/噪聲而非真正穩健的增益 |

---

## 附錄 A：關鍵數字速查表

| 指標 | 值 | 來源/方法 | 備註 |
|------|-----|-----------|------|
| 公開最高單模型 LB | **0.946** | OverfitOracle: SED + pseudo + XC | 無集成 |
| 公開最高單模型 LB | **0.946** | Arunodhayan (7th) | 方法未透露 |
| 純 CNN 單模型 LB | **0.943** | Antoine Masq: EfficientNetv2_b0 + pseudo + TTA | 無 Perch，10min 推論 |
| 純 CNN 集成 LB | **0.948** | Antoine Masq: 多個 run 集成 | 無 Perch |
| HGNetV2_b0 單模型 | **0.931** → **0.943** | Natsume: 加 pseudo labels +0.012 | 20 epochs |
| 無 fold 單模型 | **0.921** | Zejun_ (11th): 無 pseudo, 基於 loss checkpoint | |
| Perch→ProtoSSM | **0.922** → **0.930+** | OpPrime 描述 | |
| + Tucker 5-fold SED | → **0.946** | OpPrime 描述 | |
| 弱模型集成提升 | **+0.03** | cmasch (16th): 用 <0.90 模型 | 多樣性關鍵 |
| 5-fold 預期增益 | **+0.006 ~ +0.01** | !!!!! (126th) | 經驗值 |
| Pseudo 迭代輪數 | **5 輪**, 0.919→0.936 | Tucker Arrants | 預期 plateau |
| CE + hard labels 增益 | ~0.88→~0.89 LB | Nick Pellegrin | 單 fold SED |
| Ghost species mean_p | **0.747** (最低) | 517063 Southern Orange-legged Leaf Frog | Perch chain |
| 標籤重複 | **2×** (1478→739 唯一) | 官方確認為失誤 | |
| 多標籤行 | **45%** 標記 ≥5 物種 | kta_jpn 觀察 | |

---

## 附錄 B：方法推薦 vs 不推薦速查

### 推薦（已驗證有效）

- BCE loss（多數人的最佳 loss）
- Pseudo labels — 簡單串接到訓練集（Natsume 方法）
- Soft labels（Natsume）
- MixUp（單獨使用，不混合其他增強）
- 5 秒隨機 clip 訓練
- 多樣性集成（不同架構、不同 mel）
- Xeno-canto API 拉取額外資料（按物種+品質過濾，限制每物種/錄音者上限）
- Perch 蒸餾到 HGNetV2 backbone 再微調
- LB 作為主要驗證（CV 不可靠）
- Loss 作為 early stopping 信號
- Neotropical soundscapes 作為輔助驗證集
- OOF pseudo labels（避免洩漏的最佳做法）
- 解壓資料到 Colab SSD（解決 IO 瓶頸）
- CE loss 搭配未正規化 soft targets（ranking loss 技巧，進階用法）

### 不推薦 / 已驗證失敗

- ASL Loss / Focal BCE / CE 在 CNN/SED 上（BNBU 經驗）
- SuMix + CutMix + MixUp 組合（不如單獨 MixUp）
- 更長 chunk（15s）用於 clip-level 模型
- 過濾 pseudo labels（按機率或 cross-entropy，Natsume 驗證損害 LB）
- 針對 ghost species 訓練 specialist 模型（Dhruv 驗證無改善或退步）
- Label smoothing（Tucker: 差異不大，已關閉）
- EMA（Tucker: 不使用，但值得快速測試）
- 直接從 Drive 讀取訓練資料（IO 瓶頸，每 epoch 20 min）
- 訓練 backbone（在某些設定下顯著降低 LB）
- MixUp + CE loss 組合（不兼容）

---

## 附錄 C：重要參賽者索引

| 參賽者 | 排名 | 主要貢獻 |
|--------|------|----------|
| OverfitOracle | 485th | 單模型 0.946（SED + pseudo + XC） |
| Arunodhayan | **7th** | 單模型 0.946 |
| Ali Ozan Memetoglu | **3rd** | Pseudo 替換方法（+0.01），fold AUC 差異大 |
| Zejun_ | **11th** | 多樣性理念，no fold 單模型 0.921，LB 作 CV |
| cmasch | **16th** | 弱模型集成 +0.03，higher-res mels |
| Jack | **21st** | 積極探討 pseudo labels（自認失敗）、fold 策略 |
| tennogh | 33rd | Perch 內方法多樣性觀察 |
| shanzhong8 | 36th | |
| Mattia Angeli | 44th | 高分 notebook cascade 反思 |
| Antoine Masq | 49th | 純 CNN 0.943（EfficientNetv2_b0 + custom ASL + pseudo + TTA） |
| MarkDjadchenko | 74th | 積極提問者（pseudo 方法細節） |
| MR.h | 74th | 訓練視窗討論 |
| noobanot | 123rd | 集成增益有限的經驗 |
| !!!!! | 126th | 5-fold 增益預期 +0.006~0.01 |
| Tucker Arrants | 203rd | Distilled SED notebook 作者、pseudo OOF 建議、CE vs BCE 深度分析、Perch 特徵空間觀察 |
| Nick Pellegrin | 932nd | Training recipe 系統性討論、pseudo labels 洩漏問題 |
| Natsume | 918th | HGNetV2 + pseudo 詳盡實驗分享 |
| BNBU Anpeng Yuan | 520th | 失敗方法經驗分享 |
| MengYe | 586th | Asymmetric loss + HGNet = 0.90+ |
| Dhruv Pai Dukle | 1040th | Ghost species specialist 模型失敗經驗 |
| Stefan Kahl | **Host** | 官方回覆標籤方法論 |
| sergheibrinza | — | 外部資料宣告範本 |
| kta_jpn | 482nd | 標籤方法論深度調查 |

---

## 附錄 D：外部資源清單

| 資源 | 類型 | 用途 |
|------|------|------|
| Xeno-canto API | 音訊資料 | 額外鳥類錄音（按物種+品質過濾） |
| AnuraSet (Zenodo) | 音訊資料 | 兩棲類訓練增強 |
| InsectSet459 (Zenodo) | 音訊資料 | 昆蟲預訓練 |
| iNaturalist API | 音訊資料 | 稀有物種補充 |
| BirdCLEF 2025 Data | 音訊資料 | 鳥類預訓練 |
| Google Perch 2.0 (HuggingFace) | 預訓練模型 | 音訊 embedding 提取、知識蒸餾 |
| EfficientNet / timm | 預訓練模型 | CNN backbone |
| ESP-Aves2 (HuggingFace) | 預訓練模型 | BEATs-based 生物聲學骨幹（授權待確認） |
| `honglihang/background-noise` (Kaggle) | 音訊資料 | 背景噪音增強（注意含動物聲音） |
| Tucker Arrants' pseudo cache | Kaggle Dataset | 蒸餾 SED 預計算資料 |
| 去年冠軍 (Nikita Babich) 推論 notebook | 參考架構 | SED 架構參考 |
| 去年第 2 名 (Volodymyr) 方法 | 參考方法 | Pseudo 替換策略參考 |
| Neotropical soundscapes（線上） | 驗證資料 | 輔助驗證集（Tucker 建議） |

---

## 附錄 E：全部優化方向總表（51 項）

以下彙整討論中提及的**所有**優化方向，按類別分組，每項標註來源與具體數據。

### E.1 模型架構（7 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 1 | SED（Sound Event Detection）架構，產出 frame-level 預測 | OverfitOracle (485th), Nick Pellegrin (932nd) | 單模型 0.946 |
| 2 | Perch + ProtoSSM + MLP-probe + ResidualSSM chain | Ghost species 帖 | 0.922→0.930+ |
| 3 | Perch 蒸餾到 HGNetV2/EfficientNet backbone 再微調 | Natsume (918th) | HGNetV2_b0 蒸餾後 0.931 |
| 4 | 純 CNN pipeline（EfficientNetv2_b0，不用 Perch） | Antoine Masq (49th) | 單模型 0.943 |
| 5 | 多 backbone（`tf_efficientnetv2_s_in21k`, `eca_nfnet_l0`） | sergheibrinza 外部宣告 | — |
| 6 | BEATs-based backbone（ESP-Aves2） | 授權討論帖 | 授權待確認 |
| 7 | Higher-resolution mel spectrograms | cmasch (16th) | 下一步嘗試 |

### E.2 Loss 函數（6 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 8 | **BCE（BCEWithLogitsLoss）** — 多數人的最佳 loss | Tucker (203rd), BNBU (520th) | Tucker: CV 和 LB 分數更好 |
| 9 | **CE（CrossEntropyLoss）作為 ranking loss** — 未正規化 soft targets，不 normalize to sum=1 | Tucker (203rd), 去年冠軍 Nikita Babich | CE 收斂快，處理類別不平衡更好 |
| 10 | **Focal BCE** | 本 repo C_mlp_focal | γ=2.0, α=0.25 |
| 11 | **Custom ASL（Asymmetric Loss）** | Antoine Masq (49th), MengYe (586th) | ASL + HGNet = 0.90+ LB |
| 12 | **0.5×clip + 0.5×frame 混合 loss**，推論只用 frame max logits | Nick Pellegrin (932nd) | 常見 SED 做法 |
| 13 | CE + hard labels（pos_weight 調整） | Nick Pellegrin | fold-0 SED 從 ~0.88→~0.89 |

### E.3 資料增強（4 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 14 | **MixUp（單獨使用）** | 多人確認有效, BNBU (520th) | 單獨 MixUp = 單獨 SuMix |
| 15 | SuMix + CutMix + MixUp 組合 | BNBU (520th) | **失敗** — 比單獨 MixUp 更差 |
| 16 | Background noise 增強（`honglihang/background-noise`） | Nick Pellegrin, Tucker | Tucker 跳過（clip 含昆蟲/蛙聲） |
| 17 | Label smoothing（~0.02） | Nick Pellegrin, Tucker | Tucker: 差異不大，已關閉 |

### E.4 Pseudo-labeling（7 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 18 | **Teacher model 生成 pseudo labels，直接串接訓練集** | Natsume (918th) | 0.931→0.943（+0.012） |
| 19 | 不過濾 pseudo labels（不按機率或 cross-entropy 過濾） | Natsume | 過濾反而損害 LB |
| 20 | 使用 **soft labels**（而非 hard labels） | Natsume | — |
| 21 | **OOF pseudo labels**（每 fold 獨立生成，避免洩漏） | Tucker (203rd) | 最佳做法 |
| 22 | **Pseudo 替換法**: 以機率 x 將 XC clip 替換為 teacher top species 匹配的 soundscape chunk | Ali Ozan Memetoglu (3rd) | +0.01 LB，類似去年第 2 名 |
| 23 | 多輪 pseudo 迭代（iterative self-training） | Tucker (203rd) | 5 輪，0.919→0.936，預期 plateau |
| 24 | Pseudo-blending 時微調 backbone | Natsume (918th) | 是 |

### E.5 外部資料（6 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 25 | **Xeno-canto API** 拉取額外音訊（按物種+品質過濾，限制每物種/錄音者上限） | OverfitOracle (485th), Antoine Masq (49th) | 單模型 0.946 的關鍵要素 |
| 26 | **AnuraSet**（兩棲類訓練增強） | sergheibrinza | Zenodo CC BY 1.0 |
| 27 | **InsectSet459**（昆蟲預訓練） | sergheibrinza | Zenodo CC BY 4.0 |
| 28 | **iNaturalist Research-Grade Audio**（稀有物種補充） | sergheibrinza | API: cc0/cc-by/cc-by-nc |
| 29 | **BirdCLEF 2025 Competition Data**（鳥類預訓練） | sergheibrinza | CC BY-NC-SA 4.0 |
| 30 | Neotropical soundscapes（線上找的輔助驗證集） | Tucker (203rd) | 驗證泛化能力 |

### E.6 訓練策略（8 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 31 | **5 秒隨機 clip** 訓練（非 10s/15s/20s） | Natsume (918th), MR.h (74th) | 5 秒 clip-level 最佳 |
| 32 | 20 秒 clip + 中間 5 秒標籤（SED 模型） | Nick Pellegrin, 去年冠軍 | SED frame-level 才適用 |
| 33 | Secondary labels 設為 **soft（0.5）** vs hard（1.0） | Tucker, Natsume | Tucker: 差別不大 |
| 34 | **只訓練 head，凍結 backbone** | MarkDjadchenko (74th) | 訓練 backbone 在某些設定顯著降 LB |
| 35 | Backbone 先凍後解凍（multi-stage training） | Nick Pellegrin 提問 | 未有明確回覆 |
| 36 | EMA Checkpoints | Nick Pellegrin, Tucker | Tucker 不用，但值得快速測試 |
| 37 | 基於 **loss** 做 early stopping（不依賴 val AUC） | Zejun_ (11th) | 搭配 LB 提交驗證 |
| 38 | No fold, single model, 全資料訓練 | Zejun_ (11th) | 0.921 LB |

### E.7 驗證策略（4 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 39 | **LB 作為主要驗證**（local CV 不可靠） | Zejun_ (11th) | "Just take LB as CV" |
| 40 | OOF labeled soundscapes 做驗證集 | Nick Pellegrin | fold-0 只有 63/234 物種 |
| 41 | Neotropical soundscapes 作輔助驗證 | Tucker (203rd) | 驗證泛化到類似生態區 |
| 42 | 用 **S22** 站點做驗證 / 用 S22 以外做驗證 | Jack (21st) | 待確認 |

### E.8 集成與後處理（7 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 43 | **多樣性集成**（不同架構 + 不同 mel） | Zejun_ (11th), Tucker (203rd) | "Diversity is all you need" |
| 44 | 弱模型（<0.90）加入集成 | cmasch (16th) | 集成 +0.03 |
| 45 | 計算模型間 **correlation** 來設定集成權重 | Zejun_ (11th) | 0.02+ 集成提升可能 |
| 46 | 5-fold 集成 | !!!!! (126th) | 預期 +0.006~0.01 LB |
| 47 | **Sliding window + smoothing**（推論端 TTA） | Antoine Masq (49th) | 經典 TTA |
| 48 | 一個額外模型 > overlap/TTA | Zejun_ (11th) | — |
| 49 | 後處理小型定向修正 | cmasch (16th) | — |

### E.9 環境/工程優化（2 項）

| # | 方向 | 來源 | 具體數據 |
|---|------|------|----------|
| 50 | 解壓資料到 **Colab SSD**（不從 Drive 直讀） | Tucker (203rd) | 每 epoch 20min→40s |
| 51 | 修復 soundscape **2× 重複標籤** | Stefan Kahl (Host) | 官方確認失誤，去重到 739 |
