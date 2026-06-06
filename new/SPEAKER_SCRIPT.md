# BirdCLEF+ 2026 · 第二部分講稿(Route B)

**報告定位:** 本組共做兩個嘗試 — **Route A(第一個,隊友負責)+ Route B(第二個,本人負責)**
**範圍:** Page 9-14(約 5 分 45 秒)
**風格:** 正式書面口語,可直接照念

---

## 📄 Page 9 · 嘗試 Transfer Learning(交棒頁)

**時長:40-50 秒 · 交棒 + 雙路線敘事,語速略慢**

> 謝謝 XXX 的介紹。
>
> 剛才 XXX 介紹的是本組的**第一個嘗試 Route A** — 復刻去年冠軍的 **EfficientNet-B2 + self-training** 架構,整體 val_auc 達到 0.9806,soundscape 子集 0.6997。
>
> 接下來是**本組的第二個嘗試 Route B**,由我負責,走**完全不同的方向** — 用 **Google Perch v2 做 Transfer Learning**。
>
> 為什麼本組決定並行試兩條路?第一次簡報原本規劃的是 **Mel-spectrogram 加 CNN ResNet 從頭訓練**,目標 ROC-AUC 0.85。但在有限算力與有限標註資料下,從頭訓 ResNet 容易過擬合。Google 已經在 **10,000 多個鳥類物種上大規模訓練** Perch,站在巨人肩膀上,也是當前 representation learning 的主流做法 — 因此 Route B 決定試看看這條路能走多遠,與 Route A 做對照。
>
> 左邊是第一次簡報的預計規劃,右邊是 Route B 此次實際採用。接下來幾頁會展開這個方法的架構與實驗。

**轉場:** 念完平順切下一頁,不要停頓。
**提醒:** XXX 改成隊友實際名字(例如 MM、CC 等)。

---

## 📄 Page 10 · 整體架構

**時長:50-60 秒 · 視覺主角,順著箭頭由左往右講**

> 這是 Route B 的完整架構,跟 Route A 的對照性會在最後一頁說明。
>
> 左邊綠色是**輸入** — 5 秒音訊、32 kHz 單聲道、共 160,000 個 sample。
>
> 中間紫色是 **Perch v2,完全凍結不訓練** — Google 預訓練的權重,約 3000 萬個參數。它內部有三個模組:
>
> - **PCEN mel-spectrogram** 把波形轉成時頻圖,同時自動處理背景噪音
> - **Conformer Encoder Blocks** 混合 Convolution 捕捉局部時頻 pattern,加 Self-Attention 捕捉長距離依賴,是音訊領域的 SOTA 架構
> - 最後 **Temporal Pooling** 把時間軸壓縮
>
> 最終吐出 **1536 維 embedding**。
>
> 右邊粉色是 **自訂設計的 MLP Head,這才是本路線實際訓練的部分** — 約 157 萬可訓練參數,3 層 Linear 堆疊含 BatchNorm、ReLU、Dropout。最後用 sigmoid 輸出 234 類獨立機率。
>
> 最右紅色是**輸出** — 234 類機率,multi-label,每 5 秒一列,寫入 submission.csv。
>
> 核心概念:**訓練與推論共用這條 pipeline,只有粉色 head 的 157 萬參數被學習**。這跟 Route A 全網路 fine-tune 的路線,是兩種很不同的哲學。

**轉場:** 「下一頁展開 head 的內部細節 — 」

---

## 📄 Page 11 · MLP Head 架構

**時長:50-60 秒 · 技術細節頁,每個元件要清楚帶出理由**

> 這頁聚焦 MLP Head 的內部設計。
>
> 吃 1536 維 Perch embedding,經過 3 層 Linear,最後吐 234 類機率。**每一層的設計都有明確理由 —**
>
> **第一,BatchNorm 穩定深層 MLP 的 gradient 分布**。沒有它,深層網路容易 gradient 爆炸或消失,難以收斂。
>
> **第二,Dropout 0.3 抑制過擬合**。在 1536 維高維特徵空間上,模型特別容易記住雜訊,Dropout 在訓練時隨機丟 30% 神經元,強迫模型學習 robust 特徵。
>
> **第三,ReLU 提供非線性** — 這個最關鍵。沒有 ReLU,三層 Linear 堆疊會數學上等價於一層 Linear,多層設計就失去意義。
>
> **第四,漸進壓縮 1536 到 768 到 384 再到 234**,平衡表達容量與過擬合風險。
>
> **第五,sigmoid 輸出搭配 BCE 或 Focal Loss**,因為是 multi-label 場景,每類要獨立預測機率。
>
> 這個 head 共 **157 萬可訓練參數,是本路線唯一訓練的模組**。

**轉場:** 「知道 head 長什麼樣,下一頁說明實驗怎麼設計 — 」

---

## 📄 Page 12 · 實驗設計

**時長:45-55 秒 · 觀念解釋,關鍵是讓觀眾懂 ablation 的精神**

> Route B 的核心研究問題是:**「加深 head 有沒有用?」「Focal Loss 有沒有用?」**所以設計了三個 variants 做系統對照。
>
> **Variant A 是最樸素的 baseline** — 單層 Linear probe,直接量化 Perch embedding 本身的可分性,看看什麼都不加光靠 Perch 有多強。
>
> **Variant B 在 A 的基礎上只改一件事 — 把 head 加深成 3 層 MLP**。其他條件完全不變,為了量化「加深 head 的淨貢獻」。
>
> **Variant C 在 B 的基礎上又只改一件事 — 把 BCE 換成 Focal Loss**。架構完全跟 B 一樣,只換 loss 函數,為了量化「換 loss 的淨貢獻」。
>
> 這是 **ablation study 的控制變因精神** — 每次只改一件事,才能獨立歸因每個設計決策貢獻多少。所有 variants 共用**相同 seed、optimizer、data split**,結果可以直接對照比較。

**轉場:** 「實驗設計清楚後,下一頁就是完整結果 — 」

---

## 📄 Page 13 · 實驗結果(核心頁)

**時長:70-85 秒 · 講慢一點,數字要清晰帶出**

> 這頁是 Route B 的完整實驗結果,結構跟 Route A 的 STAGE 1 / STAGE 2 並排,方便大家對照。
>
> **STAGE 1 是 Embedding Extraction**,在 Kaggle P100 GPU 上跑。輸入 **35,549 個 train_audio 檔**加 **1,478 個 soundscape 標註段**,產出 **23 萬 3 千多筆 5 秒 frame 的 1536 維 embedding**。
>
> 關鍵優化是**固定 BATCH_SIZE=32 的 buffer 設計** — 原本速度是 2 秒一個檔,發現是 XLA 對每個不同 batch size 重新編譯造成。固定大小後提升到 **每秒 12 個檔,10 倍加速**。總耗時 47 分鐘。
>
> **STAGE 2 是 Head Ablation**,在本機 RTX 5070 跑。
>
> **Variant A linear probe baseline,val_auc = 0.9540**。這個數字本身就很驚人 — 證實 Perch embedding 的線性可分性已經非常強,光用單層就能達到公開 baseline 水準。
>
> **Variant B 把 head 升級為 3 層 MLP,val_auc = 0.9655**,淨提升 **+0.0115**,也就是 1.15 個百分點。這個數字證明**非線性 head 確實有價值**,Perch 空間中有線性無法切分的結構。
>
> **Variant C 換上 Focal Loss,val_auc = 0.9660**,是三者最佳。但 B 到 C 的提升只有 **+0.0005,落在 noise 範圍,是 null result**。
>
> 這個 null result 反而是有趣的發現 — 在 Perch 這種高品質 embedding 上,Focal Loss 想壓低的「簡單負樣本」其實是有用的梯度信號,壓掉反而虧。

**轉場:** 「知道 Route B 的數字後,下一頁正式跟 Route A 對比 — 」

---

## 📄 Page 14 · 結果比較與觀察(結尾頁)

**時長:60-75 秒 · 雙路線對比 + 呼應兩嘗試敘事**

> 最後對比本組的兩個嘗試。Route A 是剛才介紹的 EfficientNet + self-training,Route B 是 Perch + MLP。
>
> **整體 val_auc**,Route A 是 0.9806,Route B 是 **0.9660**,Route A 略高。
>
> **train_audio 子集 AUC** 大約持平,Route A 約 0.98,Route B 是 **0.9600**。
>
> 但是 **soundscape 子集就差很大** — Route A 只有 **0.6997**,Route B 是 **0.9957**,**方向完全相反**。
>
> 三個原因解釋這個差距。
>
> **第一**,Route B 把 soundscape 標註段直接納入訓練,模型第一階段就學過這個分布。
>
> **第二**,Perch 的預訓練資料原生就含 iNaturalist 野外錄音,encoder 對多物種共現場景本來就穩健。
>
> **第三**,Route A 第一階段 supervised 訓練只用 train_audio,soundscape 要到 self-training 階段才逐步補上,所以 sc_auc 較低。
>
> **但必須誠實說** — Route B 的 soundscape val 只有 **244 筆**,樣本小方差大,不能直接宣稱已解決 domain gap。**最終判準要等 Kaggle Public Leaderboard 才算準**。
>
> 本組的兩個嘗試剛好呈現**互補特性** — Route A 對 train_audio 分布適應性強,Route B 對 soundscape 的 domain generalization 較穩。下一步規劃是 **先上 Phase 3 submission 拿到 Public LB,再把 Route A 跟 Route B 做 ensemble,發揮兩路線的互補性**。
>
> 以上是本組第二個嘗試 Route B 的進度。謝謝大家,接下來是 Q&A 時間。

**轉場:** 如果後面還有物種展示頁,改說「謝謝大家,接下來交給 XXX 介紹 Pantanal 的代表物種。」

---

## ⏱️ 整體時間分配

| 頁 | 主題 | 時長 | 累計 |
|:---:|---|:---:|:---:|
| 9 | 交棒 + 雙路線敘事 | **45s** | 0:45 |
| 10 | 整體架構 | 55s | 1:40 |
| 11 | MLP Head 細節 | 55s | 2:35 |
| 12 | 實驗設計 | 50s | 3:25 |
| 13 | **實驗結果(核心)** | **80s** | 4:45 |
| 14 | **雙路線對比 + 結尾** | **70s** | **5:55** |

**總計約 5 分 55 秒**,適合 6-7 分鐘報告時段。

---

## 🔗 連貫性重點(本版改動)

Page 9 開場**明確交棒**:「謝謝 XXX / 剛才 Route A / 接下來 Route B 由我負責」
Page 10 點一句「**跟 Route A 的對照性在最後一頁**」預告尾頁
Page 13 強調「**結構跟 Route A 的 STAGE 1/2 並排,方便對照**」
Page 14 收尾**呼應兩嘗試**:「**本組的兩個嘗試剛好呈現互補特性**...下一步做 ensemble」

整條敘事線:**Route A 結果 → 交棒 → Route B 方法 → Route B 結果 → 兩路線對比 → Ensemble 計畫**

---

## 🎯 練習提示

1. **Page 13 是核心**,多練幾次,數字念準:
   - `0.9540` · `0.9655` · `0.9660`
   - `+0.0115` · `+0.0005`

2. **Page 14 的誠實聲明不要漏** — 「244 筆方差大」「等 LB 驗證」
   這些句子展現學術誠信,教授最看重。

3. **Page 9 開場交棒**要念清楚隊友名字,不要讓觀眾摸不清誰負責什麼。

4. **Page 14 的「互補特性」**是壓軸亮點,不只是你個人的結論,是**整組的研究發現**。

5. 若時間不夠,最容易省的是:
   - Page 10 的「PCEN / Conformer / Pooling」三個細節(可壓成 1 句)
   - Page 11 的第 4 點「漸進壓縮」(可省)
   - **不可省:** Page 9 交棒、Page 13 的所有數字、Page 14 的誠實聲明 + 互補結論

---

## 🚨 臨場可能卡的地方

| 情境 | 備案 |
|---|---|
| 忘詞卡住 | 深呼吸,看投影片上的字,直接念表格內容即可 |
| 被中斷提問 | 「這個問題我先記下,待 Q&A 一起回答,先把這頁講完」 |
| 時間超時 | 跳過 Page 11 的元件細節,直接進 Page 12 |
| 時間太快 | Page 13 多停頓,強調每個 Variant 的 delta |
| 隊友交棒卡住 | Page 9 可以短版:「剛才是 Route A,接下來 Route B 由我負責,走 Transfer Learning 路線」 |

祝報告順利 ✨
