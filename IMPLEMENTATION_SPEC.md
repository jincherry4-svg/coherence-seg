# 實作規格書：句子重組輔助任務之半監督文本分割模型

> **給實作者（Claude Code / Opus / Sonnet）的說明**
> 本文件是完整的實作規格，所有架構決策都已定案，請勿更改設計，遇到規格未涵蓋的細節時選擇最簡單、最可測試的實作方式並在 README 中記錄。請嚴格按照里程碑 M0 → M4 的順序實作，每個里程碑通過驗收標準後才進入下一個。全程使用 Traditional Chinese 註解與說明文件。

---

## 0. 專案目標與一句話摘要

在 Longformer 為基底的文本分割（text segmentation）模型上，加入一個自監督輔助任務——**句子挖空重組（sentence unshuffling / slot-candidate matching）**——並以此為橋樑引入無標註資料與 Mean Teacher 一致性訓練，目標在 WikiSection 資料集上超越 baseline（Longformer 邊界分類器）的 Pk / WindowDiff。

任務直覺：隨機抽走文件中部分句子、打亂後附在文件尾端，模型必須把每個候選句放回正確的空缺位置。這迫使編碼器學習句子層級的語意連貫性，而語意連貫正是判斷主題邊界的核心能力。

**最終交付物**：
1. 可執行的完整 repo（結構見 §2）
2. 每個里程碑的訓練 config 與 wandb 紀錄
3. 多 seed 實驗結果表（Pk、WindowDiff、boundary F1，mean ± std）
4. 通過的完整單元測試
5. README（環境安裝、資料準備、重現每個實驗的指令）

---

## 1. 環境與相依套件

- **本專案的開發、測試、訓練、評估全部在 Google Colab（A100，bf16）內完成**，執行環境細節見 §9；GTX 1080 Ti 的縮小設定僅為備用。
- Python 3.10（以 Colab 當前預裝版本為準）、PyTorch ≥ 2.1（CUDA）。
- 套件（寫入 `requirements.txt` 並 pin 版本）：

```
torch
transformers>=4.38
datasets
pytorch-lightning>=2.1
omegaconf
wandb
protobuf==3.20.3        # 必須 pin，否則與 wandb 在 Colab 上衝突（已知問題）
pysbd
scipy
scikit-learn
segeval
numpy
pytest
```

- 隨機性：所有隨機源（python / numpy / torch / cuda）由單一 `seed` 統一設定，使用 `lightning.seed_everything(seed, workers=True)`。

---

## 2. Repo 結構

```
coherence-seg/
├── configs/
│   ├── base.yaml            # 共用設定
│   ├── m0_baseline.yaml
│   ├── m2_reorder.yaml
│   ├── m3_meanteacher.yaml
│   ├── m4_full.yaml
│   └── sanity_1080ti.yaml
├── src/
│   ├── data/
│   │   ├── wikisection.py   # 有標註資料載入（沿用 SpokenNLP 格式）
│   │   ├── unlabeled.py     # 無標註 Wikipedia 載入與切句
│   │   ├── corruption.py    # ★ 挖空/打亂/拼接/對齊核心模組
│   │   └── collate.py       # 動態 collate_fn
│   ├── models/
│   │   ├── encoder.py       # Longformer 包裝、global attention
│   │   ├── heads.py         # 邊界分類頭 + 句子重組配對頭
│   │   ├── ema.py           # EMA 教師
│   │   └── lit_module.py    # LightningModule（loss 組合、排程）
│   ├── losses.py            # focal / BCE、配對 CE、consistency MSE、ramp-up
│   ├── eval/
│   │   ├── metrics.py       # Pk、WindowDiff、boundary F1
│   │   └── decode.py        # 匈牙利演算法解碼（僅評估用）
│   └── train.py             # 進入點：python -m src.train --config configs/xxx.yaml
├── tests/                   # pytest，見 §10
├── scripts/
│   ├── prepare_wikisection.sh
│   └── prepare_unlabeled.py
├── notebooks/
│   └── colab_runner.ipynb   # Colab 唯一執行入口（見 §9）
└── README.md
```

---

## 3. 資料

### 3.1 有標註資料：WikiSection

- 主要資料集 `wiki_section_disease`（en_disease），次要 `en_city`。
- **格式與前處理完全沿用 SpokenNLP repo**（alibaba-damo-academy/SpokenNLP，EMNLP 2023 "Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling" 的官方程式碼）的資料格式：文件 = 句子列表 + 每句一個邊界標籤。
- **邊界標籤慣例：以 SpokenNLP repo 的實際慣例為準**（實作前先閱讀其資料處理與評估程式碼確認「標籤 1 代表該句是段落最後一句」還是「第一句」），確認後在 `src/data/wikisection.py` 頂部以註解明確記錄，並在單元測試中固定該慣例。整個 codebase 只允許一種慣例。
- 內部統一資料表示（所有 Dataset 的輸出）：

```python
{
  "doc_id": str,
  "sentences": list[str],      # n 句
  "labels": list[int] | None,  # n 個 0/1；無標註資料為 None
}
```

### 3.2 無標註資料

- 用 HuggingFace `datasets` 載入 `wikimedia/wikipedia`（英文，最新可用 snapshot），streaming 模式。
- 以 `pysbd` 切句（`Segmenter(language="en", clean=False)`）。
- 過濾條件：20 ≤ 句數 ≤ 150，且 tokenize 後總長 ≤ 3500 tokens（保留候選句附加空間）。
- 第一版抽 **50,000 篇**存成 jsonl（`scripts/prepare_unlabeled.py` 負責，含 seed）。

### 3.3 Tokenizer 與特殊 token

- `LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")`
- 加入特殊 token：

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["[SLOT]", "[CAND]"]})
model.longformer.resize_token_embeddings(len(tokenizer))  # 絕對不可漏
```

- **每個句子前插入一個 `<s>`（bos_token）作為句子表徵錨點**（與 SpokenNLP 的做法一致）；句子表徵 = 該 `<s>` 位置的最後一層 hidden state。

---

## 4. 輸入構造模組（`src/data/corruption.py`）— 全案最關鍵模組

### 4.1 演算法（訓練時在 collate_fn 中動態執行，每個 epoch 挖不同句子）

輸入：一篇文件的 `sentences`（n 句）、`labels`（或 None）、挖空比例 `p`、隨機源 `rng`（`numpy.random.default_rng`，由全域 seed 派生）。

1. **決定挖空數量** `m = clip(round(p * n), 2, min(10, n - 2))`。若 `n < 6` 則跳過挖空（m=0，該文件只算分割 loss）。
2. **抽樣挖空位置**：從索引 `1 .. n-1` 均勻抽 m 個（**永不挖第 0 句**，保留文件開頭上下文）。
3. **構造殘缺文件**：token 序列為

```
<s> tok(sent_0) <s> tok(sent_1) ... 其中被挖空的句 i 整句替換為單一 [SLOT] token
```

即：被挖走的句子在原位置留下恰好一個 `[SLOT]`（不保留其 `<s>`）。
4. **構造候選區**：把 m 個被抽走的句子以 `rng.permutation` 打亂，附加在文件尾端：

```
</s> [CAND] tok(sent_a) [CAND] tok(sent_b) ... [CAND] tok(sent_c)
```

5. **配對標籤** `match_labels`：長度 m 的整數陣列，`match_labels[j] = k` 表示第 j 個槽位（依文件內出現順序）的正確答案是第 k 個候選（依候選區出現順序）。
6. **分割 loss mask** `seg_mask`：長度 = 殘留句數（含槽位），**槽位位置一律 mask 掉**（不計分割 loss）；有標註文件其餘位置為 1，無標註文件全 0。
7. **長度保證**：構造後總長 > 4096 則整篇丟棄（訓練時）；驗證/測試集**永不挖空**（完整文件），過長文件依 SpokenNLP 的原始處理方式切塊。
8. 同時輸出**原始未挖空版本**的 token 序列與句子錨點索引（M4 的教師輸入需要，M0–M3 可延遲構造但介面先留好）。

### 4.2 collate 輸出（batch dict，全部 padding 對齊）

```python
{
  "input_ids", "attention_mask", "global_attention_mask",   # 殘缺文件（學生輸入）
  "sent_anchor_idx",    # (B, S) 每個殘留句的 <s>/[SLOT] token 位置；padding = -1
  "slot_idx",           # (B, M) [SLOT] token 位置；padding = -1
  "cand_idx",           # (B, M) [CAND] token 位置；padding = -1
  "match_labels",       # (B, M) padding = -100
  "seg_labels",         # (B, S) padding = -100
  "seg_mask",           # (B, S) 0/1
  "clean_input_ids", "clean_attention_mask", "clean_global_attention_mask",
  "clean_sent_anchor_idx",                                   # 教師輸入（M4）
  "student_to_clean_sent_map",  # (B, S) 學生殘留句 → 原文句索引，供一致性對齊
}
```

### 4.3 Global attention

`global_attention_mask = 1` 的位置：token 0、所有句子 `<s>` 錨點、所有 `[SLOT]`、所有 `[CAND]`、`</s>` 分隔符。其餘為 0（local attention）。

---

## 5. 模型（`src/models/`）

### 5.1 編碼器

`LongformerModel("allenai/longformer-base-4096")`，hidden size d=768。開啟 `gradient_checkpointing_enable()`（config 可關）。

**權重來源（已定案）**：使用 AllenAI 原始預訓練權重，經 HF transformers 載入。
不使用 allenai/longformer GitHub repo 的舊版自訂 CUDA kernel 實作——該 repo 已停止
維護且綁定舊版 PyTorch，AllenAI 官方亦說明 HF sliding-window 實作適用於下游微調；
SpokenNLP baseline 同樣使用 HF 版，維持載入器一致是可比性的前提。

### 5.2 邊界分類頭（主任務）

```
sent_repr (B,S,768) → Linear(768,768) → GELU → Dropout(0.1) → Linear(768,1) → logit
```

Loss：**focal loss（gamma=2, alpha=0.75）**，config 可切換成帶 `pos_weight` 的 BCE。只在 `seg_mask=1` 的位置計算，對有效位置取平均。

### 5.3 句子重組配對頭（輔助任務）

```python
slot_h = gather(hidden, slot_idx)      # (B, M, d)
cand_h = gather(hidden, cand_idx)      # (B, M, d)
scores = torch.einsum("bmd,dk,bnk->bmn", slot_h, W, cand_h) / sqrt(d)  # W 為可學習 (d,d) 雙線性矩陣
L_reorder = cross_entropy(scores.flatten(0,1), match_labels.flatten(), ignore_index=-100)
```

- 訓練用逐槽位獨立 CE（不加一對一約束）。
- **重組準確率**（逐槽位 argmax 命中率）必須記錄到 wandb，這是 curriculum 的儀表板。
- 評估/分析時提供 `scipy.optimize.linear_sum_assignment` 的一對一解碼函式（`src/eval/decode.py`），不參與訓練。

### 5.4 EMA 教師（M4）

- 教師 = 學生完整模型（編碼器＋邊界頭）的參數指數移動平均，`torch.no_grad()`、`eval()` 模式、不參與梯度。
- 每個 optimizer step 後更新：`decay = min(0.999, (1 + step) / (10 + step))`。
- 用 `copy.deepcopy` 初始化，自行實作輕量 EMA 類別（`src/models/ema.py`），不依賴外部套件。

---

## 6. 損失函數與排程（`src/losses.py`）

總損失：

```
L = L_seg + λ1(t) · L_reorder + λ2(t) · L_consistency
```

- **ramp-up**（Tarvainen & Valpola 的 sigmoid ramp-up）：`w(t) = w_max * exp(-5 * (1 - min(t/T, 1))^2)`，t 為 global step。
- `λ1`: `w_max = 0.5`，`T = 2000` steps（config 可調）。
- `λ2`（僅 M4）: `w_max = 1.0`，`T = 4000` steps。
- **L_consistency**（僅 M4）：教師吃 `clean_*` 完整文件輸出邊界機率 `sigmoid(logit_T)`；學生吃殘缺文件輸出 `sigmoid(logit_S)`；透過 `student_to_clean_sent_map` 對齊後，**只在學生殘留的真實句子（非槽位）上**計算 MSE。有標註與無標註資料都算。

### Curriculum（挖空比例排程）

| 階段 | 條件 | p |
|---|---|---|
| 起始 | step 0 | 固定 m ∈ {2,3}（p 忽略，直接抽 2–3 句） |
| 升級 1 | 重組準確率移動平均（window=200 steps）> 0.60 | p = 0.15 |
| 升級 2 | 準確率再度 > 0.60 | p = 0.25（上限） |

實作為 LightningModule 內的狀態機，透過 dataset 的共享變數（或 callback 設定 dataset 屬性）傳遞當前 p。升級事件記錄到 wandb。

---

## 7. 訓練流程（`src/models/lit_module.py` + `src/train.py`）

- 框架：**PyTorch Lightning**。
- **資料混合（M3 起）**：使用 Lightning 的 `CombinedLoader(mode="max_size_cycle")` 同時供給 labeled 與 unlabeled 兩個 dataloader。loss 分配由 flags 控制：labeled batch 算 `L_seg + λ1·L_reorder + λ2·L_consistency`；unlabeled batch 在 **M3** 只算 `λ2·L_consistency`（`unlabeled_reorder=false`，無標註文件仍會挖空，但僅作為學生端的強增強供一致性使用）；**M4** 起 `unlabeled_reorder=true`，unlabeled batch 加算 `λ1·L_reorder`。M0–M2 只有 labeled loader。
- Optimizer：AdamW，兩組參數：encoder lr=2e-5、新頭與雙線性矩陣 lr=1e-4，weight_decay=0.01。
- LR 排程：linear warmup（前 10% steps）+ linear decay。
- 精度：A100 用 `precision="bf16-mixed"`。
- Batch：labeled 每 GPU 2 篇文件、gradient accumulation 4（有效 batch 8）；unlabeled 同規格。
- Epochs：10（disease 資料集小，以 val Pk 做 early stopping，patience=3）。
- Gradient clipping：1.0。
- Checkpoint：`ModelCheckpoint(monitor="val/pk", mode="min", save_top_k=1, save_last=True)`，路徑指向可掛載 Google Drive 的目錄（config 變數 `ckpt_dir`）。支援 `--resume` 續訓。
- wandb 必記項目：`train/L_seg`、`train/L_reorder`、`train/L_consistency`、`train/reorder_acc`、`train/lambda1`、`train/lambda2`、`train/p_corrupt`、`val/pk`、`val/wd`、`val/boundary_f1`、learning rate、config 全文與 git commit hash。

---

## 8. 評估（`src/eval/metrics.py`）

- **Pk 與 WindowDiff 以 SpokenNLP repo 的評估腳本為準**（把該腳本邏輯移植進來並註明出處），確保與已復現的 baseline 數字直接可比。
- 同時用 `segeval` 套件實作第二套計算，**單元測試要求兩套在同一份預測上的 Pk 差異 < 1e-6**；若不一致，以 SpokenNLP 為準並在 README 記錄差異原因（常見原因：邊界表示慣例、視窗 k 的計算方式）。
- boundary F1 用 `sklearn.metrics.f1_score`（正類 = 邊界）。
- 推論時邊界判定：`sigmoid(logit) > threshold`，threshold 在驗證集上以 F1 掃描 {0.1, 0.2, ..., 0.9} 選定後固定用於測試集。
- **多 seed 流程**：每組設定跑 seeds = [42, 43, 44, 45, 46]，輸出腳本自動彙整 mean ± std，並提供 paired bootstrap（10,000 次重抽，`scipy`）比較兩組設定的 Pk 差異顯著性。

---

## 9. 執行環境：Google Colab（全案唯一執行環境）

### 9.1 基本策略

- **GitHub repo 是唯一真實來源**：程式碼永遠 commit 回 GitHub，每個 Colab session 開始時 clone / pull 到 `/content/coherence-seg`。Colab 本地檔案系統視為隨時會消失。
- **Google Drive 負責持久化**：掛載於 `/content/drive`，存放三類東西——處理好的資料（jsonl）、checkpoint、實驗結果彙整。
- **訓練資料先複製到本地再讀**：session 開始時把 Drive 上的資料 jsonl 複製到 `/content/data/`（Drive 直接讀 I/O 太慢，會拖垮 dataloader）；checkpoint 則直接寫 Drive。

### 9.2 執行入口：`notebooks/colab_runner.ipynb`

全專案唯一的執行入口 notebook，cell 依序為（每個 cell 必須**冪等可重跑**）：

1. GPU 檢查（`!nvidia-smi`，確認 A100 與 bf16 可用）
2. 掛載 Google Drive
3. clone / pull repo + `!pip install -r requirements.txt`
4. 從 **Colab Secrets**（`google.colab.userdata`）讀取 `WANDB_API_KEY` 設入環境變數——金鑰不得寫死在任何檔案
5. 資料同步：Drive → `/content/data/`
6. `!pytest tests/ -v`
7. 訓練 cell：`!python -m src.train --config configs/xxx.yaml --seed 42`
8. 續訓 cell：同上加 `--resume`（自動尋找 ckpt_dir 下的 `last.ckpt`）
9. 結果與 log 回存 Drive

### 9.3 斷線防護（必須實作）

- `ModelCheckpoint` 除 `save_top_k=1` 外，加 `every_n_train_steps=500` 的 `save_last=True`，直接寫入 Drive 路徑。
- `--resume` 恢復時，global step、LR 排程、λ ramp-up、curriculum 狀態機、EMA 教師參數（M4）全部必須正確續接——這是 §10 之外的隱性驗收條件，實作後用「訓練 300 步 → 中斷 → resume 再 300 步」與「連續 600 步」對照 loss 曲線驗證。

### 9.4 設定檔

- `configs/base.yaml`（Colab A100）：max_len=4096、bf16、如 §7。
- `configs/sanity_1080ti.yaml`：max_len=1024、fp16、batch 1 + accumulation 8、gradient checkpointing 開、unlabeled 只取 500 篇、1 epoch。僅作**煙霧測試**（Colab 或 1080 Ti 皆可），不看指標。
- **設備疊加 config**：`lab_1080ti.yaml`（實驗室 GTX 1080 Ti，fp16、batch 1、本地 ckpt_dir）與 `colab_t4.yaml`（Colab T4，fp16、batch 1）。用法：`python -m src.train --config <里程碑config> <設備config>`，多個 config 由左至右疊加合併；不帶設備 config 即為 A100 預設（bf16、batch 2）。兩者 max_len 皆維持 4096 以確保結果可比。

### 9.5 三種硬體的分配策略

| 資源 | 特性 | 分配 |
|---|---|---|
| 實驗室 GTX 1080 Ti（11GB, Pascal） | 免費常駐、無 Tensor Core（fp16 只省記憶體不加速）、不支援 bf16 | 單元測試、煙霧測試、**M0 與 M2 的多 seed 主力**（過夜排程跑） |
| Colab T4（16GB, Turing） | 免費額度、fp16 有 Tensor Core、中速 | M0/M2 溢出的 seed、`prepare_unlabeled.py` 資料準備、M3 的備援（batch 1，少數超長文件可能 OOM 被丟棄） |
| Colab A100（40GB, Ampere） | 稀缺/付費、bf16、最快 | **每個里程碑 seed 42 的首次驗證**（快速發現問題）＋ **M3/M4 的全部 seed**（教師前向傳播吃緊，只有 A100 寬鬆） |

**嚴格規則：同一個里程碑 config 的 5 個 seed 必須跑在同一種設備上**（M0/M2 全在
1080 Ti 或全在 T4；M3/M4 全在 A100），避免 fp16/bf16 的數值差異混進 seed 變異。
最終 ABLATION.md 必須記錄每組設定使用的設備。單一 run 不得中途換設備續訓。

---

## 10. 單元測試（pytest，全部必須通過）

1. **重建測試**：對 corruption 輸出，依 `match_labels` 把候選句放回槽位後，還原的 token 序列 == 原文 token 序列（隨機 100 篇、多 seed）。
2. **標籤對齊測試**：挖空後 `seg_labels` 與殘留句一一對應，槽位位置的 `seg_mask == 0`；無標註文件 `seg_mask` 全 0。
3. **索引測試**：`slot_idx`、`cand_idx`、`sent_anchor_idx` 指向的 token id 分別為 `[SLOT]`、`[CAND]`、`<s>`。
4. **global attention 測試**：mask 為 1 的位置恰為 §4.3 所列集合。
5. **長度測試**：任何輸出序列 ≤ 4096。
6. **配對頭測試**：手工構造 3 槽位 batch，驗證 einsum 分數矩陣形狀與 CE 對 ignore_index 的行為。
7. **EMA 測試**：兩步更新後教師參數等於手算的移動平均；教師無梯度。
8. **一致性對齊測試**：`student_to_clean_sent_map` 把學生殘留句正確映射回原文句索引。
9. **決定性測試**：同 seed 下 corruption 輸出完全一致。
10. **指標交叉驗證**：SpokenNLP 移植版與 segeval 的 Pk/WD 在 20 組隨機預測上一致（< 1e-6）。

---

## 11. 里程碑與驗收標準（嚴格依序執行）

### M0 — Baseline 復現
純 Longformer + 邊界分類頭（無挖空、無輔助任務），wiki_section_disease。
**驗收**：訓練收斂、val Pk 落在合理範圍（與使用者先前復現的 SpokenNLP baseline 同量級）、eval 流程與多 seed 腳本可用。

### M1 — 輸入構造模組
完成 `corruption.py` + `collate.py` + 測試 1–5、9。
**驗收**：pytest 全綠；可視化腳本（印出一篇文件挖空前後的句子與配對答案）人工檢查正確。

### M2 — 加入重組輔助任務（Stage A 核心）
labeled 資料上 `L = L_seg + λ1·L_reorder`，含 curriculum。
**驗收**：reorder_acc 曲線脫離隨機水準（> 1/m 顯著）並隨訓練上升；5 seeds 的 val Pk 相對 M0 的變化有紀錄（改善與否都如實回報，這是核心實驗結果）。

### M3 — Mean Teacher（Stage B 核心）
在 M2 之上加入 EMA 教師、L_consistency 與 50k 無標註資料（CombinedLoader）。
教師看完整文件、學生看挖空文件、非槽位句 MSE。**無標註資料只透過一致性損失
貢獻（`unlabeled_reorder=false`）**——無標註文件仍會被挖空，但那是作為學生端
的強增強，不算 L_reorder。如此 M3−M2 的差異純粹隔離出 Mean Teacher 的貢獻。
**驗收**：測試 7、8 通過；λ2 ramp-up 與 ema_decay 曲線正確；訓練穩定（L_seg
不發散）；5 seeds 結果對比 M2 有完整紀錄。

### M4 — 無標註重組全開（full）
在 M3 之上設 `unlabeled_reorder=true`：無標註 batch 同時計 λ1·L_reorder 與
λ2·L_consistency。M4−M3 的差異 = 無標註重組訊號的貢獻。
**驗收**：unlabeled 的 L_reorder 曲線正常下降；5 seeds 結果對比 M3 有完整紀錄；
最終彙整 M0–M4 的 ablation 總表（Pk / WD / F1，mean ± std）。

---

## 12. 已知陷阱清單（實作時逐條自查）

1. `resize_token_embeddings` 必須在載入預訓練權重之後、訓練之前執行一次。
2. 邊界標籤慣例全 repo 唯一，並以 SpokenNLP 評估腳本反向驗證。
3. 槽位位置絕不計分割 loss；驗證/測試集絕不挖空。
4. ramp-up 以 global step 計，續訓時要從 checkpoint 恢復 step 計數。
5. Colab 上 `protobuf==3.20.3` 必須 pin；wandb 登入用環境變數 `WANDB_API_KEY`。
6. Longformer 對 `attention_mask` 與 `global_attention_mask` 的 padding 敏感，padding 位置兩者皆為 0。
7. 動態挖空發生在 collate（CPU），注意 `num_workers>0` 時每個 worker 的 rng 需以 `worker_init_fn` 派生獨立 seed，否則各 worker 產生相同挖空。
8. bf16 下 focal loss 的 log 運算注意數值穩定（用 `logsigmoid` 實作）。
9. 教師 forward 記得 `eval()` + `no_grad()`，且教師的 dropout 必須關閉。
10. 每次實驗把 config、commit hash、seed 寫入 wandb run config。
11. **Colab session 隨時可能中斷**：長訓練開跑前先確認 `last.ckpt` 有按 500 steps 正常寫入 Drive；恢復後 global step、curriculum 狀態、EMA 教師必須正確續接（§9.3 的對照驗證）。
12. **Drive I/O 慢**：訓練資料一律先複製到 `/content` 本地再讀；checkpoint 寫入頻率不要高於 500 steps，否則寫 Drive 會成為瓶頸。
13. **Colab 重啟後環境歸零**：`colab_runner.ipynb` 所有 cell 必須冪等；未 commit 的程式碼修改在 session 結束即消失，任何修改完成後立刻 commit + push 回 GitHub。

---

## 13. 執行指令範例（寫入 README）

以下指令在 Colab 中透過 `notebooks/colab_runner.ipynb` 的 cell 以 `!` 前綴執行（例如 `!pytest tests/ -v`）；notebook 中已依 §9.2 順序排好，README 同時保留純命令列版本以利日後遷移。

```bash
# 環境
pip install -r requirements.txt

# 資料
bash scripts/prepare_wikisection.sh
python scripts/prepare_unlabeled.py --n 50000 --seed 42

# 測試
pytest tests/ -v

# 各里程碑（--seed 可覆寫）
python -m src.train --config configs/m0_baseline.yaml --seed 42
python -m src.train --config configs/m2_reorder.yaml --seed 42
python -m src.train --config configs/m3_meanteacher.yaml --seed 42
python -m src.train --config configs/m4_full.yaml --seed 42

# 多 seed 彙整
python -m src.eval.aggregate --runs <wandb_group> 
```
