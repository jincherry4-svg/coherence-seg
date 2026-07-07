# PROJECT_MASTER_PLAN：從今天到論文完成的總任務書

> 撰寫日：2026-07-07。撰寫者：Claude Fable（本輪接手者）。
> 本文件取代 `DEBUG_HANDOFF_REPORT.md` 成為**新對話的第一份必讀文件**，
> 設計依據仍是 `IMPLEMENTATION_SPEC.md`，操作細節仍看 `RUNBOOK.md`。
> 給任何接手模型（Opus/Sonnet/Fable/Claude Code）：先讀本文件 §0 現況快照，
> 再讀 §7 的接手 prompt 範本，**不要重新診斷已結案的問題**。

---

## §0 現況快照（2026-07-07 交接當下）

### 已結案（勿重新診斷）
1. checkpoint 路徑解析（exp_tag/run_tag 分離 + resolve_resume 診斷）——Colab 實測驗證。
2. Drive 路徑 `LongformerSC/` 前綴——Colab 實測驗證。
3. PyTorch 2.6 `weights_only`——操作者已確認看到 `Restored all states ... global_step=2000`。
4. `run_all_seeds.sh` DONE 標記與 `train.py` exp_tag 分家——已修（patch 0001）。
5. results JSON 命名 bug（config_name 退回模型名、seed 不隨 --seed 變）——已修（patch 0001）。
6. **邊界標籤三分類修正（-100/0/1）**——已修（patch 0002），含 4 項新測試，離線 15 項測試全綠。
   已實讀 SpokenNLP 原始碼確認：`preprocess_data.py` `tokenize_method` 產出
   `-100`（段落內非末句）/`0`（段落末句非主題邊界）/`1`（主題邊界）；
   評估對齊 `postprocess_predictions.py` **para_level** 慣例＝pred/ref 成對剔除 -100。

### 等待操作者執行（依序，見 §2 Phase A）
- [ ] 在 Colab 套用 patch 0001、0002 → `pytest`（應 24 項全過）→ push 到 GitHub。
- [ ] `head` 檢查現有 jsonl 是否含 -100（結果決定是否要重做資料，見 Phase A-3）。
- [ ] **清除污染的舊 checkpoint 與結果**（見 Phase A-4，非常重要，不清會被 --resume 撿回來）。

### ⚠️ 最重要的一件事：舊結果全部作廢
`corruption.py` 的舊版 seg_mask 會把 label=-100 的位置以 -100 當 BCE 目標計入
分割 loss——**訓練從頭就被污染**。因此：
- M2 seed 42（Pk 17.39、`results/longformer-base-4096_seed42.json`）：作廢刪除。
- M2 seed 43 在 Drive 上的部分進度 checkpoint：作廢刪除。
- 修正前後的數字**不可比**，論文只使用修正後重跑的結果。

---

## §1 系統架構總覽（給接手者的一張地圖)

```
configs/                     里程碑 × 設備 兩維疊加（後者覆寫前者）
  base.yaml                  共用（A100 預設）：bf16、batch2×accum4=有效8、epochs10、
                             focal loss、λ1 ramp 2000步、λ2 ramp 4000步、ckpt每500步
  m0/m2/m3/m4_*.yaml         只覆寫 flags 與 wandb group
  sanity_1080ti.yaml         煙霧測試（64 docs、1 epoch）
  lab_1080ti.yaml            fp16、本地 ./checkpoints（不走 Drive）
  colab_t4.yaml              T4 備援

src/data/
  wikisection.py             load_jsonl（三分類驗證）、SegDataset、build_special_ids
  corruption.py              挖空重組輸入構造（純 numpy 可離線測試）：
                             學生=殘缺文件+[SLOT]/[CAND] 候選區、教師=完整文件
  collate.py                 CurriculumController（stage 0/1/2 控挖空難度）、collator
  unlabeled.py               無標註 wiki 載入（labels=None）
src/models/
  encoder.py                 Longformer-base-4096 + boundary head + bilinear matching head
  ema.py                     EmaTeacher（decay 爬向 0.999）
  lit_module.py              訓練核心：L_seg + λ1·L_reorder + λ2·L_consistency、
                             val 掃 threshold、test 固定 val threshold 並寫 results JSON
src/train.py                 進入點：config 疊加、cfg.config_name/seed 注入、
                             resolve_resume 診斷、weights_only 相容、CKPT_DIR 覆寫
src/eval/
  metrics.py                 labels_to_mass（拒 -100）、drop_ignored_pairwise（成對剔除）、
                             pk_wd_spokennlp（逐篇 Pk/WD 平均 + 攤平 F1/P/R）、scan_threshold
  aggregate.py               多 seed 彙整 → ABLATION.md、paired bootstrap 顯著性
scripts/
  run_all_seeds.sh           5 seeds 批次（DONE 標記已對齊 exp_tag）
  prepare_wikisection.sh     clone SpokenNLP 做前處理
  prepare_unlabeled.py       產無標註 wiki 資料
notebooks/colab_runner.ipynb Cell 1-10（GPU→Drive→clone→wandb→同步→pytest→訓練→續訓→多seed→彙整）

實驗設計（四里程碑 ablation，每組 5 seeds=42..46）：
  M0 = Longformer baseline
  M2 = +句子挖空重組輔助任務（curriculum 三階段）      → Stage A 貢獻
  M3 = M2 +Mean Teacher（無標註只走一致性損失）        → M3−M2 = Mean Teacher 貢獻
  M4 = M3 +無標註 batch 也計 L_reorder                → M4−M3 = 無標註重組訊號貢獻
硬體鐵律（§9.5）：同組 5 seeds 同設備；不跨 fp16↔bf16 續訓；
  1080 Ti→M0/M2 主力、A100→每組 seed42 首驗 + M3/M4 全部、T4→溢出備援。
```

---

## §2 任務書（Phase A→H，完成即論文可寫）

### Phase A：地基收尾【今天～1 天，Colab + 終端機】
1. 套用 patch 0001+0002 → `pytest tests/ -v` 24 項全過 → push。
2. `git pull` 後在 Colab 重跑 cell 6 確認（真 torch 環境下含 `test_modules.py`）。
3. **檢查資料**：`head -c 800 train.jsonl`。
   - 含 -100 → 資料正確，直接進 Phase B。
   - 不含 -100 → 當初前處理漏了三分類，回 RUNBOOK 第 2 步用 SpokenNLP 原版
     `run_process_data.sh` 重產 disease（順便產 city），舊 jsonl 覆蓋。
4. **大掃除**（防 --resume 撿回污染進度）：
   ```bash
   # Drive 與實驗室機器都要做
   rm -rf .../checkpoints/m2_reorder .../checkpoints/m0_baseline
   rm -f  .../results/longformer-base-4096_seed*.json
   find .../checkpoints -name DONE -delete
   ```
5. 用 sanity config 跑一次「訓 300 步→中斷→--resume 300 步」對照連續 600 步
   （RUNBOOK 第 4 步從未正式做過完整版；現在程式都修好了，一次做完永絕後患）。

**通過標準**：pytest 24 全過；resume 對照的 global_step/LR/λ/curriculum 連續。

### Phase B：city 資料集支援【0.5 天，Sonnet/Opus 改碼 + Colab 產資料】
1. 給 Sonnet 的任務 prompt 已寫好（見 §7 任務B）。核心：`configs/data_city.yaml`
   覆寫資料路徑；`config_name`/`exp_tag` 附加資料集後綴（如 `m2_reorder+city`），
   results JSON 與 checkpoint 目錄都不撞名；`run_all_seeds.sh` NAME 同步。
2. Colab 產 city 的 train/dev/test.jsonl 上 Drive（與 disease 同一批前處理產出）。

**通過標準**：`m0_baseline+city` sanity 跑通；兩資料集的 results 檔名可區分。

### Phase C：M0 baseline 重跑【disease 1 天 + city 1 天，多為掛機】
- disease：seed42 A100 首驗 → 5 seeds 1080 Ti 過夜（`run_all_seeds.sh`）。
- city：同上。city 文件數多（test 3893 篇 vs disease 718），訓練時間×5 有心理準備；
  若 1080 Ti 跑不完可整組移 T4（記住鐵律：整組同設備）。
- **驗收**：test Pk 與先前復現 SpokenNLP Longformer baseline 同量級。
  修正後數字會與 17.39 不同，屬預期；只與「同樣修正後」的組內互比。

### Phase D：M2 重跑【同 Phase C 節奏，這是 Stage A 核心】
- 前 30 分鐘盯 `train/reorder_acc` 從 ~1/m 爬升；`train/curriculum_stage` 0→1→2。
- 卡隨機水準 → 中止，貼曲線給 Opus（§7 除錯 prompt）。
- **產物**：M2 vs M0 在兩資料集各 5 seeds 的 Pk/WD/F1（好壞都如實記錄）。

### Phase E：M3/M4 半監督【各 1 天，A100 全 seeds】
> 注意：Mean Teacher/EMA/一致性損失**程式碼已完成**，本 Phase 是資料準備＋驗證＋執行，不是實作。
1. `prepare_unlabeled.py --n 500` 試跑 → `--n 50000` 正式產出上 Drive。
2. 先給 Opus §7 任務E 的煙霧驗證（無標註路徑在三分類修正後 labels=None 分支不受影響、
   一致性 valid 遮罩正確、50 步不崩）。
3. M3 開跑盯三線：`L_seg` 不發散、`L_consistency` 隨 λ2 ramp 出現後緩降、
   `ema_decay` 爬向 0.999。M4 加盯 `L_reorder_unlabeled` 下降。
4. 資料集策略決定點：**M3/M4 至少完成 disease**；city 視 A100 額度，額度不夠時
   論文寫法改為「主實驗 disease 全 ablation、city 驗證 M0/M2 泛化」，這是合法的敘事。

### Phase F：彙整與統計【0.5 天】
- cell 10 aggregate → `ABLATION.md`；對 M2vsM0、M3vsM2、M4vsM3 跑 paired bootstrap
  （`src/eval/aggregate.py` 內建函式），報告 p 值與效應方向。
- 把總表貼給 Opus 產三點觀察 + 實驗章節草稿（RUNBOOK 第 9 步的 prompt 照用）。

### Phase G：分析實驗（論文的「肉」，別跳過）【1–2 天】
1. **逐篇 Pk 分佈圖**：per_doc_pk 已在 results JSON 裡，畫 M0 vs M2 的直方圖/散點，
   看改善集中在長文件還是短文件（Longformer 的賣點是長文件，這張圖是論點核心）。
2. **curriculum 行為圖**：wandb 匯出 `reorder_acc` 與 `curriculum_stage` 曲線。
3. **質性案例**：挑 2–3 篇 M2 修對、M0 錯的文件，展示邊界句與上下文（`src/eval/decode.py`）。
4. **失敗分析**：挑 M2 仍錯的類型（如章節極短、標題句省略）——誠實的失敗分析是加分項。
5. 若 M3/M4 有效：畫「有效 labeled 資料量 vs Pk」的半監督賣點圖
   （可加跑 `max_train_docs` 10%/50% 的低資源 ablation，各 3 seeds 即可）。

### Phase H：論文撰寫【1–2 週，與 Phase C–G 平行起草】
章節骨架（照系上格式調整）：
1. 緒論：長文件主題分割問題、標註昂貴 → 自監督重組 + 半監督的動機。
2. 相關研究：監督式分割（CS-BERT、SpokenNLP TSSP+CSSL）、自監督輔助任務、
   Mean Teacher 系半監督。妳先前的 EMNLP 2024 survey 文獻整理直接沿用
   （維持「不含 LLM-based 方法」的既定範圍）。
3. 方法：架構圖（encoder+雙頭+EMA 教師）、挖空重組任務形式化（§4 的構造流程
   可用 `scripts/visualize_corruption.py` 產示意）、curriculum、損失組合與 ramp-up。
4. 實驗設定：資料集統計表、三分類標籤慣例與 para_level 評估（**引用 SpokenNLP
   原始碼出處，這是審查者會問的點**）、硬體與 5-seed 協定、超參數表（base.yaml 照抄）。
5. 結果：主表（M0/M2/M3/M4 × 兩資料集，mean±std + 顯著性標記）、與 SpokenNLP
   論文數字的對照列。
6. 分析：Phase G 全部產物。
7. 結論與限制：誠實寫（如 curriculum 閾值未調參、無標註資料域內性、單一骨幹）。

必備圖表清單：架構圖、挖空示意圖、主 ablation 表、per-doc Pk 分佈、
reorder_acc/curriculum 曲線、質性案例表、（選配）低資源曲線。

---

## §3 待改善程式碼（依優先序，接手者的 backlog）

| # | 位置 | 問題 | 建議 | 急迫 |
|---|---|---|---|---|
| 1 | Drive 容量 | best+last 每 seed ≈3.6GB；M3/M4×2 資料集×5 seeds 會爆 Drive 免費額度 | `run_all_seeds.sh` 每 seed 完成（test 過、results JSON 寫出）後自動刪該 seed 的 .ckpt 只留 JSON；或 ModelCheckpoint 加 `save_weights_only=True`（注意：這會犧牲該 seed 的續訓能力，只能在 DONE 後清理，不能改存檔方式） | 高（Phase E 前必做） |
| 2 | `train.py` | `exp_tag=args.config[0]` 依賴「里程碑排第一」的口頭慣例 | 開頭 assert `args.config[0]` 的檔名以 `m0/m2/m3/m4/sanity` 開頭，違反就報錯 | 高 |
| 3 | `lit_module.py` | results 寫到相對路徑 `results/`，只存在 Colab session 內 | 沒 cell 10 就會遺失；改為也寫一份到 `cfg.ckpt_dir` 同層的 Drive results/，雙保險 | 中 |
| 4 | `lit_module.py` | per_doc_pk 計算邏輯與 `aggregate.py`/`metrics.py` 三處重複 | 抽成 `metrics.per_doc_pk()` 單一來源 | 中 |
| 5 | wandb | 續訓會開新 run，曲線斷成多段 | resolve_resume 找到 ckpt 時讀出 wandb run id 傳給 WandbLogger(id=, resume="must") | 低（美觀） |
| 6 | `train.py` | EarlyStopping(patience=3) 的單位是「驗證次數」；目前一 epoch 驗一次沒問題，若日後加 val_check_interval 要重審 | 加註解說明即可 | 低 |
| 7 | `metrics.py` | scan_threshold 的掃描粒度/範圍未在論文附錄記錄 | 實驗章節附錄補一句掃描設定 | 低 |
| 8 | 測試 | 尚無「lit_module 用三分類假 batch 走一次 training_step」的整合測試（現有測試到 corruption 層為止） | 需 torch，放 Colab 端補：假 batch 含 -100，斷言 loss 有限且 -100 位置梯度為 0 | 中 |

## §4 研究層面還能優化的點（時間允許才做，論文加分項）
1. **低資源 ablation**（§2 Phase G-5）：半監督論文最有說服力的一張圖，成本低（縮小訓練集反而快）。
2. curriculum `acc_threshold: 0.60` 與 `acc_window: 200` 未調參——至少在限制章節誠實聲明；有 A100 餘裕可對 disease 掃 {0.5, 0.6, 0.7} 各 3 seeds。
3. λ1/λ2 上限與 ramp 步數同上，聲明「沿用初始設定未調參」比假裝最優誠實。
4. 無標註資料域配適：目前是通用 wiki；若時間允許，比較「與目標域同分佈的無標註」vs「通用」一組即可，是 M3/M4 故事的深度來源。
5. 與 SpokenNLP TSSP+CSSL 的直接對照：妳復現過他們的數字，主表放一列「他們論文報告值 / 妳的復現值 / 本方法」，審查者最愛這種三欄。

## §5 風險清單
- **A100 額度**：M3/M4 全 seeds 綁 A100。額度見底時的降級路徑已寫在 Phase E-4。
- **city 訓練時長**：×5 文件量；先跑 disease 全流程，city 永遠可以砍成「僅 M0/M2」。
- **Drive 爆容量**：§3 第 1 項不做，Phase E 中途會存檔失敗且錯誤訊息不直觀。
- **標籤慣例回歸**：任何人再動 data/eval 相關檔案，必跑 `tests/test_three_class_labels.py`。
- **時程**：Phase A–F 順利約 6–8 個工作天（多為掛機）；Phase G 1–2 天；Phase H 與前面平行。

---

## §6 三分類修正的技術備忘（審查者可能追問，寫論文時要用）
- 出處：SpokenNLP `emnlp2023-topic_segmentation/src/preprocess_data.py` `tokenize_method`：
  每段落 `[-100]*(n-1)+[0]`，章節末句改 1。
- 評估：其 `postprocess_predictions.py` 有 sent_level（-100 補 0）與 para_level
  （-100 成對剔除）兩種；**本 repo 採 para_level**（論文主表慣例），實作於
  `metrics.drop_ignored_pairwise` 與 lit_module 的 `keep = labels != -100`。
- 訓練：-100 位置 `seg_mask=0` 不進分割 loss（`corruption.py`）；挖空重組與
  一致性損失不受標籤值影響（其遮罩來自 is_slot/anchor，與 label 無關）。

## §7 接手 prompt 範本（開新對話直接複製）

**通用開場（每個新對話第一則）**
```
你是 coherence-seg 專案的實作工程師，接手進行中的工作（不是從零實作）。
必讀順序：PROJECT_MASTER_PLAN.md（現況與任務書）→ IMPLEMENTATION_SPEC.md
（唯一設計依據）→ RUNBOOK.md（操作步驟）。§0 已結案的問題不要重新診斷。
規則：註解與文件繁體中文；改完立即 commit+push；持久化走
MyDrive/LongformerSC/coherence-seg/；動到 data/eval 相關檔案必跑
tests/test_three_class_labels.py。讀完先以 10 行內回報你的理解與下一步，
等我確認後動工。
```

**任務B（city 支援，給 Sonnet）**：見 PROJECT_MASTER_PLAN §2 Phase B——
```
任務：新增 wiki_section_en_city 資料集支援。
1. configs/data_city.yaml：只覆寫 data.{train,dev,test}_path 指向
   /content/data/wiki_section_city/。
2. train.py：--config 含 data_*.yaml 時，config_name 與 exp_tag 附加資料集
   後綴（例 m2_reorder+city），results JSON 與 checkpoint 目錄皆不撞名；
   run_all_seeds.sh 的 NAME 同步此規則。
3. 更新 RUNBOOK 第 2 步（city 資料同批前處理產出）。
4. pytest、commit、push。約束：不動 §9.5 硬體鐵律邏輯。
```

**任務E（M3 煙霧驗證，給 Opus）**
```
任務：M3 開跑前煙霧驗證（規格書 §6 Mean Teacher）。
1. prepare_unlabeled.py --n 500 → load_unlabeled → corruption：斷言 seg_mask
   全 0、seg_labels 全 -100（labels=None 分支在三分類修正後不受影響）。
2. 一致性損失 valid 遮罩（is_slot==0 且 anchor!=-1）行為正確。
3. sanity + use_mean_teacher 跑 50 步不崩，train/ema_decay 有值。
完成後 push 並回報 M3 可正式開跑的核對表。
```

**除錯（訓練異常時）**
```
除錯任務。現象：<貼 wandb 曲線描述或完整 traceback>。環境：<設備/config/seed/step>。
請先讀 PROJECT_MASTER_PLAN.md §0 排除已結案問題，再給診斷假設（依可能性排序）
與最小驗證步驟，不要一次改多處。
```

**驗收檢查（每個 Phase 結束）**
```
驗收檢查。我完成了 <Phase X>，證據：<貼指標/輸出>。請對照
PROJECT_MASTER_PLAN.md §2 該 Phase 的通過標準逐項核對，列出缺漏，
並確認可否進入下一 Phase。
```
