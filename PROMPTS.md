# Prompt 手冊：搭配 IMPLEMENTATION_SPEC.md 使用

使用方式：每個里程碑開一個新的對話（或 Claude Code session），**先貼「通用開場」，再貼該階段的 prompt**。每階段做完後用「驗收 prompt」檢查，通過才進下一階段。`IMPLEMENTATION_SPEC.md` 必須放在 repo 根目錄，讓模型隨時能讀。

---

## 通用開場（每個新對話的第一段，必貼）

```
你是本專案的實作工程師。專案是一個以 Longformer 為基底、加入「句子挖空重組」自監督輔助任務的半監督文本分割模型。

規則：
1. 先完整閱讀 repo 根目錄的 IMPLEMENTATION_SPEC.md，這是唯一的設計依據。所有架構決策已定案，你不得更改設計；規格未涵蓋的細節選最簡單、最可測試的做法，並記錄在 README 的「實作決策」小節。
2. 接著檢視 repo 目前的程式碼與 git log，掌握前面里程碑已完成的部分，不要重寫已通過驗收的模組。
3. 程式碼註解與說明文件使用繁體中文。
4. 每完成一個檔案就跑對應的測試；任何測試失敗要修到通過才能回報完成。
5. 回報時列出：新增/修改的檔案、pytest 結果、你做的實作決策、以及你認為的風險點。
6. 本專案全程在 Google Colab（A100）上實作與執行（規格書 §9）：所有指令必須能在 notebook cell 以 ! 前綴執行；程式碼以 GitHub 為唯一真實來源，任何修改完成後立即 commit + push；持久化（資料、checkpoint、結果）一律走 Google Drive；長訓練必須支援斷線續訓。

讀完規格書與 repo 後，先用 10 行以內摘要你理解的當前任務與現況，再開始動工。
```

---

## Prompt 0 — 專案初始化

```
任務：建立專案骨架，對應規格書 §1、§2、§13。

具體要求：
1. 依 §2 建立完整目錄結構與空模組檔（含 __init__.py），每個檔案頂部寫一句中文 docstring 說明職責。
2. 建立 requirements.txt（§1，protobuf 必須 pin 3.20.3）。
3. 建立 configs/base.yaml 與 configs/sanity_1080ti.yaml 的骨架，欄位涵蓋：資料路徑、max_len、batch、accumulation、lr（encoder 與 heads 分開）、warmup 比例、epochs、seed、ckpt_dir、wandb project/group、挖空 curriculum 參數、λ1/λ2 的 w_max 與 T。用 OmegaConf 可讀的格式。
4. 建立 src/train.py 進入點：讀 config、seed_everything、印出 config，暫時不接模型。
5. 建立 README.md 骨架：安裝、資料準備、各里程碑執行指令（§13）、「實作決策」空小節。
6. 建立 .gitignore（排除資料、checkpoint、wandb）並 git init + 首次 commit。
7. 建立 notebooks/colab_runner.ipynb（規格書 §9.2）：依序包含 GPU 檢查、掛載 Drive、clone/pull + pip install、從 Colab Secrets 讀 WANDB_API_KEY、資料同步 Drive→/content、pytest、訓練 cell、續訓 cell（--resume）、結果回存 Drive。所有 cell 必須冪等可重跑，checkpoint 路徑指向 Drive。

完成後回報目錄樹、config 內容，以及 colab_runner.ipynb 各 cell 的用途清單。
```

---

## Prompt 1 — M0：Baseline 復現

```
任務：完成里程碑 M0（規格書 §11 M0），即純 Longformer + 邊界分類頭的監督式 baseline，不含任何挖空或輔助任務。

實作範圍：
1. src/data/wikisection.py：載入 SpokenNLP 格式的 wiki_section_disease，輸出 §3.1 的統一資料表示。實作前先閱讀 SpokenNLP repo 的資料處理與評估程式碼，確認邊界標籤慣例（標籤 1 是段落最後一句還是第一句），把結論寫成檔案頂部註解，全 repo 統一。
2. src/data/collate.py：M0 版 collate（無挖空）：每句前插 <s>、拼接、padding、global_attention_mask（token 0 + 所有句子 <s>）、sent_anchor_idx、seg_labels、seg_mask。
3. src/models/encoder.py 與 heads.py：Longformer 包裝 + §5.2 邊界分類頭（focal loss，gamma=2, alpha=0.75，config 可切 BCE）。記得 add_special_tokens 後 resize_token_embeddings（[SLOT]、[CAND] 現在就加好，M1 之後會用到）。
4. src/losses.py：focal loss（用 logsigmoid 實作，bf16 數值穩定）。
5. src/models/lit_module.py + src/train.py：Lightning 訓練迴圈，optimizer/排程/精度/checkpoint/wandb 依 §7。
6. src/eval/metrics.py：移植 SpokenNLP 的 Pk/WindowDiff 計算並註明出處；另用 segeval 實作第二套；threshold 掃描邏輯依 §8。
7. tests/：針對 collate 的索引正確性、metrics 兩套實作一致性（§10 測試 10）寫測試。
8. configs/m0_baseline.yaml。

先在 Colab 上用 configs/sanity_1080ti.yaml 的縮小設定跑通 1 個 epoch 作為煙霧測試（幾分鐘內完成），再回報。回報內容：pytest 結果、sanity run 的 loss 曲線走向、val Pk/WD/F1 數字。
```

M0 訓練完成後，用這段確認 baseline 合格：

```
請在完整設定（configs/m0_baseline.yaml、A100）跑 seed=42 的完整訓練，回報 val 與 test 的 Pk / WindowDiff / boundary F1，以及最佳 threshold。同時把 wandb run 連結與 commit hash 記進 README。
```

---

## Prompt 2 — M1：輸入構造模組

```
任務：完成里程碑 M1（規格書 §4、§11 M1）：挖空/打亂/拼接/對齊模組與其測試。這是全案最容易藏 bug 的模組，正確性優先於一切。

實作範圍：
1. src/data/corruption.py：嚴格依 §4.1 的 8 個步驟實作。函式簽名建議：
   corrupt_document(sentences, labels, p, rng, tokenizer, fixed_m=None) -> CorruptionOutput
   其中 fixed_m 供 curriculum 起始階段直接指定挖 2–3 句。
2. src/data/collate.py：擴充為完整版，輸出 §4.2 的全部欄位（含 clean_* 教師輸入與 student_to_clean_sent_map；M0 路徑保持可用，用 config 開關）。
3. 處理 §12 陷阱 7：num_workers>0 時以 worker_init_fn 為每個 worker 派生獨立 rng。
4. tests/：實作 §10 的測試 1、2、3、4、5、8、9，全部通過。
5. scripts/visualize_corruption.py：讀一篇文件，印出挖空前後的句子清單、槽位位置、打亂後候選順序、match_labels，供人工檢查。

回報：pytest 全部結果 + visualize 腳本對 3 篇文件的輸出。不需要跑訓練。
```

---

## Prompt 3 — M2：加入重組輔助任務（Stage A 核心）

```
任務：完成里程碑 M2（規格書 §5.3、§6、§11 M2）：在有標註資料上加入句子重組輔助任務與 curriculum。

實作範圍：
1. src/models/heads.py：加入 §5.3 配對頭（雙線性打分 + 逐槽位 CE，ignore_index=-100），並實作重組準確率計算。
2. src/losses.py：sigmoid ramp-up 函式（§6 公式），λ1 排程。
3. src/models/lit_module.py：training_step 改為 L = L_seg + λ1(t)·L_reorder；槽位位置不計分割 loss（用 seg_mask）；wandb 記錄 §7 列出的全部項目。
4. Curriculum 狀態機（§6 表格）：起始 fixed_m∈{2,3}，重組準確率 200-step 移動平均 > 0.60 時升到 p=0.15，再達標升 p=0.25。升級事件記 wandb。續訓時狀態要能從 checkpoint 恢復。
5. src/eval/decode.py：linear_sum_assignment 一對一解碼（僅分析用）。
6. tests/：§10 測試 6。
7. configs/m2_reorder.yaml。

驗證流程：先 sanity 設定跑通，確認 reorder_acc 曲線在幾百步內明顯高於隨機水準（1/m）；然後 A100 完整設定跑 seed=42。回報：pytest 結果、reorder_acc 與 λ1 曲線描述、curriculum 升級發生的 step、val/test Pk/WD/F1 與 M0 的對比。無論優於或劣於 M0 都如實回報。
```

---

## Prompt 4 — M3：Mean Teacher（Stage B 核心）

```
任務：完成里程碑 M3（規格書 §3.2、§5.4、§6、§11 M3）：無標註資料 + Mean Teacher。
注意本階段的無標註資料「只」透過一致性損失貢獻（flags.unlabeled_reorder=false），
不計 L_reorder——無標註文件仍會挖空，但僅作為學生端的強增強。

實作範圍：
1. scripts/prepare_unlabeled.py：依 §3.2 用 datasets streaming 載 wikimedia/wikipedia、pysbd 切句、過濾（20≤句數≤150）、抽 50000 篇存 jsonl，含 seed 與進度列。
2. src/data/unlabeled.py：載入 jsonl，輸出統一資料表示（labels=None）。
3. src/models/ema.py：EMA 教師（deepcopy 初始化、decay = min(0.999, (1+step)/(10+step))、每個 optimizer step 後更新、恆為 eval() + no_grad()、dropout 關閉、隨 checkpoint 保存以支援續訓）。不得依賴外部 EMA 套件。
4. src/models/lit_module.py：CombinedLoader(mode="max_size_cycle") 混合 labeled/unlabeled；教師對 clean_* 完整文件 forward 得邊界機率，學生對挖空文件的邊界機率經 student_to_clean_sent_map 對齊後，只在非槽位真實句上算 MSE；labeled 與 unlabeled batch 都算一致性；unlabeled 不算 L_seg 也不算 L_reorder（unlabeled_reorder=false 路徑要正確短路、不產生 NaN）。
5. 總損失 L = L_seg + λ1·L_reorder + λ2(t)·L_consistency，λ2 依 §6（w_max=1.0, T=4000）。
6. tests/：§10 測試 7（EMA 數值正確、無梯度）與測試 8（對齊映射）。
7. configs/m3_meanteacher.yaml。
8. 若 A100 記憶體吃緊（教師多一次前向傳播），教師 forward 用 no_grad 並可分塊，不得改變演算法。

驗證流程：先用 500 篇 unlabeled 的 sanity 設定確認兩個 loader 交替正常、L_seg 不發散、L_consistency 與 ema_decay 曲線形狀合理；再 A100 完整跑 seed=42。回報：pytest 結果、λ2 與 L_consistency 曲線描述、val/test 指標與 M2 對比。
```

---

## Prompt 5 — M4：無標註重組全開（full）

```
任務：完成里程碑 M4（規格書 §11 M4）：在 M3 之上設 flags.unlabeled_reorder=true，
使無標註 batch 同時計 λ1·L_reorder 與 λ2·L_consistency。M4−M3 的差異即
「無標註重組訊號」的貢獻。

實作範圍：
1. 確認 lit_module 中 unlabeled_reorder=true 時無標註 batch 的 L_reorder 路徑正確（含配對頭前向、match_labels、重組準確率統計併入 curriculum 的移動平均）。
2. 確認無標註文件與有標註文件共用同一套 corruption 與 curriculum 狀態。
3. configs/m4_full.yaml。
4. 補一個單元測試：以假 config 驗證 unlabeled_reorder 開/關時 loss 組成正確（可 mock 模型 forward）。

回報：pytest 結果、unlabeled 的 L_reorder 曲線描述、val/test 指標與 M3 對比、以及 M0–M4 完整 ablation 初表。
```

---

## Prompt 6 — 多 seed 實驗與最終彙整

```
任務：完成多 seed 實驗與 ablation 總表（規格書 §8、§11 M4 驗收）。

實作範圍：
1. src/eval/aggregate.py：從 wandb（或本地結果 json）撈同一 group 的多個 run，輸出各設定的 Pk / WindowDiff / boundary F1 的 mean ± std；實作 paired bootstrap（10000 次重抽）比較任兩組設定的 Pk 差異與 p-value。
2. scripts/run_all_seeds.sh：對指定 config 依序跑 seeds 42 43 44 45 46（Colab 環境下可分次執行、支援續跑，已完成的 seed 自動跳過）。
3. 對 M0、M2、M3、M4 四組設定各跑滿 5 seeds。
4. 產出最終 ablation 總表（markdown），寫入 results/ABLATION.md：四組設定 × 三個指標的 mean±std，加上 M2 vs M0、M3 vs M2、M4 vs M3 的 bootstrap p-value。

回報：總表全文與你對結果的三點觀察（哪個元件貢獻最大、變異程度、是否有 seed 敏感問題）。
```

---

## 常用輔助 prompt

**驗收檢查（每個里程碑完成後貼）**

```
請對照 IMPLEMENTATION_SPEC.md §11 中本里程碑的驗收標準逐條自查，輸出一張核對表（標準 / 是否通過 / 證據）。同時對照 §12 的十條陷阱清單逐條確認。任何一條不通過就先修復再重新回報，不要進入下一步。
```

**除錯（結果異常時貼，附上現象）**

```
現象：<貼上錯誤訊息 / 異常的 loss 或指標曲線描述>

請按以下順序診斷，不要直接猜測改 code：
1. 先跑 pytest 全部測試，確認基礎模組正確。
2. 用 scripts/visualize_corruption.py 檢查 3 篇文件的挖空輸出。
3. 檢查 wandb 曲線：L_seg、L_reorder、reorder_acc、λ 排程是否符合預期形狀。
4. 列出 3 個最可能的原因與各自的最小驗證實驗，經我確認後再動手修改。
修復後說明根因，並補一個能抓到此 bug 的回歸測試。
```

**新對話接手（session 中斷或換模型時貼）**

```
這是一個進行中的專案。請先讀 IMPLEMENTATION_SPEC.md，再讀 README 的「實作決策」小節與 git log，然後跑一次 pytest 確認現況。完成後回報：目前處於哪個里程碑、已通過與未完成的部分、pytest 結果。在我確認你的理解正確之前，不要修改任何程式碼。
```

**Colab 斷線復原（session 中斷後貼）**

```
Colab session 剛才中斷了。請依 notebooks/colab_runner.ipynb 的順序重建環境（掛載 Drive、pull repo、安裝套件、資料同步），然後：
1. 檢查 Drive 上 ckpt_dir 的 last.ckpt 時間戳與檔案完整性。
2. 用 --resume 恢復訓練，並在恢復後的前 50 步確認：global step 接續正確、learning rate 與 λ ramp-up 數值連續、curriculum 的 p 值與中斷前一致、（M4）EMA 教師已載入。
3. 把上述五項的實際數值回報給我，確認無誤後才讓訓練繼續跑完。
```

**續訓正確性驗證（M0 完成後、開始長訓練前貼一次）**

```
請依規格書 §9.3 驗證斷線續訓的正確性：用 sanity 設定訓練 300 步後手動中斷，--resume 再跑 300 步；另外連續跑 600 步作為對照。比較兩者的 loss 曲線與最終權重差異，確認 global step、LR 排程、λ ramp-up、curriculum 狀態、隨機源都正確續接。回報對照結果；若不一致，修復後補回歸測試。
```

**降級到 1080 Ti sanity check（備用：在實驗室機器上驗證時貼）**

```
請用 configs/sanity_1080ti.yaml 在單張 GTX 1080 Ti 上跑通當前里程碑的訓練 1 個 epoch。此設定僅驗證程式正確性，不看指標。若 OOM，依序嘗試：確認 gradient checkpointing 已開、max_len 降到 512、accumulation 提高。不得為了省記憶體修改模型架構。
```
