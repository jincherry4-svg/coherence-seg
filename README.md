# coherence-seg：句子重組輔助任務之半監督文本分割

以 Longformer 為基底的文本分割模型，加入「句子挖空重組」自監督輔助任務
（槽位–候選配對），並透過無標註資料與 Mean Teacher 引入半監督訊號。
架構參考 [SpokenNLP emnlp2023-topic_segmentation](https://github.com/alibaba-damo-academy/SpokenNLP/tree/main/emnlp2023-topic_segmentation)
（Yu et al., EMNLP 2023）的 Longformer 句級序列標註設計，輔助任務為其 TSSP 的泛化強化版。

**設計唯一依據：`IMPLEMENTATION_SPEC.md`。給實作模型的指令：`PROMPTS.md`。**

## 快速開始（Colab）

打開 `notebooks/colab_runner.ipynb`，依 cell 順序執行（規格書 §9.2）。
本機 / 實驗室機器：

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                       # 離線可跑（不需下載模型）
python scripts/visualize_corruption.py           # 離線檢視挖空輸出
bash scripts/prepare_wikisection.sh /content/data
python scripts/prepare_unlabeled.py --n 50000 --seed 42
python -m src.train --config configs/m0_baseline.yaml --seed 42
```

## 里程碑（規格書 §11，嚴格依序）

| 里程碑 | config | 內容 |
|---|---|---|
| M0 | `m0_baseline.yaml` | 純 Longformer + 邊界頭 baseline |
| M2 | `m2_reorder.yaml` | + 句子重組輔助任務 + curriculum |
| M3 | `m3_meanteacher.yaml` | + EMA 教師與一致性損失（無標註只走一致性；挖空=強增強） |
| M4 | `m4_full.yaml` | 全開：無標註 batch 加算 L_reorder |
| — | `sanity_1080ti.yaml` | 煙霧測試（Colab 或 1080 Ti，不看指標） |

多 seed：`bash scripts/run_all_seeds.sh configs/m2_reorder.yaml`；
彙整：`python -m src.eval.aggregate`。

## 邊界標籤慣例（全 repo 唯一）

`labels[i] == 1` = 第 i 句是**段落最後一句**（B-EOP）。已對照 SpokenNLP 原始碼
確認（其 jsonl 的 "1" 映射為 B-EOP；其模型空間 class 0 = 邊界，本 repo 用單
logit sigmoid 故無此反轉問題）。Pk/WD 採其 example-level 作法：逐篇轉 mass
後以 segeval 計算再平均（`src/eval/metrics.py`）。

## 實作決策（規格未涵蓋處）

- 序列首 token 即第 0 句的 `<s>` 錨點，不另加 CLS；global attention 已涵蓋。
- 挖空後超過 max_len 的文件退回「不挖空版本」，仍超長才丟棄（訓練集）。
- Curriculum 狀態以 multiprocessing 共享記憶體傳遞給 dataloader workers
  （Linux fork 模式下有效；Colab 預設即 fork）。
- 配對頭對 padding 候選填 `-inf` 分數，防止搶走 softmax 機率。
- 教師（M4）的 buffers 直接複製學生當前值，參數走 EMA。

## 測試

`tests/` 全部離線可跑（FakeTokenizer，不需網路與模型權重），涵蓋規格書
§10 的測試 1–10（測試 1–5、8、9 在 `test_corruption.py`；4、6、7、10 在
`test_modules.py`）。
