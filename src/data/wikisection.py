"""WikiSection（SpokenNLP 格式）與無標註資料的載入。

【邊界標籤慣例 — 全 repo 唯一，已對照 SpokenNLP 原始碼確認】
出處：SpokenNLP（emnlp2023-topic_segmentation）`src/preprocess_data.py` 的
`tokenize_method`：每個段落產生 `[-100]*(len(p_sents)-1) + [0]`，章節最後
一句再改為 1（原始碼註解："label of final sentence of topic is 1, final
sentence of each paragraph is 0, other sentences is -100"）。即**三分類**：

    -100 = 段落內非末句 → 忽略（不計 loss、不進評估）
       0 = 段落最後一句、但非主題邊界
       1 = 主題（章節）邊界（章節最後一句）

評估慣例對齊其 `postprocess_predictions.py` 的 para_level 作法：把 -100
位置**成對剔除**後，只在段落末句序列上計算 Pk/WD/F1（本 repo 的
lit_module 於 val/test 以 `labels != -100` 過濾，見 src/eval/metrics.py）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from .collate import CurriculumController
from .corruption import CorruptionOutput, SpecialIds, corrupt_document


@dataclass
class DocExample:
    doc_id: str
    sentences: list[str]
    labels: Optional[list[int]]  # 三分類 -100/0/1（見模組頂部）；無標註為 None


_VALID_LABELS = {-100, 0, 1}


def load_jsonl(path: str, labeled: bool = True, max_docs: Optional[int] = None) -> list[DocExample]:
    """讀取 SpokenNLP 格式 jsonl：每行 {"sentences": [...], "labels": [...]}。

    labels 允許三種值：-100（段落內非末句，忽略）、0（段落末句非主題邊界）、
    1（主題邊界）。其他值直接報錯，避免標籤慣例悄悄跑掉。
    """
    docs: list[DocExample] = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            if max_docs is not None and len(docs) >= max_docs:
                break
            obj = json.loads(line)
            sents = obj["sentences"]
            labels = None
            if labeled:
                labels = [int(v) for v in obj["labels"]]
                assert len(labels) == len(sents)
                bad = set(labels) - _VALID_LABELS
                assert not bad, (
                    f"{path} 第 {line_no} 行含非法標籤值 {bad}；"
                    f"合法值僅 -100/0/1（SpokenNLP 三分類，見模組頂部註解）"
                )
            docs.append(DocExample(str(obj.get("example_id", line_no)), sents, labels))
    return docs


def build_special_ids(tokenizer) -> SpecialIds:
    """從 HF tokenizer 取得特殊 token id。呼叫前 tokenizer 必須已加入 [SLOT]/[CAND]。"""
    slot = tokenizer.convert_tokens_to_ids("[SLOT]")
    cand = tokenizer.convert_tokens_to_ids("[CAND]")
    assert slot != tokenizer.unk_token_id and cand != tokenizer.unk_token_id, (
        "請先 tokenizer.add_special_tokens({'additional_special_tokens': ['[SLOT]', '[CAND]']}) "
        "並 model.resize_token_embeddings(len(tokenizer))（規格書 §3.3 / §12 陷阱 1）"
    )
    return SpecialIds(
        bos=tokenizer.bos_token_id,
        eos=tokenizer.eos_token_id,
        slot=slot,
        cand=cand,
        pad=tokenizer.pad_token_id,
    )


class SegDataset(Dataset):
    """回傳 CorruptionOutput 的 Dataset。挖空在 __getitem__ 動態執行。

    - 訓練集：依 curriculum controller 的當前狀態挖空。
    - 驗證/測試集或 M0：controller.enabled=0，永不挖空（規格書 §4.1 步驟 7）。
    - 每篇文件的 per-sentence token ids 首次存取時計算並快取。
    """

    def __init__(
        self,
        docs: list[DocExample],
        tokenizer,
        ids: SpecialIds,
        controller: CurriculumController,
        max_len: int = 4096,
        base_seed: int = 42,
    ):
        self.docs = docs
        self.tokenizer = tokenizer
        self.ids = ids
        self.controller = controller
        self.max_len = max_len
        self._token_cache: dict[int, list[list[int]]] = {}
        self._rng = np.random.default_rng(base_seed)

    def set_rng(self, rng: np.random.Generator) -> None:  # worker_init_fn 用
        self._rng = rng

    def __len__(self) -> int:
        return len(self.docs)

    def _sent_token_ids(self, idx: int) -> list[list[int]]:
        if idx not in self._token_cache:
            doc = self.docs[idx]
            self._token_cache[idx] = [
                self.tokenizer.encode(s, add_special_tokens=False) for s in doc.sentences
            ]
        return self._token_cache[idx]

    def __getitem__(self, idx: int) -> Optional[CorruptionOutput]:
        doc = self.docs[idx]
        toks = self._sent_token_ids(idx)
        p, fixed_m = self.controller.sample_params(self._rng)
        out = corrupt_document(
            toks, doc.labels, self.ids, self._rng, p=p, fixed_m=fixed_m, max_len=self.max_len
        )
        if out is None:  # 超長：退回不挖空版本
            out = corrupt_document(
                toks, doc.labels, self.ids, self._rng, p=0.0, fixed_m=0, max_len=self.max_len
            )
        if out is None:
            # 仍超長 → 截斷保留（取能裝進 max_len 的最長句子前綴），而非整篇丟棄。
            # 舊行為會讓最長的一批文件從 train/val/test 完全消失，評估集不完整、
            # 與 SpokenNLP（tokenizer 截斷保留前段）不可比。
            budget = self.max_len - 1
            cum, m_fit = 0, 0
            for t in toks:
                cost = 1 + len(t)
                if cum + cost > budget:
                    break
                cum += cost
                m_fit += 1
            if m_fit == 0:
                return None
            trunc_labels = doc.labels[:m_fit] if doc.labels is not None else None
            out = corrupt_document(
                toks[:m_fit], trunc_labels, self.ids, self._rng,
                p=0.0, fixed_m=0, max_len=self.max_len,
            )
        return out
