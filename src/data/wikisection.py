"""WikiSection（SpokenNLP 格式）與無標註資料的載入。

【邊界標籤慣例 — 全 repo 唯一，已對照 SpokenNLP 原始碼確認】
SpokenNLP（emnlp2023-topic_segmentation）的 jsonl 中，labels[i] == "1" 代表
第 i 句是「段落最後一句」（其 dataset script 將 "1" 映射為 B-EOP，註解為
"end sentence of topic"；label_to_id 為 {'B-EOP': 0, 'O': 1}，即其二分類
模型空間中 class 0 = 邊界）。本 repo 內部一律使用整數 1 = 邊界（段落最後
一句）、0 = 非邊界，僅在需要與 SpokenNLP 模型輸出直接對照時才需注意其
class index 反轉；Pk/WD 計算只依賴「1 = 關閉一個段落」的 mass 轉換，與
本慣例一致（見 src/eval/metrics.py）。
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
    labels: Optional[list[int]]  # 1 = 段落最後一句；無標註為 None


def load_jsonl(path: str, labeled: bool = True, max_docs: Optional[int] = None) -> list[DocExample]:
    """讀取 SpokenNLP 格式 jsonl：每行 {"sentences": [...], "labels": ["0"/"1", ...]}。"""
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
        if out is None:  # 超長：退回不挖空版本；仍超長則丟棄（collate 會過濾 None）
            out = corrupt_document(
                toks, doc.labels, self.ids, self._rng, p=0.0, fixed_m=0, max_len=self.max_len
            )
        return out
