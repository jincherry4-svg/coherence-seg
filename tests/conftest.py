"""測試共用 fixtures。FakeTokenizer 讓 corruption/collate 測試離線執行。"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.corruption import SpecialIds


class FakeTokenizer:
    """極簡 word-level tokenizer：每個新詞分配一個遞增 id（從 100 起）。"""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.bos_token_id, self.pad_token_id, self.eos_token_id = 0, 1, 2
        self.unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = 100 + len(self.vocab)
            ids.append(self.vocab[w])
        return ids


@pytest.fixture
def ids() -> SpecialIds:
    return SpecialIds(bos=0, eos=2, slot=90, cand=91, pad=1)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def make_doc(rng: np.random.Generator, n: int, min_len=3, max_len=12):
    """隨機文件：每句 token ids（>=100 避免撞特殊 id）與 0/1 邊界標籤。"""
    sents = [rng.integers(100, 5000, size=int(rng.integers(min_len, max_len))).tolist()
             for _ in range(n)]
    labels = rng.integers(0, 2, size=n).tolist()
    labels[-1] = 1  # 文件最後一句必為段落結尾
    return sents, labels


def expected_effective_labels(labels, is_slot):
    """複製 corruption.py 的【邊界標籤移交】規則，供測試對照：
    被挖走的邊界句（label==1）把邊界資訊移交給前一個最近的非槽位句。"""
    eff = [int(v) for v in labels]
    n = len(eff)
    for i in range(n):
        if is_slot[i] and int(labels[i]) == 1:
            j = i - 1
            while j >= 0 and is_slot[j]:
                j -= 1
            if j >= 0:
                eff[j] = 1
    return eff
