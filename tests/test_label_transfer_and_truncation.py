"""邊界標籤移交（label transfer）與超長文件截斷保留的測試。

背景（源自操作者早期 TSSP 復現筆記本的洞見）：
1. 挖走邊界句（label==1）後若不移交邊界資訊，前一保留句 label 仍為 0 且
   mask=1，等於在教模型「主題轉換前一句不是邊界」的錯誤答案。
2. 舊行為把超過 max_len 的文件整篇丟棄（train/val/test 皆然），導致評估集
   不完整；新行為改為截斷保留前段。
只依賴 numpy，可離線執行。
"""

from __future__ import annotations

import numpy as np

from src.data.corruption import corrupt_document
from tests.conftest import expected_effective_labels


def _boundary_doc(rng, n=24):
    sents = [rng.integers(100, 5000, size=6).tolist() for _ in range(n)]
    labels = [1 if (i + 1) % 4 == 0 else 0 for i in range(n)]
    return sents, labels


def test_boundary_label_transfer_happens(ids):
    transfers_seen = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        sents, labels = _boundary_doc(rng)
        out = corrupt_document(sents, labels, ids, rng, p=0.35)
        if out is None or out.m_slots == 0:
            continue
        eff = expected_effective_labels(labels, out.is_slot)
        for i in range(out.n_sentences):
            if out.is_slot[i]:
                assert out.seg_labels[i] == -100 and out.seg_mask[i] == 0
            else:
                assert out.seg_labels[i] == eff[i]
                if eff[i] != labels[i]:
                    assert eff[i] == 1
                    transfers_seen += 1
    assert transfers_seen > 0, "20 個 seed 都沒觀察到移交——機制疑似未生效"


def test_transfer_skips_over_adjacent_slots(ids):
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        sents, labels = _boundary_doc(rng, n=16)
        out = corrupt_document(sents, labels, ids, rng, fixed_m=5)
        if out is None:
            continue
        for i in range(1, out.n_sentences):
            if out.is_slot[i] and labels[i] == 1 and out.is_slot[i - 1]:
                j = i - 1
                while j >= 0 and out.is_slot[j]:
                    j -= 1
                if j >= 0:
                    assert out.seg_labels[j] == 1, "邊界未跳過相鄰槽位正確移交"
                    return
    raise AssertionError("未能構造出連續槽位含邊界句的情境")


class _FakeTokenizer:
    def __init__(self, tokens_per_sent=50):
        self.k = tokens_per_sent

    def encode(self, s, add_special_tokens=False):
        return list(range(100, 100 + self.k))


def test_overlong_doc_truncated_not_dropped(ids):
    from src.data.collate import CurriculumController
    from src.data.wikisection import DocExample, SegDataset

    n_sents = 30
    doc = DocExample("long_doc", ["s"] * n_sents,
                     [1 if (i + 1) % 5 == 0 else 0 for i in range(n_sents)])
    ds = SegDataset([doc], _FakeTokenizer(50), ids,
                    CurriculumController(enabled=False), max_len=512, base_seed=0)
    out = ds[0]
    assert out is not None, "超長文件被丟棄了——截斷保留邏輯未生效"
    assert out.n_sentences == 10
    assert len(out.seg_labels) == 10
    assert out.seg_labels[4] == 1 and out.seg_labels[9] == 1


def test_normal_doc_unaffected_by_truncation_path(ids):
    from src.data.collate import CurriculumController
    from src.data.wikisection import DocExample, SegDataset

    doc = DocExample("ok_doc", ["s"] * 8, [0, 0, 0, 1, 0, 0, 0, 1])
    ds = SegDataset([doc], _FakeTokenizer(10), ids,
                    CurriculumController(enabled=False), max_len=512, base_seed=0)
    out = ds[0]
    assert out is not None and out.n_sentences == 8
    assert out.seg_labels == [0, 0, 0, 1, 0, 0, 0, 1]
