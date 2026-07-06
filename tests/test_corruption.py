"""規格書 §10 測試 1、2、3、5、8、9：corruption 模組正確性。"""

import numpy as np
import pytest

from src.data.corruption import corrupt_document, reconstruct, _assemble_clean
from tests.conftest import make_doc


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reconstruction_equals_original(ids, seed):
    """測試 1：依 match_labels 還原後 == 原文（100 篇 × 多 seed）。"""
    rng = np.random.default_rng(seed)
    for _ in range(20):
        n = int(rng.integers(6, 60))
        sents, labels = make_doc(rng, n)
        out = corrupt_document(sents, labels, ids, rng, p=0.25)
        assert out is not None
        clean, _ = _assemble_clean(sents, ids)
        assert reconstruct(out, sents, ids) == clean == out.clean_input_ids


def test_label_alignment_and_mask(ids, rng):
    """測試 2：槽位 seg_mask=0 / seg_labels=-100；非槽位標籤一致；無標註全 0。"""
    sents, labels = make_doc(rng, 30)
    out = corrupt_document(sents, labels, ids, rng, p=0.2)
    assert out.n_sentences == 30 and out.m_slots > 0
    for i in range(30):
        if out.is_slot[i]:
            assert out.seg_labels[i] == -100 and out.seg_mask[i] == 0
        else:
            assert out.seg_labels[i] == labels[i] and out.seg_mask[i] == 1
    out_u = corrupt_document(sents, None, ids, rng, p=0.2)
    assert all(v == 0 for v in out_u.seg_mask)
    assert all(v == -100 for v in out_u.seg_labels)


def test_special_token_positions(ids, rng):
    """測試 3：slot_idx / cand_idx / sent_anchor_idx 指向正確 token。"""
    sents, labels = make_doc(rng, 25)
    out = corrupt_document(sents, labels, ids, rng, p=0.25)
    for pos in out.slot_idx:
        assert out.input_ids[pos] == ids.slot
    for pos in out.cand_idx:
        assert out.input_ids[pos] == ids.cand
    for i, pos in enumerate(out.sent_anchor_idx):
        expected = ids.slot if out.is_slot[i] else ids.bos
        assert out.input_ids[pos] == expected
    # 第 0 句永不挖空
    assert not out.is_slot[0]


def test_max_len_guard(ids, rng):
    """測試 5：超長回傳 None；正常輸出 ≤ max_len。"""
    sents, labels = make_doc(rng, 40, min_len=8, max_len=12)
    assert corrupt_document(sents, labels, ids, rng, p=0.2, max_len=50) is None
    out = corrupt_document(sents, labels, ids, rng, p=0.2, max_len=4096)
    assert len(out.input_ids) <= 4096 and len(out.clean_input_ids) <= 4096


def test_student_to_clean_map(ids, rng):
    """測試 8：槽位保留原位，映射為恆等。"""
    sents, labels = make_doc(rng, 15)
    out = corrupt_document(sents, labels, ids, rng, p=0.2)
    assert out.student_to_clean_sent_map == list(range(15))


def test_determinism(ids):
    """測試 9：同 seed 輸出完全一致。"""
    r1, r2 = np.random.default_rng(123), np.random.default_rng(123)
    sents, labels = make_doc(np.random.default_rng(9), 20)
    o1 = corrupt_document(sents, labels, ids, r1, p=0.25)
    o2 = corrupt_document(sents, labels, ids, r2, p=0.25)
    assert o1.input_ids == o2.input_ids and o1.match_labels == o2.match_labels


def test_small_doc_skips_corruption(ids, rng):
    """n < 6 一律不挖空（§4.1 步驟 1）。"""
    sents, labels = make_doc(rng, 4)
    out = corrupt_document(sents, labels, ids, rng, p=0.5)
    assert out.m_slots == 0 and out.slot_idx == [] and out.input_ids == out.clean_input_ids
