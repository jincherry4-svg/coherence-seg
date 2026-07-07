"""三分類邊界標籤（-100/0/1，SpokenNLP preprocess_data.py 慣例）的專屬測試。

涵蓋：
1. corrupt_document：-100 位置與槽位的 seg_mask 皆為 0；0/1 非槽位為 1。
2. load_jsonl：允許 -100，拒絕非法值。
3. metrics：labels_to_mass 對 -100 大聲報錯；drop_ignored_pairwise 成對剔除
   後的 Pk 與「上游先過濾」完全一致（對齊 SpokenNLP para_level）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.corruption import corrupt_document
from src.eval.metrics import drop_ignored_pairwise, labels_to_mass, pk_wd_spokennlp


def make_three_class_doc(rng: np.random.Generator, n_paras: int = 6, para_len: int = 4):
    """仿 SpokenNLP tokenize_method：每段 [-100]*(len-1)+[0]，最後一句改 1。"""
    sents, labels = [], []
    for _ in range(n_paras):
        for _ in range(para_len - 1):
            sents.append(rng.integers(100, 5000, size=5).tolist())
            labels.append(-100)
        sents.append(rng.integers(100, 5000, size=5).tolist())
        labels.append(0)
    # 每兩段一個主題邊界，且文件最後一句必為 1
    for i in range(2 * para_len - 1, len(labels), 2 * para_len):
        labels[i] = 1
    labels[-1] = 1
    return sents, labels


def test_seg_mask_excludes_ignore_and_slots(ids):
    """使用者指定情境：同篇含 -100 與 0/1，挖空後——
    -100 位置與槽位 seg_mask == 0；0/1 非槽位 seg_mask == 1。"""
    rng = np.random.default_rng(11)
    sents, labels = make_three_class_doc(rng)
    assert -100 in labels and 0 in labels and 1 in labels  # 三種值都要在場
    out = corrupt_document(sents, labels, ids, rng, p=0.25)
    assert out is not None and out.m_slots > 0
    for i in range(out.n_sentences):
        orig = labels[out.student_to_clean_sent_map[i]]
        if out.is_slot[i]:
            assert out.seg_mask[i] == 0 and out.seg_labels[i] == -100
        elif orig == -100:
            # 核心修復點：-100 非槽位不得計入分割 loss
            assert out.seg_mask[i] == 0 and out.seg_labels[i] == -100
        else:
            assert out.seg_mask[i] == 1 and out.seg_labels[i] == orig


def test_load_jsonl_accepts_minus100_rejects_others(tmp_path):
    from src.data.wikisection import load_jsonl

    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(
        {"example_id": "d0", "sentences": ["a", "b", "c", "d"],
         "labels": [-100, 0, -100, 1]}) + "\n")
    docs = load_jsonl(str(good))
    assert docs[0].labels == [-100, 0, -100, 1]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(
        {"sentences": ["a", "b"], "labels": [2, 1]}) + "\n")
    with pytest.raises(AssertionError):
        load_jsonl(str(bad))


def test_labels_to_mass_rejects_minus100():
    with pytest.raises(ValueError):
        labels_to_mass([-100, 0, 1])
    assert labels_to_mass([1, 1, 0, 0, 1, 1]) == [1, 1, 3, 1]


def test_pairwise_drop_matches_prefiltered_pk():
    """含 -100 的 ref 丟給 pk_wd_spokennlp，結果須與手動先過濾完全一致。"""
    ref_full = [-100, 0, -100, 1, -100, 0, -100, 1, -100, 0, -100, 1]
    pred_full = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # -100 位置的值應被忽略
    pred_f, ref_f = drop_ignored_pairwise(pred_full, ref_full)
    assert ref_f == [0, 1, 0, 1, 0, 1]
    assert pred_f == [1, 1, 0, 1, 0, 1]
    res_auto = pk_wd_spokennlp([pred_full], [ref_full])
    res_manual = pk_wd_spokennlp([pred_f], [ref_f])
    assert res_auto["pk"] == pytest.approx(res_manual["pk"])
    assert res_auto["wd"] == pytest.approx(res_manual["wd"])
