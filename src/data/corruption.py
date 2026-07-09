"""句子挖空重組的輸入構造模組（規格書 §4）。

職責：給定「每句已 tokenize 的 token id 列表」，執行挖空、打亂、拼接、
標籤對齊，輸出學生（殘缺文件＋候選區）與教師（完整文件）兩份序列。

本模組刻意只依賴 numpy，不依賴 transformers，以便單元測試離線執行。
標籤慣例（與 SpokenNLP EMNLP 2023 一致，三分類，出處：
emnlp2023-topic_segmentation/src/preprocess_data.py 的 tokenize_method）：
labels[i] == 1  代表第 i 句是「主題（章節）最後一句」＝主題邊界；
labels[i] == 0  代表段落最後一句但非主題邊界；
labels[i] == -100 代表段落內非末句，一律忽略（不計 loss、不進評估）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SpecialIds:
    """組裝序列所需的特殊 token id。由 tokenizer 端提供（見 wikisection.py）。"""

    bos: int  # 每句的錨點 <s>
    eos: int  # 文件與候選區之間的分隔 </s>
    slot: int  # [SLOT]
    cand: int  # [CAND]
    pad: int


@dataclass
class CorruptionOutput:
    """單篇文件構造結果。所有 idx 皆為序列內的 token 位置（0-based）。"""

    # --- 學生輸入（殘缺文件 + 候選區）---
    input_ids: list[int]
    sent_anchor_idx: list[int]  # 長度 n：每個句位（含槽位）的錨點 token 位置
    is_slot: list[bool]  # 長度 n：該句位是否為槽位
    slot_idx: list[int]  # 長度 m：[SLOT] token 位置（依文件順序）
    cand_idx: list[int]  # 長度 m：[CAND] token 位置（依候選區順序）
    match_labels: list[int]  # 長度 m：第 j 個槽位的正確候選索引
    seg_labels: list[int]  # 長度 n：邊界標籤；槽位或無標註為 -100
    seg_mask: list[int]  # 長度 n：1 = 計入分割 loss
    # --- 教師輸入（完整文件）---
    clean_input_ids: list[int]
    clean_sent_anchor_idx: list[int]  # 長度 n
    student_to_clean_sent_map: list[int]  # 長度 n：學生句位 → 原文句索引
    n_sentences: int = 0
    m_slots: int = 0


def _assemble_clean(
    sent_token_ids: list[list[int]], ids: SpecialIds
) -> tuple[list[int], list[int]]:
    """完整文件：<s> tok... <s> tok... </s>。回傳 (input_ids, anchor_idx)。"""
    input_ids: list[int] = []
    anchor_idx: list[int] = []
    for toks in sent_token_ids:
        anchor_idx.append(len(input_ids))
        input_ids.append(ids.bos)
        input_ids.extend(toks)
    input_ids.append(ids.eos)
    return input_ids, anchor_idx


def choose_m(n: int, p: float, fixed_m: Optional[int], rng: np.random.Generator) -> int:
    """依規格 §4.1 步驟 1 決定挖空數量。n < 6 一律回傳 0。"""
    if n < 6:
        return 0
    if fixed_m is not None:  # curriculum 起始階段：直接指定 2 或 3
        return int(min(fixed_m, n - 2))
    m = int(round(p * n))
    return int(np.clip(m, 2, min(10, n - 2)))


def corrupt_document(
    sent_token_ids: list[list[int]],
    labels: Optional[list[int]],
    ids: SpecialIds,
    rng: np.random.Generator,
    p: float = 0.0,
    fixed_m: Optional[int] = None,
    max_len: int = 4096,
) -> Optional[CorruptionOutput]:
    """對單篇文件執行挖空構造。

    Args:
        sent_token_ids: 每句的 token id（不含任何特殊 token）。
        labels: 每句 0/1 邊界標籤（1 = 段落最後一句）；無標註傳 None。
        p / fixed_m: 挖空比例或固定句數（curriculum 用，fixed_m 優先）。
        max_len: 構造後（含候選區）的長度上限；超過回傳 None，由呼叫端丟棄。

    Returns:
        CorruptionOutput；序列超長時回傳 None。m=0 時學生輸入等於完整文件。
    """
    n = len(sent_token_ids)
    if n == 0:
        return None
    has_labels = labels is not None
    if has_labels:
        assert len(labels) == n, "labels 長度必須等於句數"

    clean_ids, clean_anchor = _assemble_clean(sent_token_ids, ids)

    m = choose_m(n, p, fixed_m, rng)
    if m > 0:
        # 步驟 2：永不挖第 0 句
        selected = np.sort(rng.choice(np.arange(1, n), size=m, replace=False))
        perm = rng.permutation(m)  # 候選區第 k 位 = 第 perm[k] 個被選句
        # match_labels[j] = k s.t. perm[k] == j（逆置換）
        match_labels = np.argsort(perm).tolist()
    else:
        selected = np.array([], dtype=int)
        perm = np.array([], dtype=int)
        match_labels = []
    selected_set = set(selected.tolist())

    # --- 組裝殘缺文件 ---
    input_ids: list[int] = []
    sent_anchor_idx: list[int] = []
    is_slot: list[bool] = []
    slot_idx: list[int] = []
    for i, toks in enumerate(sent_token_ids):
        sent_anchor_idx.append(len(input_ids))
        if i in selected_set:
            is_slot.append(True)
            slot_idx.append(len(input_ids))
            input_ids.append(ids.slot)  # 整句以單一 [SLOT] 取代（§4.1 步驟 3）
        else:
            is_slot.append(False)
            input_ids.append(ids.bos)
            input_ids.extend(toks)
    input_ids.append(ids.eos)  # 文件 / 候選區分隔

    # --- 候選區 ---
    cand_idx: list[int] = []
    for k in range(m):
        orig_sent = int(selected[perm[k]])
        cand_idx.append(len(input_ids))
        input_ids.append(ids.cand)
        input_ids.extend(sent_token_ids[orig_sent])

    if len(input_ids) > max_len or len(clean_ids) > max_len:
        return None

    # --- 標籤與 mask 對齊（§4.1 步驟 6）---
    # 三分類慣例（SpokenNLP preprocess_data.py）：-100 = 段落內非末句（忽略）、
    # 0 = 段落末句但非主題邊界、1 = 主題邊界。
    # seg_mask = 1 僅限「非槽位、有標註、且 label 為 0/1」的位置；
    # label 為 -100 的位置必須排除於分割 loss（否則 -100 會被當成 BCE 目標值污染訓練）。
    #
    # 【邊界標籤移交】若被挖走的是主題邊界句（label==1），挖空後主題轉換點
    # 實際上落在它前一個「保留」句之後——若不處理，該保留句 label 仍為 0 且
    # mask=1，等於在教模型「主題轉換前一句不是邊界」這個錯誤答案。
    # 因此把邊界資訊移交給前一個最近的非槽位句（覆寫其 label 為 1）。
    effective_labels = list(labels) if has_labels else None
    if has_labels:
        for i in range(n):
            if i in selected_set and int(labels[i]) == 1:
                j = i - 1
                while j >= 0 and j in selected_set:
                    j -= 1
                if j >= 0:
                    effective_labels[j] = 1
    seg_labels: list[int] = []
    seg_mask: list[int] = []
    for i in range(n):
        if is_slot[i] or not has_labels or int(effective_labels[i]) == -100:
            seg_labels.append(-100)
            seg_mask.append(0)
        else:
            seg_labels.append(int(effective_labels[i]))
            seg_mask.append(1)

    return CorruptionOutput(
        input_ids=input_ids,
        sent_anchor_idx=sent_anchor_idx,
        is_slot=is_slot,
        slot_idx=slot_idx,
        cand_idx=cand_idx,
        match_labels=match_labels,
        seg_labels=seg_labels,
        seg_mask=seg_mask,
        clean_input_ids=clean_ids,
        clean_sent_anchor_idx=clean_anchor,
        student_to_clean_sent_map=list(range(n)),  # 槽位保留原位，映射為恆等
        n_sentences=n,
        m_slots=m,
    )


def reconstruct(out: CorruptionOutput, sent_token_ids: list[list[int]], ids: SpecialIds) -> list[int]:
    """依 match_labels 把候選句放回槽位，重建完整文件（單元測試用，§10 測試 1）。

    重建規則：第 j 個槽位（文件順序）的答案是第 match_labels[j] 個候選。
    """
    # 從學生序列還原每個候選句的 token（[CAND] 之後到下一個 [CAND] 或序列尾）
    cand_tokens: list[list[int]] = []
    for k, start in enumerate(out.cand_idx):
        end = out.cand_idx[k + 1] if k + 1 < len(out.cand_idx) else len(out.input_ids)
        cand_tokens.append(out.input_ids[start + 1 : end])

    rebuilt: list[int] = []
    slot_j = 0
    doc_end = out.cand_idx[0] - 1 if out.cand_idx else len(out.input_ids) - 1  # </s> 位置
    i = 0
    while i < doc_end:
        tok = out.input_ids[i]
        if tok == ids.slot:
            rebuilt.append(ids.bos)
            rebuilt.extend(cand_tokens[out.match_labels[slot_j]])
            slot_j += 1
            i += 1
        else:
            rebuilt.append(tok)
            i += 1
    rebuilt.append(ids.eos)
    return rebuilt
