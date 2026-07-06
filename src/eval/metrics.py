"""Pk / WindowDiff / boundary F1（規格書 §8）。

主要實作移植自 SpokenNLP emnlp2023-topic_segmentation 的
src/metrics/seqeval.py（Alibaba DAMO Academy, Apache-2.0）：
把「1 = 段落最後一句」的標籤序列轉成 mass（[1,1,0,0,1,1] → [1,1,3,1]），
逐篇以 segeval 計算 Pk / WD 後取平均；F1/P/R 在展平的標籤上以 micro 方式計算。

第二套純 Python 實作（reference_pk / reference_wd）依 Beeferman 1999 /
Pevzner & Hearst 2002 的定義撰寫，單元測試要求與 segeval 在同一份預測上
一致（§10 測試 10）。
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

try:  # segeval 為主要實作；離線環境缺套件時 reference 版仍可用
    from segeval.window.pk import pk as _segeval_pk
    from segeval.window.windowdiff import window_diff as _segeval_wd

    HAS_SEGEVAL = True
except ImportError:  # pragma: no cover
    HAS_SEGEVAL = False


def labels_to_mass(labels: list[int]) -> list[int]:
    """[1,1,0,0,1,1] → [1,1,3,1]。1 = 該句關閉一個段落；殘尾自成一段。"""
    mass, cur = [], 0
    for v in labels:
        cur += 1
        if v == 1:
            mass.append(cur)
            cur = 0
    if cur > 0:
        mass.append(cur)
    return mass


# ---------- 純 Python 參考實作（與 segeval 定義對齊） ----------


def _mass_to_positions(mass: list[int]) -> list[int]:
    """mass → 每個單位所屬段落 id 序列，如 [2,3] → [0,0,1,1,1]。"""
    pos = []
    for seg_id, m in enumerate(mass):
        pos.extend([seg_id] * m)
    return pos


def _segeval_window_size(ref_mass: list[int]) -> int:
    """與 segeval __compute_window_size__ 一致：round(Decimal 平均段長 / 2)，最小 2。

    注意 round(Decimal) 為 banker's rounding，與 segeval 的 fnc_round=round 相同。
    """
    avg = (sum(Decimal(m) for m in ref_mass) / Decimal(len(ref_mass))) / Decimal(2)
    k = int(round(avg))
    return k if k > 1 else 2


def reference_pk(hyp_mass: list[int], ref_mass: list[int]) -> float:
    """Beeferman Pk。視窗比較 position[i] 與 position[i+k] 是否同段。"""
    n = sum(ref_mass)
    assert n == sum(hyp_mass), "hyp 與 ref 的總句數必須一致"
    k = _segeval_window_size(ref_mass)
    ref_pos = _mass_to_positions(ref_mass)
    hyp_pos = _mass_to_positions(hyp_mass)
    errors, total = 0, 0
    for i in range(n - k):
        same_ref = ref_pos[i] == ref_pos[i + k]
        same_hyp = hyp_pos[i] == hyp_pos[i + k]
        errors += int(same_ref != same_hyp)
        total += 1
    return errors / total if total else 0.0


def reference_wd(hyp_mass: list[int], ref_mass: list[int]) -> float:
    """Pevzner & Hearst WindowDiff。視窗內邊界數不同即記一次錯。"""
    n = sum(ref_mass)
    assert n == sum(hyp_mass)
    k = _segeval_window_size(ref_mass)

    def boundaries(mass: list[int]) -> list[int]:
        b = [0] * n  # b[i] = 單位 i 與 i+1 之間是否有邊界
        acc = 0
        for m in mass[:-1]:
            acc += m
            b[acc - 1] = 1
        return b

    rb, hb = boundaries(ref_mass), boundaries(hyp_mass)
    errors, total = 0, 0
    for i in range(n - k):
        r_cnt = sum(rb[i : i + k])
        h_cnt = sum(hb[i : i + k])
        errors += int(r_cnt != h_cnt)
        total += 1
    return errors / total if total else 0.0


# ---------- 主要入口 ----------


def pk_wd_spokennlp(
    predictions: list[list[int]], references: list[list[int]], use_segeval: bool = True
) -> dict[str, float]:
    """逐篇計算 Pk/WD 後平均（SpokenNLP 的 example-level 作法）。

    Args:
        predictions/references: 每篇文件的 0/1 標籤序列（1 = 段落最後一句）。
    """
    pks, wds = [], []
    for pred, ref in zip(predictions, references):
        hyp_mass, ref_mass = labels_to_mass(pred), labels_to_mass(ref)
        if sum(hyp_mass) != sum(ref_mass) or sum(ref_mass) == 0:
            continue  # 與 SpokenNLP 相同：異常樣本跳過
            
        if use_segeval and HAS_SEGEVAL:
            # 【防禦安全鎖】動態計算當前文件的視窗大小 k 與總句數
            k = _segeval_window_size(ref_mass)
            total_sentences = sum(ref_mass)
            
            # 如果單篇文件的總句數太短（小於或等於視窗長度），segeval 計算分母會 <= 0 導致除以零暴斃
            if total_sentences <= k:
                # 針對極短文本，指標安全計為 0.0
                pks.append(0.0)
                wds.append(0.0)
            else:
                try:
                    pks.append(float(_segeval_pk(hyp_mass, ref_mass)))
                    wds.append(float(_segeval_wd(hyp_mass, ref_mass)))
                except Exception:
                    pks.append(0.0)
                    wds.append(0.0)
        else:
            pks.append(reference_pk(hyp_mass, ref_mass))
            wds.append(reference_wd(hyp_mass, ref_mass))
            
    flat_p = [v for doc in predictions for v in doc]
    flat_r = [v for doc in references for v in doc]
    return {
        "pk": float(np.mean(pks)) if pks else 1.0,
        "wd": float(np.mean(wds)) if wds else 1.0,
        "precision": float(precision_score(flat_r, flat_p, zero_division=0)),
        "recall": float(recall_score(flat_r, flat_p, zero_division=0)),
        "f1": float(f1_score(flat_r, flat_p, zero_division=0)),
    }


def scan_threshold(
    probs: list[list[float]], references: list[list[int]], grid=None
) -> tuple[float, dict[str, float]]:
    """在驗證集掃描 threshold（§8）：以 F1 選出最佳，回傳 (threshold, 該點全部指標)。"""
    grid = grid or [round(0.1 * i, 1) for i in range(1, 10)]
    best_t, best = None, None
    for t in grid:
        preds = [[int(p > t) for p in doc] for doc in probs]
        res = pk_wd_spokennlp(preds, references)
        if best is None or res["f1"] > best["f1"]:
            best_t, best = t, res
    return best_t, best
