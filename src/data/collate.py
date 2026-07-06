"""動態 collate（規格書 §4.2）與 curriculum 共享狀態。

挖空發生在 collate 階段（CPU、每個 epoch 不同），故 curriculum 的當前
p / fixed_m 透過 multiprocessing 共享記憶體傳遞，num_workers > 0（fork）
時 worker 端也能讀到主行程的最新值（§12 陷阱 7 的 rng 派生見 worker_init_fn）。
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .corruption import CorruptionOutput, SpecialIds, corrupt_document


class CurriculumController:
    """挖空難度的共享狀態機介面（規格書 §6 表格）。

    stage 0: fixed_m ∈ {2,3}；stage 1: p=0.15；stage 2: p=0.25。
    升級判斷由 LightningModule 執行（依重組準確率移動平均），這裡只存狀態。
    """

    P_BY_STAGE = (0.0, 0.15, 0.25)

    def __init__(self, enabled: bool = True):
        self.enabled = mp.Value("i", 1 if enabled else 0)
        self.stage = mp.Value("i", 0)

    def advance(self) -> int:
        with self.stage.get_lock():
            self.stage.value = min(self.stage.value + 1, 2)
            return self.stage.value

    def sample_params(self, rng: np.random.Generator) -> tuple[float, Optional[int]]:
        """回傳 (p, fixed_m)。stage 0 隨機挖 2 或 3 句。"""
        if not self.enabled.value:
            return 0.0, 0  # 關閉挖空（M0 / 驗證集）
        s = self.stage.value
        if s == 0:
            return 0.0, int(rng.integers(2, 4))
        return self.P_BY_STAGE[s], None

    # checkpoint 續訓（§9.3）
    def state_dict(self) -> dict:
        return {"enabled": int(self.enabled.value), "stage": int(self.stage.value)}

    def load_state_dict(self, d: dict) -> None:
        self.enabled.value = int(d["enabled"])
        self.stage.value = int(d["stage"])


def worker_init_fn(worker_id: int) -> None:
    """為每個 dataloader worker 派生獨立 numpy rng（§12 陷阱 7）。"""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.set_rng(np.random.default_rng(info.seed % (2**32)))


def _pad_2d(rows: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max((len(r) for r in rows), default=1)
    width = max(width, 1)
    out = torch.full((len(rows), width), pad_value, dtype=torch.long)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def _attention_and_global(
    input_rows: list[list[int]], anchor_rows: list[list[int]], extra_rows: list[list[int]], pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """回傳 (input_ids, attention_mask, global_attention_mask)。

    global attention = 1 的位置：所有句子錨點（<s> / [SLOT]）、[CAND]、</s>（§4.3；
    序列首 token 即第 0 句錨點，已涵蓋）。
    """
    input_ids = _pad_2d(input_rows, pad)
    attention_mask = (input_ids != pad).long()
    # 注意：pad id 可能與真實 token 重疊的風險由 attention 長度另行處理
    lengths = [len(r) for r in input_rows]
    attention_mask = torch.zeros_like(input_ids)
    for i, L in enumerate(lengths):
        attention_mask[i, :L] = 1
    global_mask = torch.zeros_like(input_ids)
    for i, (anchors, extras) in enumerate(zip(anchor_rows, extra_rows)):
        for pos in anchors:
            global_mask[i, pos] = 1
        for pos in extras:
            global_mask[i, pos] = 1
    return input_ids, attention_mask, global_mask


def collate_corruption_outputs(
    outs: list[CorruptionOutput], ids: SpecialIds, need_clean: bool = False
) -> dict[str, torch.Tensor]:
    """把一個 batch 的 CorruptionOutput 組成張量 dict（規格書 §4.2 欄位）。"""
    sep_positions = []  # </s> 分隔符位置（文件結尾），納入 global attention
    for o in outs:
        doc_end = o.cand_idx[0] - 1 if o.cand_idx else len(o.input_ids) - 1
        sep_positions.append([doc_end])

    input_ids, attention_mask, global_mask = _attention_and_global(
        [o.input_ids for o in outs],
        [o.sent_anchor_idx for o in outs],
        [o.cand_idx + sep for o, sep in zip(outs, sep_positions)],
        ids.pad,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_mask,
        "sent_anchor_idx": _pad_2d([o.sent_anchor_idx for o in outs], -1),
        "slot_idx": _pad_2d([o.slot_idx for o in outs], -1),
        "cand_idx": _pad_2d([o.cand_idx for o in outs], -1),
        "match_labels": _pad_2d([o.match_labels for o in outs], -100),
        "seg_labels": _pad_2d([o.seg_labels for o in outs], -100),
        "seg_mask": _pad_2d([o.seg_mask for o in outs], 0),
        "is_slot": _pad_2d([[int(b) for b in o.is_slot] for o in outs], 0),
        "student_to_clean_sent_map": _pad_2d([o.student_to_clean_sent_map for o in outs], -1),
    }
    if need_clean:
        c_ids, c_attn, c_glob = _attention_and_global(
            [o.clean_input_ids for o in outs],
            [o.clean_sent_anchor_idx for o in outs],
            [[len(o.clean_input_ids) - 1] for o in outs],  # 尾端 </s>
            ids.pad,
        )
        batch.update(
            clean_input_ids=c_ids,
            clean_attention_mask=c_attn,
            clean_global_attention_mask=c_glob,
            clean_sent_anchor_idx=_pad_2d([o.clean_sent_anchor_idx for o in outs], -1),
        )
    return batch


class SegmentationCollator:
    """供 DataLoader 使用的 collate 物件。dataset 端負責挖空，這裡只組張量。"""

    def __init__(self, ids: SpecialIds, need_clean: bool = False):
        self.ids = ids
        self.need_clean = need_clean

    def __call__(self, outs: list[Optional[CorruptionOutput]]) -> dict[str, torch.Tensor]:
        outs = [o for o in outs if o is not None]
        if not outs:  # 整個 batch 都超長被丟棄：回傳空 batch 由訓練端跳過
            return {}
        return collate_corruption_outputs(outs, self.ids, self.need_clean)
