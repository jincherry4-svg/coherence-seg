"""任務頭（規格書 §5.2、§5.3）。不依賴 transformers，方便離線測試。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def gather_positions(hidden: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """依 token 位置索引抽取表徵。

    Args:
        hidden: (B, L, d)；idx: (B, K)，padding 為 -1。
    Returns:
        (B, K, d)；padding 位置的內容無意義，由呼叫端以標籤 mask 忽略。
    """
    safe = idx.clamp(min=0)
    return hidden.gather(1, safe.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))


class BoundaryHead(nn.Module):
    """邊界分類頭：Linear → GELU → Dropout → Linear(1)（§5.2）。"""

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sent_repr: torch.Tensor) -> torch.Tensor:
        """(B, S, d) → (B, S) 邊界 logits。"""
        return self.net(sent_repr).squeeze(-1)


class MatchingHead(nn.Module):
    """槽位–候選雙線性配對頭（§5.3）：S = (H_slot W) H_cand^T / sqrt(d)。"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.bilinear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(
        self, hidden: torch.Tensor, slot_idx: torch.Tensor, cand_idx: torch.Tensor
    ) -> torch.Tensor:
        """(B, L, d) + 索引 → (B, M, M) 配對分數；padding 槽位由標籤 -100 忽略。"""
        slot_h = gather_positions(hidden, slot_idx)  # (B, M, d)
        cand_h = gather_positions(hidden, cand_idx)  # (B, M, d)
        scores = torch.einsum("bmd,bnd->bmn", self.bilinear(slot_h), cand_h) * self.scale
        # 把 padding 候選（idx == -1）的分數壓到 -inf，避免搶走 softmax 機率
        cand_pad = (cand_idx == -1).unsqueeze(1)  # (B, 1, M)
        return scores.masked_fill(cand_pad, torch.finfo(scores.dtype).min)
