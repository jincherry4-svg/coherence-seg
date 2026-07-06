"""EMA 教師（規格書 §5.4）。輕量自製，不依賴外部套件。"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..losses import ema_decay_schedule


class EmaTeacher(nn.Module):
    """學生模型的指數移動平均複本。

    - deepcopy 初始化；恆為 eval() 且所有參數 requires_grad=False。
    - update(student, step) 於每個 optimizer step 後呼叫。
    - buffers（如 LayerNorm 統計以外的註冊 buffer）直接複製學生當前值。
    - 透過 nn.Module 繼承讓 Lightning checkpoint 自動保存教師參數（§9.3）。
    """

    def __init__(self, student: nn.Module, max_decay: float = 0.999):
        super().__init__()
        self.max_decay = max_decay
        self.module = copy.deepcopy(student)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module, step: int) -> float:
        d = ema_decay_schedule(step, self.max_decay)
        s_params = dict(student.named_parameters())
        for name, t_param in self.module.named_parameters():
            t_param.mul_(d).add_(s_params[name].detach(), alpha=1.0 - d)
        s_bufs = dict(student.named_buffers())
        for name, t_buf in self.module.named_buffers():
            t_buf.copy_(s_bufs[name])
        return d

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self.module.eval()  # 保證 dropout 關閉（§12 陷阱 9）
        return self.module(*args, **kwargs)
