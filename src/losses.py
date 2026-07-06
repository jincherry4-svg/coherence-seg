"""損失函數與排程（規格書 §5.2、§5.3、§6）。全部為純 torch，可離線測試。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """二元 focal loss（§5.2）。以 logsigmoid 實作確保 bf16 數值穩定（§12 陷阱 8）。

    Args:
        logits/targets/mask: 同形狀 (B, S)。mask=0 的位置不計入；targets 在
        mask=0 處可為任意值（含 -100）。
    """
    t = targets.clamp(min=0).float()  # -100 → 0，反正會被 mask 掉
    log_p = F.logsigmoid(logits)
    log_1p = F.logsigmoid(-logits)
    p = torch.exp(log_p)
    loss_pos = -alpha * (1.0 - p).pow(gamma) * log_p
    loss_neg = -(1.0 - alpha) * p.pow(gamma) * log_1p
    loss = t * loss_pos + (1.0 - t) * loss_neg
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return (loss * m).sum() / denom


def bce_with_pos_weight(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: float = 8.0
) -> torch.Tensor:
    """備選分割 loss：帶正類權重的 BCE（config 可切換）。"""
    t = targets.clamp(min=0).float()
    loss = F.binary_cross_entropy_with_logits(
        logits, t, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def matching_cross_entropy(scores: torch.Tensor, match_labels: torch.Tensor) -> torch.Tensor:
    """逐槽位配對 CE（§5.3）。scores: (B, M, M)，match_labels: (B, M)，-100 忽略。"""
    B, M, _ = scores.shape
    return F.cross_entropy(scores.reshape(B * M, M), match_labels.reshape(B * M), ignore_index=-100)


def matching_accuracy(scores: torch.Tensor, match_labels: torch.Tensor) -> tuple[float, int]:
    """逐槽位 argmax 命中率。回傳 (accuracy, 有效槽位數)；無有效槽位回傳 (0.0, 0)。"""
    valid = match_labels != -100
    n = int(valid.sum().item())
    if n == 0:
        return 0.0, 0
    pred = scores.argmax(dim=-1)
    correct = ((pred == match_labels) & valid).sum().item()
    return correct / n, n


def sigmoid_rampup(step: int, ramp_steps: int, w_max: float) -> float:
    """Tarvainen & Valpola 的 sigmoid ramp-up（§6）。"""
    if ramp_steps <= 0:
        return w_max
    t = min(max(step, 0), ramp_steps) / ramp_steps
    return w_max * math.exp(-5.0 * (1.0 - t) ** 2)


def consistency_mse(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """師生邊界機率的 MSE（§6）。只在 valid_mask=1（非槽位真實句）計算。"""
    m = valid_mask.float()
    diff = (student_probs - teacher_probs) ** 2
    return (diff * m).sum() / m.sum().clamp(min=1.0)


def ema_decay_schedule(step: int, max_decay: float = 0.999) -> float:
    """EMA 教師 decay（§5.4）：min(max_decay, (1+step)/(10+step))。"""
    return min(max_decay, (1 + step) / (10 + step))
