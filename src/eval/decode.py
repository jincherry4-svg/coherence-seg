"""一對一配對解碼（規格書 §5.3，僅供評估/分析，不參與訓練）。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_decode(scores: np.ndarray) -> np.ndarray:
    """對 (M, M) 分數矩陣做一對一最大化指派，回傳每個槽位的候選索引。"""
    row, col = linear_sum_assignment(-scores)
    out = np.zeros(scores.shape[0], dtype=int)
    out[row] = col
    return out
