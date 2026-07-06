"""規格書 §10 測試 4、6、7、10 與 losses 基本行為。"""

import numpy as np
import pytest
import torch

from src.data.collate import collate_corruption_outputs
from src.data.corruption import corrupt_document
from src.eval.metrics import (
    HAS_SEGEVAL, labels_to_mass, pk_wd_spokennlp, reference_pk, reference_wd, scan_threshold,
)
from src.losses import (
    binary_focal_loss_with_logits, ema_decay_schedule, matching_accuracy,
    matching_cross_entropy, sigmoid_rampup,
)
from src.models.ema import EmaTeacher
from src.models.heads import MatchingHead
from tests.conftest import make_doc


# ---------- 測試 4：global attention mask ----------

def test_global_attention_positions(ids, rng):
    sents, labels = make_doc(rng, 20)
    outs = [corrupt_document(sents, labels, ids, rng, p=0.2) for _ in range(3)]
    batch = collate_corruption_outputs(outs, ids, need_clean=True)
    for b, o in enumerate(outs):
        expected = set(o.sent_anchor_idx) | set(o.cand_idx)
        expected.add(o.cand_idx[0] - 1 if o.cand_idx else len(o.input_ids) - 1)  # </s>
        got = set(torch.nonzero(batch["global_attention_mask"][b]).flatten().tolist())
        assert got == expected
        # attention_mask 覆蓋實際長度
        assert batch["attention_mask"][b].sum().item() == len(o.input_ids)
        # clean 版錨點也在 global attention 內
        got_c = set(torch.nonzero(batch["clean_global_attention_mask"][b]).flatten().tolist())
        assert set(o.clean_sent_anchor_idx) <= got_c


# ---------- 測試 6：配對頭 ----------

def test_matching_head_and_ce(ids, rng):
    torch.manual_seed(0)
    B, L, d, M = 2, 40, 16, 3
    hidden = torch.randn(B, L, d)
    slot_idx = torch.tensor([[3, 10, 20], [5, 15, -1]])
    cand_idx = torch.tensor([[25, 30, 35], [25, 30, -1]])
    head = MatchingHead(d)
    scores = head(hidden, slot_idx, cand_idx)
    assert scores.shape == (B, M, M)
    # padding 候選的分數必為極小值
    assert torch.all(scores[1, :, 2] <= torch.finfo(scores.dtype).min / 2 + 1)
    labels = torch.tensor([[1, 0, 2], [0, 1, -100]])
    loss = matching_cross_entropy(scores, labels)
    assert torch.isfinite(loss)
    acc, n = matching_accuracy(scores, labels)
    assert n == 5 and 0.0 <= acc <= 1.0


# ---------- 測試 7：EMA ----------

def test_ema_update_matches_manual():
    torch.manual_seed(1)
    student = torch.nn.Linear(4, 4)
    teacher = EmaTeacher(student, max_decay=0.999)
    w0 = teacher.module.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1.0)
    d1 = ema_decay_schedule(0)
    teacher.update(student, step=0)
    expect = w0 * d1 + student.weight.detach() * (1 - d1)
    assert torch.allclose(teacher.module.weight, expect, atol=1e-6)
    with torch.no_grad():
        student.weight.add_(-0.5)
    d2 = ema_decay_schedule(1)
    expect = expect * d2 + student.weight.detach() * (1 - d2)
    teacher.update(student, step=1)
    assert torch.allclose(teacher.module.weight, expect, atol=1e-6)
    assert all(not p.requires_grad for p in teacher.module.parameters())


# ---------- 測試 10：指標交叉驗證 ----------

@pytest.mark.skipif(not HAS_SEGEVAL, reason="segeval 未安裝")
def test_metrics_cross_validation(rng):
    for _ in range(20):
        n = int(rng.integers(10, 80))
        ref = rng.integers(0, 2, size=n).tolist(); ref[-1] = 1
        pred = rng.integers(0, 2, size=n).tolist(); pred[-1] = 1
        a = pk_wd_spokennlp([pred], [ref], use_segeval=True)
        b = pk_wd_spokennlp([pred], [ref], use_segeval=False)
        assert abs(a["pk"] - b["pk"]) < 1e-6, (a, b)
        assert abs(a["wd"] - b["wd"]) < 1e-6, (a, b)


def test_mass_conversion():
    assert labels_to_mass([1, 1, 0, 0, 1, 1]) == [1, 1, 3, 1]
    assert labels_to_mass([0, 0, 1]) == [3]
    assert labels_to_mass([0, 0, 0]) == [3]  # 殘尾自成一段


def test_perfect_prediction_gives_zero_pk():
    ref = [0, 1, 0, 0, 1, 0, 1]
    res = pk_wd_spokennlp([ref], [ref])
    assert res["pk"] == 0.0 and res["wd"] == 0.0 and res["f1"] == 1.0


def test_scan_threshold_picks_best():
    refs = [[0, 1, 0, 1]] * 4
    probs = [[0.2, 0.8, 0.3, 0.9]] * 4
    t, res = scan_threshold(probs, refs)
    assert res["f1"] == 1.0 and 0.3 <= t <= 0.8


# ---------- losses ----------

def test_focal_loss_masking_and_stability():
    logits = torch.tensor([[10.0, -10.0, 0.0]])
    targets = torch.tensor([[1, 0, -100]])
    mask = torch.tensor([[1, 1, 0]])
    loss = binary_focal_loss_with_logits(logits, targets, mask)
    assert torch.isfinite(loss) and loss.item() < 0.01  # 全對且被 mask 的不計
    # 全 mask 不得 NaN
    loss0 = binary_focal_loss_with_logits(logits, targets, torch.zeros_like(mask))
    assert loss0.item() == 0.0


def test_rampup_monotone():
    vals = [sigmoid_rampup(s, 100, 0.5) for s in range(0, 120, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[0] < 0.01 and abs(vals[-1] - 0.5) < 1e-9
