"""scan_threshold 的選擇準則測試：(Pk+WD)/2 最小、F1 tie-break。

背景：
1. F1 最佳門檻偏向高 recall（過度切分），Pk 對過切鈍感但 WD 重罰，曾造成
   Pk 正常（17.5）而 WD 爆表（35 vs 論文基準 ~20.6）的病態組合。
2. 改以「Pk 最小」為準則後暴露另一個問題：Pevzner & Hearst (2002) 指出 Pk
   對過切天生寬容，短文件視窗小時常見多個門檻 Pk 打平，此時選到「第一個
   遇到的最小 Pk」會挑中過切但 WD 差的門檻。
現行準則：(Pk+WD)/2 最小，F1 為 tie-break。只依賴 numpy/segeval，可離線執行。
"""

from src.eval.metrics import pk_wd_spokennlp, scan_threshold


def test_scan_threshold_avoids_oversegmentation_trap():
    # 低門檻（0.1）過度切分：Pk 打平但 WD 差；高門檻（0.3-0.7）完美預測。
    refs = [[0, 1, 0, 1]] * 4
    probs = [[0.2, 0.8, 0.3, 0.9]] * 4
    t, res = scan_threshold(probs, refs)
    assert res["pk"] == 0.0 and res["wd"] == 0.0 and res["f1"] == 1.0
    assert 0.3 <= t <= 0.7, f"門檻 {t} 過低：選到了 Pk 打平但 WD 差的過切門檻"


def test_wd_penalizes_oversegmentation_more_than_pk():
    ref = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]] * 3
    over = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]] * 3  # 過度切分
    res = pk_wd_spokennlp(over, ref)
    assert res["wd"] >= res["pk"]


def test_criterion_is_argmin_of_pk_wd_mean():
    """準則驗證：選出的門檻必須是全網格上 (Pk+WD)/2 的最小值。"""
    import numpy as np
    rng = np.random.default_rng(7)
    refs, probs = [], []
    for _ in range(6):
        n = 20
        ref = [1 if (i + 1) % 5 == 0 else 0 for i in range(n)]
        prob = [min(0.99, max(0.01, (0.75 if r == 1 else 0.25) + rng.normal(0, 0.15)))
                for r in ref]
        refs.append(ref)
        probs.append(prob)
    t, res = scan_threshold(probs, refs)
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    all_scores = []
    for g in grid:
        r = pk_wd_spokennlp([[int(p > g) for p in d] for d in probs], refs)
        all_scores.append((r["pk"] + r["wd"]) / 2)
    assert (res["pk"] + res["wd"]) / 2 == min(all_scores)
