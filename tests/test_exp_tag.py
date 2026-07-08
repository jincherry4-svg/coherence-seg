"""測試 src/train.py 的 compute_exp_tag：里程碑 + 資料集後綴的命名規則。

需要 pytorch_lightning（train.py 頂層 import），與 test_modules.py 同要求
真實訓練環境（Colab）執行，不在無 torch 的離線沙箱跑。
"""

from src.train import compute_exp_tag


def test_milestone_only():
    assert compute_exp_tag(["configs/m2_reorder.yaml"]) == "m2_reorder"


def test_device_config_does_not_affect_exp_tag():
    """設備 config 不得改變實驗身分，否則換設備續訓會找不到 checkpoint。"""
    assert compute_exp_tag(
        ["configs/m2_reorder.yaml", "configs/lab_1080ti.yaml"]
    ) == "m2_reorder"


def test_data_city_appends_suffix():
    assert compute_exp_tag(
        ["configs/m2_reorder.yaml", "configs/data_city.yaml"]
    ) == "m2_reorder+city"


def test_data_suffix_order_independent_of_device_position():
    """資料集 config 不論排在設備 config 前後，結果一致。"""
    a = compute_exp_tag(["configs/m0_baseline.yaml", "configs/data_city.yaml",
                         "configs/lab_1080ti.yaml"])
    b = compute_exp_tag(["configs/m0_baseline.yaml", "configs/lab_1080ti.yaml",
                         "configs/data_city.yaml"])
    assert a == b == "m0_baseline+city"


def test_no_data_config_matches_default_disease_naming():
    """不疊加 data_*.yaml 時沿用舊命名（無後綴），與既有 disease 結果相容。"""
    assert compute_exp_tag(["configs/m0_baseline.yaml"]) == "m0_baseline"
