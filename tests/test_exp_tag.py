"""測試 src/train.py 的 compute_exp_tag：里程碑 + 資料集後綴（以 cfg.data.name 為準）。

需要 pytorch_lightning（train.py 頂層 import），與 test_modules.py 同要求
真實訓練環境（Colab）執行。
"""

from src.train import compute_exp_tag


def test_milestone_only_default_disease():
    assert compute_exp_tag(["configs/m2_reorder.yaml"], "disease") == "m2_reorder"


def test_device_config_does_not_affect_exp_tag():
    assert compute_exp_tag(
        ["configs/m2_reorder.yaml", "configs/lab_1080ti.yaml"], "disease"
    ) == "m2_reorder"


def test_city_appends_suffix():
    assert compute_exp_tag(
        ["configs/m2_reorder.yaml", "configs/data_city.yaml"], "city"
    ) == "m2_reorder+city"


def test_explicit_disease_config_keeps_default_naming():
    assert compute_exp_tag(
        ["configs/m0_baseline.yaml", "configs/data_disease.yaml"], "disease"
    ) == "m0_baseline"


def test_suffix_follows_data_name_not_filename():
    assert compute_exp_tag(
        ["configs/m0_baseline.yaml", "configs/my_custom_override.yaml"], "city"
    ) == "m0_baseline+city"
