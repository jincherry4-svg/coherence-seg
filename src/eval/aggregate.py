"""多 seed 結果彙整與 paired bootstrap（規格書 §8、Prompt 6）。

輸入：results/ 下每個 run 一個 json：{"config": "m2_reorder", "seed": 42,
"per_doc_pk": [...], "pk": ..., "wd": ..., "f1": ...}
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def load_runs(pattern: str):
    groups = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        groups[r["config"]].append(r)
    return groups


def summarize(groups) -> str:
    lines = ["| config | Pk | WD | F1 | seeds |", "|---|---|---|---|---|"]
    for name, runs in groups.items():
        def ms(key):
            v = np.array([r[key] for r in runs])
            return f"{v.mean() * 100:.2f} ± {v.std() * 100:.2f}"
        lines.append(f"| {name} | {ms('pk')} | {ms('wd')} | {ms('f1')} | {len(runs)} |")
    return "\n".join(lines)


def paired_bootstrap(a_docs: list[float], b_docs: list[float], n_boot=10000, seed=0) -> float:
    """回傳 p-value：H0 = 兩組逐篇 Pk 無差異。a/b 為同一測試集的逐篇 Pk。"""
    rng = np.random.default_rng(seed)
    a, b = np.array(a_docs), np.array(b_docs)
    assert len(a) == len(b)
    observed = a.mean() - b.mean()
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(min((diffs >= 0).mean(), (diffs <= 0).mean()) * 2) if observed != 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/*.json")
    ap.add_argument("--out", default="results/ABLATION.md")
    args = ap.parse_args()
    groups = load_runs(args.pattern)
    table = summarize(groups)
    with open(args.out, "w") as f:
        f.write("# Ablation 總表\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
