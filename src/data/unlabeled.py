"""無標註 Wikipedia 資料載入（規格書 §3.2）。

jsonl 由 scripts/prepare_unlabeled.py 產生，格式與 WikiSection 相同但無 labels。
"""

from __future__ import annotations

from .wikisection import DocExample, load_jsonl


def load_unlabeled(path: str, max_docs=None) -> list[DocExample]:
    return load_jsonl(path, labeled=False, max_docs=max_docs)
