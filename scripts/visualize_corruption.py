"""人工檢查挖空輸出（規格書 §11 M1 驗收）。離線可跑（用 FakeTokenizer 或 HF tokenizer）。

用法：python scripts/visualize_corruption.py [--data path/to/train.jsonl]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from src.data.corruption import SpecialIds, corrupt_document  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--n_docs", type=int, default=3)
    args = ap.parse_args()

    if args.data:
        from transformers import AutoTokenizer
        from src.data.wikisection import build_special_ids, load_jsonl

        tok = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        tok.add_special_tokens({"additional_special_tokens": ["[SLOT]", "[CAND]"]})
        ids = build_special_ids(tok)
        docs = load_jsonl(args.data, max_docs=args.n_docs)
        get_sents = lambda d: [tok.encode(s, add_special_tokens=False) for s in d.sentences]
        show = lambda d, i: d.sentences[i][:60]
    else:  # 離線示範模式
        from tests.conftest import FakeTokenizer, make_doc

        ids = SpecialIds(bos=0, eos=2, slot=90, cand=91, pad=1)
        rng0 = np.random.default_rng(0)

        class D:  # 假文件
            def __init__(self, n):
                self.toks, self.labels = make_doc(rng0, n)
                self.sentences = [f"sent_{i}(len={len(t)})" for i, t in enumerate(self.toks)]

        docs = [D(12), D(20), D(8)]
        get_sents = lambda d: d.toks
        show = lambda d, i: d.sentences[i]

    rng = np.random.default_rng(42)
    for k, doc in enumerate(docs):
        out = corrupt_document(get_sents(doc), getattr(doc, "labels", None), ids, rng, p=0.2)
        print(f"\n===== 文件 {k}（{out.n_sentences} 句，挖 {out.m_slots} 句）=====")
        for i in range(out.n_sentences):
            tag = "[SLOT]" if out.is_slot[i] else f"label={out.seg_labels[i]}"
            print(f"  句 {i:>3} {tag:>10} | {show(doc, i)}")
        print(f"  match_labels（槽位 j → 候選 k）: {out.match_labels}")


if __name__ == "__main__":
    main()
