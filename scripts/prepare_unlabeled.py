"""下載無標註 Wikipedia 並切句過濾（規格書 §3.2）。

用法：python scripts/prepare_unlabeled.py --n 50000 --seed 42 --out /content/data/unlabeled/wiki_unlabeled_50k.jsonl
"""

import argparse
import json
import os

import pysbd
from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/unlabeled/wiki_unlabeled_50k.jsonl")
    ap.add_argument("--min_sents", type=int, default=20)
    ap.add_argument("--max_sents", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seg = pysbd.Segmenter(language="en", clean=False)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)

    kept = 0
    with open(args.out, "w") as f, tqdm(total=args.n) as bar:
        for ex in ds:
            if kept >= args.n:
                break
            sents = [s.strip() for s in seg.segment(ex["text"]) if s.strip()]
            if not (args.min_sents <= len(sents) <= args.max_sents):
                continue
            # 粗略長度過濾（詞數近似 token 數的下界；精確 4096 檢查由 corruption 端負責）
            if sum(len(s.split()) for s in sents) > 3000:
                continue
            f.write(json.dumps({"example_id": ex["id"], "sentences": sents}) + "\n")
            kept += 1
            bar.update(1)
    print(f"wrote {kept} docs to {args.out}")


if __name__ == "__main__":
    main()
