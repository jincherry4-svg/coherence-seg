#!/bin/bash
# 取得 SpokenNLP 前處理後的 WikiSection 資料（規格書 §3.1）。
# 完全沿用官方 repo 的資料處理流程以確保與 baseline 可比。
set -e
DATA_DIR=${1:-/content/data}
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/SpokenNLP" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/alibaba-damo-academy/SpokenNLP.git "$DATA_DIR/SpokenNLP"
  cd "$DATA_DIR/SpokenNLP" && git sparse-checkout set emnlp2023-topic_segmentation
fi
echo "請依 $DATA_DIR/SpokenNLP/emnlp2023-topic_segmentation/README.md 下載原始 WikiSection"
echo "並執行其 run_process_data.sh，將輸出的 train/dev/test.jsonl 放到 $DATA_DIR/wiki_section_disease/"
