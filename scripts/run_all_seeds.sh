#!/bin/bash
# 多 seed 批次執行（規格書 §8）。已完成的 seed（存在 done 標記）自動跳過。
# 用法：bash scripts/run_all_seeds.sh configs/m2_reorder.yaml [configs/lab_1080ti.yaml]
set -e
CONFIGS=("$@")
NAME=$(IFS=-; parts=(); for c in "${CONFIGS[@]}"; do parts+=("$(basename "$c" .yaml)"); done; echo "${parts[*]}")
MARK_DIR=${CKPT_DIR:-/content/drive/MyDrive/coherence-seg/checkpoints}/$NAME
for SEED in 42 43 44 45 46; do
  MARK="$MARK_DIR/seed$SEED/DONE"
  if [ -f "$MARK" ]; then echo "seed $SEED 已完成，跳過"; continue; fi
  python -m src.train --config "${CONFIGS[@]}" --seed "$SEED" --resume
  mkdir -p "$(dirname "$MARK")" && touch "$MARK"
done
