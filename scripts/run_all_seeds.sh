#!/bin/bash
# 多 seed 批次執行（規格書 §8）。已完成的 seed（存在 done 標記）自動跳過。
# 用法：bash scripts/run_all_seeds.sh configs/m2_reorder.yaml [configs/lab_1080ti.yaml]
set -e
CONFIGS=("$@")
# NAME = 里程碑 config + 資料集後綴（若疊加了 configs/data_*.yaml），與
# src/train.py 的 exp_tag 定義完全對齊（例：m2_reorder configs/data_city.yaml
# → NAME=m2_reorder+city）。舊行為把設備 config 也串進去，導致 DONE 標記寫到
# m0_baseline-lab_1080ti/ 之類的目錄，而實際 checkpoint 卻在 m0_baseline/ ——
# 兩棵目錄樹分家，跳過/重跑判斷全錯（見 DEBUG_HANDOFF_REPORT.md 殘留問題、
# IMPLEMENTATION_SPEC.md §12 陷阱 16）。資料集後綴則確保不同資料集（disease/city）
# 的 DONE 標記各自獨立、不互相撞名或誤判完成。
NAME=$(basename "${CONFIGS[0]}" .yaml)
for c in "${CONFIGS[@]}"; do
  base=$(basename "$c" .yaml)
  case "$base" in
    data_*) NAME="${NAME}+${base#data_}" ;;
  esac
done
MARK_DIR=${CKPT_DIR:-/content/drive/MyDrive/LongformerSC/coherence-seg/checkpoints}/$NAME
for SEED in 42 43 44 45 46; do
  MARK="$MARK_DIR/seed$SEED/DONE"
  if [ -f "$MARK" ]; then echo "seed $SEED 已完成，跳過"; continue; fi
  python -m src.train --config "${CONFIGS[@]}" --seed "$SEED" --resume
  mkdir -p "$(dirname "$MARK")" && touch "$MARK"
done
