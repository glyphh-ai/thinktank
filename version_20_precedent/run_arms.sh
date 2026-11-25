#!/bin/bash
# Train and score each arm in turn: same examples, same steps, same seed.
# Run: bash run_arms.sh [seed] [arms...]
cd "$(dirname "$0")"
PY=../version_19_glyph/.venv/bin/python
MODEL=mlx-community/Qwen3-1.7B-4bit
SEED=${1:-0}
shift
ARMS=${*:-move decision after}
for arm in $ARMS; do
  tag="${arm}_17b_s${SEED}"
  $PY -m mlx_lm lora --model "$MODEL" --train --data "data/$arm" \
    --mask-prompt --num-layers 16 --batch-size 8 --iters 1000 --learning-rate 1e-4 \
    --steps-per-report 50 --steps-per-eval 250 --val-batches 10 --max-seq-length 768 \
    --seed "$SEED" --adapter-path "adapters/$tag" > "train_$tag.log" 2>&1
  $PY eval.py --arm "$tag" --adapter "adapters/$tag" --model "$MODEL"
done
