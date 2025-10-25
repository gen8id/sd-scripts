#!/bin/bash

DEFAULT_CONFIG="config-24g.json"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}
GPU_ID=${2:-0}
FORCE_REPEATS=${3:-""}

VRAM_SIZE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU_ID | awk '{print int($1/1024)}')

if [ $VRAM_SIZE -ge 20 ]; then
    PRECISION="bf16"
else
    CONFIG_FILE="config-16g.json"
    PRECISION="fp16"
fi

PARAMS=$(python3 << EOF
import json
import os
import re
from pathlib import Path

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

train_dir = config['folders']['train_data_dir']
batch_size = config['training'].get('batch_size', 1)
force_repeats = $FORCE_REPEATS if "$FORCE_REPEATS" else None

image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# 전체 이미지 수
total_raw_images = 0

for folder_name in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # 이미지 개수
    image_count = sum(1 for file in os.listdir(folder_path)
                     if Path(file).suffix.lower() in image_extensions)
    total_raw_images += image_count

if total_raw_images == 0:
    print("ERROR")
    exit(1)

# 목표 steps
vram = $VRAM_SIZE
target_total_steps = 1800 if vram >= 20 else 1500

# 반복 횟수 결정
if force_repeats is not None:
    optimal_repeats = force_repeats
else:
    # 이미지 수에 따른 자동 계산
    if total_raw_images < 20:
        optimal_repeats = max(80, min(200, target_total_steps // (total_raw_images * 10)))
    elif total_raw_images < 50:
        optimal_repeats = max(30, min(80, target_total_steps // (total_raw_images * 10)))
    elif total_raw_images < 100:
        optimal_repeats = max(15, min(30, target_total_steps // (total_raw_images * 10)))
    else:
        optimal_repeats = max(5, min(20, target_total_steps // (total_raw_images * 10)))

# 실제 계산
images_per_epoch = total_raw_images * optimal_repeats
steps_per_epoch = images_per_epoch // batch_size
actual_epochs = max(1, round(target_total_steps / steps_per_epoch))
actual_epochs = min(max(actual_epochs, 5), 30)
actual_total_steps = actual_epochs * steps_per_epoch

print(f"{total_raw_images}:{optimal_repeats}:{actual_epochs}:{steps_per_epoch}:{actual_total_steps}")
EOF
)

if [[ $PARAMS == ERROR* ]]; then
    echo "Error: No images found!"
    exit 1
fi

IFS=':' read -r IMAGE_COUNT REPEATS EPOCHS STEPS_PER_EPOCH TOTAL_STEPS <<< "$PARAMS"

echo "==================================="
echo "Auto-calculated Training Config"
echo "==================================="
echo "GPU: $GPU_ID (${VRAM_SIZE}GB VRAM)"
echo "Precision: $PRECISION"
echo "Config: $CONFIG_FILE"
echo "Raw images: $IMAGE_COUNT"
echo "Repeats: $REPEATS $([ ! -z "$FORCE_REPEATS" ] && echo "(forced)" || echo "(auto)")"
echo "Images per epoch: $((IMAGE_COUNT * REPEATS))"
echo "Steps per epoch: $STEPS_PER_EPOCH"
echo "Epochs: $EPOCHS"
echo "Total steps: $TOTAL_STEPS"
echo "==================================="

export CUDA_VISIBLE_DEVICES="$GPU_ID"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision $PRECISION \
  sdxl_train_network.py \
  --config_file="$CONFIG_FILE" \
  --max_train_epochs=$EPOCHS \
  --dataset_repeats=$REPEATS \
  --mixed_precision="$PRECISION"