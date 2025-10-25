#!/bin/bash
# 모델 다운로드
if [ ! -f /app/sdxl_train_captioner/models/sd_xl_base_1.0.safetensors ]; then
    wget -O /app/sdxl_train_captioner/models/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
fi
