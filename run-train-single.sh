#!/bin/bash
set -e  # 에러 발생 시 스크립트 종료

# 로그 디렉토리
LOG_DIR="/app/sdxl_train_captioner/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 로그 파일 경로
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

trap 'echo "❌ Error occurred. Check log: '"$LOG_FILE"'"' ERR

# 학습 스크립트 실행
echo "Starting training... Logging to $LOG_FILE"

# cd /app/sdxl_train_captioner/sd-scripts

python3 -u ./run_train_single.py "$@" 2>&1 | tee "$LOG_FILE"
