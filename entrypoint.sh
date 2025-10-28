#!/bin/bash
# ===============================================
# Bitflow LoRA Trainer Entrypoint Script
# ===============================================

# ----- Color setup (safe for TTY only) -----
if [ -t 1 ]; then
  RED="\033[1;31m"
  GREEN="\033[1;32m"
  YELLOW="\033[1;33m"
  BLUE="\033[1;34m"
  MAGENTA="\033[1;35m"
  CYAN="\033[1;36m"
  RESET="\033[0m"
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""; RESET=""
fi

# ----- Logging helpers -----
log_time() {
  date +"%Y-%m-%d %H:%M:%S"
}

log_info() {
  echo -e "[$(log_time)] ${BLUE}[INFO]${RESET} $*"
}

log_warn() {
  echo -e "[$(log_time)] ${YELLOW}[WARN]${RESET} $*"
}

log_error() {
  echo -e "[$(log_time)] ${RED}[ERROR]${RESET} $*"
}

log_ok() {
  echo -e "[$(log_time)] ${GREEN}[OK]${RESET} $*"
}

# ----- Notice -----
echo -e "${CYAN}───────────────────────────────────────────────${RESET}"
echo -e "${YELLOW} ⚠️  이 프로그램은 Bitflow, Inc.의 자산이며, 재배포는 허용되지 않습니다.${RESET}"
echo -e "${YELLOW} 💬 구매 또는 사용 관련 문의는 admin@bitflow.ai 로 연락 부탁드립니다.${RESET}"
echo -e "${CYAN}───────────────────────────────────────────────${RESET}"
echo ""

# ----- Directory setup -----
log_info "필요한 디렉토리를 생성 중입니다..."
mkdir -p /app/sdxl_train_captioner/dataset/captioning/background
mkdir -p /app/sdxl_train_captioner/dataset/captioning/mainchar
mkdir -p /app/sdxl_train_captioner/dataset/training/background
mkdir -p /app/sdxl_train_captioner/dataset/training/mainchar
mkdir -p /app/sdxl_train_captioner/output_models
mkdir -p /app/sdxl_train_captioner/logs
log_ok "디렉토리 구성이 완료되었습니다."

# ----- Model download -----
MODEL_PATH="/app/sdxl_train_captioner/models/sd_xl_base_1.0.safetensors"
MODEL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

if [ ! -f "$MODEL_PATH" ]; then
    log_warn "모델 파일이 존재하지 않습니다. 다운로드를 시작합니다..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
    if [ $? -eq 0 ]; then
        log_ok "모델 다운로드가 완료되었습니다."
    else
        log_error "모델 다운로드 중 오류가 발생했습니다!"
        exit 1
    fi
else
    log_info "모델 파일이 이미 존재합니다. 다운로드를 건너뜁니다."
fi

# ----- Start the watcher -----
log_info "cap-watcher.py 실행 중..."
python cap-watcher.py

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  log_ok "프로세스가 정상적으로 종료되었습니다."
else
  log_error "cap-watcher.py 종료 코드: $EXIT_CODE"
fi
