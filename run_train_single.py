#!/usr/bin/env python3
"""
SDXL LoRA 단일 학습 스크립트 (고급 사용자용)
- 특정 폴더만 선택 학습
- 세밀한 파라미터 조정 가능
- Config 오버라이드
"""

import os
import sys
import logging
import re
import toml
import subprocess
import argparse
from pathlib import Path
from run_train_auto import TrainingConfig, LoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 터미널로
        logging.FileHandler("/app/sdxl_train_captioner/logs/train_single_debug.log")  # 파일로
    ]
)

def get_vram_size(gpu_id=0):
    """NVIDIA GPU VRAM 크기 감지 (GB)"""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
            "-i", str(gpu_id)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        vram_mb = int(result.stdout.strip())
        return vram_mb // 1024
    except:
        return 24  # 기본값


def count_images(folder_path):
    """폴더 내 이미지 개수 세기 (1단계 하위 폴더까지 포함)"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    count = 0

    folder_path = Path(folder_path)

    # 상위 폴더 안의 직접 이미지
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            count += 1

    # 바로 아래 1단계 하위 폴더 이미지
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            for file in subfolder.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    count += 1

    return count


def calculate_auto_params(image_count, vram_size, batch_size=1):
    """이미지 수 기반 자동 파라미터 계산"""
    target_steps = 1800 if vram_size >= 20 else 1500

    # Repeats 계산
    if image_count < 20:
        repeats = max(80, min(200, target_steps // (image_count * 10)))
    elif image_count < 50:
        repeats = max(30, min(80, target_steps // (image_count * 10)))
    elif image_count < 100:
        repeats = max(15, min(30, target_steps // (image_count * 10)))
    else:
        repeats = max(5, min(20, target_steps // (image_count * 10)))

    # Epochs 계산
    images_per_epoch = image_count * repeats
    steps_per_epoch = images_per_epoch // batch_size
    epochs = max(1, round(target_steps / steps_per_epoch))
    epochs = min(max(epochs, 5), 30)

    return {
        'repeats': repeats,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': epochs * steps_per_epoch
    }


def main():

    parser = argparse.ArgumentParser(
        description="SDXL LoRA 단일 학습 (고급 설정)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""(생략: 사용 예시 동일)"""
    )

    # -- 인자 정의 (생략: 기존과 동일) --

    # 필수 인자
    parser.add_argument(
        "--folder",
        required=True,
        help="학습할 폴더 경로 (예: ../dataset/training/01_alice)"
    )

    # 기본 설정
    parser.add_argument(
        "--config",
        default="config-24g.toml",
        help="Config 파일 (기본: config-24g.toml)"
    )

    parser.add_argument(
        "--output",
        help="출력 LoRA 이름 (기본: 폴더명에서 추출)"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID (기본: 0)"
    )

    # 학습 파라미터
    parser.add_argument(
        "--epochs",
        type=int,
        help="총 Epoch 수 (기본: 자동 계산)"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        help="이미지 반복 횟수 (기본: 자동 계산)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="배치 사이즈 (기본: config 값)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (기본: config 값, 보통 1e-4)"
    )

    parser.add_argument(
        "--dim",
        type=int,
        help="Network dimension (기본: config 값, 보통 32)"
    )

    parser.add_argument(
        "--alpha",
        type=int,
        help="Network alpha (기본: config 값, 보통 16)"
    )

    parser.add_argument(
        "--resolution",
        help="해상도 (예: 1024,1024 또는 768,768)"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        help="N epoch마다 저장 (기본: config 값)"
    )

    # 고급 옵션
    parser.add_argument(
        "--optimizer",
        help="Optimizer (예: AdamW8bit, Lion, Prodigy)"
    )

    parser.add_argument(
        "--scheduler",
        help="LR Scheduler (예: cosine, constant, polynomial)"
    )

    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="자동 계산 비활성화 (epochs/repeats 수동 지정 필수)"
    )

    # Resume 옵션
    parser.add_argument(
        "--resume",
        help="이어서 학습할 LoRA 파일 경로 (예: ../output_models/alice-epoch-010.safetensors)"
    )
    parser.add_argument("--logging_dir", type=str, default="../logs/tensorboard",
                        help="TensorBoard 로그 저장 폴더")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="로그를 기록할 플랫폼 (tensorboard)")

    args = parser.parse_args()
    print("🧩 전달된 인자:", sys.argv)
    print("🧩 argparse 결과:", args)

    # ==========================================
    # 1. 기본 검증
    # ==========================================
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        sys.exit(1)

    image_count = count_images(folder_path)
    if image_count == 0:
        print(f"❌ 이미지가 없습니다: {folder_path}")
        sys.exit(1)

    # ==========================================
    # 2. Resume 파일 검증 (최우선 처리)
    # ==========================================
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            print(f"❌ Resume 파일이 존재하지 않습니다: {resume_path}")
            sys.exit(1)
        if resume_path.suffix.lower() != '.safetensors':
            print(f"⚠️  경고: Resume 파일이 .safetensors가 아닙니다: {resume_path}")

    # ==========================================
    # 3. VRAM 및 Config 설정
    # ==========================================
    vram_size = get_vram_size(args.gpu)

    if vram_size >= 20:
        precision = "bf16"
        config_file = args.config if args.config else "config-24g.toml"
    else:
        precision = "fp16"
        config_file = args.config if args.config else "config-16g.toml"
        print(f"⚠️ VRAM {vram_size}GB < 20GB, fp16 모드로 전환")

    if not os.path.exists(config_file):
        print(f"❌ Config 파일 없음: {config_file}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    batch_size = args.batch_size or config['training'].get('batch_size', 1)

    # ==========================================
    # 4. 출력 이름 결정
    # ==========================================
    if args.output:
        output_name = args.output
    else:
        folder_name = folder_path.name
        output_name = folder_name

    # ==========================================
    # 5. 학습 파라미터 결정
    # ==========================================
    if args.no_auto:
        if not args.epochs or not args.repeats:
            print("❌ --no-auto 사용 시 --epochs와 --repeats 필수입니다")
            sys.exit(1)
        epochs = args.epochs
        repeats = args.repeats
    else:
        auto_params = calculate_auto_params(image_count, vram_size, batch_size)
        epochs = args.epochs or auto_params['epochs']
        repeats = args.repeats or auto_params['repeats']

    steps_per_epoch = (image_count * repeats) // batch_size
    total_steps = epochs * steps_per_epoch

    # ==========================================
    # 6. 학습 정보 출력
    # ==========================================
    print(f"\n{'=' * 70}")
    print(f"🎯 SDXL LoRA Training - Single Mode")
    print(f"{'=' * 70}")
    print(f"📁 Folder:         {folder_path}")
    print(f"💾 Output:         {output_name}.safetensors")
    print(f"📋 Config:         {config_file}")
    print(f"🖥️  GPU:           {args.gpu} ({vram_size}GB VRAM)")
    print(f"⚡ Precision:       {precision}")

    if resume_path:
        print(f"🔄 Resume from:      {resume_path}")
        print(f"   (이어학습 모드 - 기존 가중치에서 계속)")

    print(f"{'-' * 70}")
    print(f"📊 Training Parameters")
    print(f"{'-' * 70}")
    print(f"  Images:            {image_count}")
    print(f"  Repeats:           {repeats}" + (" (manual)" if args.repeats else " (auto)"))
    print(f"  Epochs:            {epochs}" + (" (manual)" if args.epochs else " (auto)"))
    print(f"  Batch size:        {batch_size}" + (" (override)" if args.batch_size else ""))
    print(f"  Images/epoch:      {image_count * repeats}")
    print(f"  Steps/epoch:       {steps_per_epoch}")
    print(f"  Total steps:       {total_steps}")
    print(f"{'=' * 70}\n")

    # 오버라이드된 파라미터 표시 (이전과 동일)
    overrides = []
    if args.lr:
        print(f"  Learning rate:     {args.lr} (override)")
        overrides.append(('lr', args.lr))
    if args.dim:
        print(f"  Network dim:       {args.dim} (override)")
        overrides.append(('dim', args.dim))
    if args.alpha:
        print(f"  Network alpha:     {args.alpha} (override)")
        overrides.append(('alpha', args.alpha))
    if args.resolution:
        print(f"  Resolution:        {args.resolution} (override)")
        overrides.append(('resolution', args.resolution))
    if args.optimizer:
        print(f"  Optimizer:         {args.optimizer} (override)")
        overrides.append(('optimizer', args.optimizer))
    if args.scheduler:
        print(f"  LR Scheduler:      {args.scheduler} (override)")
        overrides.append(('scheduler', args.scheduler))
    if args.save_every:
        print(f"  Save every:        {args.save_every} epochs (override)")
        overrides.append(('save_every', args.save_every))

    print(f"{'=' * 70}\n")

    # ==========================================
    # 7. 사용자 확인
    # ==========================================
    try:
        response = input("학습을 시작하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 학습 취소됨")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ 학습 취소됨")
        sys.exit(0)

    # ==========================================
    # 8. TrainingConfig 생성
    # ==========================================
    try:
        training_config = TrainingConfig(
            config_file=config_file,
            gpu_id=args.gpu,
            force_repeats=args.repeats
        )

        # Resume 설정 추가
        if resume_path:
            training_config.resume = str(resume_path)

        # 추가 오버라이드 설정
        if args.batch_size:
            training_config.batch_size = args.batch_size
        if args.lr:
            training_config.learning_rate = args.lr
        if args.dim:
            training_config.network_dim = args.dim
        if args.alpha:
            training_config.network_alpha = args.alpha
        if args.resolution:
            training_config.resolution = args.resolution


    except Exception as e:
        print(f"❌ TrainingConfig 생성 실패: {e}")
        sys.exit(1)

    # ==========================================
    # 9. Trainer 생성 및 학습 실행
    # ==========================================
    trainer = LoRATrainer(training_config)

    # folder_info 생성
    folder_name = folder_path.name
    folder_split_len = len(folder_name.split('_'))
    parts = folder_name.split('_', 1)
    if len(parts) == 2 and parts[0].isdigit():
        order = int(parts[0])
    else:
        order = 0

    folder_info = {
        'order': order,
        'name': folder_name,
        'path': str(folder_path),
        'folder': folder_name,
        'category': folder_path.parent.name,
        'output_name': output_name,  # 출력 이름 전달
        'epochs': epochs,
        'repeats': repeats,
    }

    if folder_split_len == 3:
        folder_info['class'] = folder_name.split('_')[2]

    # 학습 실행
    print("\n🚀 학습 시작...\n")
    success = trainer.train_single_lora(folder_info)

    if not success:
        print("\n❌ 학습 실패")
        sys.exit(1)
    else:
        print(f"\n✅ 학습 완료: {output_name}.safetensors")
        if resume_path:
            print(f"   (기존: {resume_path.name} → 새로운: {output_name}.safetensors)")
        sys.exit(0)


if __name__ == "__main__":
    main()