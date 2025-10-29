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
        epilog="""
사용 예시:
  # 기본 (자동 계산)
  python run_train_single.py --folder ../dataset/training/01_alice

  # 수동 파라미터 지정
  python run_train_single.py --folder ../dataset/training/01_alice --epochs 20 --repeats 30

  # Learning rate 조정
  python run_train_single.py --folder ../dataset/training/01_alice --lr 0.0002

  # Network dim 변경
  python run_train_single.py --folder ../dataset/training/01_alice --dim 64 --alpha 32

  # 이어서 학습 (Resume)
  python run_train_single.py --folder ../dataset/training/01_alice --resume ../output_models/alice-epoch-010.safetensors --epochs 30

  # 전체 커스텀
  python run_train_single.py \
    --folder ../dataset/training/01_alice \
    --output alice_v2 \
    --config config-24g.toml \
    --gpu 0 \
    --epochs 25 \
    --repeats 40 \
    --lr 0.00015 \
    --dim 64 \
    --alpha 32 \
    --batch-size 2
        """
    )

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

    args = parser.parse_args()

    # 폴더 확인
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        sys.exit(1)

    # 이미지 개수
    image_count = count_images(folder_path)
    if image_count == 0:
        print(f"❌ 이미지가 없습니다: {folder_path}")
        sys.exit(1)

    # VRAM 감지
    vram_size = get_vram_size(args.gpu)

    # Config 자동 선택
    if vram_size >= 20:
        precision = "bf16"
        config_file = args.config if args.config else "config-24g.toml"
    else:
        precision = "fp16"
        config_file = args.config if args.config else "config-16g.toml"
        print(f"⚠️ VRAM {vram_size}GB < 20GB, fp16 모드로 전환")

    # Config 로드
    if not os.path.exists(config_file):
        print(f"❌ Config 파일 없음: {config_file}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    batch_size = args.batch_size or config['training'].get('batch_size', 1)

    # 출력 이름 결정
    if args.output:
        output_name = args.output
    else:
        # 폴더명에서 추출 (01_alice_woman → alice)
        folder_name = folder_path.name
        parts = folder_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            base_name = parts[1]
        else:
            base_name = folder_name

        # 클래스 접미사 제거 (_woman, _man 등)
        base_name = re.sub(r'_[a-zA-Z0-9]+$', '', base_name)
        output_name = base_name

    # 📌 Resume 시 출력 이름 재결정 로직 보완
    # 요청 명령어에서는 output_name이 따로 지정되지 않아, 폴더명에서 추출된 이름이 사용됩니다.
    # 만약 --resume에 있는 파일명 기반으로 output_name을 지정하고 싶다면 아래 로직을 사용해야 하지만,
    # 현재는 폴더명 기반으로 진행합니다.

    # 파라미터 결정
    if args.no_auto:
        # 수동 모드
        if not args.epochs or not args.repeats:
            print("❌ --no-auto 사용 시 --epochs와 --repeats 필수입니다")
            sys.exit(1)
        epochs = args.epochs
        repeats = args.repeats
        steps_per_epoch = (image_count * repeats) // batch_size
        total_steps = epochs * steps_per_epoch
    else:
        # 자동 계산 (오버라이드 가능)
        auto_params = calculate_auto_params(image_count, vram_size, batch_size)
        epochs = args.epochs or auto_params['epochs']
        repeats = args.repeats or auto_params['repeats']
        steps_per_epoch = (image_count * repeats) // batch_size
        total_steps = epochs * steps_per_epoch

    # 학습 정보 출력
    print(f"\n{'=' * 70}")
    print(f"🎯 SDXL LoRA Training - Single Mode")
    print(f"{'=' * 70}")
    print(f"📁 Folder:           {folder_path}")
    print(f"💾 Output:           {output_name}.safetensors")
    print(f"📋 Config:           {config_file}")
    print(f"🖥️  GPU:            {args.gpu} ({vram_size}GB VRAM)")
    print(f"⚡ Precision:        {precision}")

    # Resume 정보
    if args.resume:
        # resume 파일이 존재하지 않을 경우를 대비하여 절대 경로로 변환하여 확인
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            # 상대 경로인 경우 현재 스크립트 실행 위치 기준으로 처리
            resume_path = Path(os.getcwd()) / resume_path

        if not resume_path.exists():
            print(f"❌ Resume 파일을 찾을 수 없습니다: {resume_path}")
            sys.exit(1)
        print(f"🔄 Resume from:      {resume_path}")

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

    # 사용자 확인
    try:
        response = input("학습을 시작하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 학습 취소됨")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ 학습 취소됨")
        sys.exit(0)

    # accelerate 명령어 구성
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", precision,
        "sdxl_train_network.py",
        f"--config_file={config_file}",
        f"--train_data_dir={folder_path}",
        f"--output_name={output_name}",
        f"--max_train_epochs={epochs}",
        f"--dataset_repeats={repeats}"
    ]

    # resume(이어받기) 처리 (중복된 로직 제거하고, 필요한 부분만 남김)
    if args.resume:
        resume_path = str(Path(args.resume).resolve())
        if os.path.exists(resume_path):
            # LoRA 가중치 로드: Kohya_SS에서 학습 재개 시 일반적으로 사용
            cmd.append(f"--network_weights={resume_path}")
            print(f"\n🔄 LoRA 가중치 로드 (--network_weights): {resume_path}\n")
        else:
            print(f"❌ Resume 파일이 존재하지 않습니다: {resume_path}")
            sys.exit(1)

    # 오버라이드 추가
    if args.batch_size:
        cmd.append(f"--train_batch_size={args.batch_size}")
    if args.lr:
        cmd.append(f"--learning_rate={args.lr}")
    if args.dim:
        cmd.append(f"--network_dim={args.dim}")
    if args.alpha:
        cmd.append(f"--network_alpha={args.alpha}")
    if args.resolution:
        cmd.append(f"--resolution={args.resolution}")

    try:
        training_config = TrainingConfig(
            config_file=args.config,
            gpu_id=args.gpu,            # argparse --gpu
            force_repeats=args.repeats, # argparse --repeats
            folder_path=folder_path,    # 학습할 폴더
            output_name=output_name,    # 저장할 LoRA 이름
            epochs=epochs,
            batch_size=batch_size,
            lr=args.lr,
            dim=args.dim,
            alpha=args.alpha,
            resolution=args.resolution,
            resume=args.resume
        )
        # 학습 실행
        trainer = LoRATrainer(training_config)
        trainer.run_batch_training()

    except KeyboardInterrupt:
        print("\n\n⚠️ 프로그램 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()