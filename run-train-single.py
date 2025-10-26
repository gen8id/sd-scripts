#!/usr/bin/env python3
"""
SDXL LoRA 단일 학습 스크립트 (고급 사용자용)
- 특정 폴더만 선택 학습
- 세밀한 파라미터 조정 가능
- Config 오버라이드
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path


def get_vram_size(gpu_id=0):
    """NVIDIA GPU VRAM 크기 감지 (GB)"""
    try:
        cmd = f"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_id}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        vram_mb = int(result.stdout.strip())
        return vram_mb // 1024
    except:
        return 24  # 기본값


def count_images(folder_path):
    """폴더 내 이미지 개수 세기"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    count = 0
    for file in os.listdir(folder_path):
        if Path(file).suffix.lower() in extensions:
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
  python train_single.py --folder ../dataset/training/01_alice

  # 수동 파라미터 지정
  python train_single.py --folder ../dataset/training/01_alice --epochs 20 --repeats 30

  # Learning rate 조정
  python train_single.py --folder ../dataset/training/01_alice --lr 0.0002

  # Network dim 변경
  python train_single.py --folder ../dataset/training/01_alice --dim 64 --alpha 32

  # 이어서 학습 (Resume)
  python train_single.py --folder ../dataset/training/01_alice --resume ../output_models/alice-epoch-010.safetensors

  # Resume + 파라미터 변경
  python train_single.py --folder ../dataset/training/01_alice --resume ../output_models/alice-epoch-010.safetensors --epochs 30 --lr 0.00008

  # 전체 커스텀
  python train_single.py \\
    --folder ../dataset/training/01_alice \\
    --output alice_v2 \\
    --config config-24g.json \\
    --gpu 0 \\
    --epochs 25 \\
    --repeats 40 \\
    --lr 0.00015 \\
    --dim 64 \\
    --alpha 32 \\
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
        default="config-24g.json",
        help="Config 파일 (기본: config-24g.json)"
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
        if args.config == "config-24g.json":
            config_file = "config-24g.json"
    else:
        precision = "fp16"
        config_file = "config-16g.json"
        print(f"⚠️ VRAM {vram_size}GB < 20GB, fp16 모드로 전환")

    # Config 로드
    if not os.path.exists(config_file):
        print(f"❌ Config 파일 없음: {config_file}")
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    batch_size = args.batch_size or config['training'].get('batch_size', 1)

    # 출력 이름 결정
    if args.output:
        output_name = args.output
    else:
        # 폴더명에서 추출 (01_alice → alice)
        folder_name = folder_path.name
        parts = folder_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            output_name = parts[1]
        else:
            output_name = folder_name

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
    print(f"📁 Folder:         {folder_path}")
    print(f"💾 Output:         {output_name}.safetensors")
    print(f"📋 Config:         {config_file}")
    print(f"🖥️  GPU:            {args.gpu} ({vram_size}GB VRAM)")
    print(f"⚡ Precision:      {precision}")

    # Resume 정보
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ Resume 파일을 찾을 수 없습니다: {args.resume}")
            sys.exit(1)
        print(f"🔄 Resume from:    {args.resume}")

    print(f"{'-' * 70}")
    print(f"📊 Training Parameters")
    print(f"{'-' * 70}")
    print(f"  Images:          {image_count}")
    print(f"  Repeats:         {repeats}" + (" (manual)" if args.repeats else " (auto)"))
    print(f"  Epochs:          {epochs}" + (" (manual)" if args.epochs else " (auto)"))
    print(f"  Batch size:      {batch_size}" + (" (override)" if args.batch_size else ""))
    print(f"  Images/epoch:    {image_count * repeats}")
    print(f"  Steps/epoch:     {steps_per_epoch}")
    print(f"  Total steps:     {total_steps}")

    # 오버라이드된 파라미터 표시
    overrides = []
    if args.lr:
        print(f"  Learning rate:   {args.lr} (override)")
        overrides.append(('lr', args.lr))
    if args.dim:
        print(f"  Network dim:     {args.dim} (override)")
        overrides.append(('dim', args.dim))
    if args.alpha:
        print(f"  Network alpha:   {args.alpha} (override)")
        overrides.append(('alpha', args.alpha))
    if args.resolution:
        print(f"  Resolution:      {args.resolution} (override)")
        overrides.append(('resolution', args.resolution))
    if args.optimizer:
        print(f"  Optimizer:       {args.optimizer} (override)")
        overrides.append(('optimizer', args.optimizer))
    if args.scheduler:
        print(f"  LR Scheduler:    {args.scheduler} (override)")
        overrides.append(('scheduler', args.scheduler))
    if args.save_every:
        print(f"  Save every:      {args.save_every} epochs (override)")
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
        f"--dataset_repeats={repeats}",
        f"--mixed_precision={precision}"
    ]

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
    if args.optimizer:
        cmd.append(f"--optimizer_type={args.optimizer}")
    if args.scheduler:
        cmd.append(f"--lr_scheduler={args.scheduler}")
    if args.save_every:
        cmd.append(f"--save_every_n_epochs={args.save_every}")

    # Resume 추가
    if args.resume:
        cmd.append(f"--network_weights={args.resume}")
        print(f"\n🔄 Resuming from: {args.resume}\n")

    # 환경 변수 설정
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 실행
    try:
        print(f"\n🚀 Starting training...\n")
        subprocess.run(cmd, env=env, check=True)
        print(f"\n✅ 학습 완료: {output_name}.safetensors")
        print(f"{'=' * 70}\n")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 실패: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ 학습 중단됨")
        sys.exit(1)


if __name__ == "__main__":
    main()