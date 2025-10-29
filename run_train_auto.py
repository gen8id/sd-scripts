#!/usr/bin/env python3
"""
SDXL LoRA 일괄 학습 스크립트
- 학습 폴더 하위의 여러 캐릭터/개념을 자동으로 개별 LoRA 학습
- VRAM에 따른 자동 설정 (bf16/fp16)
- 이미지 수에 따른 최적 파라미터 자동 계산
"""

import os
import sys
import toml
import math # 상단에 import math 추가 필요
import subprocess
import argparse
from pathlib import Path


class TrainingConfig:
    """학습 설정 관리"""

    def __init__(self, config_file, gpu_id=0, force_repeats=None):
        self.config_file = config_file
        self.gpu_id = gpu_id
        self.force_repeats = force_repeats

        # VRAM 감지
        self.vram_size = self.get_vram_size()

        # VRAM에 따른 설정
        if self.vram_size >= 20:
            self.config_file = "config-24g.toml"
            self.precision = "bf16"  # 항상 bf16
            self.target_steps = 1800
        else:
            self.config_file = "config-16g.toml"
            self.precision = "fp16"
            self.target_steps = 1500

        print(f"🧠 VRAM 감지 결과: {self.vram_size}GB / Precision={self.precision}")

        # Config 파일 로드
        self.load_config()


    def get_vram_size(self):
        """NVIDIA GPU VRAM 크기 감지 (GB)"""
        try:
            # 명령어를 리스트로 그대로 전달 (shell=False)
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i", str(self.gpu_id)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            vram_mb = int(result.stdout.strip().split("\n")[0])
            vram_gb = vram_mb // 1024
            return vram_gb

        except Exception as e:
            print(f"⚠️ VRAM 감지 실패 ({e}) — 기본값(24GB, bf16) 사용")
            self.precision = "bf16"
            return 24

    def load_config(self):
        """config.toml 로드"""
        if not os.path.exists(self.config_file):
            print(f"❌ Config 파일 없음: {self.config_file}")
            sys.exit(1)

        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = toml.load(f)

        self.train_dir = self.config['folders']['train_data_dir']
        self.output_dir = self.config['folders']['output_dir']
        self.batch_size = self.config['training'].get('batch_size', 1)


class LoRATrainer:
    """단일 LoRA 학습 실행"""

    def __init__(self, training_config):
        self.config = training_config
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def find_training_folders(self):
        """학습 폴더 찾기 (순서_이름_클래스 패턴) - 다중 카테고리 지원"""
        train_dir = self.config.train_dir

        if not os.path.isdir(train_dir):
            print(f"❌ 학습 디렉토리 없음: {train_dir}")
            return []

        folders = []

        # mainchar와 background 폴더 탐색
        category_folders = ['mainchar', 'background']

        for category in category_folders:
            category_path = os.path.join(train_dir, category)

            if not os.path.isdir(category_path):
                print(f"⚠️  카테고리 폴더 없음: {category_path}")
                continue

            # 각 카테고리 내부의 학습 폴더 탐색
            for item in os.listdir(category_path):
                item_path = os.path.join(category_path, item)
                if not os.path.isdir(item_path):
                    continue

                # 패턴: 01_alice_woman, 02_bob, 5_style
                parts = item.split('_')  # 언더스코어(_)로 모두 분리

                # 최소 2개 요소(순서, 이름)가 있어야 하며, 첫 번째가 숫자여야 함
                if len(parts) >= 2 and parts[0].isdigit():
                    order = int(parts[0])
                    # 이름 추출: 두 번째 요소부터 끝까지를 다시 조인
                    # 예: 'alice_woman'
                    name_parts = parts[1:]

                    # 📌 세 번째 요소 (클래스) 처리
                    class_word = None
                    if len(name_parts) >= 2:
                        # 이름과 클래스를 분리
                        # 이름: parts[1] (예: alice)
                        # 클래스: parts[2] (예: woman)
                        name_token = name_parts[0]
                        class_word = name_parts[1]
                    else:
                        # 세 번째 요소가 없으면, 전체를 이름 토큰으로 사용 (예: alice)
                        name_token = name_parts[0]

                    folders.append({
                        'order': order,
                        'name': name_token,  # 순수 ID (예: alice)
                        'class': class_word,  # 클래스 (예: woman) - 없으면 None
                        'path': item_path,
                        'folder': item,
                        'category': category
                    })

        if not folders:
            print(f"❌ 학습 폴더를 찾을 수 없습니다!")
            print(f"   경로: {train_dir}")
            print(f"   찾는 위치: {train_dir}/mainchar/, {train_dir}/background/")
            print(f"   패턴: N_ID_CLASS 또는 N_ID")
            return []

        # 순서대로 정렬
        folders.sort(key=lambda x: x['order'])

        print(f"✅ 발견된 학습 폴더: {len(folders)}개")
        for f in folders:
            class_display = f['class'] if f['class'] else '(None)'
            print(f"   [{f['category']}] {f['order']:02d}_{f['name']}_({class_display})")

        return folders

    def count_images(self, folder_path):
        """폴더 내 이미지 개수 세기"""
        count = 0
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in self.image_extensions:
                count += 1
        return count

    def calculate_training_params(self, image_count):
        """
        이미지 수, 배치 사이즈, 목표 스텝을 기반으로
        최적의 repeats와 epochs를 계산합니다. (24GB+ VRAM 환경)
        """

        # --- 1. 설정값 가져오기 (config에서 8, 3000, 10을 가져온다고 가정) ---
        batch_size = self.config.batch_size
        target_steps = self.config.target_steps
        force_repeats = getattr(self.config, 'force_repeats', None)
        target_epochs = getattr(self.config, 'target_epochs', 15)  # 15로 고정하여 체크포인트 수 제어

        if image_count <= 0 or batch_size <= 0:
            # 안전장치
            return {"repeats": 1, "epochs": 1, "steps_per_epoch": 1, "total_steps": 1}

        # --- 2. '자동 계산'의 경우 (추천 로직: Epochs를 고정하고 Repeats 역산) ---
        if force_repeats is None:

            epochs = target_epochs  # Epochs 고정 (예: 10)

            # 1. 목표 스텝 달성에 필요한 'repeats'를 역산하여 계산
            # repeats = (target_steps * batch_size) / (image_count * epochs)

            denominator = image_count * epochs
            repeats_float = (target_steps * batch_size) / denominator
            # repeats는 정수이며 최소 1 (반올림 사용)
            repeats = max(1, round(repeats_float))

            # 2. 결정된 repeats로 steps_per_epoch 다시 계산 (올림 사용으로 0 스텝 방지)
            steps_per_epoch = math.ceil((image_count * repeats) / batch_size)

        else:
            # --- 3. '강제 repeats'가 설정된 경우 (기존 로직) ---
            repeats = force_repeats
            steps_per_epoch = math.ceil((image_count * repeats) / batch_size)
            epochs = max(1, round(target_steps / steps_per_epoch))
            epochs = max(epochs, 5)  # 최소 5 epoch 보장

        # --- 4. 최종 결과 계산 ---
        total_steps = steps_per_epoch * epochs

        return {
            "repeats": repeats,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps
        }

    def train_single_lora(self, folder_info):
        """단일 LoRA 학습"""
        name = folder_info['name']  # ID 토큰 (예: alice)
        class_word = folder_info['class']  # Class 토큰 (예: woman)
        folder = folder_info['folder']
        folder_path = folder_info['path']
        category = folder_info['category']

        # 이미지 개수 계산
        num_images = self.count_images(folder_path)
        if num_images == 0:
            print(f"❌ {name}: 이미지 없음")
            return False

        # 학습 파라미터 자동 계산
        params = self.calculate_training_params(num_images)
        repeats = params['repeats']
        epochs = params['epochs']

        print(f"\n{'=' * 70}")
        print(f"🎯 Training LoRA: {name}")
        print(f"{'=' * 70}")
        print(f"📊 Training Configuration")
        print(f"{'-' * 70}")
        print(f"  Category:        {category}")
        print(f"  Folder:          {folder}")
        if class_word:
            print(f"  Class Token:     {class_word} (정규화 사용)")
        else:
            print(f"  Class Token:     (없음) (정규화 미사용)")
        print(f"  Images:          {num_images}")
        print(f"  Repeats:         {repeats} (auto)")
        print(f"  Epochs:          {epochs}")
        print(f"  Total steps:     {params['total_steps']}")  # 수정: params['total_steps'] 사용
        print(f"{'-' * 70}")

        # train_data_dir는 카테고리 폴더 (01_alic3_woman의 부모)
        train_data_dir = os.path.join(self.config.train_dir, category)
        output_name = folder_info.get('output_name', folder)

        cmd = [
            'accelerate', 'launch',
            '--num_cpu_threads_per_process', '1',
            '--mixed_precision', self.config.precision,
            'sdxl_train_network.py',
            f'--config_file={self.config.config_file}',
            f'--train_data_dir={train_data_dir}',  # 카테고리 폴더
            f'--output_name={output_name}',  # ID 토큰만 사용
            f'--max_train_epochs={epochs}',
            f'--dataset_repeats={repeats}'
            f'-logging_dir ../logs/tensorboard'
            f'-report_to tensorboard'
        ]

        # Resume 처리
        if hasattr(self.config, 'resume') and self.config.resume:
            # cmd.append(f"--network_weights={self.config.resume}")
            cmd.append(f"--resume={self.config.resume}")
            print(f"   🔄 Loading weights: {Path(self.config.resume).name}")


        # 📌 세 번째 요소(Class)가 있을 경우, --class_tokens 인자 추가
        # if class_word:
        #     # Kohya_SS의 Dreambooth/LoRA는 Class 토큰을 --class_tokens에 전달하여
        #     # 정규화 이미지 폴더(예: reg_woman)를 찾고 학습을 수행합니다.
        #     cmd.append(f'--class_tokens={class_word}')

        # 실행
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_id)

            print(f"🚀 Starting training...\n")
            print(f"📂 Train dir: {train_data_dir}")
            result = subprocess.run(cmd, env=env, check=True)

            print(f"\n✅ {name} 학습 완료!")
            print(f"{'=' * 70}\n")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ {name} 학습 실패: {e}")
            print(f"{'=' * 70}\n")
            return False
        except KeyboardInterrupt:
            print(f"\n⚠️ 사용자에 의해 중단됨")
            return False

    def run_batch_training(self):
        """일괄 학습 실행"""
        folders = self.find_training_folders()

        if not folders:
            print("❌ 학습 폴더를 찾을 수 없습니다!")
            print(f"   경로: {self.config.train_dir}")
            print(f"   패턴: 01_name, 02_name, ...")
            return

        print(f"\n{'=' * 70}")
        print(f"🚀 SDXL LoRA Batch Training")
        print(f"{'=' * 70}")
        print(f"📁 학습 폴더: {self.config.train_dir}")
        print(f"💾 출력 폴더: {self.config.output_dir}")
        print(f"🖥️ GPU: {self.config.gpu_id} ({self.config.vram_size}GB)")
        print(f"⚡ Precision: {self.config.precision}")
        print(f"📋 Config: {self.config.config_file}")
        print(f"\n발견된 학습 폴더: {len(folders)}개")
        print(f"{'-' * 70}")
        for f in folders:
            img_count = self.count_images(f['path'])
            print(f"  {f['folder']} ({img_count} images)")
        print(f"{'=' * 70}\n")

        # 사용자 확인
        try:
            response = input("학습을 시작하시겠습니까? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ 학습 취소됨")
                return
        except KeyboardInterrupt:
            print("\n❌ 학습 취소됨")
            return

        # 학습 실행
        results = []
        for i, folder in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}] Processing: {folder['name']}...")
            success = self.train_single_lora(folder)
            results.append({
                'name': folder['name'],
                'success': success
            })

            # 실패 시 계속 진행할지 물어봄
            if not success:
                try:
                    response = input("❓ 계속 진행하시겠습니까? (Y/n): ")
                    if response.lower() in ['n', 'no']:
                        print("⚠️ 나머지 학습 건너뜀")
                        break
                except KeyboardInterrupt:
                    print("\n⚠️ 나머지 학습 건너뜀")
                    break

        # 결과 요약
        print(f"\n{'=' * 70}")
        print(f"📊 Training Summary")
        print(f"{'=' * 70}")
        success_count = sum(1 for r in results if r['success'])
        fail_count = len(results) - success_count

        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"{status} {r['name']}")

        print(f"{'-' * 70}")
        print(f"✅ 성공: {success_count}/{len(results)}")
        if fail_count > 0:
            print(f"❌ 실패: {fail_count}/{len(results)}")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SDXL LoRA 일괄 학습 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python train_batch.py
  python train_batch.py config-16g.toml
  python train_batch.py config-24g.toml 0 15

폴더 구조:
  training/
  ├── 01_alice/
  │   └── *.jpg
  ├── 02_bob/
  │   └── *.jpg
  └── 03_background/
      └── *.jpg
        """
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="config-24g.toml",
        help="Config 파일 (기본: config-24g.toml)"
    )

    parser.add_argument(
        "gpu_id",
        nargs="?",
        type=int,
        default=0,
        help="GPU ID (기본: 0)"
    )

    parser.add_argument(
        "repeats",
        nargs="?",
        type=int,
        default=None,
        help="강제 반복 횟수 (기본: 자동 계산)"
    )

    args = parser.parse_args()

    try:
        # 설정 로드
        training_config = TrainingConfig(
            config_file=args.config,
            gpu_id=args.gpu_id,
            force_repeats=args.repeats
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