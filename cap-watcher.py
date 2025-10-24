import time
import argparse
import os
import traceback
from pathlib import Path
from caption_utils import generate_caption, Config, load_models


def process_image(image_path):
    """새 이미지 감지 시 BLIP 캡션 생성"""
    try:
        image_path = Path(image_path).as_posix()
        print(f"[🖼️] New image detected: {image_path}")
        caption = generate_caption(Path(image_path))
        if not caption:
            print(f"[❌] Caption failed: {image_path}")
            return

    except Exception as e:
        print(f"❌ 처리 실패 ({image_path.name}): {e}")
        traceback.print_exc()


def watch_folder(folder):
    """폴더 감시 루프"""
    processed = set()
    print(f"[👀] Watching folder: {folder}")
    while True:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) and path not in processed:
                process_image(path)
                processed.add(path)
        time.sleep(3)


def main():

    parser = argparse.ArgumentParser(description="BLIP + WD14 하이브리드 캡션 생성")
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="처리할 디렉토리 목록 (기본: config에서 설정)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 캡션 덮어쓰기"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="디바이스 (cuda/cpu)"
    )
    parser.add_argument(
        "--char",
        type=str,
        default="",
        help="모든 캡션 앞에 붙일 캐릭터명 (예: 'sakura_char')"
    )

    args = parser.parse_args()

    # Config 설정
    config = Config()
    if args.dirs:
        config.DATASET_DIRS = args.dirs
    config.OVERWRITE_EXISTING = args.overwrite
    config.CHARACTER_PREFIX = args.char
    config.DEVICE = args.device

    print("=" * 60)
    print("🎨 BLIP + WD14 하이브리드 캡션 생성기")
    print("=" * 60)
    print(f"📁 대상 디렉토리: {config.DATASET_DIRS}")
    print(f"💾 덮어쓰기: {config.OVERWRITE_EXISTING}")
    print(f"🖥️ 디바이스: {config.DEVICE}")
    if config.CHARACTER_PREFIX:
        print(f"🏷️  캐릭터 프리픽스: '{config.CHARACTER_PREFIX}'")
    print("=" * 60)
    # 모델 로드
    print("\n🔄 모델 로딩 중...")
    load_models(config)

    watch_folder(config.WATCH_DIR)


if __name__ == "__main__":
    main()
