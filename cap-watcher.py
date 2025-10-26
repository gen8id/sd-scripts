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
        print(f"[□] Image file detected: {image_path}")
        cnt = generate_caption(Path(image_path))
        if cnt == 0:
            print(f"[S] Skipped existing: {image_path}")

    except Exception as e:
        print(f"X 처리 실패 ({image_path.name}): {e}")
        traceback.print_exc()


def watch_folders(folders):
    """여러 폴더를 감시하는 루프 (1단계 하위 폴더, 수정 감지)"""
    processed = {}  # path: mtime
    print(f"[👀] Watching folders: {', '.join(folders)}")

    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    while True:
        for folder in folders:
            if not os.path.isdir(folder):
                continue

            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                try:
                    # 파일인 경우
                    if os.path.isfile(item_path):
                        if item.lower().endswith(image_extensions):
                            mtime = os.path.getmtime(item_path)
                            if item_path not in processed or processed[item_path] != mtime:
                                print(f"[🔍] {'New' if item_path not in processed else 'Modified'}: {item}")
                                process_image(item_path)
                                processed[item_path] = mtime

                    # 폴더인 경우 (1단계만)
                    elif os.path.isdir(item_path):
                        for f in os.listdir(item_path):
                            file_path = os.path.join(item_path, f)
                            if os.path.isfile(file_path) and f.lower().endswith(image_extensions):
                                mtime = os.path.getmtime(file_path)
                                if file_path not in processed or processed[file_path] != mtime:
                                    status = 'New' if file_path not in processed else 'Modified'
                                    print(f"[🔍] {status}: [{item}] {f}")
                                    process_image(file_path)
                                    processed[file_path] = mtime

                except Exception as e:
                    print(f"[❌] Error: {item_path} - {e}")

        time.sleep(3)


def main():

    parser = argparse.ArgumentParser(description="BLIP + WD14 하이브리드 캡션 생성")
    # nargs = "+",
    parser.add_argument(
        "--dirs",
        default="../dataset/captioning",
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
        config.WATCH_DIR = args.dirs
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

    watch_folders(config.WATCH_DIRS)


if __name__ == "__main__":
    main()
