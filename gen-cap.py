"""
BLIP + WD14 하이브리드 캡션 생성기 (수정 버전)
실사 LoRA 학습을 위한 통합 캡션 생성 스크립트

필요 환경: kohya_ss (sd-scripts)
"""
import argparse
import traceback
from pathlib import Path
from caption_utils import get_image_files, generate_caption, Config, load_models
from tqdm import tqdm


# ==============================
# 🚀 메인 프로세스
# ==============================

def process_directory(directory):
    """단일 디렉토리 처리"""
    
    print(f"\n📁 처리 중: {directory}")
    
    # 이미지 파일 찾기
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {directory}")
        return 0
    
    print(f"□ {len(image_files)}개 이미지 발견")
    
    succ_cnt = 0
    totl_cnt = image_files

    for image_path in tqdm(image_files, desc="캡션 생성"):

        try:
            cnt = generate_caption(image_path)
            if cnt == 0:
                print(f"[S] Caption skipped: {image_path}")
            else:
                succ_cnt += 1
            
        except Exception as e:
            print(f"X 처리 실패 ({image_path.name}): {e}")
            traceback.print_exc()
            continue
    
    print(f"V 완료: {succ_cnt}개 생성, {(totl_cnt - succ_cnt)}개 스킵")
    return succ_cnt


# ==============================
# 🎯 메인 함수
# ==============================

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
    
    # 각 디렉토리 처리
    total_success = 0
    
    for directory in config.DATASET_DIRS:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"⚠️ 디렉토리 없음: {directory}")
            continue
        
        count = process_directory(directory)
        total_success += count
    
    # 완료 메시지
    print("\n" + "=" * 60)
    print(f"🎉 전체 완료!")
    print(f"✅ 총 {total_success}개 캡션 생성됨")
    print("=" * 60)
    
    # 결과 예시 출력
    print("\n📝 생성 예시:")
    for directory in config.DATASET_DIRS:
        txt_files = list(Path(directory).glob("*.txt"))
        if txt_files:
            example_file = txt_files[0]
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"\n{example_file.name}:")
            print(f"  {content}")
            break


if __name__ == "__main__":
    main()