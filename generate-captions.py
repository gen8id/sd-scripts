"""
BLIP + WD14 하이브리드 캡션 생성기 (수정 버전)
실사 LoRA 학습을 위한 통합 캡션 생성 스크립트

필요 환경: kohya_ss (sd-scripts)
"""

import os
import sys
import traceback
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import logging
import argparse
import nltk
from nltk.stem import WordNetLemmatizer

# sd-scripts 기준 상대 경로
nltk_models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "nltk_data")
os.makedirs(nltk_models_dir, exist_ok=True)

# NLTK 데이터 다운로드
try:
    nltk.data.path.append(nltk_models_dir)
    nltk.download('wordnet', download_dir=nltk_models_dir, quiet=True)
    nltk.download('omw-1.4', download_dir=nltk_models_dir, quiet=True)
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"⚠️ NLTK 초기화 경고: {e}")
    lemmatizer = None


# Kohya_ss 모듈 임포트
try:
    from finetune.wd14_tagger_gen8id import preprocess_image, run_batch
except ImportError:
    print("❌ Kohya_ss 환경에서 실행해주세요!")
    print("경로: sd-scripts/ 폴더 안에서 실행")
    traceback.print_exc()
    sys.exit(1)


# ==============================
# ⚙️ 설정 (수정 가능)
# ==============================

class Config:
    # 데이터셋 경로
    DATASET_DIRS = [
        "../dataset/mainchar",
        "../dataset/bg",
    ]
    
    # 모델 설정
    BLIP_MODEL_PATH = "Salesforce/blip-image-captioning-large"
    BLIP_CACHE_DIR = "../models/blip-image-captioning-large"
    WD14_MODEL_PATH = "SmilingWolf/wd-v1-4-moat-tagger-v2"
    WD14_CACHE_DIR = "../models/wd-v1-4-moat-tagger-v2"

    # WD14 임계값
    WD14_GENERAL_THRESHOLD = 0.35
    WD14_CHARACTER_THRESHOLD = 0.85
    
    # BLIP 설정
    BLIP_MAX_LENGTH = 75
    BLIP_NUM_BEAMS = 1
    
    # 제거할 WD14 메타 태그
    REMOVE_TAGS = [
        "1girl", "1boy", "solo", "looking at viewer",
        "simple background", "white background", "grey background",
        "highres", "absurdres", "lowres", "bad anatomy",
        "signature", "watermark", "artist name", "dated",
        "rating:safe", "rating:questionable", "rating:explicit",
    ]
    
    # 출력 설정
    OUTPUT_ENCODING = "utf-8"
    OVERWRITE_EXISTING = True
    CREATE_BACKUP = True
    
    # 디바이스
    DEVICE = "cuda"
    
    # 캐릭터 프리픽스 (CLI에서 설정)
    CHARACTER_PREFIX = ""


# ==============================
# 🔧 유틸리티 함수
# ==============================

def lemmatize_tags(tags_list):
    """태그를 기본형으로 변환"""
    if lemmatizer is None:
        return tags_list
    return [lemmatizer.lemmatize(tag.lower()) for tag in tags_list]


def normalize_tags(tags_str):
    """태그 정규화: 소문자 변환, 공백 정리, 중복 제거"""
    if not tags_str:
        return []
    tags = [tag.strip().lower() for tag in tags_str.split(',')]
    # 중복 제거 (순서 유지)
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    return unique_tags


def remove_unwanted_tags(tags_list, remove_list):
    """불필요한 태그 제거"""
    remove_set = set(tag.lower() for tag in remove_list)
    return [tag for tag in tags_list if tag not in remove_set]


def merge_captions(blip_caption, wd14_tags, remove_tags):
    """
    BLIP 캡션과 WD14 태그 병합
    
    형식: [BLIP 문장], [WD14 태그들]
    """
    # BLIP 정규화
    blip_normalized = blip_caption.strip().lower() if blip_caption else ""
    
    # WD14 태그 정규화 및 필터링
    wd14_normalized = normalize_tags(wd14_tags)
    wd14_lemmatized = lemmatize_tags(wd14_normalized)
    wd14_filtered = remove_unwanted_tags(wd14_lemmatized, remove_tags)
    
    # BLIP 문장의 단어들 추출 (중복 제거용)
    blip_words = set(blip_normalized.replace(',', ' ').split()) if blip_normalized else set()
    
    # WD14에서 BLIP에 이미 포함된 단어 제거
    wd14_deduped = []
    for tag in wd14_filtered:
        # 태그가 BLIP 문장에 포함되지 않으면 추가
        if not any(word in tag or tag in word for word in blip_words):
            wd14_deduped.append(tag)
    
    # 최종 병합: BLIP (문장) + WD14 (태그)
    if blip_normalized and wd14_deduped:
        merged = f"{blip_normalized}, {', '.join(wd14_deduped)}"
    elif blip_normalized:
        merged = blip_normalized
    elif wd14_deduped:
        merged = ', '.join(wd14_deduped)
    else:
        merged = ""
    
    return merged


# ==============================
# 🎨 캡션 생성 함수
# ==============================

def generate_blip_caption(image_path, model, processor, config):
    """BLIP으로 자연어 캡션 생성"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(image, return_tensors="pt").to(config.DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.BLIP_MAX_LENGTH,
                num_beams=config.BLIP_NUM_BEAMS,
            )
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
        
    except Exception as e:
        print(f"⚠️ BLIP 생성 실패 ({image_path.name}): {e}")
        return ""


def generate_wd14_tags(image_path, tagger, config):
    """WD14로 태그 생성"""
    try:
        # WD14Tagger.tag() 메서드 호출
        tags_str = tagger.tag(
            str(image_path),
            general_threshold=config.WD14_GENERAL_THRESHOLD,
            character_threshold=config.WD14_CHARACTER_THRESHOLD,
        )
        return tags_str if tags_str else ""
        
    except Exception as e:
        print(f"⚠️ WD14 생성 실패 ({image_path.name}): {e}")
        return ""


# ==============================
# 📁 파일 처리
# ==============================

def get_image_files(directory):
    """디렉토리에서 이미지 파일 찾기"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def create_backup(caption_path):
    """기존 캡션 파일 백업"""
    if caption_path.exists():
        backup_dir = caption_path.parent / "caption_backup"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / caption_path.name
        import shutil
        shutil.copy2(caption_path, backup_path)


# ==============================
# 🚀 메인 프로세스
# ==============================

def process_directory(directory, blip_model, blip_processor, wd14_tagger, config):
    """단일 디렉토리 처리"""
    
    print(f"\n📁 처리 중: {directory}")
    
    # 이미지 파일 찾기
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {directory}")
        return 0
    
    print(f"📸 {len(image_files)}개 이미지 발견")
    
    success_count = 0
    skip_count = 0
    
    for image_path in tqdm(image_files, desc="캡션 생성"):
        
        # 캡션 파일 경로
        caption_path = image_path.with_suffix('.txt')
        
        # 기존 파일 존재 확인
        if caption_path.exists() and not config.OVERWRITE_EXISTING:
            skip_count += 1
            continue
        
        # 백업 생성
        if config.CREATE_BACKUP and caption_path.exists():
            create_backup(caption_path)
        
        try:
            # 1. BLIP 캡션 생성
            blip_caption = generate_blip_caption(
                image_path, blip_model, blip_processor, config
            )
            
            # 2. WD14 태그 생성
            wd14_tags = generate_wd14_tags(image_path, wd14_tagger, config)
            
            # 3. 병합
            merged_caption = merge_captions(
                blip_caption, wd14_tags, config.REMOVE_TAGS
            )
            
            # 4. 캐릭터명 prefix 추가
            if config.CHARACTER_PREFIX:
                char_token = config.CHARACTER_PREFIX.strip()
                merged_caption = f"{char_token}, {merged_caption}"
            
            # 5. 저장
            if merged_caption:
                with open(caption_path, 'w', encoding=config.OUTPUT_ENCODING) as f:
                    f.write(merged_caption)
                success_count += 1
            else:
                print(f"⚠️ 빈 캡션: {image_path.name}")
            
        except Exception as e:
            print(f"❌ 처리 실패 ({image_path.name}): {e}")
            traceback.print_exc()
            continue
    
    print(f"✅ 완료: {success_count}개 생성, {skip_count}개 스킵")
    return success_count


# ==============================
# 🏷️ WD14 Tagger 클래스
# ==============================

class WD14Tagger:
    def __init__(
        self,
        config,
        model_dir=None,
        repo_id=None,
        onnx=True,
        general_threshold=0.35,
        character_threshold=0.85,
        device=None,
    ):
        self.config = config
        self.model_dir = model_dir or config.WD14_CACHE_DIR
        self.repo_id = repo_id or config.WD14_MODEL_PATH
        self.onnx = onnx
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.device = device
        
        # ✅ tag_freq 초기화
        self.tag_freq = {}
        
        # 모델 초기화
        self._init_model()

    def _init_model(self):
        """모델 로딩 및 레이블 CSV 처리"""
        from huggingface_hub import hf_hub_download

        model_location = os.path.join(self.model_dir, self.repo_id.replace("/", "_"))
        os.makedirs(model_location, exist_ok=True)

        # ONNX 모델 다운로드
        if self.onnx:
            import onnx
            import onnxruntime as ort

            onnx_path = os.path.join(model_location, "model.onnx")
            if not os.path.exists(onnx_path):
                print(f"    다운로드 중: {self.repo_id}/model.onnx")
                hf_hub_download(
                    repo_id=self.repo_id, 
                    filename="model.onnx", 
                    local_dir=model_location
                )

            model = onnx.load(onnx_path)
            self.input_name = model.graph.input[0].name
            del model

            # Provider 설정
            providers = ["CPUExecutionProvider"]
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "ROCMExecutionProvider" in available_providers:
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]

            self.ort_sess = ort.InferenceSession(onnx_path, providers=providers)

        # CSV 다운로드 및 로드
        csv_file = os.path.join(model_location, "selected_tags.csv")
        if not os.path.exists(csv_file):
            print(f"    다운로드 중: {self.repo_id}/selected_tags.csv")
            hf_hub_download(
                repo_id=self.repo_id,
                filename="selected_tags.csv",
                local_dir=model_location
            )

        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)
        
        header, rows = lines[0], lines[1:]
        assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category"

        self.rating_tags = [row[1] for row in rows if row[2] == "9"]
        self.general_tags = [row[1] for row in rows if row[2] == "0"]
        self.character_tags = [row[1] for row in rows if row[2] == "4"]

    def tag(self, image_path, general_threshold=None, character_threshold=None):
        """
        단일 이미지 태깅 - 태그 문자열 반환
        """
        try:
            # 이미지 전처리
            img = Image.open(image_path).convert("RGB")
            img_array = preprocess_image(img)
            img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
            
            # 추론
            preds = self.ort_sess.run(None, {self.input_name: img_array})[0]
            
            # Threshold 설정
            gen_thresh = general_threshold if general_threshold is not None else self.general_threshold
            char_thresh = character_threshold if character_threshold is not None else self.character_threshold
            
            # 태그 추출
            tags = []
            
            # Character 태그 먼저 (있으면)
            for i, tag in enumerate(self.character_tags):
                if preds[0][i] >= char_thresh:
                    tags.append(tag.replace('_', ' '))
            
            # General 태그
            for i, tag in enumerate(self.general_tags):
                if preds[0][len(self.character_tags) + i] >= gen_thresh:
                    tags.append(tag.replace('_', ' '))
            
            return ', '.join(tags)
            
        except Exception as e:
            print(f"⚠️ WD14 태깅 실패: {e}")
            traceback.print_exc()
            return ""


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
    
    try:
        # BLIP 로드
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("  → BLIP 모델 로딩...")
        blip_processor = BlipProcessor.from_pretrained(
            config.BLIP_MODEL_PATH, 
            cache_dir=config.BLIP_CACHE_DIR
        )
        blip_model = BlipForConditionalGeneration.from_pretrained(
            config.BLIP_MODEL_PATH,
            cache_dir=config.BLIP_CACHE_DIR
        ).to(config.DEVICE)
        blip_model.eval()
        
        # WD14 로드
        print("  → WD14 Tagger 로딩...")
        wd14_tagger = WD14Tagger(
            config=config,
            model_dir=config.WD14_CACHE_DIR,
            general_threshold=config.WD14_GENERAL_THRESHOLD,
            character_threshold=config.WD14_CHARACTER_THRESHOLD,
        )
        
        print("✅ 모델 로딩 완료!\n")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 각 디렉토리 처리
    total_success = 0
    
    for directory in config.DATASET_DIRS:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"⚠️ 디렉토리 없음: {directory}")
            continue
        
        count = process_directory(
            directory, blip_model, blip_processor, wd14_tagger, config
        )
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