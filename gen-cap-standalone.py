"""
독립 실행형 BLIP + WD14 하이브리드 캡션 생성기
실사 LoRA 학습을 위한 통합 캡션 생성 스크립트

설치 필요:
pip install transformers pillow torch torchvision onnxruntime-gpu huggingface_hub
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import torch


# ==============================
# ⚙️ 설정 (수정 가능)
# ==============================

class Config:
    # 데이터셋 경로
    DATASET_DIRS = [
        "./dataset/mainchar",  # 메인 캐릭터
        "./dataset/bg",        # 배경/보조
    ]
    
    # 모델 설정
    BLIP_MODEL = "Salesforce/blip-image-captioning-large"
    WD14_MODEL = "SmilingWolf/wd-v1-4-moat-tagger-v2"
    
    # WD14 임계값
    WD14_GENERAL_THRESHOLD = 0.35
    WD14_CHARACTER_THRESHOLD = 0.85
    
    # BLIP 설정
    BLIP_MAX_LENGTH = 75
    BLIP_NUM_BEAMS = 1  # 1=greedy, >1=beam search
    
    # 제거할 WD14 메타 태그
    REMOVE_TAGS = [
        # 메타 태그
        "1girl", "1boy", "solo", "2girls", "3girls", "multiple girls",
        "looking at viewer", "facing viewer", "solo focus",
        # 배경
        "simple background", "white background", "grey background",
        "transparent background", "gradient background",
        # 품질/메타데이터
        "highres", "absurdres", "lowres", "bad anatomy",
        "signature", "watermark", "artist name", "dated",
        "commentary", "username",
        # Danbooru 메타
        "rating:safe", "rating:questionable", "rating:explicit",
        "safe", "questionable", "explicit",
    ]
    
    # 출력 설정
    OUTPUT_ENCODING = "utf-8"
    OVERWRITE_EXISTING = False
    CREATE_BACKUP = True
    
    # 디바이스
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 캡션 포맷
    # "blip_first": BLIP 문장이 먼저
    # "tags_first": WD14 태그가 먼저
    CAPTION_FORMAT = "blip_first"


# ==============================
# 🔧 유틸리티 함수
# ==============================

def normalize_tags(tags_str):
    """태그 정규화: 소문자, 공백 정리, 중복 제거"""
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


def merge_captions(blip_caption, wd14_tags, remove_tags, format_type="blip_first"):
    """
    BLIP 캡션과 WD14 태그 병합
    """
    # BLIP 정규화
    blip_normalized = blip_caption.strip().lower() if blip_caption else ""
    
    # WD14 태그 정규화 및 필터링
    wd14_normalized = normalize_tags(wd14_tags)
    wd14_filtered = remove_unwanted_tags(wd14_normalized, remove_tags)
    
    # BLIP 문장의 단어들 (중복 제거용)
    blip_words = set(blip_normalized.replace(',', ' ').split()) if blip_normalized else set()
    
    # WD14에서 BLIP 중복 제거
    wd14_deduped = []
    for tag in wd14_filtered:
        # 단순 중복 체크 (선택적으로 비활성화 가능)
        tag_words = set(tag.replace('_', ' ').split())
        if not tag_words.intersection(blip_words):
            wd14_deduped.append(tag)
    
    # 최종 병합
    if format_type == "blip_first":
        # BLIP 문장 + WD14 태그
        if blip_normalized and wd14_deduped:
            merged = f"{blip_normalized}, {', '.join(wd14_deduped)}"
        elif blip_normalized:
            merged = blip_normalized
        elif wd14_deduped:
            merged = ', '.join(wd14_deduped)
        else:
            merged = ""
    else:
        # WD14 태그 + BLIP 문장
        if wd14_deduped and blip_normalized:
            merged = f"{', '.join(wd14_deduped)}, {blip_normalized}"
        elif wd14_deduped:
            merged = ', '.join(wd14_deduped)
        elif blip_normalized:
            merged = blip_normalized
        else:
            merged = ""
    
    return merged


# ==============================
# 🎨 BLIP 캡션 생성
# ==============================

class BLIPCaptioner:
    def __init__(self, model_name, device, max_length=75, num_beams=1):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print(f"  → BLIP 모델 로딩... ({model_name})")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
    
    def generate(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption.strip()
        
        except Exception as e:
            print(f"⚠️ BLIP 실패 ({Path(image_path).name}): {e}")
            return ""


# ==============================
# 🏷️ WD14 태그 생성
# ==============================

class WD14Tagger:
    def __init__(self, model_name, device, general_thresh=0.35, character_thresh=0.85):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        
        print(f"  → WD14 모델 로딩... ({model_name})")
        
        # 모델 다운로드
        model_path = hf_hub_download(model_name, filename="model.onnx")
        tags_path = hf_hub_download(model_name, filename="selected_tags.csv")
        
        # ONNX 세션
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 태그 로드
        import pandas as pd
        self.tags_df = pd.read_csv(tags_path)
        self.general_thresh = general_thresh
        self.character_thresh = character_thresh
    
    def generate(self, image_path):
        try:
            import numpy as np
            
            # 이미지 전처리
            image = Image.open(image_path).convert("RGB")
            image = image.resize((448, 448))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # 추론
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: image_array})[0]
            
            # 태그 필터링
            tags = []
            for i, score in enumerate(output[0]):
                tag_type = self.tags_df.iloc[i]['category']
                threshold = self.character_thresh if tag_type == 4 else self.general_thresh
                
                if score >= threshold:
                    tag_name = self.tags_df.iloc[i]['name'].replace('_', ' ')
                    tags.append(tag_name)
            
            return ', '.join(tags)
        
        except Exception as e:
            print(f"⚠️ WD14 실패 ({Path(image_path).name}): {e}")
            return ""


# ==============================
# 📁 파일 처리
# ==============================

def get_image_files(directory):
    """이미지 파일 찾기"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    
    dir_path = Path(directory)
    for ext in extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def create_backup(caption_path):
    """백업 생성"""
    if caption_path.exists():
        backup_dir = caption_path.parent / "caption_backup"
        backup_dir.mkdir(exist_ok=True)
        
        import shutil
        backup_path = backup_dir / caption_path.name
        shutil.copy2(caption_path, backup_path)


# ==============================
# 🚀 메인 프로세스
# ==============================

def process_directory(directory, blip_captioner, wd14_tagger, config):
    """디렉토리 처리"""
    print(f"\n📁 처리 중: {directory}")
    
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"⚠️ 이미지 없음: {directory}")
        return 0
    
    print(f"📸 {len(image_files)}개 이미지 발견")
    
    success_count = 0
    skip_count = 0
    
    for image_path in tqdm(image_files, desc="캡션 생성"):
        caption_path = image_path.with_suffix('.txt')
        
        # 기존 파일 확인
        if caption_path.exists() and not config.OVERWRITE_EXISTING:
            skip_count += 1
            continue
        
        # 백업
        if config.CREATE_BACKUP and caption_path.exists():
            create_backup(caption_path)
        
        try:
            # BLIP 생성
            blip_caption = blip_captioner.generate(image_path)
            
            # WD14 생성
            wd14_tags = wd14_tagger.generate(image_path)
            
            # 병합
            merged = merge_captions(
                blip_caption, wd14_tags, 
                config.REMOVE_TAGS, 
                config.CAPTION_FORMAT
            )
            
            # 저장
            if merged:
                with open(caption_path, 'w', encoding=config.OUTPUT_ENCODING) as f:
                    f.write(merged)
                success_count += 1
            else:
                print(f"⚠️ 빈 캡션: {image_path.name}")
        
        except Exception as e:
            print(f"❌ 실패 ({image_path.name}): {e}")
            continue
    
    print(f"✅ 완료: {success_count}개 생성, {skip_count}개 스킵")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="BLIP + WD14 하이브리드 캡션")
    parser.add_argument("--dirs", nargs="+", help="처리할 디렉토리")
    parser.add_argument("--overwrite", action="store_true", help="덮어쓰기")
    parser.add_argument("--device", default=None, help="cuda/cpu")
    parser.add_argument("--format", choices=["blip_first", "tags_first"], help="캡션 포맷")
    
    args = parser.parse_args()
    
    config = Config()
    if args.dirs:
        config.DATASET_DIRS = args.dirs
    if args.overwrite:
        config.OVERWRITE_EXISTING = True
    if args.device:
        config.DEVICE = args.device
    if args.format:
        config.CAPTION_FORMAT = args.format
    
    print("=" * 60)
    print("🎨 BLIP + WD14 하이브리드 캡션 생성기")
    print("=" * 60)
    print(f"📁 대상: {config.DATASET_DIRS}")
    print(f"💾 덮어쓰기: {config.OVERWRITE_EXISTING}")
    print(f"🖥️ 디바이스: {config.DEVICE}")
    print(f"📝 포맷: {config.CAPTION_FORMAT}")
    print("=" * 60)
    
    # 모델 로드
    print("\n🔄 모델 로딩 중...")
    
    try:
        blip_captioner = BLIPCaptioner(
            config.BLIP_MODEL,
            config.DEVICE,
            config.BLIP_MAX_LENGTH,
            config.BLIP_NUM_BEAMS
        )
        
        wd14_tagger = WD14Tagger(
            config.WD14_MODEL,