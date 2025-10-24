from pathlib import Path
import torch
from coltran.model import ColTranModel  # 예시용 (HuggingFace 또는 GitHub fork)
from coltran.utils import color_transfer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 ColTran 모델 로드
model = ColTranModel.from_pretrained("google/coltran")  # 또는 체크포인트 경로
model.to(device).eval()

# 🔹 기준(reference) 이미지
ref_path = Path("ref_tone.jpg")

# 🔹 톤 보정할 이미지 폴더
input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

# 🔹 모든 이미지를 순차 처리
for img_path in input_dir.glob("*.png"):
    try:
        result = color_transfer(model, img_path, ref_path, device=device)
        out_path = output_dir / img_path.name
        result.save(out_path)
        print(f"[✅] {img_path.name} done.")
    except Exception as e:
        print(f"[❌] {img_path.name} failed: {e}")
