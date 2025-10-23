import os
import sys
import logging
import numpy as np
import torch
from PIL import Image

# 현재 파일의 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from library.utils import resize_image

logger = logging.getLogger(__name__)

IMAGE_SIZE = 448

def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    image = resize_image(image, image.shape[0], image.shape[1], IMAGE_SIZE, IMAGE_SIZE)

    image = image.astype(np.float32)
    return image




class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            # tensor = torch.tensor(image) # これ Tensor に変換する必要ないな……(;･∀･)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image, img_path)


# run_batch를 main 밖으로 올리고 필요한 것들을 매개변수로 받음
def run_batch(path_imgs, model=None, ort_sess=None, input_name=None, args=None,
              general_tags=None, character_tags=None, rating_tags=None,
              undesired_tags=None, tag_freq=None, always_first_tags=None):
    
    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    
    for image_path, image in path_imgs:
        combined_tags = []
        rating_tag_text = ""
        character_tag_text = ""
        general_tag_text = ""

        if args.onnx:
            probs = ort_sess.run(None, {input_name: np.array([image])})[0]
            probs = probs[:1]
        else:
            probs = model(np.array([image]), training=False)
            probs = probs.numpy()

        prob = probs[0]

        # general & character tags 처리
        for i, p in enumerate(prob[4:]):
            if i < len(general_tags) and p >= args.general_threshold:
                tag_name = general_tags[i]
                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    general_tag_text += caption_separator + tag_name
                    combined_tags.append(tag_name)
            elif i >= len(general_tags) and p >= args.character_threshold:
                tag_name = character_tags[i - len(general_tags)]
                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    character_tag_text += caption_separator + tag_name
                    if args.character_tags_first:
                        combined_tags.insert(0, tag_name)
                    else:
                        combined_tags.append(tag_name)

        # rating tags 처리
        if args.use_rating_tags or args.use_rating_tags_as_last_tag:
            ratings_probs = prob[:4]
            rating_index = ratings_probs.argmax()
            found_rating = rating_tags[rating_index]
            if found_rating not in undesired_tags:
                tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                rating_tag_text = found_rating
                if args.use_rating_tags:
                    combined_tags.insert(0, found_rating)
                else:
                    combined_tags.append(found_rating)

        # always_first_tags 처리
        if always_first_tags:
            for tag in always_first_tags:
                if tag in combined_tags:
                    combined_tags.remove(tag)
                    combined_tags.insert(0, tag)

        # 기존 파일이 있으면 append
        caption_file = os.path.splitext(image_path)[0] + args.caption_extension
        tag_text = caption_separator.join(combined_tags)
        if args.append_tags and os.path.exists(caption_file):
            with open(caption_file, "rt", encoding="utf-8") as f:
                existing_tags = [t.strip() for t in f.read().split(stripped_caption_separator) if t.strip()]
            new_tags = [t for t in combined_tags if t not in existing_tags]
            tag_text = caption_separator.join(existing_tags + new_tags)

        with open(caption_file, "wt", encoding="utf-8") as f:
            f.write(tag_text + "\n")
            if args.debug:
                logger.info(f"{image_path}: rating={rating_tag_text}, char={character_tag_text}, gen={general_tag_text}")
