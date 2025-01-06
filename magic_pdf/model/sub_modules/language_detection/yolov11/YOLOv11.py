# Copyright (c) Opendatalab. All rights reserved.
from collections import Counter
from uuid import uuid4

from PIL import Image
from loguru import logger
from ultralytics import YOLO

language_dict = {
    "ch": "中文简体",
    "en": "英语",
    "japan": "日语",
    "korean": "韩语",
    "fr": "法语",
    "german": "德语",
    "ar": "阿拉伯语",
    "ru": "俄语"
}


def split_images(image, result_images=None):
    """
    对输入文件夹内的图片进行处理,若图片竖向(y方向)分辨率超过400,则进行拆分，
    每次平分图片,直至拆分出的图片竖向分辨率都满足400以下,将处理后的图片(拆分后的子图片)保存到输出文件夹。
    避免保存因裁剪区域超出图片范围导致出现的无效黑色图片部分。
    """
    if result_images is None:
        result_images = []

    width, height = image.size
    long_side = max(width, height)  # 获取较长边长度

    if long_side <= 400:
        result_images.append(image)
        return result_images

    new_long_side = long_side // 2
    sub_images = []

    if width >= height:  # 如果宽度是较长边
        for x in range(0, width, new_long_side):
            # 判断裁剪区域是否超出图片范围，如果超出则不进行裁剪保存操作
            if x + new_long_side > width:
                continue
            box = (x, 0, x + new_long_side, height)
            sub_image = image.crop(box)
            sub_images.append(sub_image)
    else:  # 如果高度是较长边
        for y in range(0, height, new_long_side):
            # 判断裁剪区域是否超出图片范围，如果超出则不进行裁剪保存操作
            if y + new_long_side > height:
                continue
            box = (0, y, width, y + new_long_side)
            sub_image = image.crop(box)
            sub_images.append(sub_image)

    for sub_image in sub_images:
        split_images(sub_image, result_images)

    return result_images


def resize_images_to_224(image):
    """
    若分辨率小于224则用黑色背景补齐到224*224大小,若大于等于224则调整为224*224大小,并保存到输出文件夹中。
    """
    try:
        width, height = image.size
        if width < 224 or height < 224:
            new_image = Image.new('RGB', (224, 224), (0, 0, 0))
            paste_x = (224 - width) // 2
            paste_y = (224 - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        else:
            image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # uuid = str(uuid4())
        # image.save(f"/tmp/{uuid}.jpg")
        return image
    except Exception as e:
        logger.exception(e)


class YOLOv11LangDetModel(object):
    def __init__(self, weight, device):
        self.model = YOLO(weight)
        self.device = device

    def do_detect(self, images: list):
        all_images = []
        for image in images:
            width, height = image.size
            # logger.info(f"image size: {width} x {height}")
            if width < 100 and height < 100:
                continue
            temp_images = split_images(image)
            for temp_image in temp_images:
                all_images.append(resize_images_to_224(temp_image))

        images_lang_res = self.batch_predict(all_images, batch_size=8)
        logger.info(f"images_lang_res: {images_lang_res}")
        if len(images_lang_res) > 0:
            count_dict = Counter(images_lang_res)
            language = max(count_dict, key=count_dict.get)
        else:
            language = None
        return language


    def predict(self, image):
        results = self.model.predict(image, verbose=False, device=self.device)
        predicted_class_id = int(results[0].probs.top1)
        predicted_class_name = self.model.names[predicted_class_id]
        return predicted_class_name


    def batch_predict(self, images: list, batch_size: int) -> list:
        images_lang_res = []
        for index in range(0, len(images), batch_size):
            lang_res = [
                image_res.cpu()
                for image_res in self.model.predict(
                    images[index: index + batch_size],
                    verbose = False,
                    device=self.device,
                )
            ]
            for res in lang_res:
                predicted_class_id = int(res.probs.top1)
                predicted_class_name = self.model.names[predicted_class_id]
                images_lang_res.append(predicted_class_name)

        return images_lang_res