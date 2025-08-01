# Copyright (c) Opendatalab. All rights reserved.
import os

import cv2
import numpy as np
import onnxruntime
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class PaddleOrientationClsModel:
    def __init__(self, ocr_engine):
        self.sess = onnxruntime.InferenceSession(
            os.path.join(auto_download_and_get_model_root_path(ModelPath.paddle_orientation_classification), ModelPath.paddle_orientation_classification)
        )
        self.ocr_engine = ocr_engine
        self.less_length = 256
        self.cw, self.ch = 224, 224
        self.std = [0.229, 0.224, 0.225]
        self.scale = 0.00392156862745098
        self.mean = [0.485, 0.456, 0.406]
        self.labels = ["0", "90", "180", "270"]

    def preprocess(self, img):
        # PIL图像转cv2
        img = np.array(img)
        # 放大图片，使其最短边长为256
        h, w = img.shape[:2]
        scale = 256 / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        img = cv2.resize(img, (w_resize, h_resize), interpolation=1)
        # 调整为224*224的正方形
        h, w = img.shape[:2]
        cw, ch = 224, 224
        x1 = max(0, (w - cw) // 2)
        y1 = max(0, (h - ch) // 2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)
        if w < cw or h < ch:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({cw}, {ch})."
            )
        img = img[y1:y2, x1:x2, ...]
        # 正则化
        split_im = list(cv2.split(img))
        std = [0.229, 0.224, 0.225]
        scale = 0.00392156862745098
        mean = [0.485, 0.456, 0.406]
        alpha = [scale / std[i] for i in range(len(std))]
        beta = [-mean[i] / std[i] for i in range(len(std))]
        for c in range(img.shape[2]):
            split_im[c] = split_im[c].astype(np.float32)
            split_im[c] *= alpha[c]
            split_im[c] += beta[c]
        img = cv2.merge(split_im)
        # 5. 转换为 CHW 格式
        img = img.transpose((2, 0, 1))
        imgs = [img]
        x = np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x

    def predict(self, img):
        bgr_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:

            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            # Check if table is rotated by analyzing text box aspect ratios
            if det_res:
                vertical_count = 0
                is_rotated = False

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                if vertical_count >= len(det_res) * 0.3:
                    is_rotated = True
                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if is_rotated:
                    x = self.preprocess(img)
                    (result,) = self.sess.run(None, {"x": x})
                    label = self.labels[np.argmax(result)]

                    if label == "90":
                        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                        img = cv2.rotate(np.asarray(img), rotation)
                    elif label == "180":
                        rotation = cv2.ROTATE_180
                        img = cv2.rotate(np.asarray(img), rotation)
                    elif label == "270":
                        rotation = cv2.ROTATE_90_CLOCKWISE
                        img = cv2.rotate(np.asarray(img), rotation)
                    else:
                        img = np.array(img)
        return img
