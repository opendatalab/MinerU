import os

from PIL import Image
import cv2
import numpy as np
import onnxruntime
from loguru import logger
from tqdm import tqdm

from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class PaddleTableClsModel:
    def __init__(self):
        self.sess = onnxruntime.InferenceSession(
            os.path.join(auto_download_and_get_model_root_path(ModelPath.paddle_table_cls), ModelPath.paddle_table_cls)
        )
        self.less_length = 256
        self.cw, self.ch = 224, 224
        self.std = [0.229, 0.224, 0.225]
        self.scale = 0.00392156862745098
        self.mean = [0.485, 0.456, 0.406]
        self.labels = [AtomicModel.WiredTable, AtomicModel.WirelessTable]

    def preprocess(self, input_img):
        # 放大图片，使其最短边长为256
        h, w = input_img.shape[:2]
        scale = 256 / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        img = cv2.resize(input_img, (w_resize, h_resize), interpolation=1)
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

    def predict(self, input_img):
        if isinstance(input_img, Image.Image):
            np_img = np.asarray(input_img)
        elif isinstance(input_img, np.ndarray):
            np_img = input_img
        else:
            raise ValueError("Input must be a pillow object or a numpy array.")
        x = self.preprocess(np_img)
        result = self.sess.run(None, {"x": x})
        idx = np.argmax(result)
        conf = float(np.max(result))
        return self.labels[idx], conf

    def list_2_batch(self, img_list, batch_size=16):
        """
        将任意长度的列表按照指定的batch size分成多个batch

        Args:
            img_list: 输入的列表
            batch_size: 每个batch的大小，默认为16

        Returns:
            一个包含多个batch的列表，每个batch都是原列表的一个子列表
        """
        batches = []
        for i in range(0, len(img_list), batch_size):
            batch = img_list[i : min(i + batch_size, len(img_list))]
            batches.append(batch)
        return batches

    def batch_preprocess(self, imgs):
        res_imgs = []
        for img in imgs:
            img = np.asarray(img)
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
            res_imgs.append(img)
        x = np.stack(res_imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x
    def batch_predict(self, img_info_list, batch_size=16):
        imgs = [item["wired_table_img"] for item in img_info_list]
        imgs = self.list_2_batch(imgs, batch_size=batch_size)
        label_res = []
        with tqdm(total=len(img_info_list), desc="Table-wired/wireless cls predict", disable=True) as pbar:
            for img_batch in imgs:
                x = self.batch_preprocess(img_batch)
                result = self.sess.run(None, {"x": x})
                for img_res in result[0]:
                    idx = np.argmax(img_res)
                    conf = float(np.max(img_res))
                    label_res.append((self.labels[idx],conf))
                pbar.update(len(img_batch))
            for img_info, (label, conf) in zip(img_info_list, label_res):
                img_info['table_res']["cls_label"] = label
                img_info['table_res']["cls_score"] = round(conf, 3)
