# Copyright (c) Opendatalab. All rights reserved.
"""ONNX 后端的 PP-OCRv6 推理封装。

与 ``PytorchPaddleOCR``（torch 后端）公开接口一致，
内部用 onnxruntime 推理 PaddlePaddle 官方发布的 PP-OCRv6 ONNX 模型。

限制：
- 仅支持 ch 语系（中英日 + 拉丁系 50 语言），不支持多语种切换
- 不支持 seal 模式
- 不支持角度分类（use_angle_cls）

预处理/后处理参数与 PaddlePaddle 官方 ``inference.yml`` 一似。
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import pyclipper
from loguru import logger
from shapely.geometry import Polygon
from tqdm import tqdm

from ...utils.ocr_utils import (
    check_img,
    get_rotate_crop_image_for_text_rec,
    merge_det_boxes,
    preprocess_image,
    sorted_boxes,
    update_det_boxes,
)
from ..utils.onnxruntime_provider import ort_session

__all__ = ["PPOCRv6ONNX"]


# ------------------------------------------------------------------
# 共享工具
# ------------------------------------------------------------------


def _load_character_dict(dict_path: str) -> List[str]:
    """加载字符表，支持 txt 文件或 inference.yml 格式。"""
    if dict_path.endswith(".yml") or dict_path.endswith(".yaml"):
        import yaml

        with open(dict_path, encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        chars = yml.get("PostProcess", {}).get("character_dict", [])
        if not isinstance(chars, list):
            raise ValueError(f"character_dict in {dict_path} is not a list")
        return chars
    # txt 文件：每行一个字符
    return [line.decode("utf-8").strip("\n").strip("\r\n") for line in Path(dict_path).read_bytes().splitlines()]


# ------------------------------------------------------------------
# 文本检测器（det）
# ------------------------------------------------------------------
class TextDetectorONNX:
    """PP-OCRv6 det 的 ONNX 推理封装。

    预处理: DetResize (limit_side_len=960, limit_type='max') + Normalize (ImageNet) + ToCHW
    后处理: DBPostProcess (thresh/box_thresh/unclip_ratio) + filter
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        use_dilation: bool = True,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.session = ort_session(model_path, device, intra_op_num_threads)
        self.input_name = self.session.get_inputs()[0].name

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = 3
        self.score_mode = "fast"
        self.use_dilation = use_dilation
        self.dilation_kernel = np.array([[1, 1], [1, 1]]) if use_dilation else None

    # ---- 预处理 ----
    def _resize_image(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """与 DetResizeForTest.resize_image_type0 一致。"""
        h, w = img.shape[:2]

        if self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                ratio = float(self.limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif self.limit_type == "min":
            if min(h, w) < self.limit_side_len:
                ratio = float(self.limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        else:
            ratio = 1.0

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        if max(resize_h, resize_w) > self.max_side_limit:
            ratio = float(self.max_side_limit) / max(resize_h, resize_w)
            resize_h = int(resize_h * ratio)
            resize_w = int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if resize_w <= 0 or resize_h <= 0:
            return None, None

        resized = cv2.resize(img, (resize_w, resize_h))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return resized, (ratio_h, ratio_w)

    def _preprocess(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """返回 (chw_float32, shape_list) 或 None。"""
        resized, ratios = self._resize_image(img)
        if resized is None:
            return None

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (resized.astype(np.float32) / 255.0 - mean) / std
        chw = norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        src_h, src_w = img.shape[:2]
        ratio_h, ratio_w = ratios
        shape_list = np.array([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)
        return chw, shape_list

    # ---- 后处理（DBPostProcess）----
    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
        rect = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(rect), key=lambda x: x[0])
        idx_1, idx_4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
        idx_2, idx_3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
        box = np.array([points[idx_1], points[idx_2], points[idx_3], points[idx_4]])
        return box, min(rect[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        b = box.copy()
        b[:, 0] -= xmin
        b[:, 1] -= ymin
        cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        poly = Polygon(box)
        if not poly.is_valid or poly.area == 0:
            return box
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def _boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]
        contours = contours[: self.max_candidates]

        boxes: List[np.ndarray] = []
        scores: List[float] = []
        for c in contours:
            points, sside = self._get_mini_boxes(c)
            if sside < self.min_size:
                continue
            score = self._box_score_fast(pred, points)
            if self.box_thresh > score:
                continue
            box = self._unclip(points)
            if len(box) == 0:
                continue
            box, sside = self._get_mini_boxes(box.reshape(-1, 1, 2))
            if sside < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int16), []
        return np.array(boxes), scores

    def _postprocess(self, pred: np.ndarray, shape_list: np.ndarray) -> np.ndarray:
        """DB 后处理 + 过滤，返回 [N, 4, 2] int16。"""
        seg = pred[0, 0, :, :]  # [H, W]
        bitmap = seg > self.thresh
        if self.dilation_kernel is not None:
            bitmap = cv2.dilate(np.array(bitmap).astype(np.uint8), self.dilation_kernel)

        src_h, src_w = int(shape_list[0]), int(shape_list[1])
        boxes, _ = self._boxes_from_bitmap(seg, bitmap, src_w, src_h)
        return self._filter_det_res(boxes, (src_h, src_w))

    @staticmethod
    def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most
        return np.array([tl, tr, br, bl], dtype="float32")

    def _filter_det_res(self, dt_boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        img_height, img_width = image_shape
        dt_boxes_new: List[np.ndarray] = []
        for box in dt_boxes:
            box = self._order_points_clockwise(box)
            box = np.clip(box, 0, [img_width - 1, img_height - 1]).astype(np.int16)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        if not dt_boxes_new:
            return np.zeros((0, 4, 2), dtype=np.int16)
        return np.array(dt_boxes_new)

    # ---- 推理 ----
    def __call__(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        t0 = time.perf_counter()
        preprocessed = self._preprocess(img)
        if preprocessed is None:
            return None, 0.0
        chw, shape_list = preprocessed
        pred = self.session.run(None, {self.input_name: chw})[0]
        dt_boxes = self._postprocess(pred, shape_list[0])
        return dt_boxes, time.perf_counter() - t0

    def batch_predict(
        self,
        img_list: List[np.ndarray],
        max_batch_size: int = 8,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-det Predict",
        tqdm_progress_bar: Optional[Any] = None,
    ) -> List[Tuple[Optional[np.ndarray], float]]:
        if not img_list:
            return []

        pbar = tqdm_progress_bar
        should_close = False
        if pbar is None:
            pbar = tqdm(total=len(img_list), desc=tqdm_desc, disable=not tqdm_enable)
            should_close = True

        results: List[Tuple[Optional[np.ndarray], float]] = [(None, 0.0)] * len(img_list)
        try:
            for i, img in enumerate(img_list):
                dt_boxes, elapse = self(img)
                results[i] = (dt_boxes, elapse)
                pbar.update(1)
        finally:
            if should_close:
                pbar.close()
        return results


# ------------------------------------------------------------------
# 文本识别器（rec）
# ------------------------------------------------------------------
class TextRecognizerONNX:
    """PP-OCRv6 rec 的 ONNX 推理封装。

    预处理: RecResizeImg (动态宽度, imgH=48) + Normalize (/127.5 - 1) + padding
    后处理: CTC 解码
    """

    def __init__(
        self,
        model_path: str,
        dict_path: str,
        device: Optional[str] = None,
        rec_image_shape: Tuple[int, int, int] = (3, 48, 320),
        rec_batch_num: int = 6,
        drop_score: float = 0.5,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.session = ort_session(model_path, device, intra_op_num_threads)
        self.input_name = self.session.get_inputs()[0].name

        self.img_c, self.img_h, self.img_w = rec_image_shape
        self.rec_batch_num = rec_batch_num
        self.drop_score = drop_score

        # 字符表: ["blank"] + dict_chars + [" "]
        # dict_path 可以是 txt 文件（每行一个字符）或 inference.yml（含 character_dict 列表）
        chars = _load_character_dict(dict_path)
        self.character = ["blank"] + chars + [" "]

    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_c, img_h, img_w = self.img_c, self.img_h, self.img_w
        max_wh_ratio = max(max_wh_ratio, img_w / img_h)
        img_w = int(img_h * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(img_w, int(max(math.ceil(img_h * ratio), 1)))

        resized = cv2.resize(img, (resized_w, img_h))
        norm = resized.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        padded = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padded[:, :, :resized_w] = norm
        return padded

    def _decode(self, pred: np.ndarray) -> Tuple[str, float]:
        """CTC 解码单个 prediction。"""
        idx = pred.argmax(axis=1)
        prob = pred.max(axis=1)
        # collapse consecutive duplicates
        selection = np.ones(len(idx), dtype=bool)
        selection[1:] = idx[1:] != idx[:-1]
        # remove blank (index 0)
        selection &= idx != 0
        chars = [self.character[i] for i in idx[selection]]
        text = "".join(chars)
        conf = float(prob[selection].mean()) if selection.any() else 0.0
        return text, conf

    def __call__(
        self,
        img_list: List[np.ndarray],
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        tqdm_progress_bar: Optional[Any] = None,
    ) -> Tuple[List[Tuple[str, float]], float]:
        if not img_list:
            return [], 0.0

        t0 = time.perf_counter()
        img_num = len(img_list)

        # 按宽高比排序（加速 batch 内 padding 效率）
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))

        rec_res: List[Tuple[str, float]] = [("", 0.0)] * img_num

        pbar = tqdm_progress_bar
        should_close = False
        if pbar is None:
            pbar = tqdm(total=img_num, desc=tqdm_desc, disable=not tqdm_enable)
            should_close = True

        try:
            batch_num = self.rec_batch_num
            for beg in range(0, img_num, batch_num):
                end = min(img_num, beg + batch_num)
                batch_indices = indices[beg:end]
                max_wh_ratio = width_list[batch_indices[-1]]

                norm_img_batch = []
                for idx in batch_indices:
                    norm_img = self._resize_norm_img(img_list[idx], max_wh_ratio)
                    norm_img_batch.append(norm_img[np.newaxis, ...])
                batch_tensor = np.concatenate(norm_img_batch, axis=0).astype(np.float32)

                preds = self.session.run(None, {self.input_name: batch_tensor})[0]
                # preds shape: [B, T, C]

                for i, idx in enumerate(batch_indices):
                    text, conf = self._decode(preds[i])
                    rec_res[idx] = (text, conf)
                pbar.update(end - beg)
        finally:
            if should_close:
                pbar.close()

        return rec_res, time.perf_counter() - t0


# ------------------------------------------------------------------
# 组合类：PPOCRv6ONNX
# ------------------------------------------------------------------
class PPOCRv6ONNX:
    """PP-OCRv6 的 ONNX 推理封装。

    与 ``PytorchPaddleOCR`` 的 ``ocr()`` 接口兼容。
    不支持 seal 模式和多语种切换。
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        dict_path: str,
        device: Optional[str] = None,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.5,
        enable_merge_det_boxes: bool = True,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.text_detector = TextDetectorONNX(
            model_path=det_model_path,
            device=device,
            box_thresh=det_db_box_thresh,
            unclip_ratio=det_db_unclip_ratio,
            intra_op_num_threads=intra_op_num_threads,
        )
        self.text_recognizer = TextRecognizerONNX(
            model_path=rec_model_path,
            dict_path=dict_path,
            device=device,
            rec_batch_num=rec_batch_num,
            drop_score=drop_score,
            intra_op_num_threads=intra_op_num_threads,
        )
        self.drop_score = drop_score
        self.is_seal = False
        self.enable_merge_det_boxes = enable_merge_det_boxes
        self.lang = "ch"

        logger.debug(
            "PPOCRv6ONNX loaded: det={}, rec={}",
            Path(det_model_path).name,
            Path(rec_model_path).name,
        )

    def ocr(
        self,
        img: Union[np.ndarray, List[np.ndarray], str, bytes],
        det: bool = True,
        rec: bool = True,
        mfd_res: Optional[List[dict]] = None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        tqdm_progress_bar: Optional[Any] = None,
    ) -> List[Optional[List]]:
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det:
            logger.error("When input a list of images, det must be false")
            return [None]

        img = check_img(img)
        imgs = [img]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if det and rec:
                # 完整 OCR：det + crop + rec
                ocr_res: List[Optional[List]] = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self._det_rec(img, mfd_res)
                    if (dt_boxes is None or len(dt_boxes) == 0) and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                    ocr_res.append(tmp_res)
                return ocr_res

            elif det and not rec:
                # 仅检测
                ocr_res: List[Optional[List]] = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, _elapse = self.text_detector(img)
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    dt_boxes = sorted_boxes(dt_boxes)
                    if self.enable_merge_det_boxes:
                        dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res:
                        dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                return ocr_res

            elif not det and rec:
                # 仅识别
                ocr_res: List[Optional[List]] = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_res, _elapse = self.text_recognizer(
                        img,
                        tqdm_enable=tqdm_enable,
                        tqdm_desc=tqdm_desc,
                        tqdm_progress_bar=tqdm_progress_bar,
                    )
                    ocr_res.append(rec_res)
                return ocr_res

            return [None]

    def _det_rec(self, img: np.ndarray, mfd_res: Optional[List[dict]] = None) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """det + crop + rec 的完整流程。"""
        ori_im = img
        dt_boxes, _elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return np.array([]), []

        dt_boxes = sorted_boxes(dt_boxes)
        if self.enable_merge_det_boxes:
            dt_boxes = merge_det_boxes(dt_boxes)
        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        img_crop_list: List[np.ndarray] = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image_for_text_rec(ori_im, tmp_box)
            if img_crop is not None:
                img_crop_list.append(img_crop)

        if not img_crop_list:
            return np.array([]), []

        rec_res, _elapse = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            _text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        if not filter_boxes:
            return np.array([]), []
        return np.array(filter_boxes), filter_rec_res

    def __call__(self, img: np.ndarray, mfd_res: Optional[List[dict]] = None):
        """便捷调用，等价于 ocr(img, det=True, rec=True)。"""
        if img is None:
            return None, None
        boxes, rec_res = self._det_rec(img, mfd_res)
        if len(boxes) == 0:
            return None, None
        return boxes, rec_res


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="PP-OCRv6 ONNX local inference smoke test")
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--det", default=None, help="Path to det inference.onnx.")
    parser.add_argument("--rec", default=None, help="Path to rec inference.onnx.")
    parser.add_argument("--dict", default=None, help="Path to character dict file.")
    parser.add_argument("--device", default="cpu", help="cpu / cuda.")
    parser.add_argument("--output", default=None, help="Save result JSON to this path.")
    args = parser.parse_args()

    # 默认从 model_registry 获取路径
    if args.det is None or args.rec is None:
        from ...utils.model_registry import PP_OCR_V6_SMALL_DET_ONNX, PP_OCR_V6_SMALL_REC_ONNX

        if args.det is None:
            args.det = str(PP_OCR_V6_SMALL_DET_ONNX.onnx.ensure())
        if args.rec is None:
            args.rec = str(PP_OCR_V6_SMALL_REC_ONNX.onnx.ensure())
    if args.dict is None:
        # rec 的 dict 内嵌在 inference.yml 里，但我们直接用 ModelScope 下载的 dict
        # 从 rec 的 model_registry config path 读取
        from ...utils.model_registry import PP_OCR_V6_SMALL_REC_ONNX

        rec_dir = PP_OCR_V6_SMALL_REC_ONNX.onnx.local_path().parent
        # 尝试从 inference.yml 提取 dict，或使用内置的
        args.dict = str(rec_dir / "inference.yml")

    print(f"det: {args.det}")
    print(f"rec: {args.rec}")
    print(f"dict: {args.dict}")

    # 对于 dict，我们需要从 inference.yml 提取 character_dict
    # 但更简单的方式是直接用 RapidOCR 下载的 dict 文件
    # 这里先用一个 fallback
    if not os.path.exists(args.dict) or args.dict.endswith(".yml"):
        # 尝试使用 rec onnx 内嵌的 dict（从 inference.yml 解析）
        import yaml

        with open(args.dict, encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        chars = yml.get("PostProcess", {}).get("character_dict", [])
        # 写入临时 dict 文件
        tmp_dict = "/tmp/ppocrv6_dict.txt"
        with open(tmp_dict, "w", encoding="utf-8") as f:
            for c in chars:
                f.write(c + "\n")
        args.dict = tmp_dict
        print(f"extracted dict to: {args.dict} ({len(chars)} chars)")

    model = PPOCRv6ONNX(
        det_model_path=args.det,
        rec_model_path=args.rec,
        dict_path=args.dict,
        device=args.device,
    )

    img = cv2.imread(args.image)
    results = model.ocr(img)
    print(f"\ndetected {len(results[0] or [])} text lines")
    for item in results[0] or []:
        box, (text, score) = item
        print(f"  conf={score:.2f}  {text[:80]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nresult saved to {args.output}")
