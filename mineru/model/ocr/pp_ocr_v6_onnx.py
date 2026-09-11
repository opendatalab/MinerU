# Copyright (c) Opendatalab. All rights reserved.
"""ONNX 后端的 PP-OCRv6 推理封装。

与 ``PytorchPaddleOCR``（torch 后端）公开接口一致，
内部用 onnxruntime 推理 PaddlePaddle 官方发布的 PP-OCRv6 ONNX 模型。

限制：
- 仅支持 ch 语系（中英日 + 拉丁系 50 语言），不支持多语种切换
- seal 使用专用检测图、多边形裁剪和共享 Small Rec
- 不支持角度分类（use_angle_cls）

预处理/后处理参数与 PaddlePaddle 官方 ``inference.yml`` 一似。
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..runtime.onnx import ort_session
from .._internal.pytorchocr.data.imaug.operators import DetResizeForTest, NormalizeImage
from .db_postprocess import DBPostProcess
from .resources import PPOCRV6_DICT_PATH
from .seal_crop import CropByPolys, SortPolyBoxes
from .geometry import merge_det_boxes, sorted_boxes, update_det_boxes
from .image import check_img, get_rotate_crop_image_for_text_rec, preprocess_image, resize_text_recognition_image

DetectionBoxes = np.ndarray | list[np.ndarray]

__all__ = ["DetectionBoxes", "PPOCRv6ONNX", "TextDetectorONNX", "TextRecognizerONNX"]


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
        use_dilation: bool = False,
        box_type: Literal["quad", "poly"] = "quad",
        intra_op_num_threads: int = 0,
    ) -> None:
        """加载 CPU ONNX 会话并设置推理参数。"""
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
        self.box_type = box_type
        self.resize_op = DetResizeForTest(limit_side_len=limit_side_len, limit_type=limit_type, max_side_limit=max_side_limit)
        self.normalize_op = NormalizeImage(
            scale=1.0 / 255.0,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order="hwc",
        )
        self.postprocess_op = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            box_type=box_type,
        )

    # ---- 预处理 ----
    def _resize_image(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """复用 Torch 路径的纯 NumPy/OpenCV 缩放逻辑。"""
        if img.size == 0 or min(img.shape[:2]) == 0:
            return None, None
        resized, ratios = self.resize_op.resize_image_type0(img)
        return resized, tuple(ratios)

    def _preprocess(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """返回 (chw_float32, shape_list) 或 None。"""
        resized, ratios = self._resize_image(img)
        if resized is None:
            return None

        norm = self.normalize_op({"image": resized})["image"]
        chw = np.ascontiguousarray(norm.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)

        src_h, src_w = img.shape[:2]
        ratio_h, ratio_w = ratios
        shape_list = np.array([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)
        return chw, shape_list

    def _postprocess(self, pred: np.ndarray, shape_list: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """复用 DB 后处理，印章多边形只裁剪坐标，不转成四边形。"""
        boxes = self.postprocess_op({"maps": pred}, shape_list[np.newaxis, :])[0]["points"]
        src_h, src_w = int(shape_list[0]), int(shape_list[1])
        if self.box_type == "poly":
            return [np.clip(np.asarray(box), 0, [src_w - 1, src_h - 1]).astype(np.int32) for box in boxes]
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
            return np.zeros((0, 4, 2), dtype=np.float32)
        # OpenCV 透视裁剪要求 float32 点集；与 Torch 检测器的输出契约一致。
        return np.asarray(dt_boxes_new, dtype=np.float32)

    # ---- 推理 ----
    def __call__(self, img: np.ndarray) -> tuple[DetectionBoxes | None, float]:
        """单图复用同尺寸合批路径，保持四边形及印章多边形的返回类型。"""
        return self.batch_predict([img], max_batch_size=1)[0]

    def _run_det_batch(self, items: list[tuple[int, np.ndarray, np.ndarray, float]]) -> list[tuple[int, DetectionBoxes, float]]:
        """执行同尺寸检测批次；共享推理耗时均摊到各图，后处理使用各自原图信息。"""
        pixels = np.concatenate([item[1] for item in items], axis=0)
        started = time.perf_counter()
        predictions = self.session.run(None, {self.input_name: pixels})[0]
        elapsed = (time.perf_counter() - started) / len(items)
        if predictions.ndim != 4 or predictions.shape[0] != len(items):
            raise ValueError("OCR detector output batch does not match input batch")
        results = []
        for position, (index, _, shape_list, preprocess_seconds) in enumerate(items):
            started = time.perf_counter()
            boxes = self._postprocess(predictions[position : position + 1], shape_list[0])
            results.append((index, boxes, preprocess_seconds + elapsed + time.perf_counter() - started))
        return results

    def batch_predict(
        self,
        img_list: List[np.ndarray],
        max_batch_size: int = 8,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-det Predict",
        tqdm_progress_bar: Optional[Any] = None,
    ) -> list[tuple[DetectionBoxes | None, float]]:
        """按预处理后尺寸分桶，桶满即真批处理，最后恢复输入顺序。"""
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not img_list:
            return []

        pbar = tqdm_progress_bar
        should_close = False
        if pbar is None:
            pbar = tqdm(total=len(img_list), desc=tqdm_desc, disable=not tqdm_enable)
            should_close = True

        results: list[tuple[DetectionBoxes | None, float]] = [(None, 0.0)] * len(img_list)
        buckets: dict[tuple[int, ...], list[tuple[int, np.ndarray, np.ndarray, float]]] = {}

        def flush(items: list[tuple[int, np.ndarray, np.ndarray, float]]) -> None:
            """回填已完成批次并更新进度，释放桶内的预处理张量。"""
            for index, boxes, elapsed in self._run_det_batch(items):
                results[index] = (boxes, elapsed)
            pbar.update(len(items))
            items.clear()

        try:
            for i, img in enumerate(img_list):
                started = time.perf_counter()
                prepared = self._preprocess(img)
                if prepared is None:
                    pbar.update(1)
                    continue
                pixels, shape_list = prepared
                key = tuple(pixels.shape[1:])
                items = buckets.setdefault(key, [])
                items.append((i, pixels, shape_list, time.perf_counter() - started))
                if len(items) == max_batch_size:
                    flush(buckets.pop(key))
            for items in buckets.values():
                flush(items)
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
        """加载 CPU ONNX 会话并设置推理参数。"""
        self.session = ort_session(model_path, device, intra_op_num_threads)
        self.input_name = self.session.get_inputs()[0].name

        self.img_c, self.img_h, self.img_w = rec_image_shape
        self.rec_batch_num = rec_batch_num
        self.drop_score = drop_score

        # 字符表: ["blank"] + dict_chars + [" "]
        # dict_path 可以是 txt 文件（每行一个字符）或 inference.yml（含 character_dict 列表）
        chars = _load_character_dict(dict_path)
        self.character = ["blank"] + chars + [" "]
        outputs = self.session.get_outputs()
        classes = outputs[0].shape[-1]
        if isinstance(classes, int) and classes != len(self.character):
            raise ValueError(f"OCR CTC output has {classes} classes but dictionary requires {len(self.character)}")

    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """复用标准识别输入处理，包含最小宽度与超长文字宽度上限。"""
        return resize_text_recognition_image(img, max_wh_ratio, (self.img_c, self.img_h, self.img_w))

    def _decode(self, pred: np.ndarray) -> Tuple[str, float]:
        """CTC 解码单个 prediction。"""
        if pred.ndim != 2 or pred.shape[1] != len(self.character):
            raise ValueError(f"Unexpected OCR CTC output shape: {pred.shape}; expected (*, {len(self.character)})")
        idx = pred.argmax(axis=1)
        prob = pred.max(axis=1)
        # collapse consecutive duplicates
        selection = np.ones(len(idx), dtype=bool)
        selection[1:] = idx[1:] != idx[:-1]
        # remove blank (index 0)
        selection &= idx != 0
        chars = [self.character[i] for i in idx[selection]]
        text = "".join(chars)
        conf = float(prob[selection].mean()) if selection.any() else 1.0
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
    普通文字和印章分别使用显式检测配置，识别器与字符表共用。
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        dict_path: str = str(PPOCRV6_DICT_PATH),
        device: Optional[str] = None,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.5,
        enable_merge_det_boxes: bool = True,
        drop_score: float = 0.5,
        rec_batch_num: int = 6,
        intra_op_num_threads: int = 0,
        lang: Literal["ch", "seal"] = "ch",
    ) -> None:
        """按普通文字或印章模式初始化 CPU 检测与识别模型。"""
        if lang not in {"ch", "seal"}:
            raise ValueError(f"Unsupported ONNX OCR mode: {lang}")
        self.lang = lang
        self.is_seal = lang == "seal"
        self.device = "cpu"
        self.text_detector = TextDetectorONNX(
            model_path=det_model_path,
            device=device,
            limit_side_len=736 if self.is_seal else 960,
            limit_type="min" if self.is_seal else "max",
            thresh=0.2 if self.is_seal else 0.3,
            box_thresh=0.6 if self.is_seal else det_db_box_thresh,
            unclip_ratio=0.5 if self.is_seal else det_db_unclip_ratio,
            box_type="poly" if self.is_seal else "quad",
            use_dilation=False,
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
        self.drop_score = 0.0 if self.is_seal else drop_score
        self.enable_merge_det_boxes = enable_merge_det_boxes and not self.is_seal
        self._seal_sort_boxes = SortPolyBoxes()
        self._seal_crop_by_polys = CropByPolys(det_box_type="poly")

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
        """保持统一 OCR 调用协议，并按模式执行检测、裁剪和识别。"""
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
                    dt_boxes = self._seal_sort_boxes(dt_boxes) if self.is_seal else sorted_boxes(dt_boxes)
                    if self.enable_merge_det_boxes:
                        dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res and not self.is_seal:
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

    def _det_rec(
        self,
        img: np.ndarray,
        mfd_res: Optional[List[dict]] = None,
    ) -> tuple[DetectionBoxes, list[tuple[str, float]]]:
        """det + crop + rec 的完整流程。"""
        ori_im = img
        dt_boxes, _elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return np.array([]), []

        if self.is_seal:
            dt_boxes = self._seal_sort_boxes(dt_boxes)
            img_crop_list = self._seal_crop_by_polys(ori_im, dt_boxes)
        else:
            dt_boxes = sorted_boxes(dt_boxes)
            if self.enable_merge_det_boxes:
                dt_boxes = merge_det_boxes(dt_boxes)
            if mfd_res:
                dt_boxes = update_det_boxes(dt_boxes, mfd_res)
            img_crop_list = []
            crop_boxes = []
            for box in dt_boxes:
                crop = get_rotate_crop_image_for_text_rec(ori_im, copy.deepcopy(box))
                if crop is not None:
                    img_crop_list.append(crop)
                    crop_boxes.append(box)
            dt_boxes = crop_boxes

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
        return filter_boxes, filter_rec_res

    def __call__(
        self,
        img: np.ndarray,
        mfd_res: Optional[List[dict]] = None,
    ) -> tuple[DetectionBoxes | None, list[tuple[str, float]] | None]:
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

    parser = argparse.ArgumentParser(description="PP-OCRv6 ONNX local inference smoke test")
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--det", default=None, help="Path to det inference.onnx.")
    parser.add_argument("--rec", default=None, help="Path to rec inference.onnx.")
    parser.add_argument("--dict", default=None, help="Path to character dict file.")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="ONNX Runtime CPU.")
    parser.add_argument("--output", default=None, help="Save result JSON to this path.")
    args = parser.parse_args()

    from ..registry import MINERU_4_MODELS_ONNX

    args.det = args.det or str(MINERU_4_MODELS_ONNX.ocr_det.ensure())
    args.rec = args.rec or str(MINERU_4_MODELS_ONNX.ocr_rec.ensure())
    args.dict = args.dict or str(PPOCRV6_DICT_PATH)

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
