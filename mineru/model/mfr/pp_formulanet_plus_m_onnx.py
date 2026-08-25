# Copyright (c) Opendatalab. All rights reserved.
"""ONNX 后端的 PP-FormulaNet-Plus-M 推理封装。

与 ``UnimernetModel``（torch/transformers 后端）公开接口完全一致，
内部用 onnxruntime 推理 RapidDoc 转出的 PP-FormulaNet_plus-M ONNX 模型。

ONNX 模型输出 token IDs（int64），不是 logits——自回归循环已 bake 进计算图，
一次前向推理即可得到完整 LaTeX token 序列。

预处理/后处理参数与 RapidDoc ``inference.yml`` 一致。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

from ..utils.onnxruntime_provider import ort_session
from .post_process import post_process_formula

__all__ = ["PPFormulaNetPlusMONNX"]

# 预处理常量（与 inference.yml 一致）
_INPUT_SIZE = (384, 384)
_MEAN = np.array([0.7931, 0.7931, 0.7931], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.1738, 0.1738, 0.1738], dtype=np.float32).reshape(1, 1, 3)

# tokenizer 特殊 token IDs
_BOS_TOKEN_ID = 0
_PAD_TOKEN_ID = 1
_EOS_TOKEN_ID = 2


class PPFormulaNetPlusMONNX:
    """PP-FormulaNet-Plus-M 的 ONNX 推理封装。

    与 ``UnimernetModel`` 的 ``predict`` / ``batch_predict`` 接口完全一致。

    差异：
    - 构造函数接收 onnx 文件路径 + yml 配置路径，而非 HF 模型目录
    - 不依赖 torch / transformers，仅用 onnxruntime + numpy + cv2
    - 模型输出 token IDs（非 logits），自回归循环已 bake 进 ONNX 图
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[str] = None,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.device = device
        self.session = ort_session(model_path, device, intra_op_num_threads)
        self.input_name = self.session.get_inputs()[0].name

        # 从 yml 加载 tokenizer
        with open(config_path, encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        char_dict = yml["PostProcess"]["character_dict"]

        from tokenizers import Tokenizer as TokenizerFast

        fast_str = json.dumps(char_dict["fast_tokenizer_file"])
        self.tokenizer = TokenizerFast.from_buffer(fast_str.encode("utf-8"))

        logger.debug(
            "PPFormulaNetPlusMONNX loaded: {} (config={})",
            Path(model_path).name,
            Path(config_path).name,
        )

    # ------------------------------------------------------------------
    # bbox 工具（与 UnimernetModel._normalize_bbox 一致）
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_bbox(bbox: Any, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None:
            return None
        xmin, ymin, xmax, ymax = [float(v) for v in bbox]
        xmin = math.floor(xmin)
        ymin = math.floor(ymin)
        xmax = math.ceil(xmax)
        ymax = math.ceil(ymax)
        height, width = image.shape[:2]
        xmin = max(0, min(width, xmin))
        xmax = max(0, min(width, xmax))
        ymin = max(0, min(height, ymin))
        ymax = max(0, min(height, ymax))
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _item_to_bbox(item: dict, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        return PPFormulaNetPlusMONNX._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(
        self, mfd_res: list, image: np.ndarray, interline_enable: bool = True
    ) -> Tuple[List[dict], List[Tuple[dict, Tuple[int, int, int, int]]]]:
        formula_list = []
        crop_targets = []

        for item in mfd_res or []:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label not in ["inline_formula", "display_formula"]:
                continue
            if not interline_enable and label == "display_formula":
                continue

            new_item = dict(item)
            new_item.setdefault("latex", "")
            formula_list.append(new_item)

            bbox = self._item_to_bbox(new_item, image)
            if bbox is not None:
                crop_targets.append((new_item, bbox))

        return formula_list, crop_targets

    # ------------------------------------------------------------------
    # 预处理（与 RapidDoc PPPreProcess 一致）
    # ------------------------------------------------------------------
    @staticmethod
    def _crop_margin(pil_img: Image.Image) -> Image.Image:
        data = np.array(pil_img.convert("L"))
        max_val, min_val = data.max(), data.min()
        if max_val == min_val:
            return pil_img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        if coords is None:
            return pil_img
        a, b, w, h = cv2.boundingRect(coords)
        return pil_img.crop((a, b, w + a, h + b))

    @staticmethod
    def _img_decode(img: np.ndarray) -> Optional[np.ndarray]:
        pil = Image.fromarray(img).convert("RGB")
        pil = PPFormulaNetPlusMONNX._crop_margin(pil)
        if pil.height == 0 or pil.width == 0:
            return None

        # resize: shortest edge to min(input_size), thumbnail to max
        size = min(_INPUT_SIZE)
        if pil.height <= pil.width:
            new_w = int(size * pil.width / pil.height)
            pil = pil.resize((new_w, size), resample=Image.BILINEAR)
        else:
            new_h = int(size * pil.height / pil.width)
            pil = pil.resize((size, new_h), resample=Image.BILINEAR)
        pil.thumbnail((_INPUT_SIZE[1], _INPUT_SIZE[0]))

        delta_w = _INPUT_SIZE[1] - pil.width
        delta_h = _INPUT_SIZE[0] - pil.height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return np.array(ImageOps.expand(pil, padding))

    @staticmethod
    def _transform(img: np.ndarray) -> np.ndarray:
        img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squeezed = np.squeeze(gray)
        return cv2.merge([squeezed, squeezed, squeezed])

    @staticmethod
    def _format_image(img: np.ndarray) -> np.ndarray:
        im_h, im_w = img.shape[:2]
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = img[:, :, 0]
        img = np.pad(img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1))
        img = img[:, :, np.newaxis].transpose(2, 0, 1)
        return img[np.newaxis, :]

    @classmethod
    def _preprocess(cls, img: np.ndarray) -> Optional[np.ndarray]:
        decoded = cls._img_decode(img)
        if decoded is None:
            return None
        return cls._format_image(cls._transform(decoded))

    # ------------------------------------------------------------------
    # 后处理（tokenizer decode）
    # ------------------------------------------------------------------
    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        if token_ids.ndim == 2:
            ids = [int(x) for x in token_ids[0].tolist()]
        else:
            ids = [int(x) for x in token_ids.tolist()]

        # 截断到 eos token
        for i, tid in enumerate(ids):
            if tid == _EOS_TOKEN_ID:
                ids = ids[: i + 1]
                break

        raw_text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return post_process_formula(raw_text)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _infer_batch(self, crops: List[np.ndarray]) -> List[str]:
        """对一批裁剪图做推理，返回 LaTeX 字符串列表。"""
        if not crops:
            return []

        results: List[str] = [""] * len(crops)
        valid_indices: List[int] = []
        valid_inputs: List[np.ndarray] = []

        for i, crop in enumerate(crops):
            inp = self._preprocess(crop)
            if inp is not None:
                valid_indices.append(i)
                valid_inputs.append(inp)

        if not valid_inputs:
            return results

        with tqdm(total=len(valid_inputs), desc="MFR Predict") as pbar:
            for i, inp in enumerate(valid_inputs):
                preds = self.session.run(None, {self.input_name: inp.astype(np.float32)})[0]
                latex = self._decode_tokens(preds)
                results[valid_indices[i]] = latex
                pbar.update(1)

        return results

    # ------------------------------------------------------------------
    # 公开接口（与 UnimernetModel 一致）
    # ------------------------------------------------------------------
    def predict(
        self,
        mfd_res: list,
        image: np.ndarray,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        return self.batch_predict(
            [mfd_res],
            [image],
            batch_size=batch_size,
            interline_enable=interline_enable,
        )[0]

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        if not images_mfd_res:
            return []

        if len(images_mfd_res) != len(images):
            raise ValueError("images_mfd_res and images must have the same length.")

        images_formula_list: List[List[dict]] = []
        all_crops: List[np.ndarray] = []
        crop_to_formula: List[Tuple[int, int]] = []  # (page_idx, formula_idx_in_page)

        for page_idx, (mfd_res, image) in enumerate(zip(images_mfd_res, images)):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res,
                image,
                interline_enable=interline_enable,
            )

            for formula_idx, (formula_item, (xmin, ymin, xmax, ymax)) in enumerate(crop_targets):
                bbox_img = image[ymin:ymax, xmin:xmax]
                all_crops.append(bbox_img)
                crop_to_formula.append((page_idx, len(images_formula_list), formula_idx))

            images_formula_list.append(formula_list)

        if not all_crops:
            return images_formula_list

        # 批量推理
        latex_results = self._infer_batch(all_crops)

        # 回填 latex
        for (page_idx, _, formula_idx), latex in zip(crop_to_formula, latex_results):
            images_formula_list[page_idx][formula_idx]["latex"] = latex

        return images_formula_list
