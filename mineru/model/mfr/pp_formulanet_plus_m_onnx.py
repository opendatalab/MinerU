# Copyright (c) Opendatalab. All rights reserved.
"""ONNX 后端的 PP-FormulaNet-Plus-M 推理封装。

与 ``UnimernetModel``（torch/transformers 后端）公开接口完全一致，
内部用 onnxruntime CPU 推理从 MinerU Torch 权重导出的 PP-FormulaNet_plus-M ONNX 模型。

ONNX 模型输出 token IDs（int64），不是 logits——自回归循环已 bake 进计算图，
一次前向推理即可得到完整 LaTeX token 序列。

预处理/后处理参数与 RapidDoc ``inference.yml`` 一致。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from ..runtime.onnx import ort_session
from .pp_formulanet.processors import LatexImageFormat, UniMERNetDecode, UniMERNetImgDecode, UniMERNetTestTransform

__all__ = ["PPFormulaNetPlusMONNX", "CPU_BATCH_SIZE"]

CPU_BATCH_SIZE = 8

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
        """初始化 CPU 公式模型，并复用 Torch Plus-M 的纯 Python 处理器。"""
        self.device = "cpu"
        self.session = ort_session(model_path, device, intra_op_num_threads)
        self.input_name = self.session.get_inputs()[0].name

        # 从 yml 加载 tokenizer
        with open(config_path, encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        char_dict = yml["PostProcess"]["character_dict"]

        self.decoder = UniMERNetDecode(character_list=char_dict)
        self.tokenizer = self.decoder.tokenizer
        self.image_decoder = UniMERNetImgDecode(input_size=(384, 384))
        self.image_transform = UniMERNetTestTransform()
        self.image_formatter = LatexImageFormat()

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
        """裁剪公式坐标到有效图像范围。"""
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
        """读取并规范化公式条目的坐标。"""
        return PPFormulaNetPlusMONNX._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(
        self, mfd_res: list, image: np.ndarray, interline_enable: bool = True
    ) -> Tuple[List[dict], List[Tuple[dict, Tuple[int, int, int, int]]]]:
        """复制有效类别的公式，只有可裁剪条目进入推理任务。"""
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

    def _preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """复用现有 Plus-M 去边、缩放、灰度归一化和张量格式化。"""
        if img.size == 0:
            return None
        decoded = self.image_decoder.img_decode(img)
        if decoded is None:
            return None
        return self.image_formatter.format(self.image_transform.transform(decoded))

    # ------------------------------------------------------------------
    # 后处理（tokenizer decode）
    # ------------------------------------------------------------------
    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """在首个 EOS 截断并调用共享 Plus-M tokenizer 与公式修复。"""
        if token_ids.ndim == 2:
            ids = [int(x) for x in token_ids[0].tolist()]
        else:
            ids = [int(x) for x in token_ids.tolist()]

        # 截断到 eos token
        for i, tid in enumerate(ids):
            if tid == _EOS_TOKEN_ID:
                ids = ids[: i + 1]
                break

        return self.decoder(np.asarray([ids], dtype=np.int64))[0]

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _infer_batch(self, crops: List[np.ndarray], batch_size: int = CPU_BATCH_SIZE) -> List[str]:
        """按原图面积排序，以最多 8 张的 CPU 批次推理，再恢复原顺序。"""
        if not crops:
            return []

        results: List[str] = [""] * len(crops)
        valid_indices: List[int] = []
        valid_inputs: List[np.ndarray] = []

        ordered_indices = sorted(range(len(crops)), key=lambda i: (crops[i].shape[0] * crops[i].shape[1], i))
        for i in ordered_indices:
            inp = self._preprocess(crops[i])
            if inp is not None:
                valid_indices.append(i)
                valid_inputs.append(inp)

        if not valid_inputs:
            return results

        with tqdm(total=len(valid_inputs), desc="MFR Predict") as pbar:
            step = min(CPU_BATCH_SIZE, max(1, batch_size))
            for start in range(0, len(valid_inputs), step):
                inputs = np.concatenate(valid_inputs[start : start + step], axis=0).astype(np.float32)
                preds = self.session.run(None, {self.input_name: inputs})[0]
                if preds.ndim != 2 or preds.shape[0] != inputs.shape[0]:
                    raise ValueError("Formula ONNX output batch does not match input batch")
                for offset, tokens in enumerate(preds):
                    results[valid_indices[start + offset]] = self._decode_tokens(tokens)
                pbar.update(inputs.shape[0])

        return results

    # ------------------------------------------------------------------
    # 公开接口（与 UnimernetModel 一致）
    # ------------------------------------------------------------------
    def predict(
        self,
        mfd_res: list,
        image: np.ndarray,
        batch_size: int = CPU_BATCH_SIZE,
        interline_enable: bool = True,
    ) -> list:
        """识别单页公式并保留输入公式的顺序。"""
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
        batch_size: int = CPU_BATCH_SIZE,
        interline_enable: bool = True,
    ) -> list:
        """按 CPU batch=8 上限跨页识别公式，并按原目标回填，避免无效框导致错位。"""
        if not images_mfd_res:
            return []

        if len(images_mfd_res) != len(images):
            raise ValueError("images_mfd_res and images must have the same length.")

        images_formula_list: List[List[dict]] = []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        all_crops: List[np.ndarray] = []
        crop_to_formula: list[dict] = []

        for page_idx, (mfd_res, image) in enumerate(zip(images_mfd_res, images)):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res,
                image,
                interline_enable=interline_enable,
            )

            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                all_crops.append(bbox_img)
                crop_to_formula.append(formula_item)

            images_formula_list.append(formula_list)

        if not all_crops:
            return images_formula_list

        # 批量推理
        latex_results = self._infer_batch(all_crops, batch_size=batch_size)

        # 回填 latex
        for formula_item, latex in zip(crop_to_formula, latex_results):
            formula_item["latex"] = latex

        return images_formula_list
