# Copyright (c) Opendatalab. All rights reserved.
import math
import os
import zipfile
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from loguru import logger

from ..._internal.pytorchocr.base_ocr_v20 import BaseOCRV20

from ..utils import build_mfr_batch_groups
from .processors import (
    LatexImageFormat,
    ToBatch,
    UniMERNetDecode,
    UniMERNetImgDecode,
    UniMERNetTestTransform,
)


# MFR 独立精度策略：auto 为 CPU fp32、非 CPU fp16；可改为 fp32 做精度对照。
MFR_INFERENCE_PRECISION = "auto"


class FormulaRecognizer(BaseOCRV20):
    def __init__(
        self,
        weight_dir,
        device="cpu",
        *,
        model_name="PP-FormulaNet_plus-M",
    ):
        """按明确的模型变体加载权重和配置，默认保持 plus-M 兼容。"""
        variants = {"PP-FormulaNet_plus-M", "PP-FormulaNet_plus-S"}
        if model_name not in variants:
            raise ValueError(f"Unsupported formula model: {model_name}")
        self.model_name = model_name
        self.weights_path = os.path.join(
            weight_dir,
            f"{model_name}.pth",
        )
        self.yaml_path = str(
            Path(__file__).resolve().parents[2]
            / "_internal"
            / "pytorchocr"
            / "utils"
            / "resources"
            / "pp_formulanet_arch_config.yaml"
        )
        self.infer_yaml_path = os.path.join(
            weight_dir,
            f"{model_name}_inference.yml",
        )

        # 同一配置文件按模型名选择完整架构，避免 S/M 参数混用。
        with open(self.yaml_path, encoding="utf-8") as config_file:
            network_config = yaml.safe_load(config_file)[model_name]["Architecture"]
        # 新版 ZIP 权重使用 mmap；旧版 pickle 权重仍支持普通读取。
        weights = torch.load(
            self.weights_path, map_location="cpu", weights_only=True,
            mmap=zipfile.is_zipfile(self.weights_path),
        )
        # 只创建结构，避免父类临时解码器及随机初始化占用实际内存。
        with torch.device("meta"):
            super(FormulaRecognizer, self).__init__(network_config)
        self.net.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.device = torch.device(device) if isinstance(device, str) else device
        self.ocr_inference_dtype = self._resolve_inference_dtype(
            self.device, precision_override=MFR_INFERENCE_PRECISION
        )
        self.net.to(device=self.device, dtype=self.ocr_inference_dtype)
        self.net.eval()
        if model_name == "PP-FormulaNet_plus-M" and self.device.type == "cpu":
            # CPU 仅使用增量缓存，attention 仍保持原有 eager 算子。
            self.net.head.use_growing_cache = True
        if model_name == "PP-FormulaNet_plus-M" and self.device.type in {"mps", "cuda"}:
            # 仅为经过适配的公式解码器启用快速 attention；CPU 和其他模型保持原路径。
            self.net.head.set_fast_attention()
        logger.info(
            "MFR loaded: model={}, device={}, dtype={}, attention={}, cache={}",
            model_name, self.device, self.ocr_inference_dtype,
            "sdpa" if getattr(self.net.head.decoder.model.decoder.layers[0].self_attn, "use_sdpa", False) else "eager",
            "growing" if getattr(self.net.head, "use_growing_cache", False) else "concat",
        )

        with open(self.infer_yaml_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.pre_tfs = {
            "UniMERNetImgDecode": UniMERNetImgDecode(input_size=(384, 384)),
            "UniMERNetTestTransform": UniMERNetTestTransform(
                paddle_compatible=model_name == "PP-FormulaNet_plus-S"
            ),
            "LatexImageFormat": LatexImageFormat(),
            "ToBatch": ToBatch(),
        }

        self.post_op = UniMERNetDecode(
            character_list=data["PostProcess"]["character_dict"]
        )

    @staticmethod
    def _normalize_bbox(bbox, image):
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
    def _item_to_bbox(item, image):
        return FormulaRecognizer._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(self, mfd_res, image, interline_enable=True):
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

    def predict(
        self,
        mfd_res,
        image,
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
        """按原有面积分组逐批预处理和识别，并保持原页序与检测字段回填。"""
        if not images_mfd_res:
            return []

        if len(images_mfd_res) != len(images):
            raise ValueError("images_mfd_res and images must have the same length.")

        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []

        for mfd_res, image in zip(images_mfd_res, images):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res,
                image,
                interline_enable=interline_enable,
            )

            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)
                backfill_list.append(formula_item)

            images_formula_list.append(formula_list)

        if not image_info:
            return images_formula_list

        image_info.sort(key=lambda x: x[0])
        sorted_areas = [x[0] for x in image_info]
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]
        index_mapping = {
            new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)
        }

        # Plus-S/M 统一直接使用请求上限，后续按面积动态缩小批次。
        formula_requested_batch_size = max(1, batch_size)
        batch_groups = build_mfr_batch_groups(
            sorted_areas,
            formula_requested_batch_size,
        )
        # 公共分组器可能将不足 16 张的尾部整体收下；公式网络严格遵守实际 batch 上限。
        batch_groups = [
            group[start:start + formula_requested_batch_size]
            for group in batch_groups
            for start in range(0, len(group), formula_requested_batch_size)
        ]
        # 参照公共分组器合并较小尾批，Plus-S/M 均允许合并到请求上限。
        # 保留至少两组，避免样本不足请求 batch 时破坏公共分组器的拆分语义。
        while (
            len(batch_groups) >= 3
            and len(batch_groups[-1]) < len(batch_groups[-2])
            and len(batch_groups[-2]) + len(batch_groups[-1]) <= formula_requested_batch_size
        ):
            tail_group = batch_groups.pop()
            batch_groups[-1].extend(tail_group)

        rec_formula = []
        with torch.inference_mode():
            with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar:
                for batch_group in batch_groups:
                    # CPU 和设备端均只保留当前批次的归一化输入。
                    batch_imgs = self.pre_tfs["UniMERNetImgDecode"](
                        imgs=sorted_images[batch_group[0]:batch_group[-1] + 1]
                    )
                    batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
                    batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)
                    inp = self.pre_tfs["ToBatch"](imgs=batch_imgs)[0]
                    del batch_imgs
                    batch_data = torch.from_numpy(inp).to(
                        device=self.device, dtype=self.ocr_inference_dtype
                    )
                    batch_preds = self.net(batch_data).cpu().numpy()
                    rec_formula += self.post_op(batch_preds)
                    del inp, batch_data, batch_preds
                    pbar.update(len(batch_group))

        unsorted_results = [""] * len(rec_formula)
        for new_idx, latex in enumerate(rec_formula):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
