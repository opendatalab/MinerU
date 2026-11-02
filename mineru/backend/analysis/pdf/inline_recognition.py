"""表外行内切片的 VLM 文本识别准备、定界符校验及局部 MFR 回退。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ....model.vlm.text_recognition import TextRecognitionResult, raise_unrecoverable_text_error
from ....utils.timing import stage_timer
from .constants import BATCH_RATIO, MFR_BASE_BATCH_SIZE
from .formulas import _build_formula_inputs, _normalize_hybrid_formula_bbox
from .tables import _build_table_external_mfr_inputs

if TYPE_CHECKING:
    from PIL.Image import Image

    from ....model.runtime.hybrid import HybridLocalModelContext

_DELIMITER = re.compile(r"\\[()\[\]]")
_TEXT_DOLLAR_ESCAPE = re.compile(r"(?<!\\)(?P<pairs>(?:\\\\)*)\\\$")


@dataclass
class _InlineCrop:
    """显式保存切片在窗口内的落点，不用识别文本反查公式。"""

    page_index: int
    formula_index: int
    image: Image


@dataclass
class InlineRecognitionBatch:
    """窗口拥有公式副本和切片资源；原始 MFD 继续独立服务 OCR 遮罩。"""

    formulas: list[list[dict[str, Any]]]
    crops: list[_InlineCrop] = field(default_factory=list)

    @property
    def images(self) -> list[Image]:
        """按稳定任务顺序交付图片，不向模型传递页面资源或裸 PDF 句柄。"""
        return [crop.image for crop in self.crops]

    def close(self) -> None:
        """在识别及回填退出后关闭当前窗口持有的全部切片。"""
        for crop in self.crops:
            crop.image.close()
        self.crops.clear()


def prepare_inline_recognition(
    images_layout_res: list[list[dict[str, Any]]],
    model_list: list[list[dict[str, Any]]],
    images: list[Image],
) -> InlineRecognitionBatch:
    """依据最终表格归属选框，按现有 MFR 的取整和裁边规则创建独立切片。"""
    if not (len(images_layout_res) == len(model_list) == len(images)):
        raise ValueError("Inline recognition page count mismatch")
    inputs = _build_table_external_mfr_inputs(
        _build_formula_inputs(images_layout_res), model_list, [image.size for image in images]
    )
    batch = InlineRecognitionBatch([[] for _ in images])
    try:
        for page_index, (formulas, image) in enumerate(zip(inputs, images, strict=True)):
            for formula in formulas:
                bbox = _normalize_hybrid_formula_bbox(formula.get("bbox"))
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                box = (
                    max(0, min(image.width, math.floor(x0))),
                    max(0, min(image.height, math.floor(y0))),
                    max(0, min(image.width, math.ceil(x1))),
                    max(0, min(image.height, math.ceil(y1))),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                crop = image.crop(box)
                batch.crops.append(_InlineCrop(page_index, len(batch.formulas[page_index]), crop))
                batch.formulas[page_index].append(formula)
        return batch
    except BaseException:
        batch.close()
        raise


def _unescape_text_dollars(text: str) -> str:
    """仅解除 VLM 普通文本中美元前的单层转义，保留成对反斜杠和其他命令。"""
    return _TEXT_DOLLAR_ESCAPE.sub(r"\g<pairs>$", text)


def normalize_inline_text(text: str | None) -> str:
    """规范公式定界符和公式外的美元转义，拒绝空或残缺响应。"""
    if text is None or not text.strip() or text.strip() == "[Non-Text]":
        raise ValueError("empty VLM inline response")
    parts: list[str] = []
    cursor = 0
    opening: re.Match[str] | None = None
    for match in _DELIMITER.finditer(text):
        # 偶数个连续反斜杠表示转义文字，不作为公式边界。
        preceding = match.start() - 1
        while preceding >= 0 and text[preceding] == "\\":
            preceding -= 1
        if (match.start() - preceding - 1) % 2:
            continue
        token = match.group()
        if token in {r"\(", r"\["}:
            if opening is not None:
                raise ValueError("nested VLM inline delimiter")
            opening = match
        else:
            expected = {r"\)": r"\(", r"\]": r"\["}[token]
            if opening is None or opening.group() != expected:
                raise ValueError("unmatched VLM inline delimiter")
            latex = text[opening.end() : match.start()].strip()
            if not latex:
                raise ValueError("empty VLM inline formula")
            parts.extend((_unescape_text_dollars(text[cursor : opening.start()]), f"\\({latex}\\)"))
            cursor = match.end()
            opening = None
    if opening is not None:
        raise ValueError("unclosed VLM inline delimiter")
    parts.append(_unescape_text_dollars(text[cursor:]))
    return "".join(parts).strip()


def finish_inline_recognition(
    batch: InlineRecognitionBatch,
    outputs: list[TextRecognitionResult],
    context: HybridLocalModelContext,
    np_images: list[np.ndarray],
    *,
    page_offset: int,
) -> list[list[dict[str, Any]]]:
    """在本地模型锁内仅对失败片段合批回退，任何二次失败均显式终止。"""
    if len(outputs) != len(batch.crops):
        raise ValueError(f"Inline recognition count mismatch: inputs={len(batch.crops)}, outputs={len(outputs)}")
    fallback_inputs: list[list[dict[str, Any]]] = [[] for _ in batch.formulas]
    failed_crops: list[list[_InlineCrop]] = [[] for _ in batch.formulas]
    for crop, output in zip(batch.crops, outputs, strict=True):
        formula = batch.formulas[crop.page_index][crop.formula_index]
        try:
            if isinstance(output, Exception):
                raise_unrecoverable_text_error(output)
                raise output
            content = normalize_inline_text(output)
        except Exception as error:
            raise_unrecoverable_text_error(error)
            logger.warning(
                "Inline VLM fallback to MFR: page={}, formula_index={}, bbox={}, reason={}: {}",
                page_offset + crop.page_index + 1,
                crop.formula_index,
                formula["bbox"],
                type(error).__name__,
                error,
            )
            fallback_inputs[crop.page_index].append(dict(formula))
            failed_crops[crop.page_index].append(crop)
        else:
            formula["content"] = content
            formula["content_format"] = "text_with_inline_math"
    fallback_count = sum(len(page) for page in failed_crops)
    if fallback_count:
        with stage_timer("pdf.inline_mfr_fallback"):
            fallback_results = context.mfr_model.batch_predict(
                fallback_inputs,
                np_images,
                batch_size=BATCH_RATIO * MFR_BASE_BATCH_SIZE,
                interline_enable=False,
            )
        if len(fallback_results) != len(failed_crops):
            raise ValueError("Inline MFR fallback page count mismatch")
        for crops, results in zip(failed_crops, fallback_results, strict=True):
            if len(crops) != len(results):
                raise ValueError("Inline MFR fallback result count mismatch")
            for crop, result in zip(crops, results, strict=True):
                formula = batch.formulas[crop.page_index][crop.formula_index]
                if result.get("bbox") != formula["bbox"] or result.get("label") != "inline_formula":
                    raise ValueError("Inline MFR fallback result identity mismatch")
                latex = result.get("latex")
                if not isinstance(latex, str) or not latex.strip():
                    raise ValueError(
                        f"Empty inline MFR fallback: page={page_offset + crop.page_index + 1}, bbox={formula['bbox']}"
                    )
                formula["latex"] = latex.strip()
    if batch.crops:
        logger.info(
            "Inline recognition: total={}, vlm_success={}, mfr_fallback={}",
            len(batch.crops),
            len(batch.crops) - fallback_count,
            fallback_count,
        )
    return batch.formulas


__all__ = ["InlineRecognitionBatch", "prepare_inline_recognition", "finish_inline_recognition", "normalize_inline_text"]
