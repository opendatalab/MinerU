# Copyright (c) Opendatalab. All rights reserved.
"""公式识别后处理的公共逻辑。

从 ``pp_formulanet_plus_m/processors.py`` 的 ``UniMERNetDecode`` 类中提取，
供 torch 版和 ONNX 版共同复用，避免代码重复。
"""

from __future__ import annotations

import re

from ftfy import fix_text

from .utils import fix_pp_formulanet_latex

_CHINESE_TEXT_WRAPPING_PATTERN = re.compile(r"\\text\s*{\s*([^}]*?[\u4e00-\u9fff]+[^}]*?)\s*}")


def remove_chinese_text_wrapping(formula: str) -> str:
    """去掉中文文字外的 ``\\text{}`` 包裹。

    与 ``UniMERNetDecode.remove_chinese_text_wrapping`` 行为一致。
    """
    replaced = _CHINESE_TEXT_WRAPPING_PATTERN.sub(lambda m: m.group(1), formula)
    return replaced.replace('"', "")


def post_process_formula(text: str) -> str:
    """公式文本的完整后处理流程。

    与 ``UniMERNetDecode.post_process`` 行为一致：
    1. 去掉中文 ``\\text{}`` 包裹
    2. ftfy 修复 Unicode
    3. 修复 LaTeX 格式
    """
    text = remove_chinese_text_wrapping(text)
    text = fix_text(text)
    text = fix_pp_formulanet_latex(text)
    return text
