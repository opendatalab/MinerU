from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mineru.model.mfr import utils as mfr_utils
from mineru.model.mfr.pp_formulanet.processors import UniMERNetDecode


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\mathring \mathrm { A }", r"\mathring{\mathrm { A }}"),
        (r"\mathring\mathrm{A}", r"\mathring{\mathrm{A}}"),
        (r"\mathring{\mathrm{A}}", r"\mathring{\mathrm{A}}"),
        (r"\mathring {\mathrm{A}}", r"\mathring {\mathrm{A}}"),
        (r"\mathring A", r"\mathring A"),
        (r"\mathrm{A}", r"\mathrm{A}"),
    ],
)
def test_fix_mathring_font_arguments_is_narrow_and_idempotent(source: str, expected: str) -> None:
    """验证 mathring 只修复未分组 mathrm 参数，并保持合法输入幂等。"""
    fixed = mfr_utils.fix_mathring_font_arguments(source)

    assert fixed == expected
    assert mfr_utils.fix_mathring_font_arguments(fixed) == expected


def test_fix_mathring_font_arguments_repairs_multiple_occurrences() -> None:
    """验证同一公式中的多个未分组 mathring 参数均会修复。"""
    source = r"\mathring \mathrm{A} + \mathring\mathrm { B }"

    assert mfr_utils.fix_mathring_font_arguments(source) == (r"\mathring{\mathrm{A}} + \mathring{\mathrm { B }}")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\left x \right y", r"\left x \right y"),
        (r"\Dot{x}", r"\Dot{x}"),
        (r"{x", r"{x"),
        (r"\upalpha + \emph{x}", r"\alpha + {x}"),
    ],
)
def test_pp_formulanet_latex_repairs_preserve_current_output(source: str, expected: str) -> None:
    """验证 PP-FormulaNet 保留分隔符、命令和大括号的既有修复行为。"""
    assert mfr_utils.fix_pp_formulanet_latex(source) == expected


def test_pp_formulanet_fix_latex_uses_shared_mathring_repair() -> None:
    """验证 PP-FormulaNet 的真实 processor 入口应用共享修复。"""
    source = r"R , { \mathring \mathrm { A } }"

    tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens: source)
    processor = SimpleNamespace(tokenizer=tokenizer)
    assert UniMERNetDecode.token2str(processor, [np.array([0, 7, 2, 1])]) == [r"R , { \mathring{\mathrm { A }} }"]


def test_latex_rm_whitespace_name_is_removed() -> None:
    """验证旧职责不符名称不再作为兼容入口保留。"""
    assert not hasattr(mfr_utils, "latex_rm_whitespace")
