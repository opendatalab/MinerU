from __future__ import annotations

import pytest
import torch

from mineru.model.mfr import utils as mfr_utils
from mineru.model.mfr.pp_formulanet_plus_m.processors import UniMERNetDecode
from mineru.model.mfr.unimernet.unimernet_hf.modeling_unimernet import UnimernetModel


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

    assert mfr_utils.fix_mathring_font_arguments(source) == (
        r"\mathring{\mathrm{A}} + \mathring{\mathrm { B }}"
    )


@pytest.mark.parametrize(
    ("source", "pp_expected", "unimernet_expected"),
    [
        (r"\left x \right y", r"\left x \right y", r"\left. x \right. y"),
        (r"\Dot{x}", r"\Dot{x}", r"\dot{x}"),
        (r"{x", r"{x", r"x"),
        (r"\upalpha + \emph{x}", r"\alpha + {x}", r"\alpha + {x}"),
    ],
)
def test_model_specific_latex_repairs_keep_existing_differences(
    source: str,
    pp_expected: str,
    unimernet_expected: str,
) -> None:
    """验证共享重构不会把任一模型的专属修复带入另一模型。"""
    assert mfr_utils.fix_pp_formulanet_latex(source) == pp_expected
    assert mfr_utils.fix_unimernet_latex(source) == unimernet_expected


def test_pp_formulanet_fix_latex_uses_shared_mathring_repair() -> None:
    """验证 PP-FormulaNet 的真实 processor 入口应用共享修复。"""
    source = r"R , { \mathring \mathrm { A } }"

    assert UniMERNetDecode.fix_latex(None, source) == r"R , { \mathring{\mathrm { A }} }"


def test_unimernet_decode_entry_uses_renamed_latex_repair() -> None:
    """验证 UniMERNet 解码入口通过新名称应用共享修复。"""

    class _TokenizerStub:
        """为轻量解码入口返回固定 LaTeX 文本。"""

        @staticmethod
        def token2str(_token_ids: object) -> list[str]:
            """返回含未分组 mathring 参数的模型文本。"""
            return [r"R , { \mathring \mathrm { A } }"]

    class _ModelStub:
        """仅提供解码方法实际访问的 tokenizer。"""

        tokenizer = _TokenizerStub()

    result = UnimernetModel._decode_generate_outputs(
        _ModelStub(),
        torch.tensor([[0, 1, 2]]),
        return_full_result=False,
    )

    assert result == {"fixed_str": [r"R , { \mathring{\mathrm { A }} }"]}


def test_latex_rm_whitespace_name_is_removed() -> None:
    """验证旧职责不符名称不再作为兼容入口保留。"""
    assert not hasattr(mfr_utils, "latex_rm_whitespace")
