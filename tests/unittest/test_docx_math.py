from __future__ import annotations

import pytest
from lxml import etree

from mineru.render._internal.docx.math import DocxFormulaError, latex_to_omml, split_formula_tag

_OFFICE_MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (r"x + y\tag{1}", ("x + y", "1")),
        (r"x + y\tag { A_{n} }  ", ("x + y", "A_{n}")),
        (r"x + y\tag{\mathrm{A}_{n}}", ("x + y", r"\mathrm{A}_{n}")),
        (r"x + y\tag{\{1\}}", ("x + y", r"\{1\}")),
    ],
)
def test_split_formula_tag_strips_only_balanced_terminal_tag(
    content: str,
    expected: tuple[str, str | None],
) -> None:
    """验证末端 tag 支持空白、嵌套花括号和转义花括号。"""
    assert split_formula_tag(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        r"x + y\tag{1} + z",
        r"x + y\tag{1",
        r"x + y\tag{1}}",
        r"x + y\tagged{1}",
        r"x + y\\tag{1}",
        "plain formula  ",
    ],
)
def test_split_formula_tag_preserves_non_terminal_or_malformed_content(content: str) -> None:
    """验证非末端、损坏或并非命令的 tag 文本不会被误剥离。"""
    assert split_formula_tag(content) == (content, None)


def test_latex_to_omml_returns_inline_equation_with_bound_namespace() -> None:
    """验证行内公式返回可独立序列化的 m:oMath 节点。"""
    equation = latex_to_omml(r"x^2 + \frac{a}{b}", display=False)

    assert equation.tag == etree.QName(_OFFICE_MATH_NAMESPACE, "oMath")
    assert equation.getparent() is None
    assert equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}f") is not None
    reparsed = etree.fromstring(etree.tostring(equation))
    assert reparsed.tag == equation.tag


def test_latex_to_omml_wraps_display_matrix_in_math_paragraph() -> None:
    """验证块公式以 m:oMathPara 包装，并保留矩阵 OMML。"""
    paragraph = latex_to_omml(r"\begin{matrix}a&b\\c&d\end{matrix}", display=True)

    assert paragraph.tag == etree.QName(_OFFICE_MATH_NAMESPACE, "oMathPara")
    equations = paragraph.findall(f"{{{_OFFICE_MATH_NAMESPACE}}}oMath")
    assert len(equations) == 1
    assert equations[0].find(f".//{{{_OFFICE_MATH_NAMESPACE}}}m") is not None


@pytest.mark.parametrize("latex", [r"\bar p", r"\vec{u}"])
def test_latex_to_omml_repairs_group_character_property_closing_tag(latex: str) -> None:
    """验证横线和向量符号不会因第三方库的错误闭合标签退化为文本。"""
    equation = latex_to_omml(latex, display=False)

    assert equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}groupChr") is not None
    etree.fromstring(etree.tostring(equation))


def test_latex_to_omml_hides_square_root_degree_placeholder() -> None:
    """验证普通平方根包含隐藏 degree，避免 Word/LibreOffice 显示占位框。"""
    equation = latex_to_omml(r"\sqrt{x}", display=False)

    radical = equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}rad")
    assert radical is not None
    assert radical.find(f"{{{_OFFICE_MATH_NAMESPACE}}}deg") is not None
    degree_hidden = radical.find(f"{{{_OFFICE_MATH_NAMESPACE}}}radPr/{{{_OFFICE_MATH_NAMESPACE}}}degHide")
    assert degree_hidden is not None
    assert degree_hidden.get(f"{{{_OFFICE_MATH_NAMESPACE}}}val") == "1"


@pytest.mark.parametrize("latex", [r"^{2}", r"_{0}"])
def test_latex_to_omml_uses_zero_width_script_base(latex: str) -> None:
    """验证无显式底数的上下标使用零宽字符抑制可见方框。"""
    equation = latex_to_omml(latex, display=False)
    base = equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}sSup/{{{_OFFICE_MATH_NAMESPACE}}}e")
    if base is None:
        base = equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}sSub/{{{_OFFICE_MATH_NAMESPACE}}}e")

    assert base is not None
    assert "\u200b" in "".join(base.itertext())


def test_latex_to_omml_removes_explicitly_empty_operator_limits() -> None:
    """验证空上下限不会在积分或其他算子旁生成可见占位框。"""
    equation = latex_to_omml(r"\int_{}^{} x + E_{}", display=False)
    serialized = etree.tostring(equation, encoding="unicode")

    assert "<m:t></m:t>" not in serialized
    assert "<m:sub>" not in serialized and "<m:sup>" not in serialized


def test_latex_to_omml_normalizes_no_bar_genfrac_in_binomial_formula() -> None:
    """验证 Office 无横线分式可转为带定界符的双行 OMML matrix。"""
    formula = (
        r"\left(x+a\right)^{n}=\sum_{k=0}^{n}"
        r"\left(\genfrac{}{}{0pt}{}{n}{k}\right)x^{k}a^{n-k}"
    )

    equation = latex_to_omml(formula, display=False)
    matrix = equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}m")
    delimiter = equation.find(f".//{{{_OFFICE_MATH_NAMESPACE}}}d")

    assert matrix is not None
    assert delimiter is not None
    rows = matrix.findall(f"{{{_OFFICE_MATH_NAMESPACE}}}mr")
    assert ["".join(row.itertext()) for row in rows] == ["n", "k"]


@pytest.mark.parametrize(
    ("latex", "expected_matrix_count"),
    [
        (r"\genfrac{}{}{0pt}{}{a_{i}}{\frac{b}{c}}", 1),
        (r"\genfrac{}{}{0pt}{}{a}{b}+\genfrac{}{}{0pt}{}{c}{d}", 2),
        (r"\genfrac{}{}{0pt}{}{\genfrac{}{}{0pt}{}{a}{b}}{c}", 2),
    ],
)
def test_latex_to_omml_normalizes_nested_and_multiple_genfrac(
    latex: str,
    expected_matrix_count: int,
) -> None:
    """验证 genfrac 参数可含嵌套结构，且一条公式可转换多个实例。"""
    equation = latex_to_omml(latex, display=False)

    assert len(equation.findall(f".//{{{_OFFICE_MATH_NAMESPACE}}}m")) == expected_matrix_count


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{a",
        r"\genfrac{(}{)}{1pt}{}{a}{b}",
        "x\x00y",
    ],
)
def test_latex_to_omml_wraps_conversion_failures_with_original_cause(latex: str) -> None:
    """验证损坏或非规范输入失败时保留原异常链。"""
    with pytest.raises(DocxFormulaError, match="无法转换为 OMML") as exc_info:
        latex_to_omml(latex, display=True)

    assert exc_info.value.__cause__ is not None
