from __future__ import annotations

from _span_test_utils import inline as _inline

from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess

from PIL import Image
import pytest

from mineru.render import render_latex
from mineru.types import (
    AlgorithmBodyBlock,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeAnnotationBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    IndexBlock,
    ListBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    PageFootnoteBlock,
    PageInfo,
    ParagraphTitleBlock,
    TableAnnotationBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
)

_TEX_LIVE_FILES = (
    "ctexart.cls",
    "geometry.sty",
    "amsmath.sty",
    "amssymb.sty",
    "graphicx.sty",
    "longtable.sty",
    "array.sty",
    "booktabs.sty",
    "multirow.sty",
    "enumitem.sty",
    "xcolor.sty",
    "ulem.sty",
    "xeCJKfntef.sty",
    "fvextra.sty",
    "hyperref.sty",
    "FandolSong-Regular.otf",
    "FandolSong-Bold.otf",
    "FandolHei-Regular.otf",
    "FandolHei-Bold.otf",
    "FandolKai-Regular.otf",
    "FandolFang-Regular.otf",
)


def _middle(*pages: PageInfo) -> MiddleJson:
    """构造无需源坐标布局的严格测试 MiddleJson。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """按调用方顺序构造一页测试内容。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def test_latex_document_uses_tex_live_preamble_default_planner_and_escaping() -> None:
    """验证固定 TeX Live 模板、续段合并、辅助块过滤、标题和普通文本转义。"""
    middle = _middle(
        _page(
            0,
            PageAuxTextBlock(type="header", index=0, content=_inline("HEADER")),
            DocTitleBlock(type="doc_title", index=1, level=1, content=_inline("中文标题")),
            TextBlock(type="text", index=2, content=_inline("first\ninter-"), anchor="body-target"),
        ),
        _page(
            3,
            TextBlock(type="text", index=0, content=_inline("national 50% & x_y {#} \\path"), continues_prev=True),
            PageFootnoteBlock(type="page_footnote", index=1, anchor="note", content=_inline("页面脚注")),
        ),
    )
    original = deepcopy(middle)

    rendered = render_latex(middle, document_title="Title & Metadata")

    assert rendered.startswith("% !TeX program = xelatex\n")
    assert r"\documentclass[UTF8,fontset=fandol,11pt]{ctexart}" in rendered
    assert "unicode-math" not in rendered and r"\setCJKmainfont" not in rendered
    assert r"\hypersetup{pdftitle={Title \& Metadata}}" in rendered
    assert r"first\MinerULineBreak{}international 50\% \& x\_y \{\#\} \textbackslash{}path\par" in rendered
    assert "HEADER" not in rendered
    assert "页面脚注" in rendered and r"\footnotesize\color{MinerUGray}" in rendered
    assert r"\write18" not in rendered and "minted" not in rendered and "shellesc" not in rendered.casefold()
    assert rendered.endswith("\\end{document}\n")
    assert rendered == render_latex(middle, document_title="Title & Metadata")
    assert middle == original


def test_latex_renders_inline_styles_formulas_links_and_safe_anchors() -> None:
    """验证全部 InlineSpan 样式、原始公式、内外部链接及哈希 anchor。"""
    title = ParagraphTitleBlock(
        type="paragraph_title",
        index=0,
        level=2,
        anchor="章节 1/#",
        content=_inline("章节一"),
    )
    text = TextBlock(
        type="text",
        index=1,
        content=[
            {"type": "text", "content": "Styled", "styles": ["bold", "italic", "underline"]},
            {"type": "text", "content": " dot", "styles": ["emphasis"]},
            {"type": "text", "content": " strike", "styles": ["strikethrough"]},
            {"type": "text", "content": "2", "styles": ["superscript"]},
            {"type": "text", "content": "i", "styles": ["subscript"]},
            {"type": "code_inline", "content": "a_b%  c"},
            {"type": "equation_inline", "content": r"x_1^2"},
            {"type": "hyperlink", "url": "#章节 1/#", "content": _inline("jump")},
            {"type": "hyperlink", "url": "https://example.com/a_b?q=1&x=2", "content": _inline("external")},
        ],
    )
    equation = EquationBlock(type="equation", index=2, content=r"\frac{1}{1-x^2}\tag{7}")
    aligned = EquationBlock(type="equation", index=3, content=r"\begin{align}a&=b\\c&=d\end{align}")

    rendered = render_latex(_middle(_page(0, title, text, equation, aligned)))

    assert r"\textbf{\textit{\uline{Styled}}}" in rendered
    assert r"\CJKunderdot[format=\normalcolor]{ dot}" in rendered
    assert r"\sout{ strike}" in rendered
    assert r"\textsuperscript{2}" in rendered and r"\textsubscript{i}" in rendered
    assert r"\texttt{a\_b\%\ \ c}" in rendered and r"\(x_1^2\)" in rendered
    assert r"\hypertarget{mineru-" in rendered and r"\hyperlink{mineru-" in rendered
    assert r"\href{https://example.com/a\_b?q=1\&x=2}{external}" in rendered
    assert "\\begin{equation*}\n\\frac{1}{1-x^2}\\tag{7}\n\\end{equation*}" in rendered
    assert rendered.count(r"\begin{align}") == 1 and r"\begin{equation*}\begin{align}" not in rendered


def test_latex_renders_native_lists_and_linked_index() -> None:
    """验证有序、显式、嵌套列表和目录链接保持来源语义。"""
    target = ParagraphTitleBlock(type="paragraph_title", index=0, level=2, anchor="target", content=_inline("Target"))
    ordered = ListBlock(
        type="list",
        index=1,
        sub_type="text",
        content=[
            TextBlock(type="text", content=_inline("2. Second")),
            TextBlock(type="text", content=_inline("4. Fourth")),
            ListBlock(type="list", sub_type="text", content=[TextBlock(type="text", content=_inline("- Nested"))]),
        ],
    )
    explicit = ListBlock(
        type="list",
        index=2,
        sub_type="text",
        content=[
            TextBlock(type="text", content=_inline("[A] Alpha")),
            TextBlock(type="text", content=_inline("plain")),
        ],
    )
    index = IndexBlock(
        type="index",
        index=3,
        content=[ParagraphTitleBlock(type="paragraph_title", level=2, anchor="target", content=_inline("Target\t12"))],
    )

    rendered = render_latex(_middle(_page(0, target, ordered, explicit, index)))

    assert r"\begin{enumerate}[leftmargin=*,nosep]" in rendered
    assert r"\item[{2.}] Second" in rendered and r"\item[{4.}] Fourth" in rendered
    assert r"\item Nested" in rendered
    assert r"\begin{description}[style=nextline,leftmargin=3em,nosep]" in rendered
    assert r"\item[{[A]}] Alpha" in rendered and r"\item[{}] plain" in rendered
    assert "Target 12" not in rendered
    assert r"\hyperlink{mineru-" in rendered and "{Target}" in rendered


def test_latex_renders_visual_blocks_complex_tables_and_code_in_source_order() -> None:
    """验证图片、复杂表格、图表、代码和视觉说明按子块来源顺序输出。"""
    image = ImageBlock(
        type="image",
        index=0,
        content=[
            ImageAnnotationBlock(type="image_caption", content=_inline("Before image")),
            ImageBodyBlock(type="image_body", index=0, content="Alt", image_path="images/a b.png"),
            ImageAnnotationBlock(type="image_footnote", content=_inline("After image")),
        ],
    )
    table_html = (
        "<table><thead><tr><th rowspan='2'>H</th><th colspan='2'><b>Group</b></th></tr></thead>"
        "<tbody><tr><td><eq>x^2</eq></td><td>before<table><tr><td>Nested</td></tr></table></td></tr></tbody></table>"
    )
    table = TableBlock(
        type="table",
        index=1,
        content=[
            TableBodyBlock(type="table_body", index=1, content=table_html),
            TableAnnotationBlock(type="table_caption", content=_inline("Table caption")),
        ],
    )
    chart = ChartBlock(
        type="chart",
        index=2,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=2,
                content="<table><tr><td>Data</td></tr></table>",
                image_path="images/chart.png",
            ),
            ChartAnnotationBlock(type="chart_footnote", content=_inline("Chart note")),
        ],
    )
    code = CodeBlock(
        type="code",
        index=3,
        sub_type="code",
        guess_lang="python",
        content=[
            CodeBodyBlock(type="code_body", index=3, content="print('x_y%')\n\\end{MinerUVerbatim1}"),
            CodeAnnotationBlock(type="code_caption", content=_inline("Code caption")),
        ],
    )
    algorithm = CodeBlock(
        type="code",
        index=4,
        sub_type="algorithm",
        content=[
            AlgorithmBodyBlock(
                type="algorithm_body",
                index=4,
                content=[{"type": "text", "content": "Step "}, {"type": "equation_inline", "content": "x^2"}],
            )
        ],
    )

    rendered = render_latex(
        _middle(_page(0, image, table, chart, code, algorithm)),
        asset_base_path="document assets",
    )

    assert rendered.index("Before image") < rendered.index(r"\detokenize{document assets/images/a b.png}")
    assert rendered.index(r"\detokenize{document assets/images/a b.png}") < rendered.index("After image")
    assert r"\begin{longtable}" in rendered and r"\multirow{2}" in rendered and r"\multicolumn{2}" in rendered
    assert r"\cline{2-3}" in rendered
    assert r"\begin{tabular}" in rendered and "Nested" in rendered and r"\(x^2\)" in rendered
    assert r"\detokenize{document assets/images/chart.png}" in rendered and "Data" in rendered
    assert "Table caption" in rendered and "Chart note" in rendered
    assert r"\DefineVerbatimEnvironment{MinerUVerbatim2}" in rendered
    assert "print('x_y%')" in rendered and "Code caption" in rendered
    assert r"{\small\ttfamily Step \(x^2\)\par}" in rendered


def test_latex_table_cell_preserves_interleaved_text_image_and_nested_table_order() -> None:
    """验证单元格文字、图片与嵌套表格严格保持 HTML 来源顺序。"""

    table = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content=(
                    "<table><tr><td><b>before<img src='images/cell.png' alt='cell'>after</b>"
                    "<table><tr><td>nested</td></tr></table>tail</td></tr></table>"
                ),
            )
        ],
    )

    rendered = render_latex(
        _middle(_page(0, table)),
        asset_base_path="assets",
    )

    before_index = rendered.index(r"\textbf{before}")
    image_index = rendered.index(r"\detokenize{assets/images/cell.png}")
    after_index = rendered.index(r"\textbf{after}")
    nested_index = rendered.index("nested")
    tail_index = rendered.index("tail")
    assert before_index < image_index < after_index < nested_index < tail_index


def test_latex_image_and_table_fallbacks_remain_visible_without_io() -> None:
    """验证远程、data URI、不支持格式和畸形表格均退化为可见内容。"""
    remote = ImageBlock(
        type="image",
        index=0,
        content=[
            ImageBodyBlock(
                type="image_body",
                index=0,
                content="Remote diagram",
                image_url="https://example.com/a.png",
            )
        ],
    )
    unsupported = ImageBlock(
        type="image",
        index=1,
        content=[ImageBodyBlock(type="image_body", index=1, content="SVG diagram", image_path="images/a.svg")],
    )
    malformed = TableBlock(
        type="table",
        index=2,
        content=[TableBodyBlock(type="table_body", index=2, content="<table><tr><td rowspan='2'>A</td></tr></table>")],
    )
    spatial = TableBlock(
        type="table",
        index=3,
        content=[TableBodyBlock(type="table_body", index=3, content="A  B\n1  2")],
    )
    inline_chart = ChartBlock(
        type="chart",
        index=4,
        content=[
            ChartBodyBlock(
                type="chart_body",
                index=4,
                content="",
                image_base64="data:image/png;base64,AAAA",
            )
        ],
    )
    empty_formula = EquationBlock(type="equation", index=5, content="")
    empty_table = TableBlock(
        type="table",
        index=6,
        content=[TableBodyBlock(type="table_body", index=6, content="")],
    )

    rendered = render_latex(
        _middle(_page(0, remote, unsupported, malformed, spatial, inline_chart, empty_formula, empty_table))
    )

    assert r"\href{https://example.com/a.png}" in rendered and "Remote diagram" in rendered
    assert "SVG diagram" in rendered and "images/a.svg" not in rendered
    assert "A" in rendered and rendered.count("table unavailable") == 1
    assert "A  B\n1  2" in rendered
    assert rendered.count(r"\DefineVerbatimEnvironment") == 2
    assert "image unavailable: chart" in rendered
    assert "formula unavailable" in rendered and "table unavailable" in rendered


def test_latex_normalizes_windows_asset_prefix_without_system_font_or_io_dependencies() -> None:
    """验证 Windows 路径前缀使用 XeLaTeX 跨平台可读的正斜杠。"""
    image = ImageBlock(
        type="image",
        index=0,
        content=[ImageBodyBlock(type="image_body", index=0, content="image", image_path="images/a.png")],
    )

    rendered = render_latex(_middle(_page(0, image)), asset_base_path=r"C:\documents\output")

    assert r"\detokenize{C:/documents/output/images/a.png}" in rendered


def test_latex_public_entry_rejects_invalid_contract_values() -> None:
    """验证专用入口拒绝旧字典、错误选项类型和控制字符路径。"""
    middle = _middle(_page(0))

    with pytest.raises(TypeError, match="MiddleJson"):
        render_latex(middle.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="asset_base_path"):
        render_latex(middle, asset_base_path=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document_title"):
        render_latex(middle, document_title=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control"):
        render_latex(middle, asset_base_path="images\nunsafe")


def test_latex_compiles_with_default_tex_live_full(tmp_path: Path) -> None:
    """在可用或强制要求的 TeX Live full 环境中真实编译代表文档。"""
    required = os.environ.get("MINERU_REQUIRE_TEXLIVE") == "1"
    latexmk = shutil.which("latexmk")
    xelatex = shutil.which("xelatex")
    kpsewhich = shutil.which("kpsewhich")
    missing_tools = [
        name for name, path in (("latexmk", latexmk), ("xelatex", xelatex), ("kpsewhich", kpsewhich)) if path is None
    ]
    if missing_tools:
        message = f"TeX Live tools are unavailable: {', '.join(missing_tools)}"
        pytest.fail(message) if required else pytest.skip(message)
    assert latexmk is not None and xelatex is not None and kpsewhich is not None

    missing_files = [
        filename
        for filename in _TEX_LIVE_FILES
        if subprocess.run([kpsewhich, filename], capture_output=True, text=True, check=False).returncode != 0
    ]
    if missing_files:
        message = f"TeX Live scheme-full files are unavailable: {', '.join(missing_files)}"
        pytest.fail(message) if required else pytest.skip(message)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 8), (40, 80, 120)).save(image_dir / "sample image.png")
    middle = _middle(
        _page(
            0,
            DocTitleBlock(type="doc_title", index=0, level=1, content=_inline("中文 TeX Live")),
            TextBlock(
                type="text",
                index=1,
                content=[
                    {"type": "text", "content": "正文 "},
                    {"type": "equation_inline", "content": r"x_1^2"},
                ],
            ),
            EquationBlock(type="equation", index=2, content=r"\frac{a}{b}\tag{1}"),
            ImageBlock(
                type="image",
                index=3,
                content=[ImageBodyBlock(type="image_body", index=3, content="sample", image_path="images/sample image.png")],
            ),
            TableBlock(
                type="table",
                index=4,
                content=[
                    TableBodyBlock(
                        type="table_body",
                        index=4,
                        content="<table><tr><th colspan='2'>Header</th></tr><tr><td>A</td><td><eq>x</eq></td></tr></table>",
                    )
                ],
            ),
            CodeBlock(
                type="code",
                index=5,
                sub_type="code",
                guess_lang="python",
                content=[CodeBodyBlock(type="code_body", index=5, content="print('TeX Live')")],
            ),
        )
    )
    tex_path = tmp_path / "main.tex"
    tex_path.write_text(render_latex(middle), encoding="utf-8")

    result = subprocess.run(
        [latexmk, "-xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert (tmp_path / "main.pdf").read_bytes().startswith(b"%PDF-")
