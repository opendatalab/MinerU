from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import base64

from PIL import Image
from pypdf import PdfReader
import pytest
from reportlab.graphics.shapes import Drawing
import ziamath

from _span_test_utils import inline as _inline
from mineru.render import render_pdf
from mineru.render._internal.pdf import formula as formula_module
from mineru.render._internal.pdf.formula import FormulaRenderer, FormulaVector, PdfFormulaError
from mineru.types import (
    AlgorithmBodyBlock,
    ChartAnnotationBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
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
    TableBlock,
    TextBlock,
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


def _png_bytes(size: tuple[int, int] = (20, 10)) -> bytes:
    """生成可由 Pillow、ReportLab 与 pypdf 测试读取的 PNG。"""
    output = BytesIO()
    Image.new("RGB", size, (30, 80, 130)).save(output, format="PNG")
    return output.getvalue()


def _png_uri() -> str:
    """返回有效 PNG data URI。"""
    return f"data:image/png;base64,{base64.b64encode(_png_bytes()).decode('ascii')}"


def _reader(payload: bytes) -> PdfReader:
    """从内存 bytes 打开 PDF 并先验证固定签名。"""
    assert payload.startswith(b"%PDF-")
    return PdfReader(BytesIO(payload))


def _page_text(reader: PdfReader, page_index: int) -> str:
    """提取指定 PDF 页的可搜索文字。"""
    return reader.pages[page_index].extract_text() or ""


def test_pdf_uses_default_planner_without_source_page_boundaries() -> None:
    """验证固定默认 PDF 合并续段、隐藏辅助块、折叠空源页且输入不变。"""
    middle = _middle(
        _page(
            0,
            PageAuxTextBlock(type="header", index=0, content=_inline("HEADER")),
            TextBlock(type="text", index=1, content=_inline("inter-")),
        ),
        _page(5),
        _page(
            9,
            TextBlock(type="text", index=0, content=_inline("national"), continues_prev=True),
            PageAuxTextBlock(type="footer", index=1, content=_inline("FOOTER")),
        ),
    )
    original = deepcopy(middle)

    payload = render_pdf(middle, document_title="PDF Contract")
    assert payload == render_pdf(middle, document_title="PDF Contract")
    reader = _reader(payload)

    assert len(reader.pages) == 1
    assert "international" in _page_text(reader, 0).replace("\n", "")
    assert "HEADER" not in _page_text(reader, 0) and "FOOTER" not in _page_text(reader, 0)
    assert reader.metadata.title == "PDF Contract"
    assert reader.metadata.creation_date is not None and reader.metadata.creation_date.year == 2000
    assert middle == original


def test_pdf_renders_inline_and_display_formulas_as_vector_paths_with_links() -> None:
    """验证行内/行间公式矢量化、标签、中文混排及内外部链接。"""
    middle = _middle(
        _page(
            0,
            ParagraphTitleBlock(type="paragraph_title", index=0, level=2, anchor="section-one", content=_inline("章节一")),
            TextBlock(
                type="text",
                index=1,
                content=[
                    {"type": "text", "content": "中文 Before "},
                    {"type": "equation_inline", "content": r"c=\pm\sqrt{a^2+b^2}"},
                    {"type": "text", "content": " after "},
                    {"type": "hyperlink", "url": "#section-one", "content": _inline("jump")},
                    {"type": "text", "content": " and "},
                    {"type": "hyperlink", "url": "https://example.com", "content": _inline("external")},
                ],
            ),
            EquationBlock(type="equation", index=2, content=r"\frac{1}{1-x^2}\tag{7}"),
        )
    )

    reader = _reader(render_pdf(middle))
    text = _page_text(reader, 0)
    resources = reader.pages[0]["/Resources"]
    xobjects = resources.get("/XObject", {})
    image_xobjects = [value for value in xobjects.values() if value.get_object().get("/Subtype") == "/Image"]

    assert "章节一" in text and "中文 Before" in text and "after" in text
    assert "jump" in text and "external" in text
    assert not image_xobjects
    assert len(reader.pages[0].get("/Annots", [])) >= 2
    stream = reader.pages[0].get_contents().get_data()
    assert b" m" in stream and (b" c" in stream or b" l" in stream)


def test_pdf_unicode_fallback_and_script_styles_avoid_black_squares() -> None:
    """验证 Latin Extended、独立重音、希腊文和西里尔文使用确定性字体回退。"""
    middle = _middle(
        _page(
            0,
            TextBlock(
                type="text",
                index=0,
                content=[
                    {"type": "text", "content": "Jędrzej J˛edrzej λ Ж 中文 x"},
                    {"type": "text", "content": "2", "styles": ["superscript"]},
                ],
            ),
        )
    )

    reader = _reader(render_pdf(middle))
    text = _page_text(reader, 0)
    stream = reader.pages[0].get_contents().get_data()

    assert "Jędrzej J˛edrzej λ Ж 中文 x2" in text.replace("\n", "")
    assert "■" not in text
    assert b" Ts" in stream


def test_pdf_formula_failures_fall_back_to_visible_latex(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ZiaMath 失败时行内与行间公式均保留可见 LaTeX。"""
    original_render = FormulaRenderer.render

    def fail_bad_formula(
        self: FormulaRenderer,
        latex: str,
        *,
        inline: bool,
        font_size: float,
        color: str = "#1f2937",
    ) -> object:
        """仅让测试公式进入稳定回退，其余公式沿用真实实现。"""
        if latex == "bad":
            raise PdfFormulaError("synthetic formula failure")
        return original_render(self, latex, inline=inline, font_size=font_size, color=color)

    monkeypatch.setattr(FormulaRenderer, "render", fail_bad_formula)
    middle = _middle(
        _page(
            0,
            TextBlock(
                type="text",
                index=0,
                content=[{"type": "text", "content": "inline "}, {"type": "equation_inline", "content": "bad"}],
            ),
            EquationBlock(type="equation", index=1, content="bad"),
        )
    )

    text = _page_text(_reader(render_pdf(middle)), 0)

    assert "$bad$" in text
    assert "bad" in text


@pytest.mark.parametrize(
    "source,inline",
    [
        (r"\frac{1}{1-x^2}", False),
        (r"\sqrt[3]{x+y}", True),
        (r"\sum_{i=1}^{n} i^2", False),
        (r"\int_0^\infty e^{-x}\,dx", False),
        (r"\begin{bmatrix}a&b\\c&d\end{bmatrix}", False),
        (r"\left(\frac{x}{y}\right)^2", True),
        (r"\begin{aligned}a&=b+c\\d&=e\end{aligned}", False),
        (r"\color{red}{x}+\boxed{y}", True),
    ],
)
def test_ziamath_vector_corpus_covers_inline_and_display_constructs(source: str, inline: bool) -> None:
    """验证分数、根式、积分、矩阵、aligned 与颜色均能生成有效矢量几何。"""
    vector = FormulaRenderer().render(source, inline=inline, font_size=12)

    assert vector.width > 0 and vector.height > 0
    assert vector.ascent >= 0 and vector.descent >= 0
    assert vector.drawing.contents


def test_formula_cache_is_bounded_and_skips_oversized_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证公式超长或唯一项超过预算时继续渲染但不扩张文档级缓存。"""
    calls: list[str] = []

    def fake_render(latex: str, *, inline: bool, font_size: float, color: str) -> FormulaVector:
        """返回固定矢量并记录实际转换次数。"""
        calls.append(latex)
        return FormulaVector(Drawing(1, 1), width=1, height=1, ascent=1, descent=0)

    monkeypatch.setattr(formula_module, "_render_ziamath_formula", fake_render)
    renderer = FormulaRenderer()
    oversized = "x" * (formula_module.MAX_FORMULA_CHARACTERS + 1)
    renderer.render(oversized, inline=True, font_size=10)
    renderer.render(oversized, inline=True, font_size=10)
    for index in range(formula_module.MAX_CACHED_FORMULAS + 3):
        renderer.render(f"x_{index}", inline=True, font_size=10)

    assert calls.count(oversized) == 2
    assert len(renderer._cache) == formula_module.MAX_CACHED_FORMULAS  # noqa: SLF001


def test_long_formulas_scale_or_wrap_without_hiding_surrounding_text() -> None:
    """验证超宽行内与行间公式缩放后仍保留前后可搜索正文。"""
    formula = "+".join(f"x_{index}^2" for index in range(45))
    middle = _middle(
        _page(
            0,
            TextBlock(
                type="text",
                index=0,
                content=[
                    {"type": "text", "content": "before long formula "},
                    {"type": "equation_inline", "content": formula},
                    {"type": "text", "content": " after long formula"},
                ],
            ),
            EquationBlock(type="equation", index=1, content=formula + r"\tag{wide}"),
        )
    )

    reader = _reader(render_pdf(middle))
    text = "\n".join(_page_text(reader, index) for index in range(len(reader.pages)))

    assert "before long formula" in text
    assert "after long formula" in text


def test_pdf_uses_asset_resolver_and_replaces_missing_or_remote_images_with_placeholders() -> None:
    """验证 sidecar 优先级、有效图片、损坏素材和远程 URL 的宽松占位策略。"""
    requested: list[str] = []

    def resolve_asset(path: str) -> bytes:
        """记录图片请求，并分别返回有效或损坏字节。"""
        requested.append(path)
        return _png_bytes() if path == "images/ok.png" else b"broken"

    middle = _middle(
        _page(
            0,
            ImageBlock(
                type="image",
                index=0,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=0,
                        content="valid image",
                        image_path="images/ok.png",
                        image_base64=_png_uri(),
                    )
                ],
            ),
            ImageBlock(
                type="image",
                index=1,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=1,
                        content="external svg",
                        image_base64=(
                            "data:image/svg+xml;base64,"
                            + base64.b64encode(
                                b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
                            ).decode("ascii")
                        ),
                    )
                ],
            ),
            ImageBlock(
                type="image",
                index=2,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=2,
                        content="broken image",
                        image_path="images/broken.png",
                    )
                ],
            ),
            ImageBlock(
                type="image",
                index=3,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=3,
                        content="remote image",
                        image_url="https://example.com/image.png",
                    )
                ],
            ),
        )
    )

    reader = _reader(render_pdf(middle, asset_resolver=resolve_asset))
    text = _page_text(reader, 0)

    assert requested == ["images/ok.png", "images/broken.png"]
    assert text.count("image unavailable") == 3
    assert "https://example.com/image.png" in text
    assert "page_idx=0, block_index=2, block_type=image_body" in text


def test_pdf_renders_tables_lists_indices_code_algorithm_and_annotations() -> None:
    """验证原生表格合并、嵌套表、目录、列表、代码算法与说明的组合输出。"""
    table = TableBlock.model_validate(
        {
            "type": "table",
            "index": 2,
            "content": [
                {
                    "type": "table_body",
                    "index": 2,
                    "content": (
                        "<table><thead><tr><th colspan='2'>Header</th></tr></thead>"
                        "<tbody><tr><td rowspan='2'>A<eq>x^2</eq></td><td>B</td></tr>"
                        "<tr><td><table><tr><td>Nested</td></tr></table></td></tr></tbody></table>"
                    ),
                },
                {"type": "table_caption", "content": _inline("Table caption")},
                {"type": "table_footnote", "content": _inline("Table note")},
            ],
        }
    )
    index = IndexBlock(
        type="index",
        index=1,
        content=[ParagraphTitleBlock(type="paragraph_title", level=2, anchor="target", content=_inline("Target\t9"))],
    )
    code = CodeBlock(
        type="code",
        index=3,
        sub_type="code",
        guess_lang="python",
        content=[
            CodeBodyBlock(type="code_body", index=3, content="print('中文')\nprint(2)"),
            {"type": "code_caption", "content": _inline("Code caption")},
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
                content=[{"type": "text", "content": "for "}, {"type": "equation_inline", "content": "n^2"}],
            )
        ],
    )
    chart = ChartBlock(
        type="chart",
        index=5,
        sub_type="bar",
        content=[
            ChartBodyBlock(type="chart_body", index=5, content="<table><tr><th>Year</th></tr><tr><td>2026</td></tr></table>"),
            ChartAnnotationBlock(type="chart_caption", content=_inline("Chart caption")),
        ],
    )
    middle = _middle(
        _page(
            0,
            ParagraphTitleBlock(type="paragraph_title", index=0, level=2, anchor="target", content=_inline("Target")),
            index,
            table,
            code,
            algorithm,
            chart,
            ListBlock(type="list", index=6, content=[TextBlock(type="text", content=_inline("1. item"))]),
            PageFootnoteBlock(type="page_footnote", index=7, anchor="note", content=_inline("Page note")),
        )
    )

    reader = _reader(render_pdf(middle))
    text = "\n".join(_page_text(reader, index) for index in range(len(reader.pages)))

    for expected in (
        "Target",
        "Header",
        "Nested",
        "Table caption",
        "Table note",
        "print('中文')",
        "Code caption",
        "Year",
        "2026",
        "Chart caption",
        "1. item",
        "Page note",
    ):
        assert expected in text


def test_pdf_formula_configuration_is_restored_across_parallel_renders() -> None:
    """验证并行 PDF 公式渲染不会泄漏 ZiaMath 的进程级 svg2 配置。"""
    previous_svg2 = ziamath.config.svg2
    middle = _middle(
        _page(
            0,
            DocTitleBlock(type="doc_title", index=0, level=1, content=_inline("Parallel")),
            EquationBlock(type="equation", index=1, content=r"\sum_{i=1}^{n} i^2"),
        )
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _index: render_pdf(middle), range(4)))

    assert all(_reader(payload).pages for payload in outputs)
    assert ziamath.config.svg2 is previous_svg2


def test_pdf_public_arguments_remain_strict() -> None:
    """验证专用 PDF 门面拒绝旧 dict、已删除 mode 与错误参数类型。"""
    middle = _middle(_page(0))

    with pytest.raises(TypeError, match="MiddleJson"):
        render_pdf(middle.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        render_pdf(middle, mode="full")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="asset_resolver"):
        render_pdf(middle, asset_resolver="images")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document_title"):
        render_pdf(middle, document_title=1)  # type: ignore[arg-type]
