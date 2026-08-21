#!/usr/bin/env python3
"""生成不含真实文档内容的 Flash PDF 单元测试夹具。"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import cidfonts, pdfmetrics
from reportlab.pdfgen.canvas import Canvas


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "unittest" / "pdfs"
FLASH_FIXTURE_NAME = "flash_table_annotations_synthetic.pdf"
CJK_FIXTURE_NAME = "native_cjk_layout_synthetic.pdf"
SAFE_WATERMARK_TEXT = "MINERU TEST WATERMARK"
NEUTRAL_METADATA_VALUE = "MinerU Test Suite"
CJK_FONT_NAME = "STSong-Light"
PAGE_WIDTH, PAGE_HEIGHT = A4


def _register_fonts() -> None:
    """注册无需读取原文或本机私有字体文件的标准 CJK 字体。"""

    if CJK_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(cidfonts.UnicodeCIDFont(CJK_FONT_NAME))


def _new_canvas(output_path: Path) -> Canvas:
    """创建具有固定元数据和确定性对象编号的 A4 PDF 画布。"""

    canvas = Canvas(
        str(output_path),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    canvas.setAuthor(NEUTRAL_METADATA_VALUE)
    canvas.setCreator(NEUTRAL_METADATA_VALUE)
    canvas.setTitle(NEUTRAL_METADATA_VALUE)
    canvas.setSubject("Synthetic PDF regression fixture")
    canvas.setKeywords("MinerU synthetic test fixture")
    canvas._doc.info.producer = NEUTRAL_METADATA_VALUE  # noqa: SLF001
    return canvas


def _draw_page_heading(canvas: Canvas, title: str, subtitle: str = "") -> None:
    """绘制统一的合成页面标题和可选副标题。"""

    canvas.setFillColor(black)
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(54, PAGE_HEIGHT - 54, title)
    if subtitle:
        canvas.setFont("Helvetica", 9)
        canvas.drawString(54, PAGE_HEIGHT - 70, subtitle)


def _draw_page_number(canvas: Canvas, page_number: int) -> None:
    """在页面底部中央绘制独立页码。"""

    canvas.setFillColor(black)
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(PAGE_WIDTH / 2, 30, str(page_number))


def _draw_table(
    canvas: Canvas,
    *,
    left: float,
    top: float,
    width: float,
    row_height: float,
    rows: list[list[str]],
) -> tuple[float, float, float, float]:
    """用矢量线和中性单元格文本绘制稳定的规则表格。"""

    column_count = len(rows[0])
    height = row_height * len(rows)
    bottom = top - height
    column_width = width / column_count
    canvas.saveState()
    canvas.setStrokeColor(black)
    canvas.setLineWidth(0.8)
    for row_index in range(len(rows) + 1):
        y = top - row_index * row_height
        canvas.line(left, y, left + width, y)
    for column_index in range(column_count + 1):
        x = left + column_index * column_width
        canvas.line(x, bottom, x, top)
    canvas.setFont("Helvetica", 8)
    for row_index, row in enumerate(rows):
        baseline = top - row_index * row_height - 0.68 * row_height
        for column_index, value in enumerate(row):
            canvas.drawString(left + column_index * column_width + 5, baseline, value)
    canvas.restoreState()
    return left, bottom, left + width, top


def _solid_image_reader(color: tuple[int, int, int]) -> ImageReader:
    """创建仅用于图像和 footer 几何回归的内存 PNG。"""

    image = Image.new("RGB", (80, 50), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    buffer.seek(0)
    return ImageReader(buffer)


def _draw_flash_fixture(output_path: Path) -> None:
    """生成八页表格、注释、跨页和 footer 行为合成夹具。"""

    canvas = _new_canvas(output_path)

    _draw_page_heading(canvas, "Synthetic Flash Table Fixture", "Neutral content for parser regression tests")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(72, 690, "This document contains only generated test content.")
    canvas.drawString(72, 668, "Each page isolates one layout behavior used by the test suite.")
    _draw_page_number(canvas, 1)
    canvas.showPage()

    _draw_page_heading(canvas, "Two Independent Vector Tables")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(72, 715, "Table 1 Synthetic Inventory")
    _draw_table(
        canvas,
        left=72,
        top=700,
        width=450,
        row_height=28,
        rows=[
            ["ITEM", "VALUE", "STATE"],
            ["ALPHA", "10", "READY"],
            ["BETA", "20", "READY"],
            ["GAMMA", "30", "DONE"],
        ],
    )
    canvas.drawString(72, 545, "Table 2 Synthetic Metrics")
    _draw_table(
        canvas,
        left=72,
        top=530,
        width=450,
        row_height=28,
        rows=[
            ["METRIC", "LOW", "HIGH"],
            ["WIDTH", "12", "48"],
            ["HEIGHT", "8", "32"],
            ["COUNT", "3", "9"],
        ],
    )
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 370, "Ordinary text below both tables must remain outside table bodies.")
    _draw_page_number(canvas, 2)
    canvas.showPage()

    _draw_page_heading(canvas, "Same-page Table Footnote Continuation")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(72, 715, "Table 3 Synthetic Footnote Merge")
    _draw_table(
        canvas,
        left=72,
        top=700,
        width=450,
        row_height=30,
        rows=[
            ["FIELD", "DETAIL", "STATUS"],
            ["A", "Generated value", "OK"],
            ["B", "Generated value", "OK"],
            ["C", "Generated value", "OK"],
        ],
    )
    canvas.setFont("Helvetica", 8)
    canvas.drawString(76, 535, "Note: This generated note belongs to Table 3.")
    canvas.drawString(92, 523, "Its first indented continuation remains on the same page.")
    canvas.drawString(92, 511, "Its second indented continuation completes the annotation.")
    _draw_page_number(canvas, 3)
    canvas.showPage()

    _draw_page_heading(canvas, "In-border Note Remains Table Content")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(72, 715, "Table 4 Bordered Note")
    _draw_table(
        canvas,
        left=72,
        top=700,
        width=450,
        row_height=34,
        rows=[
            ["KEY", "DESCRIPTION", "FLAG"],
            ["A1", "Neutral row", "YES"],
            ["A2", "Neutral row", "NO"],
            ["NOTE", "Note: This sentence is inside the border.", "TABLE"],
        ],
    )
    _draw_page_number(canvas, 4)
    canvas.showPage()

    _draw_page_heading(canvas, "Body Text Negative Samples")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 700, "This ordinary paragraph is unrelated to any table or figure.")
    canvas.drawString(72, 680, "ITEM_001 and VALUE_002 are prose tokens, not a code block.")
    canvas.drawString(72, 660, "A distant Note: marker remains body text without a visual parent.")
    canvas.drawString(72, 620, "The parser should preserve these lines as readable text.")
    _draw_page_number(canvas, 5)
    canvas.showPage()

    _draw_page_heading(canvas, "Code-like Cells Stay in a Table")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(72, 715, "Table 5 Generated Token Grid")
    _draw_table(
        canvas,
        left=72,
        top=700,
        width=450,
        row_height=30,
        rows=[
            ["TOKEN", "EXPRESSION", "RESULT"],
            ["ROW_001", "A=B+1", "PASS"],
            ["ROW_002", "C=D+2", "PASS"],
            ["ROW_003", "E=F+3", "PASS"],
        ],
    )
    canvas.setFont("Helvetica", 8)
    canvas.drawString(76, 566, "Note: Token-shaped cell text must not become a code block.")
    _draw_page_number(canvas, 6)
    canvas.showPage()

    _draw_page_heading(canvas, "Cross-page Footnote Boundary")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(72, 250, "Table 6 Page-local Annotation")
    _draw_table(
        canvas,
        left=72,
        top=235,
        width=450,
        row_height=30,
        rows=[
            ["GROUP", "VALUE", "STATE"],
            ["LOCAL", "77", "OPEN"],
            ["LOCAL", "88", "CLOSED"],
        ],
    )
    canvas.setFont("Helvetica", 8)
    canvas.drawString(76, 132, "Note: This annotation ends on page seven.")
    _draw_page_number(canvas, 7)
    canvas.showPage()

    _draw_page_heading(canvas, "Cross-page Text and Empty Footer Image")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(92, 744, "This top paragraph must not merge into the previous page footnote.")
    canvas.drawString(72, 700, "The final page also contains an image below its page number.")
    _draw_page_number(canvas, 8)
    canvas.drawImage(
        _solid_image_reader((220, 230, 240)),
        PAGE_WIDTH / 2 - 60,
        1,
        width=120,
        height=26,
        mask="auto",
    )
    canvas.showPage()
    canvas.save()


def _draw_safe_watermark_grid(canvas: Canvas) -> None:
    """在当前页底层绘制四乘四个安全旋转文本水印。"""

    canvas.saveState()
    canvas.setFillColor(HexColor("#D9DEE7"))
    canvas.setFont("Helvetica", 7)
    for row_index in range(4):
        for column_index in range(4):
            canvas.saveState()
            canvas.translate(48 + column_index * 145, 120 + row_index * 185)
            canvas.rotate(5.5)
            canvas.drawString(0, 0, SAFE_WATERMARK_TEXT)
            canvas.restoreState()
    canvas.restoreState()


def _draw_cjk_page_heading(canvas: Canvas, title: str) -> None:
    """用标准 CJK 字体绘制合成中文页面标题。"""

    canvas.setFillColor(black)
    canvas.setFont(CJK_FONT_NAME, 15)
    canvas.drawString(54, PAGE_HEIGHT - 54, title)


def _draw_mixed_line(
    canvas: Canvas,
    *,
    y: float,
    cjk_prefix: str,
    latin_suffix: str,
) -> None:
    """在同一基线绘制 CJK 与西文字体混排的物理行。"""

    line_x = 72.0
    if cjk_prefix.startswith("• "):
        canvas.setFont("Helvetica", 10)
        canvas.drawString(line_x, y, "• ")
        line_x += pdfmetrics.stringWidth("• ", "Helvetica", 10)
        cjk_prefix = cjk_prefix[2:]
    canvas.setFont(CJK_FONT_NAME, 10)
    canvas.drawString(line_x, y, cjk_prefix)
    prefix_width = pdfmetrics.stringWidth(cjk_prefix, CJK_FONT_NAME, 10)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(line_x + prefix_width, y, latin_suffix)


def _draw_link_text(
    canvas: Canvas,
    *,
    x: float,
    y: float,
    visible_text: str,
    target: str,
) -> float:
    """绘制带精确 URI 注解的西文文本并返回其右边界。"""

    font_name = "Helvetica"
    font_size = 9
    width = pdfmetrics.stringWidth(visible_text, font_name, font_size)
    canvas.setFont(font_name, font_size)
    canvas.drawString(x, y, visible_text)
    canvas.linkURL(
        target,
        (x, y - 2, x + width, y + font_size + 1),
        relative=0,
        thickness=0,
    )
    return x + width


def _draw_cjk_fixture(output_path: Path) -> None:
    """生成四页目录、字体、URL 和安全水印合成夹具。"""

    _register_fonts()
    canvas = _new_canvas(output_path)

    _draw_safe_watermark_grid(canvas)
    _draw_cjk_page_heading(canvas, "MinerU 合成原生文本测试")
    canvas.setFont(CJK_FONT_NAME, 11)
    canvas.drawString(72, 690, "本文档完全由测试生成器创建，不包含真实人员或组织信息。")
    canvas.drawString(72, 665, "页面用于验证目录、图注、链接、字体基线和水印过滤。")
    _draw_page_number(canvas, 1)
    canvas.showPage()

    _draw_safe_watermark_grid(canvas)
    _draw_cjk_page_heading(canvas, "目录")
    canvas.setFont(CJK_FONT_NAME, 9)
    for row_index in range(24):
        baseline = 742 - row_index * 26
        canvas.drawString(65, baseline, f"{row_index + 1} 合成章节 {row_index + 1}")
        canvas.drawRightString(520, baseline, str(row_index + 1))
    _draw_page_number(canvas, 2)
    canvas.showPage()

    _draw_safe_watermark_grid(canvas)
    _draw_cjk_page_heading(canvas, "混合字体和基线")
    _draw_mixed_line(canvas, y=700, cjk_prefix="• 合成支持工程师 ", latin_suffix="Support Engineer")
    _draw_mixed_line(canvas, y=660, cjk_prefix="• 混合字体工具 ", latin_suffix="Acuity Toolkit")
    _draw_mixed_line(canvas, y=620, cjk_prefix="• 测试系统 ", latin_suffix="Ubuntu 24.04")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 580, "mineru_toolkit_binary_1.0.0_linux_x86_64.tgz")
    canvas.setFont(CJK_FONT_NAME, 10)
    canvas.drawString(72, 535, "以上普通文本均应保持在基线，不应生成上下标标签。")
    _draw_page_number(canvas, 3)
    canvas.showPage()

    _draw_safe_watermark_grid(canvas)
    _draw_cjk_page_heading(canvas, "图注和链接")
    canvas.drawImage(_solid_image_reader((210, 225, 240)), 72, 565, width=190, height=105, mask="auto")
    canvas.drawImage(_solid_image_reader((235, 220, 210)), 332, 565, width=190, height=105, mask="auto")
    canvas.setFont(CJK_FONT_NAME, 9)
    canvas.drawString(72, 548, "图 1 合成示意图 A")
    canvas.drawString(332, 548, "图 2 合成示意图 B")
    wrapped_url = "https://example.com/downloads/mineru/fixture/"
    _draw_link_text(
        canvas,
        x=72,
        y=485,
        visible_text="https://example.com/downloads/",
        target=wrapped_url,
    )
    _draw_link_text(
        canvas,
        x=72,
        y=473,
        visible_text="mineru/fixture/",
        target=wrapped_url,
    )
    link_x = 72.0
    for link_index, target in enumerate(
        (
            "https://example.com/ref/a",
            "https://example.com/ref/b",
            "https://example.com/ref/c",
        )
    ):
        if link_index:
            link_x += pdfmetrics.stringWidth(" ", "Helvetica", 9)
        link_x = _draw_link_text(
            canvas,
            x=link_x,
            y=425,
            visible_text=target,
            target=target,
        )
    canvas.setFont(CJK_FONT_NAME, 9)
    canvas.drawString(72, 390, "链接均使用保留域名，不指向真实账号或资源。")
    _draw_page_number(canvas, 4)
    canvas.showPage()
    canvas.save()


def generate_fixtures(output_dir: Path) -> tuple[Path, Path]:
    """生成两份合成 PDF 并返回稳定输出路径。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    flash_path = output_dir / FLASH_FIXTURE_NAME
    cjk_path = output_dir / CJK_FIXTURE_NAME
    _draw_flash_fixture(flash_path)
    _draw_cjk_fixture(cjk_path)
    return flash_path, cjk_path


def _check_committed_fixtures() -> None:
    """重新生成夹具并逐字节校验仓库产物未发生漂移。"""

    with TemporaryDirectory(prefix="mineru-synthetic-pdf-") as temporary_directory:
        generated_paths = generate_fixtures(Path(temporary_directory))
        for generated_path in generated_paths:
            committed_path = OUTPUT_DIR / generated_path.name
            if not committed_path.is_file():
                raise SystemExit(f"missing committed fixture: {committed_path}")
            if generated_path.read_bytes() != committed_path.read_bytes():
                raise SystemExit(f"fixture is stale: {committed_path}")


def main() -> None:
    """解析命令行参数并执行生成或一致性检查。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and compare committed PDFs",
    )
    arguments = parser.parse_args()
    if arguments.check:
        _check_committed_fixtures()
        return
    generate_fixtures(OUTPUT_DIR)


if __name__ == "__main__":
    main()
