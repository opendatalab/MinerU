"""验证 Flash 原生按需截图与 MinerU 配置、选页和后处理边界。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from docvortex.analyzers.native.models import PdfModel
from docvortex.document.pdf import PDFDocument, images
from mineru.backend.analysis.pdf import window
from mineru.backend.postprocess import llm_aided
from mineru.config import LLMAidedConfig, LLMAidedFeaturesConfig, config
from mineru.parser import parse, parse_async


def _source(path: Path) -> Path:
    """构造五页 PDF，让实际输入准备负责连续和非连续选页。"""
    canvas = Canvas(str(path))
    for index in range(5):
        canvas.drawString(60, 720, f"Page {index + 1}")
        canvas.showPage()
    canvas.save()
    return path


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("page_range,indices", [("all", [0, 1, 2, 3, 4]), ("2-4", [1, 2, 3]), ("1,3,5", [0, 2, 4])])
def test_flash_sparse_render_keeps_page_mapping_and_title_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asynchronous: bool,
    page_range: str,
    indices: list[int],
) -> None:
    """真实同步和异步入口只渲染有图页，页号和整本标题开关仍由 MinerU 管理。"""
    document_sizes: list[int] = []
    rendered: list[int] = []
    created: list[Image.Image] = []

    def predict(_model: PdfModel, document: PDFDocument) -> list[list[dict[str, Any]]]:
        """替换原生语义分析，每个物理页含标题，奇数物理页额外含图片。"""
        document_sizes.append(document.page_count)
        pages: list[list[dict[str, Any]]] = []
        for index in range(document.page_count):
            blocks = [
                {
                    "type": "paragraph_title",
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "level": 2,
                    "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
                    "content": [{"type": "text", "content": f"Section {index}"}],
                }
            ]
            if index % 2:
                blocks.append({"type": "image", "bbox": [0.1, 0.3, 0.9, 0.8]})
            pages.append(blocks)
        return pages

    def raster(data: bytes, **options: Any) -> list[dict[str, Any]]:
        """以真实物理页索引生成颜色图片，同时检查宿主设置优先于引擎默认值。"""
        assert data.startswith(b"%PDF")
        assert options["timeout"] == 19 and options["threads"] == 2
        batch = list(range(options["start_page_id"], options["end_page_id"] + 1))
        rendered.extend(batch)
        result = []
        for index in batch:
            image = Image.new("RGB", (40, 20), (index * 30, 60, 120))
            created.append(image)
            result.append({"img_pil": image})
        return result

    monkeypatch.setenv("MINERU_PROCESSING_WINDOW_SIZE", "2")
    monkeypatch.setenv("MINERU_PDF_RENDER_TIMEOUT", "19")
    monkeypatch.setenv("MINERU_PDF_RENDER_THREADS", "2")
    monkeypatch.setenv("DOCVORTEX_PDF_RENDER_TIMEOUT", "31")
    monkeypatch.setattr(PdfModel, "predict", predict)
    monkeypatch.setattr(images, "load_images_from_pdf_bytes_range", raster)
    forbidden = MagicMock(side_effect=AssertionError("Flash TXT must skip inference windows"))
    monkeypatch.setattr(window, "_get_window_pdf_pages", forbidden)
    title = AsyncMock()
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(llm_aided, "LLMAidedClient", MagicMock(return_value=client))
    monkeypatch.setattr(llm_aided, "apply_llm_title_leveling", title)
    monkeypatch.setattr(
        config,
        "llm_aided",
        LLMAidedConfig(
            api_key="isolated-test",
            features=LLMAidedFeaturesConfig(title_leveling=True),
        ),
    )
    source = _source(tmp_path / "pages.pdf")
    options = {"tier": "flash", "ocr_mode": "txt", "image_analysis": False, "page_range": page_range}
    result = asyncio.run(parse_async(source, **options)) if asynchronous else parse(source, **options)

    full = page_range == "all"
    assert document_sizes == [len(indices)]
    assert rendered == list(range(1, len(indices), 2))
    assert [page.page_idx for page in result.pages] == indices
    assert result._model_output.page_index_map == ([] if full else indices)
    assert result.middle_json.is_full_document is full
    assert result._model_output.metadata.producer.name == "mineru"
    assert result.middle_json.extensions == result._model_output.extensions
    assert title.await_count == int(full)
    if full:
        assert title.await_args.args[0] is result.pages
    for index in rendered:
        assert result._model_output.pages[index][1]["image_base64"].startswith("data:image/jpeg;base64,")
    for image in created:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    forbidden.assert_not_called()


@pytest.mark.parametrize("parse_mode", ["txt", "auto"])
def test_flash_text_only_does_not_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parse_mode: str) -> None:
    """真实原生解析的纯文本 PDF 不创建页图，auto 仍只在入口分类一次。"""
    classify = MagicMock(return_value="txt")
    monkeypatch.setattr(PDFDocument, "classify", classify)
    forbidden = MagicMock(side_effect=AssertionError("Text-only PDF must not rasterize"))
    monkeypatch.setattr(images, "load_images_from_pdf_bytes_range", forbidden)
    result = parse(_source(tmp_path / "text.pdf"), tier="flash", ocr_mode=parse_mode, image_analysis=False)
    assert len(result.pages) == 5
    assert classify.call_count == int(parse_mode == "auto")
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "values,expected",
    [(None, (64, 300, 3)), (("bad", "0", "-1"), (64, 300, 3)), (("0", "17", "2"), (1, 17, 2))],
)
def test_flash_preserves_render_config_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    values: tuple[str, str, str] | None,
    expected: tuple[int, int, int],
) -> None:
    """沿用宿主现有环境变量缺省、非法值和窗口下限规则。"""
    names = ("MINERU_PROCESSING_WINDOW_SIZE", "MINERU_PDF_RENDER_TIMEOUT", "MINERU_PDF_RENDER_THREADS")
    for index, name in enumerate(names):
        monkeypatch.delenv(name, raising=False)
        if values is not None:
            monkeypatch.setenv(name, values[index])
    pages: list[list[dict[str, Any]]] = [[]]
    monkeypatch.setattr(PdfModel, "predict", MagicMock(return_value=pages))
    attach = MagicMock()
    monkeypatch.setattr(window, "attach_visual_block_images_from_pdf", attach)
    document = MagicMock(page_count=1)
    actual = window.process_pdf_windows(
        b"pdf",
        document,
        effort="flash",
        parse_mode="txt",
        image_analysis=False,
        flash_txt_mode=True,
        hybrid_model=None,
        vlm_predictor=None,
    )
    assert actual is pages
    attach.assert_called_once_with(document, pages, window_size=expected[0], timeout=expected[1], threads=expected[2])
