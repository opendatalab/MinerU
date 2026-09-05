"""验证 metafile-render 0.2 的 Only 图片经 Office 解析与四种消费者输出。"""

from __future__ import annotations

from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from _office_image_mtef_test_utils import build_image_docx
from packaging.version import Version
from PIL import Image
from pypdf import PdfReader

from mineru.backend.analyze import doc_analyze
from mineru.model.flash.office import image as office_image
from mineru.render import render_docx, render_epub, render_html, render_pdf

pytestmark = pytest.mark.skipif(
    Version(version("metafile-render")) < Version("0.2.0"),
    reason="EMF+ Only feature validation requires metafile-render 0.2.0; 0.1 remains a valid base dependency",
)
_FIXTURES = Path(__file__).parent / "fixtures" / "emfplus"


@pytest.mark.parametrize("scene", ["geometry", "fallback"])
def test_only_docx_flows_into_html_docx_pdf_epub(scene: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 Only 图片经 DOCX 解析进入全部消费端，安全 SVG 仍可提取 PNG fallback。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    package = build_image_docx((_FIXTURES / f"{scene}-only.emf").read_bytes())
    middle, _ = doc_analyze(package, effort="flash", image_analysis=False, file_suffix="docx")
    html = render_html(middle)
    assert "data:image/svg+xml;base64," in html
    for payload in (render_docx(middle), render_epub(middle)):
        with ZipFile(BytesIO(payload)) as archive:
            images = [archive.read(name) for name in archive.namelist() if name.endswith(".png")]
        assert images
        with Image.open(BytesIO(images[0])) as image:
            assert image.width > 100 and image.height > 100
            assert image.convert("RGB").getextrema() != ((255, 255), (255, 255), (255, 255))
    pdf = PdfReader(BytesIO(render_pdf(middle)))
    assert pdf.pages
    assert any(page.images for page in pdf.pages)


def test_only_partial_reports_diagnostics_and_keeps_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """样式近似的 Only 仍输出 SVG，并把部分结果诊断交给 Office 日志。"""
    warning = Mock()
    monkeypatch.setattr(office_image.logger, "warning", warning)
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    result = office_image.serialize_office_image((_FIXTURES / "fallback-only.emf").read_bytes())
    assert result is not None and result.startswith("data:image/svg+xml;base64,")
    assert any(
        "Rendered partial EMF" in str(call) and "emfplus_brush_approximation" in str(call) for call in warning.call_args_list
    )


def test_broken_only_uses_existing_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """签名有效但损坏的 Only 文件沿用原有失败兜底。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    data = (_FIXTURES / "geometry-only.emf").read_bytes()[:48]
    assert office_image.serialize_office_image(data) == office_image.get_standard_vector_placeholder_data_uri()
