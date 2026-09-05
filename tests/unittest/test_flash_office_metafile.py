"""验证独立 WMF/EMF 渲染包的 Office 集成。"""

from __future__ import annotations

import base64
import struct
import zlib
from io import BytesIO
from unittest.mock import Mock
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls
from _metafile_test_utils import (
    basic_wmf,
    build_emf,
    emf_create_brush,
    emf_create_pen,
    emf_font,
    emf_rectangle,
    emf_select_object,
    emf_stretch_dib,
    emf_text,
)
from _mtef_test_utils import build_equation_doc
from _odf_test_utils import build_odp_fixture, build_ods_fixture, build_odt_fixture
from _office_image_mtef_test_utils import build_image_docx, build_image_pptx, build_image_xlsx
from metafile_render import (
    MetafileResourceLimitError,
    render_metafile,
)
from PIL import Image

from mineru.model.flash import (
    DocModel,
    DocxModel,
    OdpModel,
    OdsModel,
    OdtModel,
    PptModel,
    PptxModel,
    RtfModel,
    XlsModel,
    XlsxModel,
)
from mineru.model.flash.office import image as office_image
from mineru.model.flash.office.legacy.officeart import OfficeArtRecord, decode_blip
from mineru.model.flash.office.pptx.pptx_converter import PptxConverter
from mineru.model.flash.office.xlsx.xlsx_converter import XlsxConverter
from mineru.utils.image_payload import extract_generated_svg_fallback


def _open_result(payload: bytes) -> Image.Image:
    """解码渲染结果并返回脱离 BytesIO 生命周期的 RGBA 图片。"""
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGBA")


def _svg_data_uri_fallback(data_uri: str) -> tuple[Image.Image, tuple[int, int], bytes]:
    """解析 MinerU SVG data URI，并返回 fallback 图片、逻辑尺寸和 SVG。"""
    assert data_uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(data_uri.split(",", 1)[1])
    fallback, logical_width, logical_height = extract_generated_svg_fallback(svg)
    return _open_result(fallback), (logical_width, logical_height), svg


def _basic_emf_records() -> list[bytes]:
    """返回覆盖画笔、画刷、文字和 DIB 的基础 EMF records。"""
    return [
        emf_create_pen(1, 0x00FF0000, width=2),
        emf_create_brush(2, 0x0000FF00),
        emf_select_object(1),
        emf_select_object(2),
        emf_rectangle(5, 5, 95, 95),
        emf_stretch_dib(),
        emf_font(3),
        emf_select_object(3),
        emf_text("EMF", 20, 50, dx=18),
    ]


def _collect_image_data_uris(value: object) -> list[str]:
    """递归收集 raw model-list 中的 image_base64 字段。"""
    if isinstance(value, dict):
        result = [value["image_base64"]] if isinstance(value.get("image_base64"), str) else []
        for item in value.values():
            result.extend(_collect_image_data_uris(item))
        return result
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_collect_image_data_uris(item))
        return result
    return []


def _replace_zip_member(package: bytes, member_name: str, payload: bytes) -> bytes:
    """替换测试 ZIP 中的单个成员并保留其他成员及压缩方式。"""
    output = BytesIO()
    with ZipFile(BytesIO(package)) as source, ZipFile(output, "w") as target:
        for info in source.infolist():
            target.writestr(info, payload if info.filename == member_name else source.read(info.filename))
    return output.getvalue()


def _build_doc_with_wmf_preview(payload: bytes) -> bytes:
    """构造 Native 失败后使用 WMF PICF 预览的 DOC。"""
    return build_equation_doc(
        [(1, b"invalid")],
        preview_storage_ids={1},
        preview_payloads={1: payload},
    )


def _build_ppt_with_wmf_preview(payload: bytes) -> bytes:
    """构造 Native 失败后使用 OfficeArt WMF 预览的 PPT。"""
    return build_equation_ppt([b"invalid"], preview_payload=payload)


def _build_xls_with_wmf_preview(payload: bytes) -> bytes:
    """构造 Native 失败后使用 OfficeArt WMF 预览的 XLS。"""
    return build_equation_xls([(1, b"invalid")], preview_payload=payload)


def test_oversized_generated_svg_degrades_to_png_for_office_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    """通过公开 API 的资源错误验证 Office 消费者可靠回退到 PNG。"""
    data = build_emf([emf_stretch_dib()])
    png = render_metafile(data, output_format="png")
    render = Mock(side_effect=[MetafileResourceLimitError("generated SVG exceeds budget"), png])
    monkeypatch.setattr(office_image, "render_metafile", render)
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    data_uri = office_image.serialize_office_image(data, part_name="image.emf", content_type="image/emf")
    assert data_uri is not None
    assert data_uri.startswith("data:image/png;base64,")
    assert [call.kwargs["output_format"] for call in render.call_args_list] == ["svg", "png"]


def test_non_windows_office_serializer_returns_generated_svg(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证非 Windows Office 图片入口返回安全 SVG 和 PNG fallback。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    data_uri = office_image.serialize_office_image(
        build_emf(_basic_emf_records()),
        part_name="/word/media/image1.emf",
        content_type="image/x-emf",
    )

    assert data_uri is not None
    fallback, logical_size, svg = _svg_data_uri_fallback(data_uri)
    assert logical_size == (144, 144)
    assert fallback.getbbox() is not None
    assert b"<path" in svg


def test_officeart_emu_size_controls_standard_wmf_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证无 placeable header 的 OfficeArt WMF 使用 ptSize EMU 定标。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    standard_wmf = basic_wmf()[22:]
    data_uri = office_image.serialize_office_image(
        standard_wmf,
        part_name="picture.wmf",
        content_type="image/wmf",
        render_size_emu=(914_400, 457_200),
    )

    assert data_uri is not None
    fallback, logical_size, _svg = _svg_data_uri_fallback(data_uri)
    assert logical_size == (144, 72)
    assert fallback.size == (288, 144)


def test_officeart_metafile_header_preserves_payload_and_emu_size() -> None:
    """验证 BLIP 解包保留 WMF bytes，并读取规范 ptSize EMU。"""
    standard_wmf = basic_wmf()[22:]
    compressed = zlib.compress(standard_wmf)
    metafile_header = bytearray(34)
    struct.pack_into("<I", metafile_header, 0, len(standard_wmf))
    struct.pack_into("<ii", metafile_header, 20, 914_400, 457_200)
    struct.pack_into("<I", metafile_header, 28, len(compressed))
    metafile_header[32:34] = b"\x00\xfe"
    record = OfficeArtRecord(
        offset=0,
        version=0,
        instance=0,
        record_type=0xF01B,
        payload=b"\x00" * 16 + bytes(metafile_header) + compressed,
    )

    decoded = decode_blip(record)

    assert decoded is not None
    assert decoded.data == standard_wmf
    assert decoded.render_size_emu == (914_400, 457_200)


def test_windows_office_serializer_prefers_svg_then_uses_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Windows 优先保留 SVG，并在引擎失败时调用原生 GDI。"""
    data = build_emf(_basic_emf_records())
    native = Mock(return_value="data:image/png;base64,native")
    generated = Mock(return_value="data:image/svg+xml;base64,generated")
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: True)
    monkeypatch.setattr(office_image, "_serialize_native_metafile", native)
    monkeypatch.setattr(office_image, "_serialize_cross_platform_metafile", generated)

    assert office_image.serialize_office_image(data) == "data:image/svg+xml;base64,generated"
    native.assert_not_called()

    generated.return_value = None
    assert office_image.serialize_office_image(data) == "data:image/png;base64,native"
    native.assert_called_once()


def test_pptx_picture_path_uses_cross_platform_metafile_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 PPTX 普通 picture 不再绕过共享 WMF/EMF 序列化入口。"""
    converter = PptxConverter()
    data = build_emf(_basic_emf_records())

    def fake_image_data(_shape: object) -> tuple[bytes, str]:
        """返回固定 EMF 载荷以隔离 shape relationship 解析。"""
        return data, "image/x-emf"

    monkeypatch.setattr(converter, "_get_shape_image_data", fake_image_data)
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    converter._handle_pictures(object())

    assert converter.cur_page[0]["image_base64"].startswith("data:image/svg+xml;base64,")


def test_xlsx_wps_cell_image_uses_cross_platform_metafile_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 WPS DISPIMG 的 cell-image 分支统一调用 Office 图片序列化器。"""
    package = BytesIO()
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/media/image1.emf", build_emf(_basic_emf_records()))
    converter = XlsxConverter()
    converter.zf = ZipFile(BytesIO(package.getvalue()))
    converter.cell_image_map = {"image-id": "media/image1.emf"}
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    try:
        html = converter._resolve_cell_image('DISPIMG("image-id")')
    finally:
        converter.zf.close()

    assert html.startswith('<img src="data:image/svg+xml;base64,')


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_image_docx),
        (PptxModel(), build_image_pptx),
        (XlsxModel(), build_image_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_modern_office_models_emit_rendered_emf_svg(
    model: object,
    builder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 DOCX/PPTX/XLSX 完整模型链都产出安全 EMF SVG。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    package = builder(build_emf(_basic_emf_records()))  # type: ignore[operator]
    pages = model.predict(BytesIO(package))  # type: ignore[attr-defined]
    images = _collect_image_data_uris(pages)

    assert images
    assert all(image.startswith("data:image/svg+xml;base64,") for image in images)
    _fallback, logical_size, _svg = _svg_data_uri_fallback(images[0])
    assert logical_size == (144, 144)


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocModel(), _build_doc_with_wmf_preview),
        (PptModel(), _build_ppt_with_wmf_preview),
        (XlsModel(), _build_xls_with_wmf_preview),
    ],
    ids=["doc", "ppt", "xls"],
)
def test_legacy_office_models_emit_rendered_wmf_svg(
    model: object,
    builder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 DOC/PPT/XLS 的 OfficeArt/PICF WMF 预览进入跨平台 SVG。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    pages = model.predict(BytesIO(builder(basic_wmf())))  # type: ignore[attr-defined,operator]
    images = _collect_image_data_uris(pages)

    assert images
    assert all(image.startswith("data:image/svg+xml;base64,") for image in images)
    fallback, logical_size, _svg = _svg_data_uri_fallback(images[0])
    assert logical_size == (144, 144)
    assert fallback.size == (288, 288)


def test_rtf_model_emf_picture_emits_rendered_svg(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 RTF emfblip 从 pict 捕获一路进入跨平台 SVG。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    emf = build_emf(_basic_emf_records())
    rtf = b"{\\rtf1 before {\\pict\\emfblip " + emf.hex().encode("ascii") + b"} after\\par}"
    images = _collect_image_data_uris(RtfModel().predict(BytesIO(rtf)))

    assert images
    assert all(image.startswith("data:image/svg+xml;base64,") for image in images)


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (OdtModel(), build_odt_fixture),
        (OdsModel(), build_ods_fixture),
        (OdpModel(), build_odp_fixture),
    ],
    ids=["odt", "ods", "odp"],
)
def test_odf_models_emit_rendered_emf_svg(
    model: object,
    builder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 ODT/ODS/ODP 包内 WMF/EMF 图片都复用统一渲染入口。"""
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)
    package = builder()  # type: ignore[operator]
    package = _replace_zip_member(package, "Pictures/pixel.png", build_emf(_basic_emf_records()))
    images = _collect_image_data_uris(model.predict(BytesIO(package)))  # type: ignore[attr-defined]

    assert images
    assert all(image.startswith("data:image/svg+xml;base64,") for image in images)
