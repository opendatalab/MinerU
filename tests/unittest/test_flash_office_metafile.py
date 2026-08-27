"""验证 Office WMF/EMF 跨平台渲染能力。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import struct
from unittest.mock import Mock
import zlib
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageFont
import pyclipper
import pytest
import pptx

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
from mineru.model.flash.office.metafile import (
    MetafileMalformedError,
    MetafileError,
    MetafileUnsupportedError,
    MetafileResourceLimitError,
    render_metafile,
)
from mineru.model.flash.office.metafile import parser as metafile_parser
from mineru.model.flash.office.metafile import render as metafile_render
from mineru.model.flash.office.metafile.geometry import FlattenBudget, PathBuilder, flatten_path, path_bounds
from mineru.model.flash.office.metafile.models import (
    ClipOperation,
    DrawImageCommand,
    DrawPathCommand,
    DrawTextCommand,
    GraphicsPath,
    Matrix,
    Pen,
    Rect,
)
from mineru.model.flash.office.legacy.officeart import OfficeArtRecord, decode_blip
from mineru.model.flash.office.pptx.pptx_converter import PptxConverter
from mineru.model.flash.office.xlsx.xlsx_converter import XlsxConverter
from mineru.utils.image_payload import extract_mineru_generated_svg_fallback

from _metafile_test_utils import (
    basic_wmf,
    build_emf,
    build_placeable_wmf,
    emf_angle_arc,
    emf_begin_path,
    emf_close_figure,
    emf_create_brush,
    emf_create_pen,
    emf_end_path,
    emf_font,
    emf_line_to,
    emf_move_to,
    emf_polybezier,
    emf_polyline_to,
    emf_record,
    emf_rectangle,
    emf_restoredc,
    emf_savedc,
    emf_select_object,
    emf_set_miter_limit,
    emf_set_text_align,
    emf_set_world_transform,
    emf_stroke_and_fill_path,
    emf_stroke_path,
    emf_stretch_dib,
    emf_text,
    emfplus_comment,
    wmf_move_to,
    wmf_record,
    wmf_set_text_align,
    wmf_textout,
)
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls
from _mtef_test_utils import build_equation_doc
from _office_image_mtef_test_utils import build_image_docx, build_image_pptx, build_image_xlsx
from _odf_test_utils import build_odp_fixture, build_ods_fixture, build_odt_fixture


def _open_result(payload: bytes) -> Image.Image:
    """解码渲染结果并返回脱离 BytesIO 生命周期的 RGBA 图片。"""
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGBA")


def _svg_data_uri_fallback(data_uri: str) -> tuple[Image.Image, tuple[int, int], bytes]:
    """解析 MinerU SVG data URI，并返回 fallback 图片、逻辑尺寸和 SVG。"""
    assert data_uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(data_uri.split(",", 1)[1])
    fallback, logical_width, logical_height = extract_mineru_generated_svg_fallback(svg)
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


def _compound_square_path(*, inner_reversed: bool, include_island: bool = False) -> GraphicsPath:
    """构造用于验证 winding 与 even-odd 的嵌套方形复合路径。"""
    builder = PathBuilder()
    for point in ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)):
        if builder.current is None:
            builder.move_to(point)
        else:
            builder.line_to(point)
    builder.close()
    inner = (
        ((25.0, 25.0), (25.0, 75.0), (75.0, 75.0), (75.0, 25.0))
        if inner_reversed
        else ((25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0))
    )
    builder.move_to(inner[0])
    for point in inner[1:]:
        builder.line_to(point)
    builder.close()
    if include_island:
        builder.move_to((40.0, 40.0))
        builder.line_to((60.0, 40.0))
        builder.line_to((60.0, 60.0))
        builder.line_to((40.0, 60.0))
        builder.close()
    return builder.build()


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


def test_emf_renders_png_jpeg_and_safe_svg() -> None:
    """验证同一 EMF 可输出三种格式且 SVG 不含活动或外部内容。"""
    data = build_emf(_basic_emf_records())

    png = render_metafile(data, output_format="png")
    jpeg = render_metafile(data, output_format="jpeg")
    svg = render_metafile(data, output_format="svg")

    assert png.media_type == "image/png"
    assert png.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert jpeg.media_type == "image/jpeg"
    assert jpeg.data.startswith(b"\xff\xd8\xff")
    assert svg.media_type == "image/svg+xml"
    assert svg.data.startswith(b'<svg xmlns="http://www.w3.org/2000/svg"')
    assert b"<script" not in svg.data
    assert b"http://" not in svg.data.replace(b"http://www.w3.org/2000/svg", b"")
    assert b"https://" not in svg.data
    assert (png.width, png.height) == (144, 144)
    assert not png.partial

    image = _open_result(png.data)
    assert image.getbbox() is not None
    assert image.getpixel((30, 30))[:3] == (255, 0, 0)
    assert image.getpixel((100, 30))[:3] == (0, 255, 0)
    assert image.getpixel((30, 110))[:3] == (0, 0, 255)
    assert image.getpixel((100, 110))[:3] == (255, 255, 255)
    assert image.getpixel((30, 90))[:3] == (255, 255, 255)


def test_alpha_dib_unpremultiplies_channels_before_compositing() -> None:
    """验证 AC_SRC_ALPHA 的 premultiplied BGRA 在 Pillow 合成前恢复为 straight RGBA。"""
    header = bytearray(40)
    struct.pack_into("<IiiHHIIiiII", header, 0, 40, 2, -1, 1, 32, 0, 8, 0, 0, 0, 0)
    command = DrawImageCommand(
        dib_header=bytes(header),
        bits=bytes((0, 0, 128, 128, 0, 64, 0, 64)),
        destination=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        source=None,
        rop=0,
        use_source_alpha=True,
    )

    decoded = metafile_render._decode_raw_alpha_dib(command)

    assert decoded is not None
    assert decoded.getpixel((0, 0)) == (255, 0, 0, 128)
    assert decoded.getpixel((1, 0)) == (0, 255, 0, 64)
    background = Image.new("RGBA", decoded.size, "white")
    background.alpha_composite(decoded)
    assert background.getpixel((0, 0)) == (255, 127, 127, 255)
    assert background.getpixel((1, 0)) == (191, 255, 191, 255)


def test_emf_save_restore_preserves_selected_objects_and_transform() -> None:
    """验证 SaveDC/RestoreDC 同时恢复 transform 与选中画刷。"""
    records = [
        emf_create_brush(1, 0x000000FF),
        emf_create_brush(2, 0x00FF0000),
        emf_select_object(1),
        emf_rectangle(0, 0, 20, 20),
        emf_savedc(),
        emf_set_world_transform((1, 0, 0, 1, 50, 0)),
        emf_select_object(2),
        emf_rectangle(0, 0, 20, 20),
        emf_restoredc(),
        emf_rectangle(25, 0, 45, 20),
    ]
    image = _open_result(render_metafile(build_emf(records)).data)

    assert image.getpixel((14, 14))[0] > 200
    assert image.getpixel((50, 14))[0] > 200
    assert image.getpixel((86, 14))[2] > 200


def test_emf_to_records_append_to_one_path_figure_and_update_current_position() -> None:
    """验证 Path bracket 内 To records 连续追加且 CloseFigure 后新建 figure。"""
    data = build_emf(
        [
            emf_begin_path(),
            emf_move_to(10, 50),
            emf_polyline_to([(20, 50), (30, 50)]),
            emf_polybezier([(40, 50), (50, 50), (60, 50)], to=True),
            emf_close_figure(),
            emf_line_to(70, 70),
            emf_end_path(),
            emf_stroke_path(),
            emf_line_to(90, 90),
        ]
    )

    document = metafile_parser.parse_metafile(data)
    commands = [command for command in document.commands if isinstance(command, DrawPathCommand)]

    assert [segment.verb for segment in commands[0].path.segments] == ["M", "L", "L", "C", "Z", "M", "L"]
    assert commands[0].path.segments[5].points == ((60.0, 50.0),)
    assert commands[1].path.segments[0].points == ((70.0, 70.0),)


def test_emf_non_to_polybezier_does_not_change_gdi_current_position() -> None:
    """验证 PolyBezier 不像 PolyBezierTo 那样更新 DC current position。"""
    data = build_emf(
        [
            emf_move_to(5, 5),
            emf_begin_path(),
            emf_polybezier([(20, 20), (30, 0), (50, 0), (60, 20)], to=False),
            emf_end_path(),
            emf_stroke_path(),
            emf_line_to(10, 5),
        ]
    )

    document = metafile_parser.parse_metafile(data)
    commands = [command for command in document.commands if isinstance(command, DrawPathCommand)]

    assert commands[1].path.segments[0].points == ((5.0, 5.0),)


@pytest.mark.parametrize(
    "data",
    [
        build_emf(
            [
                emf_set_text_align(1),
                emf_move_to(10, 50),
                emf_text("first", 900, 900, dx=None),
                emf_text("second", 800, 800, dx=None),
            ]
        ),
        build_placeable_wmf(
            [
                wmf_set_text_align(1),
                wmf_move_to(10, 50),
                wmf_textout("first", 900, 900),
                wmf_textout("second", 800, 800),
            ]
        ),
    ],
    ids=["emf", "wmf"],
)
def test_updatecp_text_without_explicit_spacing_advances_current_position(data: bytes) -> None:
    """验证 EMF/WMF 无显式 spacing 的 TA_UPDATECP 文本不会相互覆盖。"""
    commands = [command for command in metafile_parser.parse_metafile(data).commands if isinstance(command, DrawTextCommand)]

    assert len(commands) == 2
    assert commands[0].origin == (10.0, 50.0)
    assert commands[1].origin[0] > commands[0].origin[0]
    assert commands[1].origin[1] == pytest.approx(commands[0].origin[1])
    assert not commands[0].positions
    assert not commands[1].positions


@pytest.mark.parametrize(("text_align", "expected_x"), [(6, (40.0, 50.0)), (2, (30.0, 40.0))])
def test_explicit_spacing_applies_center_or_right_alignment_once(
    text_align: int,
    expected_x: tuple[float, float],
) -> None:
    """验证显式 DX 字符先按整串 alignment 平移，再从左侧 cell origin 绘制。"""
    document = metafile_parser.parse_metafile(build_emf([emf_set_text_align(text_align), emf_text("AB", 50, 50, dx=10)]))
    command = next(command for command in document.commands if isinstance(command, DrawTextCommand))
    positions, positioned_run = metafile_render._aligned_text_positions(command, Matrix(), list(command.positions))
    svg = metafile_render._svg_text_elements(command, Matrix())

    assert command.advance_end == (70.0, 50.0)
    assert positioned_run
    assert tuple(position[0] for position in positions) == expected_x
    assert all(position[1] == 50.0 for position in positions)
    assert 'text-anchor="start"' in svg


def test_rotated_text_preserves_left_and_right_anchor_without_clipping() -> None:
    """验证旋转文字围绕 GDI reference 定位，左右 anchor 均保留完整字形。"""
    font = ImageFont.load_default(size=20)
    left = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
    right = Image.new("RGBA", (180, 140), (0, 0, 0, 0))

    for layer, anchor in ((left, "la"), (right, "ra")):
        metafile_render._draw_rotated_text(
            layer,
            (90.0, 70.0),
            "Anchor",
            font=font,
            fill=(0, 0, 0, 255),
            anchor=anchor,
            rotation=35.0,
            underline=True,
            strikeout=True,
        )

    left_pixels = sum(value > 0 for value in left.getchannel("A").getdata())
    right_pixels = sum(value > 0 for value in right.getchannel("A").getdata())

    assert left.getbbox() is not None and right.getbbox() is not None
    assert right_pixels == pytest.approx(left_pixels, rel=0.05)
    assert left.getbbox()[2] > 90
    assert right.getbbox()[0] < 90


@pytest.mark.parametrize(("sweep_angle", "expected_end_y"), [(90.0, -10.0), (-90.0, 10.0)])
def test_emf_anglearc_uses_sweep_direction_and_updates_current_position(
    sweep_angle: float,
    expected_end_y: float,
) -> None:
    """验证 AngleArc 绘制起始连线、使用正确 sweep 方向并更新 current position。"""
    data = build_emf(
        [emf_angle_arc(0, 0, 10, 0.0, sweep_angle), emf_line_to(20, 20)],
        bounds=(-20, -20, 30, 30),
    )
    commands = [command for command in metafile_parser.parse_metafile(data).commands if isinstance(command, DrawPathCommand)]

    assert len(commands[0].path.segments) == 14
    assert commands[0].path.segments[0].points == ((0.0, 0.0),)
    assert commands[0].path.segments[1].points == ((10.0, 0.0),)
    assert commands[0].path.segments[-1].points[0] == pytest.approx((0.0, expected_end_y))
    assert commands[1].path.segments[0].points[0] == pytest.approx((0.0, expected_end_y))


def test_emf_stroke_and_fill_closes_every_open_figure() -> None:
    """验证 StrokeAndFillPath 在描边前显式闭合全部开放 figure。"""
    data = build_emf(
        [
            emf_begin_path(),
            emf_move_to(10, 10),
            emf_polyline_to([(40, 10), (40, 40)]),
            emf_move_to(60, 60),
            emf_polyline_to([(90, 60), (90, 90)]),
            emf_end_path(),
            emf_stroke_and_fill_path(),
        ]
    )

    command = next(command for command in metafile_parser.parse_metafile(data).commands if isinstance(command, DrawPathCommand))

    assert [segment.verb for segment in command.path.segments] == ["M", "L", "L", "Z", "M", "L", "L", "Z"]


def test_compound_path_masks_preserve_nonzero_and_evenodd_topology() -> None:
    """验证 winding、alternate、孔洞和孔洞内岛屿的 mask 拓扑。"""
    opposite = _compound_square_path(inner_reversed=True)
    same = _compound_square_path(inner_reversed=False)
    island = _compound_square_path(inner_reversed=True, include_island=True)
    self_intersecting_builder = PathBuilder()
    self_intersecting_builder.move_to((10.0, 10.0))
    self_intersecting_builder.line_to((90.0, 90.0))
    self_intersecting_builder.line_to((10.0, 90.0))
    self_intersecting_builder.line_to((90.0, 10.0))
    self_intersecting_builder.close()

    winding_hole = metafile_render._path_mask(opposite, Matrix(), (100, 100), "nonzero")
    winding_filled = metafile_render._path_mask(same, Matrix(), (100, 100), "nonzero")
    evenodd_hole = metafile_render._path_mask(same, Matrix(), (100, 100), "evenodd")
    nested_island = metafile_render._path_mask(island, Matrix(), (100, 100), "nonzero")
    self_intersecting = metafile_render._path_mask(self_intersecting_builder.build(), Matrix(), (100, 100), "evenodd")

    assert winding_hole.getpixel((20, 20)) == 255
    assert winding_hole.getpixel((30, 30)) == 0
    assert winding_filled.getpixel((50, 50)) == 255
    assert evenodd_hole.getpixel((50, 50)) == 0
    assert nested_island.getpixel((30, 30)) == 0
    assert nested_island.getpixel((50, 50)) == 255
    assert self_intersecting.getbbox() is not None
    assert self_intersecting.getpixel((50, 25)) == 255


def test_polytree_paint_handles_deep_nesting_without_python_recursion() -> None:
    """验证恶意深层 PolyTree 使用显式栈绘制且继续计入 flatten 预算。"""
    root = pyclipper.PyPolyNode()
    parent = root
    for index in range(1100):
        child = pyclipper.PyPolyNode()
        child.Contour = [
            (index, index),
            (2201 - index, index),
            (2201 - index, 2201 - index),
            (index, 2201 - index),
        ]
        child.IsHole = bool(index & 1)
        parent.Childs.append(child)
        parent = child
    budget = FlattenBudget(limit=5000)

    metafile_render._paint_polytree(Image.new("L", (10, 10)), root, budget)

    assert budget.used == 4400


def test_clip_mask_reuses_compound_path_topology_and_combine_modes() -> None:
    """验证裁剪 copy/and/or/xor/diff 与路径填充共享同一 winding 结果。"""
    outer = _compound_square_path(inner_reversed=False)
    inner_builder = PathBuilder()
    inner_builder.move_to((35.0, 35.0))
    inner_builder.line_to((65.0, 35.0))
    inner_builder.line_to((65.0, 65.0))
    inner_builder.line_to((35.0, 65.0))
    inner_builder.close()
    inner = inner_builder.build()
    copy_outer = ClipOperation(outer, "copy", "nonzero")
    copy_inner = ClipOperation(inner, "copy", "nonzero")

    intersection = metafile_render._clip_mask((copy_outer, ClipOperation(inner, "and", "nonzero")), Matrix(), (100, 100))
    union = metafile_render._clip_mask((copy_inner, ClipOperation(outer, "or", "nonzero")), Matrix(), (100, 100))
    xor = metafile_render._clip_mask((copy_inner, ClipOperation(outer, "xor", "nonzero")), Matrix(), (100, 100))
    difference = metafile_render._clip_mask((copy_outer, ClipOperation(inner, "diff", "nonzero")), Matrix(), (100, 100))

    assert intersection is not None and intersection.getpixel((50, 50)) == 255 and intersection.getpixel((20, 20)) == 0
    assert union is not None and union.getpixel((50, 50)) == 255 and union.getpixel((20, 20)) == 255
    assert xor is not None and xor.getpixel((50, 50)) == 0 and xor.getpixel((20, 20)) == 255
    assert difference is not None and difference.getpixel((50, 50)) == 0 and difference.getpixel((20, 20)) == 255


def test_adaptive_cubic_flattening_respects_error_and_point_budget() -> None:
    """验证更严格 flatness 产生更多点且离散点预算可以稳定终止。"""
    builder = PathBuilder()
    builder.move_to((0.0, 0.0))
    builder.cubic_to((0.0, 100.0), (100.0, 100.0), (100.0, 0.0))
    path = builder.build()

    coarse = flatten_path(path, flatness=10.0)[0][0]
    fine = flatten_path(path, flatness=0.1)[0][0]

    assert fine[-1] == coarse[-1] == (100.0, 0.0)
    assert len(fine) > len(coarse)
    with pytest.raises(MetafileResourceLimitError, match="max_flattened_points=2"):
        flatten_path(path, flatness=0.1, budget=FlattenBudget(limit=2))


def test_stroke_mask_honors_caps_and_svg_miter_limit() -> None:
    """验证 flat/square/round 端帽差异以及 SVG 保留 DC miter limit。"""
    subpaths = [([(20.0, 50.0), (80.0, 50.0)], False)]
    flat = metafile_render._stroke_mask(
        subpaths,
        size=(100, 100),
        width=10.0,
        pen=Pen(cap="flat"),
        dashes=(),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    square = metafile_render._stroke_mask(
        subpaths,
        size=(100, 100),
        width=10.0,
        pen=Pen(cap="square"),
        dashes=(),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    round_cap = metafile_render._stroke_mask(
        subpaths,
        size=(100, 100),
        width=10.0,
        pen=Pen(cap="round"),
        dashes=(),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    dashed = metafile_render._stroke_mask(
        subpaths,
        size=(100, 100),
        width=6.0,
        pen=Pen(cap="flat"),
        dashes=(10.0, 10.0),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    corner = [([(20.0, 80.0), (50.0, 20.0), (80.0, 80.0)], False)]
    miter = metafile_render._stroke_mask(
        corner,
        size=(100, 100),
        width=12.0,
        pen=Pen(cap="flat", join="miter"),
        dashes=(),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    bevel = metafile_render._stroke_mask(
        corner,
        size=(100, 100),
        width=12.0,
        pen=Pen(cap="flat", join="bevel"),
        dashes=(),
        miter_limit=10.0,
        budget=FlattenBudget(),
    )
    svg = render_metafile(
        build_emf(
            [
                emf_set_miter_limit(3.5),
                emf_begin_path(),
                emf_move_to(10, 10),
                emf_line_to(90, 90),
                emf_end_path(),
                emf_stroke_path(),
            ]
        ),
        output_format="svg",
    ).data

    assert flat.getpixel((16, 50)) == 0
    assert square.getpixel((16, 50)) == 255
    assert round_cap.getpixel((16, 50)) == 255
    assert dashed.getpixel((25, 50)) == 255
    assert dashed.getpixel((35, 50)) == 0
    assert miter.getbbox() is not None and bevel.getbbox() is not None
    assert miter.getbbox()[1] < bevel.getbbox()[1]
    assert b'stroke-miterlimit="3.5"' in svg


def test_bitmap_source_crop_preserves_empty_and_partial_destination_mapping() -> None:
    """验证完全越界 source 不绘制，部分交集只映射到对应 destination 子区域。"""
    image = Image.new("RGBA", (2, 2), "red")
    destination = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))

    outside, outside_fraction = metafile_render._crop_source(image, Rect(10.0, 10.0, 12.0, 12.0))
    partial, partial_fraction = metafile_render._crop_source(image, Rect(-1.0, 0.0, 1.0, 2.0))
    mirrored, mirrored_fraction = metafile_render._crop_source(image, Rect(3.0, 0.0, -1.0, 2.0))

    assert outside is None
    assert outside_fraction == Rect(0.0, 0.0, 0.0, 0.0)
    assert partial is not None and partial.size == (1, 2)
    assert partial_fraction == Rect(0.5, 0.0, 1.0, 1.0)
    assert mirrored is not None and mirrored.size == (2, 2)
    assert mirrored_fraction == Rect(0.25, 0.0, 0.75, 1.0)
    assert metafile_render._crop_destination(destination, partial_fraction) == (
        (50.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (50.0, 100.0),
    )


def test_placeable_wmf_renders_vector_content() -> None:
    """验证 placeable WMF 的 checksum、对象表和逻辑尺寸进入 PNG。"""
    result = render_metafile(basic_wmf())
    image = _open_result(result.data)

    assert result.source_format == "wmf"
    assert (result.width, result.height) == (144, 144)
    assert not result.partial
    assert image.getbbox() is not None
    assert image.getpixel((72, 72))[1] > 150


def test_wmf_roundrect_uses_record_parameter_order() -> None:
    """验证 META_ROUNDRECT 按 Height/Width/Bottom/Right/Top/Left 解码。"""
    payload = struct.pack("<hhhhhh", 100, 300, 900, 800, 200, 100)
    document = metafile_parser.parse_metafile(build_placeable_wmf([wmf_record(0x061C, payload)]))
    command = next(command for command in document.commands if isinstance(command, DrawPathCommand))

    assert path_bounds(command.path) == Rect(100.0, 200.0, 800.0, 900.0)
    assert command.path.segments[0].points == ((250.0, 200.0),)


def test_vector_only_metafile_uses_4x_antialiasing_and_2x_svg_fallback() -> None:
    """验证矢量公式保持逻辑尺寸，同时使用 4× 栅格和 2× DOCX fallback。"""
    document = metafile_parser.parse_metafile(basic_wmf())
    svg = render_metafile(basic_wmf(), output_format="svg").data
    fallback, logical_width, logical_height = extract_mineru_generated_svg_fallback(svg)

    assert metafile_render._supersample_factor(document) == 4
    assert metafile_render._svg_fallback_scale(document) == 2
    assert (logical_width, logical_height) == (144, 144)
    with Image.open(BytesIO(fallback)) as image:
        assert image.size == (288, 288)
        assert image.info["dpi"][0] == pytest.approx(192, abs=1)


def test_oversized_generated_svg_degrades_to_png_for_office_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证超过消费者上限的内部 SVG 不会进入 HTML/DOCX，而是降级为 PNG。"""
    data = build_emf([emf_stretch_dib()])
    image_element = Mock(side_effect=AssertionError("oversized fallback should stop before bitmap SVG encoding"))
    monkeypatch.setattr(metafile_render, "MAX_GENERATED_SVG_BYTES", 1)
    monkeypatch.setattr(metafile_render, "_svg_image_element", image_element)
    monkeypatch.setattr(office_image, "is_windows_environment", lambda: False)

    with pytest.raises(MetafileResourceLimitError, match="max_generated_svg_bytes"):
        render_metafile(data, output_format="svg")

    data_uri = office_image.serialize_office_image(data, part_name="image.wmf", content_type="image/wmf")

    assert data_uri is not None
    assert data_uri.startswith("data:image/png;base64,")
    image_element.assert_not_called()


def test_emfplus_dual_uses_emf_fallback_and_only_is_rejected() -> None:
    """验证 EMF+ Dual 播放 EMF fallback，Only 返回稳定不支持错误。"""
    records = [emfplus_comment(dual=True), emf_create_brush(1, 0x0000FF00), emf_select_object(1), emf_rectangle(5, 5, 95, 95)]
    dual = render_metafile(build_emf(records))

    assert dual.emfplus_mode == "dual"
    assert _open_result(dual.data).getbbox() is not None

    only_data = build_emf([emfplus_comment(dual=False), emf_rectangle(5, 5, 95, 95)])
    with pytest.raises(MetafileUnsupportedError, match=r"EMF\+ Only"):
        render_metafile(only_data)


def test_unknown_drawing_record_keeps_partial_result() -> None:
    """验证未知绘图 record 被跳过但其他内容继续输出。"""
    data = build_emf([emf_record(118, b"\x00" * 16), emf_rectangle(5, 5, 95, 95)])
    result = render_metafile(data)

    assert result.partial
    assert any(diagnostic.code == "unsupported_emf_record" for diagnostic in result.diagnostics)
    assert _open_result(result.data).getbbox() is not None


def test_malformed_record_and_emf_signature_fail_closed() -> None:
    """验证截断 record 与伪造签名均返回稳定 malformed 错误。"""
    valid = bytearray(build_emf([emf_rectangle(5, 5, 95, 95)]))
    valid[92:96] = (0x7FFFFFFC).to_bytes(4, "little")
    with pytest.raises(MetafileMalformedError):
        render_metafile(bytes(valid))
    with pytest.raises(MetafileMalformedError):
        render_metafile(b"not-a-metafile")


def test_canvas_is_downscaled_to_fixed_pixel_budget() -> None:
    """验证巨大物理 frame 只会触发等比缩放而不会分配无界画布。"""
    data = build_emf([emf_rectangle(0, 0, 100, 100)], frame=(0, 0, 254000, 254000))
    result = render_metafile(data)

    assert result.width * result.height <= 16_000_000
    assert max(result.width, result.height) <= 8192
    assert any(diagnostic.code == "canvas_downscaled" for diagnostic in result.diagnostics)


def test_fixed_record_point_and_state_limits_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证点数和 SaveDC 深度使用 metafile 专属固定安全预算。"""
    polygon_payload = (
        b"\x00" * 16
        + (3).to_bytes(4, "little")
        + b"".join(value.to_bytes(4, "little", signed=True) for value in (0, 0, 50, 0, 25, 50))
    )
    monkeypatch.setattr(metafile_parser, "MAX_POINTS_PER_RECORD", 2)
    with pytest.raises(MetafileResourceLimitError, match="max_points_per_record"):
        render_metafile(build_emf([emf_record(3, polygon_payload)]))

    monkeypatch.setattr(metafile_parser, "MAX_POINTS_PER_RECORD", 1_000_000)
    monkeypatch.setattr(metafile_parser, "MAX_STATE_DEPTH", 1)
    with pytest.raises(MetafileResourceLimitError, match="max_state_depth"):
        render_metafile(build_emf([emf_savedc(), emf_savedc(), emf_rectangle(1, 1, 10, 10)]))

    oversized_dib = bytearray(emf_stretch_dib())
    oversized_dib[84:88] = (100_000).to_bytes(4, "little", signed=True)
    oversized_dib[88:92] = (100_000).to_bytes(4, "little", signed=True)
    with pytest.raises(MetafileResourceLimitError, match="DIB dimensions"):
        render_metafile(build_emf([bytes(oversized_dib)]))

    monkeypatch.setattr(metafile_render, "MAX_RENDER_WORK_PIXELS", 1)
    with pytest.raises(MetafileResourceLimitError, match="max_render_work_pixels"):
        render_metafile(build_emf([emf_rectangle(1, 1, 10, 10)]))


def test_svg_text_escapes_markup_characters() -> None:
    """验证 EMF 文字无法把脚本或 XML 标签注入 SVG。"""
    data = build_emf([emf_font(1), emf_select_object(1), emf_text("<script>&", 5, 30)])
    svg = render_metafile(data, output_format="svg").data

    assert b"<script>" not in svg
    assert b"&lt;" in svg
    assert b"&gt;" in svg
    assert b"&amp;" in svg
    assert svg.index(b"<rect") < svg.index(b"<text")


def test_raster_operations_use_exact_bitwise_channels() -> None:
    """验证常见 ROP2/ROP3 不使用颜色明暗近似替代位运算。"""
    destination = bytes((0x0F, 0x55, 0xAA))
    source = bytes((0x33, 0x0F, 0xF0))

    assert metafile_render._bitwise_channel_bytes(destination, source, "xor") == bytes((0x3C, 0x5A, 0x5A))
    assert metafile_render._bitwise_channel_bytes(destination, source, "and") == bytes((0x03, 0x05, 0xA0))
    assert metafile_render._bitwise_channel_bytes(destination, source, "or") == bytes((0x3F, 0x5F, 0xFA))
    assert metafile_render._bitwise_channel_bytes(destination, source, "not_source") == bytes((0xCC, 0xF0, 0x0F))
    assert metafile_render._bitwise_channel_bytes(destination, source, "not_xor") == bytes((0xC3, 0xA5, 0xA5))


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


@pytest.mark.parametrize("data", [build_emf([emf_rectangle(5, 5, 95, 95)]), basic_wmf()])
def test_every_truncated_prefix_fails_without_unexpected_exception(data: bytes) -> None:
    """验证每个截断位置都只产生稳定格式错误，不泄漏 struct/Pillow 异常。"""
    for size in range(len(data)):
        with pytest.raises(MetafileError):
            render_metafile(data[:size])


@pytest.mark.parametrize("name", ["docx-icon.emf", "generic-icon.emf", "pptx-icon.emf", "xlsx-icon.emf"])
def test_real_python_pptx_emf_icons_render_without_partial_result(name: str) -> None:
    """验证依赖包内真实 Office EMF 图标的文字、alpha 和裁剪组合。"""
    template = Path(pptx.__file__).resolve().parent / "templates" / name
    result = render_metafile(template.read_bytes())
    image = _open_result(result.data)

    assert not result.partial
    assert result.diagnostics == ()
    assert image.width >= 100
    assert image.height >= 90
    assert image.getbbox() is not None
    assert image.getchannel("A").getextrema()[1] == 255
