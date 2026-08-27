# Copyright (c) Opendatalab. All rights reserved.
"""统一 WMF/EMF 图元文档的 Pillow 与安全 SVG 后端。"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from math import ceil, floor, hypot

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError
import pyclipper

from .font import load_font
from .geometry import FlattenBudget, flatten_path, transform_path
from .limits import MAX_CANVAS_PIXELS, MAX_EMBEDDED_BITMAP_BYTES, MAX_RENDER_WORK_PIXELS
from .models import (
    Brush,
    ClipOperation,
    ClipStack,
    Color,
    DrawImageCommand,
    DrawPathCommand,
    DrawTextCommand,
    GraphicsPath,
    Matrix,
    MetafileDocument,
    MetafileMalformedError,
    MetafileOutputFormat,
    MetafileResourceLimitError,
    Pen,
    Point,
    Rect,
)


_SRCCOPY = 0x00CC0020
_SRCAND = 0x008800C6
_SRCPAINT = 0x00EE0086
_SRCINVERT = 0x00660046
_DSTINVERT = 0x00550009
_BLACKNESS = 0x00000042
_WHITENESS = 0x00FF0062

_TA_RIGHT = 0x0002
_TA_CENTER = 0x0006
_TA_BOTTOM = 0x0008
_TA_BASELINE = 0x0018

_CLIPPER_SCALE = 256
_CLIPPER_COORD_LIMIT = (1 << 61) / _CLIPPER_SCALE


def _document_matrix(document: MetafileDocument, *, raster_scale: int = 1) -> Matrix:
    """把统一 device 坐标映射到最终输出像素。"""
    bounds = document.bounds.normalized()
    scale_x = document.width * raster_scale / bounds.width
    scale_y = document.height * raster_scale / bounds.height
    return Matrix(a=scale_x, d=scale_y, e=-bounds.left * scale_x, f=-bounds.top * scale_y)


def _mapped_rect(rect: Rect, matrix: Matrix) -> Rect:
    """把矩形四角变换后返回轴对齐像素包围盒。"""
    corners = [
        matrix.transform_point((rect.left, rect.top)),
        matrix.transform_point((rect.right, rect.top)),
        matrix.transform_point((rect.right, rect.bottom)),
        matrix.transform_point((rect.left, rect.bottom)),
    ]
    return Rect(
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )


def _dib_to_bmp(command: DrawImageCommand) -> bytes:
    """为分离的 DIB header 和像素数据补齐 BMP 文件头。"""
    if len(command.dib_header) < 4:
        raise MetafileMalformedError("DIB header is truncated")
    pixel_offset = 14 + len(command.dib_header)
    total_size = pixel_offset + len(command.bits)
    if total_size > MAX_EMBEDDED_BITMAP_BYTES + 14:
        raise MetafileResourceLimitError("decoded BMP exceeds the embedded bitmap byte budget")
    import struct

    return b"BM" + struct.pack("<IHHI", total_size, 0, 0, pixel_offset) + command.dib_header + command.bits


def _decode_raw_alpha_dib(command: DrawImageCommand) -> Image.Image | None:
    """对 AC_SRC_ALPHA 的常见 32 位 DIB 保留原始 alpha 字节。"""
    header = command.dib_header
    if len(header) < 40:
        return None
    import struct

    header_size, width, signed_height = struct.unpack_from("<Iii", header, 0)
    bit_count = struct.unpack_from("<H", header, 14)[0]
    compression = struct.unpack_from("<I", header, 16)[0]
    if header_size < 40 or width <= 0 or signed_height == 0 or bit_count != 32 or compression not in {0, 3, 6}:
        return None
    height = abs(signed_height)
    if width * height > MAX_CANVAS_PIXELS or len(command.bits) < width * height * 4:
        raise MetafileResourceLimitError("32-bit DIB exceeds pixel budget or is truncated")
    orientation = -1 if signed_height > 0 else 1
    try:
        return Image.frombytes(
            "RGBA",
            (width, height),
            command.bits[: width * height * 4],
            "raw",
            "BGRA",
            width * 4,
            orientation,
        )
    except (OSError, ValueError) as exc:
        raise MetafileMalformedError("32-bit DIB alpha payload cannot be decoded") from exc


def _validate_dib_dimensions(header: bytes) -> None:
    """在进入 Pillow decoder 前验证 DIB 声明尺寸与固定像素预算。"""
    if len(header) < 12:
        raise MetafileMalformedError("DIB header is truncated")
    import struct

    header_size = struct.unpack_from("<I", header, 0)[0]
    if header_size == 12:
        width, height = struct.unpack_from("<HH", header, 4)
    elif header_size >= 40 and len(header) >= 40:
        width, signed_height = struct.unpack_from("<ii", header, 4)
        height = abs(signed_height)
    else:
        raise MetafileMalformedError(f"unsupported or truncated DIB header size: {header_size}")
    if width <= 0 or height <= 0 or width * height > MAX_CANVAS_PIXELS:
        raise MetafileResourceLimitError(f"DIB dimensions exceed pixel budget: {width}x{height}")


def _decode_dib(command: DrawImageCommand) -> Image.Image:
    """严格解码 DIB，并在需要时恢复源 alpha。"""
    _validate_dib_dimensions(command.dib_header)
    if command.use_source_alpha:
        raw_alpha = _decode_raw_alpha_dib(command)
        if raw_alpha is not None:
            return raw_alpha
    try:
        with Image.open(BytesIO(_dib_to_bmp(command))) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_CANVAS_PIXELS:
                raise MetafileResourceLimitError(f"DIB dimensions exceed pixel budget: {width}x{height}")
            image.load()
            decoded = image.convert("RGBA")
    except MetafileResourceLimitError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MetafileMalformedError("embedded DIB cannot be decoded by Pillow") from exc
    if not command.use_source_alpha:
        decoded.putalpha(255)
    return decoded


def _clipper_paths(subpaths: list[tuple[list[Point], bool]]) -> list[list[tuple[int, int]]]:
    """把浮点子路径量化为 Clipper 闭合轮廓并过滤退化输入。"""
    result: list[list[tuple[int, int]]] = []
    for points, _closed in subpaths:
        converted: list[tuple[int, int]] = []
        for x, y in points:
            if abs(x) > _CLIPPER_COORD_LIMIT or abs(y) > _CLIPPER_COORD_LIMIT:
                raise MetafileResourceLimitError("flattened path coordinate exceeds Clipper range")
            point = round(x * _CLIPPER_SCALE), round(y * _CLIPPER_SCALE)
            if not converted or converted[-1] != point:
                converted.append(point)
        if converted and converted[0] == converted[-1]:
            converted.pop()
        if len(converted) >= 3:
            result.append(converted)
    return result


def _paint_polytree(mask: Image.Image, tree: pyclipper.PyPolyNode, budget: FlattenBudget | None = None) -> None:
    """按 PolyTree 层级依次绘制外轮廓、孔洞和孔洞中的岛。"""
    draw = ImageDraw.Draw(mask)

    def paint(node: pyclipper.PyPolyNode) -> None:
        """递归绘制当前节点的全部子轮廓。"""
        for child in node.Childs:
            if budget is not None:
                budget.charge(len(child.Contour))
            points = [(x / _CLIPPER_SCALE, y / _CLIPPER_SCALE) for x, y in child.Contour]
            if len(points) >= 3:
                draw.polygon(points, fill=0 if child.IsHole else 255)
            paint(child)

    paint(tree)


def _path_mask(
    path: GraphicsPath,
    matrix: Matrix,
    size: tuple[int, int],
    fill_rule: str,
    budget: FlattenBudget | None = None,
) -> Image.Image:
    """按 GDI winding/alternate 规则把复合路径转换为 L 模式 mask。"""
    transformed = transform_path(path, matrix)
    subpaths = flatten_path(transformed, budget=budget)
    mask = Image.new("L", size, 0)
    paths = _clipper_paths(subpaths)
    if not paths:
        return mask
    fill_type = pyclipper.PFT_EVENODD if fill_rule == "evenodd" else pyclipper.PFT_NONZERO
    try:
        clipper = pyclipper.Pyclipper()
        clipper.AddPaths(paths, pyclipper.PT_SUBJECT, True)
        tree = clipper.Execute2(pyclipper.CT_UNION, fill_type, fill_type)
    except pyclipper.ClipperException as exc:
        raise MetafileMalformedError("compound path cannot be resolved") from exc
    _paint_polytree(mask, tree, budget)
    return mask


def _clip_mask(
    clip: ClipStack,
    matrix: Matrix,
    size: tuple[int, int],
    budget: FlattenBudget | None = None,
) -> Image.Image | None:
    """按 GDI combine mode 顺序合成最终裁剪 mask。"""
    if not clip:
        return None
    current = Image.new("L", size, 255)
    for operation in clip:
        incoming = _path_mask(operation.path, matrix, size, operation.fill_rule, budget)
        if operation.mode == "copy":
            current = incoming
        elif operation.mode == "and":
            current = ImageChops.multiply(current, incoming)
        elif operation.mode == "or":
            current = ImageChops.lighter(current, incoming)
        elif operation.mode == "xor":
            current = ImageChops.logical_xor(current.convert("1"), incoming.convert("1")).convert("L")
        else:
            current = ImageChops.subtract(current, incoming)
    return current


def _apply_clip(layer: Image.Image, clip: ClipStack, matrix: Matrix, budget: FlattenBudget | None = None) -> None:
    """把命令级裁剪 mask 乘入 RGBA layer 的 alpha 通道。"""
    mask = _clip_mask(clip, matrix, layer.size, budget)
    if mask is None:
        return
    alpha = layer.getchannel("A")
    layer.putalpha(ImageChops.multiply(alpha, mask))


def _hatch_layer(size: tuple[int, int], brush: Brush) -> Image.Image:
    """为常见 GDI hatch brush 生成确定性 RGBA 图案层。"""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = brush.color.rgba()
    spacing = 8
    width, height = size
    if brush.hatch in {0, 4, 5}:
        for y in range(0, height + spacing, spacing):
            draw.line((0, y, width, y), fill=color)
    if brush.hatch in {1, 4, 5}:
        for x in range(0, width + spacing, spacing):
            draw.line((x, 0, x, height), fill=color)
    if brush.hatch in {2, 4}:
        for offset in range(-height, width + height, spacing):
            draw.line((offset, 0, offset + height, height), fill=color)
    if brush.hatch in {3, 5}:
        for offset in range(0, width + height * 2, spacing):
            draw.line((offset, 0, offset - height, height), fill=color)
    return layer


def _dash_polyline(
    points: list[Point],
    *,
    dashes: tuple[float, ...],
    closed: bool,
) -> list[list[Point]]:
    """按连续路径长度把折线切分为需要描绘的 dash 子段。"""
    if len(points) < 2:
        return []
    if not dashes:
        return [points]
    pattern = tuple(max(0.5, value) for value in dashes)
    pattern_index = 0
    pattern_remaining = pattern[0]
    drawing = True
    visible: list[list[Point]] = []
    current: list[Point] | None = None
    for start, end in zip(points, points[1:]):
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        length = hypot(delta_x, delta_y)
        if length <= 1e-9:
            continue
        consumed = 0.0
        while consumed < length:
            advance = min(pattern_remaining, length - consumed)
            if drawing and advance > 0:
                first = (
                    start[0] + delta_x * consumed / length,
                    start[1] + delta_y * consumed / length,
                )
                second = (
                    start[0] + delta_x * (consumed + advance) / length,
                    start[1] + delta_y * (consumed + advance) / length,
                )
                if current is None:
                    current = [first]
                elif current[-1] != first:
                    current.append(first)
                current.append(second)
            consumed += advance
            pattern_remaining -= advance
            if pattern_remaining <= 1e-9:
                if drawing and current is not None:
                    visible.append(current)
                    current = None
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_remaining = pattern[pattern_index]
                drawing = not drawing
    if current is not None:
        visible.append(current)
    if closed and len(visible) > 1 and visible[0][0] == points[0] and visible[-1][-1] == points[-1]:
        visible[0] = [*visible[-1][:-1], *visible[0]]
        visible.pop()
    return visible


def _quantize_stroke_path(points: list[Point], *, closed: bool) -> list[tuple[int, int]]:
    """把描边折线量化为 Clipper 坐标并移除重复端点。"""
    converted: list[tuple[int, int]] = []
    for x, y in points:
        if abs(x) > _CLIPPER_COORD_LIMIT or abs(y) > _CLIPPER_COORD_LIMIT:
            raise MetafileResourceLimitError("stroke coordinate exceeds Clipper range")
        point = round(x * _CLIPPER_SCALE), round(y * _CLIPPER_SCALE)
        if not converted or converted[-1] != point:
            converted.append(point)
    if closed and len(converted) > 1 and converted[0] == converted[-1]:
        converted.pop()
    return converted


def _offset_stroke_path(
    points: list[Point],
    *,
    closed: bool,
    width: float,
    pen: Pen,
    miter_limit: float,
    size: tuple[int, int],
    budget: FlattenBudget,
) -> Image.Image:
    """把单条折线扩张成遵守 GDI cap、join 和 miter limit 的描边 mask。"""
    mask = Image.new("L", size, 0)
    converted = _quantize_stroke_path(points, closed=closed)
    if len(converted) < (3 if closed else 2):
        return mask
    join_type = {
        "round": pyclipper.JT_ROUND,
        "bevel": pyclipper.JT_SQUARE,
        "miter": pyclipper.JT_MITER,
    }[pen.join]
    end_type = (
        pyclipper.ET_CLOSEDLINE
        if closed
        else {
            "round": pyclipper.ET_OPENROUND,
            "square": pyclipper.ET_OPENSQUARE,
            "flat": pyclipper.ET_OPENBUTT,
        }[pen.cap]
    )
    try:
        offset = pyclipper.PyclipperOffset(max(miter_limit, 1.0), _CLIPPER_SCALE * 0.1)
        offset.AddPath(converted, join_type, end_type)
        tree = offset.Execute2(max(width, 1.0) * _CLIPPER_SCALE / 2.0)
    except pyclipper.ClipperException as exc:
        raise MetafileMalformedError("stroke path cannot be widened") from exc
    _paint_polytree(mask, tree, budget)
    return mask


def _stroke_mask(
    subpaths: list[tuple[list[Point], bool]],
    *,
    size: tuple[int, int],
    width: float,
    pen: Pen,
    dashes: tuple[float, ...],
    miter_limit: float,
    budget: FlattenBudget,
) -> Image.Image:
    """把全部描边子路径合成为统一 L 模式覆盖 mask。"""
    mask = Image.new("L", size, 0)
    for points, closed in subpaths:
        visible = _dash_polyline(points, dashes=dashes, closed=closed)
        for segment in visible:
            segment_closed = closed and not dashes
            current = _offset_stroke_path(
                segment,
                closed=segment_closed,
                width=width,
                pen=pen,
                miter_limit=miter_limit,
                size=size,
                budget=budget,
            )
            mask = ImageChops.lighter(mask, current)
    return mask


def _render_path_command(
    command: DrawPathCommand,
    matrix: Matrix,
    size: tuple[int, int],
    raster_scale: int,
    budget: FlattenBudget,
) -> Image.Image:
    """把单条路径命令绘制到独立透明 RGBA layer。"""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    transformed = transform_path(command.path, matrix)
    subpaths = flatten_path(transformed, budget=budget)
    if command.fill and command.brush.kind != "null":
        fill_mask = _path_mask(command.path, matrix, size, command.fill_rule, budget)
        if command.brush.kind == "hatch":
            fill_layer = _hatch_layer(size, command.brush)
            fill_layer.putalpha(ImageChops.multiply(fill_layer.getchannel("A"), fill_mask))
        else:
            fill_layer = Image.new("RGBA", size, command.brush.color.rgba())
            fill_layer.putalpha(ImageChops.multiply(fill_layer.getchannel("A"), fill_mask))
        layer.alpha_composite(fill_layer)
    if command.stroke and not command.pen.null:
        scale = (abs(matrix.a) + abs(matrix.d)) / 2.0
        pen_width = command.pen.width * raster_scale if command.pen.cosmetic else command.pen.width * max(scale, 1e-9)
        width = max(1.0, min(pen_width, max(size) * 2.0))
        dash_scale = raster_scale if command.pen.cosmetic else scale
        stroke_mask = _stroke_mask(
            subpaths,
            size=size,
            width=width,
            pen=command.pen,
            dashes=tuple(value * dash_scale for value in command.pen.dashes),
            miter_limit=command.miter_limit,
            budget=budget,
        )
        stroke_layer = Image.new("RGBA", size, command.pen.color.rgba())
        stroke_layer.putalpha(ImageChops.multiply(stroke_layer.getchannel("A"), stroke_mask))
        layer.alpha_composite(stroke_layer)
    _apply_clip(layer, command.clip, matrix, budget)
    return layer


def _text_anchor(text_align: int) -> str:
    """把 GDI TextAlignmentMode 转换为 Pillow anchor。"""
    horizontal = "r" if text_align & _TA_CENTER == _TA_CENTER else "r" if text_align & _TA_RIGHT else "l"
    if text_align & _TA_CENTER == _TA_CENTER:
        horizontal = "m"
    vertical = "s" if text_align & _TA_BASELINE == _TA_BASELINE else "d" if text_align & _TA_BOTTOM else "a"
    return horizontal + vertical


def _draw_rotated_text(
    layer: Image.Image,
    position: Point,
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    anchor: str,
    rotation: float,
) -> tuple[int, int]:
    """在指定 anchor 绘制可旋转文字，并返回未旋转文字尺寸。"""
    draw = ImageDraw.Draw(layer)
    bbox = draw.textbbox((0, 0), text, font=font, anchor=anchor)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    if abs(rotation) < 1e-6:
        draw.text(position, text, font=font, fill=fill, anchor=anchor)
        return width, height
    padding = max(4, ceil(max(width, height) * 0.1))
    patch = Image.new("RGBA", (width + padding * 2, height + padding * 2), (0, 0, 0, 0))
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=fill)
    rotated = patch.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
    target = round(position[0] - rotated.width / 2), round(position[1] - rotated.height / 2)
    layer.alpha_composite(rotated, target)
    return width, height


def _render_text_command(
    command: DrawTextCommand,
    matrix: Matrix,
    size: tuple[int, int],
    budget: FlattenBudget,
) -> Image.Image:
    """把单条文字命令绘制到独立透明 RGBA layer。"""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    scale_y = max(abs(matrix.d), 1e-9)
    font_size = max(1, round(command.font_height * scale_y))
    font = load_font(command.font.face_name, font_size, command.font.weight, command.font.italic, command.font.charset)
    anchor = _text_anchor(command.text_align)
    positions = (
        [matrix.transform_point(position) for position in command.positions]
        if command.positions
        else [matrix.transform_point(command.origin)]
    )
    texts = tuple(command.text) if command.positions else (command.text,)
    if command.opaque:
        draw = ImageDraw.Draw(layer)
        if command.bounds is not None:
            rect = _mapped_rect(command.bounds, matrix).normalized()
            background_bounds = rect.left, rect.top, rect.right, rect.bottom
        else:
            text_bounds = [draw.textbbox(position, text, font=font, anchor=anchor) for position, text in zip(positions, texts)]
            background_bounds = (
                min(bounds[0] for bounds in text_bounds),
                min(bounds[1] for bounds in text_bounds),
                max(bounds[2] for bounds in text_bounds),
                max(bounds[3] for bounds in text_bounds),
            )
        draw.rectangle(background_bounds, fill=command.background_color.rgba())
    if command.positions:
        sizes: list[tuple[int, int]] = []
        for position, character in zip(positions, command.text):
            sizes.append(
                _draw_rotated_text(
                    layer,
                    position,
                    character,
                    font=font,
                    fill=command.color.rgba(),
                    anchor=anchor,
                    rotation=command.rotation,
                )
            )
    else:
        position = positions[0]
        sizes = [
            _draw_rotated_text(
                layer,
                position,
                command.text,
                font=font,
                fill=command.color.rgba(),
                anchor=anchor,
                rotation=command.rotation,
            )
        ]
    if (command.font.underline or command.font.strikeout) and positions:
        draw = ImageDraw.Draw(layer)
        width = sizes[-1][0] if sizes else font_size
        for position in positions:
            if command.font.underline:
                y = position[1] + max(1, font_size * 0.1)
                draw.line((position[0], y, position[0] + width, y), fill=command.color.rgba(), width=max(1, font_size // 14))
            if command.font.strikeout:
                y = position[1] - font_size * 0.3
                draw.line((position[0], y, position[0] + width, y), fill=command.color.rgba(), width=max(1, font_size // 14))
    _apply_clip(layer, command.clip, matrix, budget)
    return layer


def _crop_source(image: Image.Image, source: Rect | None) -> Image.Image:
    """按 GDI source rect 裁剪位图，并兼容负宽高翻转。"""
    if source is None:
        return image
    flip_x = source.width < 0
    flip_y = source.height < 0
    normalized = source.normalized()
    left = max(0, floor(normalized.left))
    top = max(0, floor(normalized.top))
    right = min(image.width, ceil(normalized.right))
    bottom = min(image.height, ceil(normalized.bottom))
    if right <= left or bottom <= top:
        return image
    cropped = image.crop((left, top, right, bottom))
    if flip_x:
        cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_y:
        cropped = cropped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return cropped


def _affine_image_layer(
    image: Image.Image,
    destination: tuple[Point, Point, Point, Point],
    matrix: Matrix,
    size: tuple[int, int],
    stretch_mode: int,
) -> Image.Image:
    """把源图片仿射映射到目标平行四边形。"""
    first, second, _third, fourth = (matrix.transform_point(point) for point in destination)
    axis_x = second[0] - first[0], second[1] - first[1]
    axis_y = fourth[0] - first[0], fourth[1] - first[1]
    determinant = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
    if abs(determinant) < 1e-9:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    inverse_00 = axis_y[1] / determinant
    inverse_01 = -axis_y[0] / determinant
    inverse_10 = -axis_x[1] / determinant
    inverse_11 = axis_x[0] / determinant
    scale_x = image.width
    scale_y = image.height
    a = inverse_00 * scale_x
    b = inverse_01 * scale_x
    c = -(inverse_00 * first[0] + inverse_01 * first[1]) * scale_x
    d = inverse_10 * scale_y
    e = inverse_11 * scale_y
    f = -(inverse_10 * first[0] + inverse_11 * first[1]) * scale_y
    resample = Image.Resampling.BICUBIC if stretch_mode == 4 else Image.Resampling.NEAREST
    return image.transform(
        size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=resample,
        fillcolor=(0, 0, 0, 0),
    )


def _render_image_command(
    command: DrawImageCommand,
    matrix: Matrix,
    size: tuple[int, int],
    budget: FlattenBudget,
) -> Image.Image:
    """解码、裁剪并仿射放置单条位图命令。"""
    image = _crop_source(_decode_dib(command), command.source)
    if command.constant_alpha < 255:
        alpha = image.getchannel("A").point(lambda value: value * command.constant_alpha // 255)
        image.putalpha(alpha)
    layer = _affine_image_layer(image, command.destination, matrix, size, command.stretch_mode)
    _apply_clip(layer, command.clip, matrix, budget)
    return layer


def _bitwise_channel_bytes(destination: bytes, source: bytes, operation: str) -> bytes:
    """对等长 RGB 字节串执行精确 AND、OR、XOR 或反相。"""
    if operation == "xor":
        return bytes(left ^ right for left, right in zip(destination, source))
    if operation == "and":
        return bytes(left & right for left, right in zip(destination, source))
    if operation == "or":
        return bytes(left | right for left, right in zip(destination, source))
    if operation == "not_or":
        return bytes((~(left | right)) & 0xFF for left, right in zip(destination, source))
    if operation == "and_not_source":
        return bytes(left & (~right & 0xFF) for left, right in zip(destination, source))
    if operation == "not_source":
        return bytes(255 - value for value in source)
    if operation == "source_and_not_destination":
        return bytes(right & (~left & 0xFF) for left, right in zip(destination, source))
    if operation == "not_and":
        return bytes((~(left & right)) & 0xFF for left, right in zip(destination, source))
    if operation == "not_xor":
        return bytes((~(left ^ right)) & 0xFF for left, right in zip(destination, source))
    if operation == "destination_or_not_source":
        return bytes(left | (~right & 0xFF) for left, right in zip(destination, source))
    if operation == "source_or_not_destination":
        return bytes(right | (~left & 0xFF) for left, right in zip(destination, source))
    if operation == "invert":
        return bytes(255 - value for value in destination)
    fill = 0 if operation == "black" else 255
    return bytes([fill]) * len(destination)


def _paste_bitwise(canvas: Image.Image, layer: Image.Image, operation: str) -> None:
    """只在 layer 覆盖框内应用精确逐通道 GDI 逻辑合成。"""
    mask = layer.getchannel("A")
    bbox = mask.getbbox()
    if bbox is None:
        return
    destination = canvas.crop(bbox).convert("RGB")
    source = layer.crop(bbox).convert("RGB")
    result = Image.frombytes(
        "RGB",
        destination.size,
        _bitwise_channel_bytes(destination.tobytes(), source.tobytes(), operation),
    ).convert("RGBA")
    local_mask = mask.crop(bbox)
    result.putalpha(ImageChops.lighter(canvas.getchannel("A").crop(bbox), local_mask))
    canvas.paste(result, bbox[:2], local_mask)


def _composite_path(canvas: Image.Image, layer: Image.Image, rop2: int) -> None:
    """按常见 ROP2 或默认 source-over 合成路径层。"""
    operations = {
        1: "black",
        2: "not_or",
        3: "and_not_source",
        4: "not_source",
        5: "source_and_not_destination",
        6: "invert",
        7: "xor",
        8: "not_and",
        9: "and",
        10: "not_xor",
        12: "destination_or_not_source",
        14: "source_or_not_destination",
        15: "or",
        16: "white",
    }
    if rop2 == 13:
        canvas.alpha_composite(layer)
    elif rop2 != 11:
        _paste_bitwise(canvas, layer, operations.get(rop2, "xor"))


def _composite_image(canvas: Image.Image, layer: Image.Image, rop: int) -> None:
    """按常见 ROP3 或 AlphaBlend 结果合成位图层。"""
    if rop in {_SRCCOPY, 0}:
        canvas.alpha_composite(layer)
    elif rop == _SRCAND:
        _paste_bitwise(canvas, layer, "and")
    elif rop == _SRCPAINT:
        _paste_bitwise(canvas, layer, "or")
    elif rop == _SRCINVERT:
        _paste_bitwise(canvas, layer, "xor")
    elif rop == _DSTINVERT:
        _paste_bitwise(canvas, layer, "invert")
    elif rop == _BLACKNESS:
        _paste_bitwise(canvas, layer, "black")
    elif rop == _WHITENESS:
        _paste_bitwise(canvas, layer, "white")
    else:
        canvas.alpha_composite(layer)


def _supersample_factor(document: MetafileDocument) -> int:
    """在画布、工作量预算内为纯矢量文档选择 1×、2× 或 4×。"""
    if any(isinstance(command, DrawImageCommand) for command in document.commands):
        return 1
    command_count = max(len(document.commands), 1)
    for factor in (4, 2):
        width = document.width * factor
        height = document.height * factor
        if (
            width <= 8192
            and height <= 8192
            and width * height <= MAX_CANVAS_PIXELS
            and width * height * command_count <= MAX_RENDER_WORK_PIXELS
        ):
            return factor
    return 1


def _render_pillow_once(document: MetafileDocument, raster_scale: int) -> Image.Image:
    """按给定整数倍率执行一次不缩放的 Pillow 栅格化。"""
    width = document.width * raster_scale
    height = document.height * raster_scale
    render_work = width * height * max(len(document.commands), 1)
    if render_work > MAX_RENDER_WORK_PIXELS:
        raise MetafileResourceLimitError(f"metafile exceeds max_render_work_pixels={MAX_RENDER_WORK_PIXELS}")
    matrix = _document_matrix(document, raster_scale=raster_scale)
    size = width, height
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))
    budget = FlattenBudget()
    for command in document.commands:
        if isinstance(command, DrawPathCommand):
            _composite_path(canvas, _render_path_command(command, matrix, size, raster_scale, budget), command.rop2)
        elif isinstance(command, DrawTextCommand):
            canvas.alpha_composite(_render_text_command(command, matrix, size, budget))
        else:
            _composite_image(canvas, _render_image_command(command, matrix, size, budget), command.rop)
    return canvas


def render_pillow(document: MetafileDocument) -> Image.Image:
    """把统一图元文档以自适应超采样渲染为最终尺寸 RGBA 图片。"""
    raster_scale = _supersample_factor(document)
    canvas = _render_pillow_once(document, raster_scale)
    if raster_scale == 1:
        return canvas
    resized = canvas.resize((document.width, document.height), Image.Resampling.LANCZOS)
    return resized.filter(ImageFilter.UnsharpMask(radius=0.6, percent=100, threshold=2))


def _svg_number(value: float) -> str:
    """以稳定且紧凑的形式序列化 SVG 浮点数。"""
    rounded = round(value, 4)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _svg_path_data(path: GraphicsPath, matrix: Matrix) -> str:
    """把统一路径转换为安全 SVG path data。"""
    parts: list[str] = []
    for segment in transform_path(path, matrix).segments:
        if segment.verb == "Z":
            parts.append("Z")
            continue
        coordinates = " ".join(f"{_svg_number(point[0])} {_svg_number(point[1])}" for point in segment.points)
        parts.append(f"{segment.verb} {coordinates}")
    return " ".join(parts)


def _svg_opacity(color: Color) -> str:
    """返回仅在颜色半透明时需要的 SVG opacity 属性。"""
    return "" if color.alpha == 255 else f' opacity="{_svg_number(color.alpha / 255.0)}"'


def _svg_pen_attributes(pen: Pen, matrix: Matrix, miter_limit: float) -> str:
    """把内部 Pen 转换为 SVG stroke 属性。"""
    if pen.null:
        return 'stroke="none"'
    scale = (abs(matrix.a) + abs(matrix.d)) / 2.0
    width = pen.width if pen.cosmetic else pen.width * max(scale, 1e-9)
    attributes = [
        f'stroke="{pen.color.svg()}"',
        f'stroke-width="{_svg_number(max(width, 1.0))}"',
        f'stroke-linecap="{pen.cap}"',
        f'stroke-linejoin="{pen.join}"',
        f'stroke-miterlimit="{_svg_number(max(miter_limit, 1.0))}"',
    ]
    if pen.color.alpha != 255:
        attributes.append(f'stroke-opacity="{_svg_number(pen.color.alpha / 255.0)}"')
    if pen.dashes:
        dash_scale = 1.0 if pen.cosmetic else scale
        attributes.append(f'stroke-dasharray="{" ".join(_svg_number(value * dash_scale) for value in pen.dashes)}"')
    return " ".join(attributes)


def _svg_brush_attributes(brush: Brush) -> str:
    """把内部 Brush 转换为基础 SVG fill 属性。"""
    if brush.kind == "null":
        return 'fill="none"'
    attributes = [f'fill="{brush.color.svg()}"']
    if brush.color.alpha != 255:
        attributes.append(f'fill-opacity="{_svg_number(brush.color.alpha / 255.0)}"')
    return " ".join(attributes)


def _svg_clip_definitions(document: MetafileDocument, matrix: Matrix) -> tuple[list[str], dict[ClipOperation, str]]:
    """为只含 copy/and 的裁剪路径生成确定性 clipPath 定义。"""
    definitions: list[str] = []
    identifiers: dict[ClipOperation, str] = {}
    ordinal = 0
    for command in document.commands:
        for operation in command.clip:
            if operation in identifiers:
                continue
            ordinal += 1
            identifier = f"mineru-clip-{ordinal}"
            identifiers[operation] = identifier
            definitions.append(
                f'<clipPath id="{identifier}"><path d="{_svg_path_data(operation.path, matrix)}" '
                f'fill-rule="{operation.fill_rule}"/></clipPath>'
            )
    return definitions, identifiers


def _wrap_svg_clip(element: str, clip: ClipStack, identifiers: dict[ClipOperation, str]) -> str:
    """按裁剪顺序用嵌套 SVG group 包裹单个图元。"""
    wrapped = element
    active: list[ClipOperation] = []
    for operation in clip:
        if operation.mode == "copy":
            active = [operation]
        elif operation.mode == "and":
            active.append(operation)
    for operation in reversed(active):
        wrapped = f'<g clip-path="url(#{identifiers[operation]})">{wrapped}</g>'
    return wrapped


def _svg_text_anchor(text_align: int) -> tuple[str, str]:
    """把 GDI 文字对齐转换为 SVG anchor 和 baseline。"""
    horizontal = "middle" if text_align & _TA_CENTER == _TA_CENTER else "end" if text_align & _TA_RIGHT else "start"
    vertical = (
        "alphabetic"
        if text_align & _TA_BASELINE == _TA_BASELINE
        else "text-after-edge"
        if text_align & _TA_BOTTOM
        else "hanging"
    )
    return horizontal, vertical


def _svg_text_elements(command: DrawTextCommand, matrix: Matrix) -> str:
    """把统一文字命令转换为一个或多个安全 SVG text 元素。"""
    anchor, baseline = _svg_text_anchor(command.text_align)
    scale_y = max(abs(matrix.d), 1e-9)
    font_size = max(1.0, command.font_height * scale_y)
    decorations = []
    if command.font.underline:
        decorations.append("underline")
    if command.font.strikeout:
        decorations.append("line-through")
    decoration_attr = f' text-decoration="{" ".join(decorations)}"' if decorations else ""
    common = (
        f'fill="{command.color.svg()}"{_svg_opacity(command.color)} '
        f'font-family="{escape(command.font.face_name, quote=True)}" font-size="{_svg_number(font_size)}" '
        f'font-weight="{command.font.weight}" font-style="{"italic" if command.font.italic else "normal"}" '
        f'text-anchor="{anchor}" dominant-baseline="{baseline}"{decoration_attr}'
    )
    elements: list[str] = []
    positions = command.positions or (command.origin,)
    texts = tuple(command.text) if command.positions else (command.text,)
    mapped_positions = tuple(matrix.transform_point(position) for position in positions)
    for mapped, text in zip(mapped_positions, texts):
        transform = ""
        if abs(command.rotation) > 1e-6:
            transform = (
                f' transform="rotate({_svg_number(command.rotation)} {_svg_number(mapped[0])} {_svg_number(mapped[1])})"'
            )
        elements.append(
            f'<text x="{_svg_number(mapped[0])}" y="{_svg_number(mapped[1])}" {common}{transform}>{escape(text)}</text>'
        )
    background = ""
    if command.opaque:
        if command.bounds is not None:
            rect = _mapped_rect(command.bounds, matrix).normalized()
        else:
            approximate_widths = tuple(max(font_size * 0.6 * len(text), font_size * 0.25) for text in texts)
            left = min(position[0] for position in mapped_positions)
            right = max(position[0] + width for position, width in zip(mapped_positions, approximate_widths))
            top = min(position[1] - font_size for position in mapped_positions)
            bottom = max(position[1] + font_size * 0.25 for position in mapped_positions)
            rect = Rect(left, top, right, bottom)
        background = (
            f'<rect x="{_svg_number(rect.left)}" y="{_svg_number(rect.top)}" width="{_svg_number(rect.width)}" '
            f'height="{_svg_number(rect.height)}" fill="{command.background_color.svg()}"'
            f"{_svg_opacity(command.background_color)}/>"
        )
    return background + "".join(elements)


def _svg_image_element(command: DrawImageCommand, matrix: Matrix) -> str:
    """把 DIB 转成内嵌 PNG 并以 SVG affine matrix 放置。"""
    image = _crop_source(_decode_dib(command), command.source)
    if command.constant_alpha < 255:
        image.putalpha(image.getchannel("A").point(lambda value: value * command.constant_alpha // 255))
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    first, second, _third, fourth = (matrix.transform_point(point) for point in command.destination)
    a = (second[0] - first[0]) / max(image.width, 1)
    b = (second[1] - first[1]) / max(image.width, 1)
    c = (fourth[0] - first[0]) / max(image.height, 1)
    d = (fourth[1] - first[1]) / max(image.height, 1)
    transform = " ".join(_svg_number(value) for value in (a, b, c, d, first[0], first[1]))
    return (
        f'<image width="{image.width}" height="{image.height}" transform="matrix({transform})" '
        f'href="data:image/png;base64,{encoded}"/>'
    )


def _svg_requires_raster(document: MetafileDocument) -> bool:
    """判断 SVG 是否需要用整图 PNG 包装保留复杂合成语义。"""
    for command in document.commands:
        if any(operation.mode not in {"copy", "and"} for operation in command.clip):
            return True
        if isinstance(command, DrawPathCommand) and command.rop2 != 13:
            return True
        if isinstance(command, DrawImageCommand) and command.rop not in {_SRCCOPY, 0}:
            return True
    return False


def _svg_fallback_scale(document: MetafileDocument) -> int:
    """为纯矢量 SVG fallback 选择最高 8× 的安全像素密度。"""
    if any(isinstance(command, DrawImageCommand) for command in document.commands):
        return 1
    command_count = max(len(document.commands), 1)
    for factor in (8, 4, 2):
        width = document.width * factor
        height = document.height * factor
        if (
            width <= 8192
            and height <= 8192
            and width * height <= MAX_CANVAS_PIXELS
            and width * height * command_count <= MAX_RENDER_WORK_PIXELS
        ):
            return factor
    return 1


def _png_fallback_bytes(document: MetafileDocument, *, pixel_scale: int = 1) -> bytes:
    """生成指定像素密度、带对应 DPI metadata 的 PNG fallback。"""
    image = render_pillow(document) if pixel_scale == 1 else _render_pillow_once(document, pixel_scale)
    output = BytesIO()
    image.save(output, format="PNG", dpi=(96 * pixel_scale, 96 * pixel_scale))
    return output.getvalue()


def _svg_fallback_metadata(encoded_png: str) -> str:
    """生成带固定标识的不可见 PNG fallback metadata。"""
    return f'<metadata id="mineru-raster-fallback" data-mime="image/png">{encoded_png}</metadata>'


def _raster_wrapped_svg(document: MetafileDocument) -> bytes:
    """把 Pillow 结果封装为没有外部引用的单图片 SVG。"""
    encoded = base64.b64encode(_png_fallback_bytes(document)).decode("ascii")
    markup = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{document.width}" height="{document.height}" '
        f'viewBox="0 0 {document.width} {document.height}" data-mineru-generated="wmf-emf">'
        f"{_svg_fallback_metadata(encoded)}"
        f'<image width="{document.width}" height="{document.height}" href="data:image/png;base64,{encoded}"/>'
        "</svg>"
    )
    return markup.encode("utf-8")


def render_svg(document: MetafileDocument) -> bytes:
    """把统一图元文档渲染为安全、自包含 SVG 字节。"""
    if _svg_requires_raster(document):
        return _raster_wrapped_svg(document)
    matrix = _document_matrix(document)
    definitions, clip_ids = _svg_clip_definitions(document, matrix)
    fallback_scale = _svg_fallback_scale(document)
    fallback = base64.b64encode(_png_fallback_bytes(document, pixel_scale=fallback_scale)).decode("ascii")
    elements: list[str] = []
    for command in document.commands:
        if isinstance(command, DrawPathCommand):
            stroke = _svg_pen_attributes(command.pen, matrix, command.miter_limit) if command.stroke else 'stroke="none"'
            fill = _svg_brush_attributes(command.brush) if command.fill else 'fill="none"'
            element = f'<path d="{_svg_path_data(command.path, matrix)}" {stroke} {fill} fill-rule="{command.fill_rule}"/>'
        elif isinstance(command, DrawTextCommand):
            element = _svg_text_elements(command, matrix)
        else:
            element = _svg_image_element(command, matrix)
        elements.append(_wrap_svg_clip(element, command.clip, clip_ids))
    defs = f"<defs>{''.join(definitions)}</defs>" if definitions else ""
    markup = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{document.width}" height="{document.height}" '
        f'viewBox="0 0 {document.width} {document.height}" data-mineru-generated="wmf-emf">'
        f"{_svg_fallback_metadata(fallback)}{defs}{''.join(elements)}</svg>"
    )
    return markup.encode("utf-8")


def encode_document(document: MetafileDocument, output_format: MetafileOutputFormat) -> tuple[bytes, str]:
    """按目标格式编码统一图元文档并返回 MIME。"""
    if output_format == "svg":
        return render_svg(document), "image/svg+xml"
    if output_format == "png":
        return _png_fallback_bytes(document), "image/png"
    image = render_pillow(document)
    output = BytesIO()
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image.convert("RGB"), (0, 0), image.getchannel("A"))
    background.save(output, format="JPEG", quality=90)
    return output.getvalue(), "image/jpeg"


__all__ = ["encode_document", "render_pillow", "render_svg"]
