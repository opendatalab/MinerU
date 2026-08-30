# Copyright (c) Opendatalab. All rights reserved.
"""从 TextObject/TextCode 恢复语义文字与页面几何。"""

from __future__ import annotations

import html
import math
import re
from collections import deque
from dataclasses import dataclass
from io import BytesIO, StringIO

from fontTools.pens.boundsPen import BoundsPen
from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from .._shared.spans import text_spans
from ....types import BBox
from .constants import MAX_DELTA_TOKENS, MAX_EXPANDED_GLYPHS, MAX_EXPANDED_TEXT_BYTES, MAX_FONT_BYTES, MAX_GLYPH_TOKENS
from .errors import OfdResourceLimitError
from .geometry import (
    Affine,
    bbox_intersection,
    bbox_union,
    canonical_angle,
    parse_affine,
    parse_st_box,
    quad_bbox,
    rect_quad,
    transform_angle,
    transform_bbox,
    transform_quad,
)
from .models import FontResource, GlyphItem, ResourceRegistry, TextLine
from .package import OfdPackage, element_text, local_name, parse_int

_HEX_ESCAPE_RE = re.compile(r"\\([0-9A-Fa-f]{4})")
_GLYPH_TOKEN_RE = re.compile(r"\S+")
_DELTA_TOKEN_RE = re.compile(r"[^,\s]+")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_TEXT_CODE_DECODE_CHUNK_SIZE = 64 * 1024


@dataclass(slots=True)
class OfdTextBudget:
    """累计限制 TextCode 文字与展开字形数量。"""

    text_bytes: int = 0
    glyph_count: int = 0
    glyph_mapping_count: int = 0
    glyph_token_count: int = 0
    delta_token_count: int = 0

    def charge(self, text: str) -> None:
        """为一次 TextCode 展开计费。"""
        glyph_count = self.glyph_count + len(text)
        if glyph_count > MAX_EXPANDED_GLYPHS:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_expanded_glyphs={MAX_EXPANDED_GLYPHS}")
        text_bytes = self.text_bytes + len(text.encode("utf-8"))
        if text_bytes > MAX_EXPANDED_TEXT_BYTES:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_expanded_text_bytes={MAX_EXPANDED_TEXT_BYTES}")
        self.glyph_count = glyph_count
        self.text_bytes = text_bytes

    def charge_glyph_mapping(self, count: int) -> None:
        """累计 CGTransform 的有效字符映射数量并限制全文展开量。"""
        self.glyph_mapping_count += count
        if self.glyph_mapping_count > MAX_EXPANDED_GLYPHS:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_expanded_glyphs={MAX_EXPANDED_GLYPHS}")

    def charge_glyph_token(self) -> None:
        """累计实际扫描的 Glyphs token 数量并限制全文解析量。"""
        self.glyph_token_count += 1
        if self.glyph_token_count > MAX_GLYPH_TOKENS:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_glyph_tokens={MAX_GLYPH_TOKENS}")

    def charge_delta_token(self) -> None:
        """累计实际扫描的 Delta token 数量并限制全文解析量。"""
        self.delta_token_count += 1
        if self.delta_token_count > MAX_DELTA_TOKENS:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_delta_tokens={MAX_DELTA_TOKENS}")


@dataclass(slots=True)
class _LoadedFont:
    """缓存 FontTools 中与字形几何相关的只读表。"""

    font: object
    glyph_order: list[str]
    glyph_set: object
    units_per_em: float
    advances: dict[str, tuple[int, int]]
    char_to_name: dict[int, str]
    name_to_char: dict[str, str]


class FontMetricResolver:
    """按 OFD 字体资源惰性读取内嵌 OpenType 指标。"""

    def __init__(self, package: OfdPackage) -> None:
        """绑定当前包并创建字体解析缓存。"""
        self.package = package
        self._cache: dict[str, _LoadedFont | None] = {}

    def _load(self, resource: FontResource | None) -> _LoadedFont | None:
        """读取一个受限字体成员，失败时缓存空结果。"""
        if resource is None or resource.font_part is None:
            return None
        if resource.font_part in self._cache:
            return self._cache[resource.font_part]
        data = self.package.read_part(resource.font_part, asset=True)
        if data is None or len(data) > MAX_FONT_BYTES:
            logger.warning(
                f"OFD_FONT_UNAVAILABLE: part={resource.font_part!r}, reason={'missing' if data is None else 'too_large'}"
            )
            self._cache[resource.font_part] = None
            return None
        try:
            from fontTools.ttLib import TTFont

            font = TTFont(BytesIO(data), lazy=True)
            glyph_order = list(font.getGlyphOrder())
            glyph_set = font.getGlyphSet()
            units_per_em = float(font["head"].unitsPerEm) if "head" in font else 1000.0
            advances = dict(font["hmtx"].metrics) if "hmtx" in font else {}
            char_to_name: dict[int, str] = {}
            if "cmap" in font:
                for table in font["cmap"].tables:
                    char_to_name.update(table.cmap)
            name_to_char: dict[str, str] = {}
            for codepoint, glyph_name in char_to_name.items():
                if glyph_name not in name_to_char and 0 <= codepoint <= 0x10FFFF:
                    name_to_char[glyph_name] = chr(codepoint)
            loaded = _LoadedFont(
                font=font,
                glyph_order=glyph_order,
                glyph_set=glyph_set,
                units_per_em=max(units_per_em, 1.0),
                advances=advances,
                char_to_name=char_to_name,
                name_to_char=name_to_char,
            )
        except Exception as exc:
            logger.warning(f"OFD_FONT_INVALID: part={resource.font_part!r}, error={type(exc).__name__}")
            loaded = None
        self._cache[resource.font_part] = loaded
        return loaded

    def resolve_character(self, resource: FontResource | None, glyph_id: int | None, fallback: str) -> str:
        """在 TextCode 使用占位符时尝试由 glyph cmap 恢复字符。"""
        if fallback != "¤" or glyph_id is None:
            return fallback
        loaded = self._load(resource)
        if loaded is None or not (0 <= glyph_id < len(loaded.glyph_order)):
            return fallback
        return loaded.name_to_char.get(loaded.glyph_order[glyph_id], fallback)

    def glyph_bbox(
        self,
        resource: FontResource | None,
        glyph_id: int | None,
        character: str,
        *,
        size: float,
        hscale: float,
        advance_hint: float | None,
    ) -> BBox:
        """返回以 glyph origin 为基准的确定性局部字形框。"""
        loaded = self._load(resource)
        glyph_name: str | None = None
        if loaded is not None:
            if glyph_id is not None and 0 <= glyph_id < len(loaded.glyph_order):
                glyph_name = loaded.glyph_order[glyph_id]
            elif character:
                glyph_name = loaded.char_to_name.get(ord(character[0]))
        if loaded is not None and glyph_name is not None and glyph_name in loaded.glyph_set:
            scale = size / loaded.units_per_em
            try:
                pen = BoundsPen(loaded.glyph_set)
                loaded.glyph_set[glyph_name].draw(pen)
                bounds = pen.bounds
            except Exception:
                bounds = None
            advance = float(loaded.advances.get(glyph_name, (loaded.units_per_em, 0))[0]) * scale * hscale
            if advance_hint is not None and advance_hint > 0:
                advance = advance_hint
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                bbox = (x0 * scale * hscale, -y1 * scale, x1 * scale * hscale, -y0 * scale)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    return bbox
            return (0.0, -0.85 * size, max(advance, 0.2 * size), 0.2 * size)
        fallback_advance = (
            advance_hint if advance_hint is not None and advance_hint > 0 else max(0.5 * size * hscale, 0.2 * size)
        )
        return (0.0, -0.85 * size, fallback_advance, 0.2 * size)

    def close(self) -> None:
        """关闭已经打开的 FontTools 字体对象。"""
        for loaded in self._cache.values():
            if loaded is None:
                continue
            close = getattr(loaded.font, "close", None)
            if callable(close):
                close()


def decode_text_code(value: str) -> str:
    """解码 OFD TextCode 中的反斜杠四位十六进制字符。"""
    return _HEX_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)


def _incomplete_hex_escape_length(value: str) -> int:
    """返回末尾可能跨分片的反斜杠十六进制前缀长度。"""
    for length in range(min(4, len(value)), 0, -1):
        suffix = value[-length:]
        if suffix[0] == "\\" and all(character in _HEX_DIGITS for character in suffix[1:]):
            return length
    return 0


def _decode_text_code_element(text_code: etree._Element, budget: OfdTextBudget) -> str:
    """分片解码一个 TextCode，并在写入完整字符串前累计文字预算。"""
    output = StringIO()
    carry = ""
    for part in text_code.itertext():
        for offset in range(0, len(part), _TEXT_CODE_DECODE_CHUNK_SIZE):
            chunk = part[offset : offset + _TEXT_CODE_DECODE_CHUNK_SIZE]
            value = f"{carry}{chunk}" if carry else chunk
            carry_length = _incomplete_hex_escape_length(value)
            if carry_length:
                complete = value[:-carry_length]
                carry = value[-carry_length:]
            else:
                complete = value
                carry = ""
            if not complete:
                continue
            decoded = decode_text_code(complete)
            budget.charge(decoded)
            output.write(decoded)
    if carry:
        budget.charge(carry)
        output.write(carry)
    return output.getvalue()


def parse_delta(value: str | None, count: int, budget: OfdTextBudget) -> list[float]:
    """流式展开普通与 g-count-value 压缩 Delta，并补齐不足项。"""
    if count <= 0:
        return []
    output: list[float] = []
    matches = iter(_DELTA_TOKEN_RE.finditer(value or ""))
    pending: deque[str] = deque()

    def next_token() -> str | None:
        """返回下一个 Delta token，并只对首次扫描计费。"""
        if pending:
            return pending.popleft()
        try:
            match = next(matches)
        except StopIteration:
            return None
        budget.charge_delta_token()
        return match.group()

    while len(output) < count:
        token = next_token()
        if token is None:
            break
        if token.casefold() == "g":
            repeat_token = next_token()
            value_token = next_token()
            if repeat_token is None or value_token is None:
                if repeat_token is not None:
                    pending.appendleft(repeat_token)
                continue
            try:
                repeat = max(0, int(repeat_token))
                repeated_value = float(value_token)
            except ValueError:
                pending.appendleft(value_token)
                pending.appendleft(repeat_token)
                continue
            if math.isfinite(repeated_value):
                output.extend([repeated_value] * min(repeat, count - len(output)))
            continue
        try:
            parsed = float(token)
        except ValueError:
            continue
        if math.isfinite(parsed):
            output.append(parsed)
    if len(output) < count:
        output.extend([output[-1] if output else 0.0] * (count - len(output)))
    return output[:count]


def _bounded_glyph_ids(glyphs_element: etree._Element, limit: int, budget: OfdTextBudget) -> list[int]:
    """按需迭代 Glyphs token，只保留有效映射所需的有限 ID。"""
    glyph_ids: list[int] = []
    for match in _GLYPH_TOKEN_RE.finditer(element_text(glyphs_element)):
        budget.charge_glyph_token()
        glyph_id = parse_int(match.group())
        if glyph_id is None:
            continue
        glyph_ids.append(glyph_id)
        if len(glyph_ids) >= limit:
            break
    return glyph_ids


def _glyph_map(text_object: etree._Element, position_count: int, budget: OfdTextBudget) -> dict[int, int]:
    """把实际 TextCode 字符位置映射到 glyph ID，并限制累计展开量。"""
    result: dict[int, int] = {}
    for element in text_object:
        if local_name(element.tag) != "CGTransform":
            continue
        code_position = parse_int(element.get("CodePosition"))
        code_count = parse_int(element.get("CodeCount"))
        glyphs_element = next((child for child in element if local_name(child.tag) == "Glyphs"), None)
        if code_position is None or code_count is None or glyphs_element is None:
            continue
        effective_count = min(code_count, max(0, position_count - code_position))
        if effective_count <= 0:
            continue
        glyph_ids = _bounded_glyph_ids(glyphs_element, effective_count, budget)
        if not glyph_ids:
            continue
        budget.charge_glyph_mapping(effective_count)
        for offset in range(effective_count):
            glyph_index = min(offset, len(glyph_ids) - 1)
            result[code_position + offset] = glyph_ids[glyph_index]
    return result


def _styles(
    text_object: etree._Element,
    font: FontResource | None,
    resolved_style: dict[str, str],
) -> tuple[str, ...]:
    """从字体资源和 TextObject 属性恢复可投影行内样式。"""
    styles: list[str] = []
    weight = parse_int(resolved_style.get("Weight") or text_object.get("Weight"))
    if (font is not None and font.bold) or (weight is not None and weight >= 600):
        styles.append("bold")
    if (font is not None and font.italic) or (resolved_style.get("Italic") or text_object.get("Italic") or "").casefold() in {
        "true",
        "1",
    }:
        styles.append("italic")
    return tuple(styles)


def format_line_spans(text: str, styles: tuple[str, ...]) -> list[dict[str, object]]:
    """把 OFD 原生文字和样式直接投影为结构化 Span。"""
    return text_spans(text, styles)


def format_line_html(text: str, styles: tuple[str, ...]) -> str:
    """把 OFD 表格单元格文字序列化为安全 HTML。"""
    rendered = html.escape(text, quote=False)
    if "superscript" in styles:
        rendered = f"<sup>{rendered}</sup>"
    elif "subscript" in styles:
        rendered = f"<sub>{rendered}</sub>"
    if "underline" in styles:
        rendered = f"<u>{rendered}</u>"
    if "bold" in styles:
        rendered = f"<strong>{rendered}</strong>"
    if "italic" in styles:
        rendered = f"<em>{rendered}</em>"
    if "strikethrough" in styles:
        rendered = f"<s>{rendered}</s>"
    return rendered


def build_text_lines(
    text_object: etree._Element,
    *,
    parent_transform: Affine,
    parent_clip: BBox,
    resources: ResourceRegistry,
    package: OfdPackage,
    font_metrics: FontMetricResolver,
    budget: OfdTextBudget,
    paint_order: int,
    layer_type: str,
    template_id: int | None,
    resolved_style: dict[str, str] | None = None,
) -> list[TextLine]:
    """把一个 TextObject 展开为按 TextCode 划分的页面文字行。"""
    style = resolved_style or {}
    if (style.get("Visible") or text_object.get("Visible") or "true").casefold() in {"false", "0"}:
        return []
    if (style.get("Alpha") or text_object.get("Alpha") or "255").strip() == "0":
        return []
    boundary = parse_st_box(text_object.get("Boundary"))
    if boundary is None:
        logger.warning(f"OFD_TEXT_INVALID_BOUNDARY: object_id={text_object.get('ID')!r}")
        return []
    boundary_page = transform_bbox(boundary, parent_transform)
    if boundary_page is None:
        return []
    object_clip = bbox_intersection(parent_clip, boundary_page)
    if object_clip is None:
        return []
    translation = Affine.translation(boundary[0], boundary[1])
    object_transform = parent_transform.compose(translation).compose(parse_affine(text_object.get("CTM")))
    read_direction = float(parse_int(text_object.get("ReadDirection")) or 0)
    char_direction = float(parse_int(text_object.get("CharDirection")) or 0)
    direction_transform = object_transform.compose(Affine.rotation(read_direction))
    font_id = parse_int(text_object.get("Font"))
    font = resources.fonts.get(font_id) if font_id is not None else None
    try:
        size = max(0.1, float(text_object.get("Size") or 1.0))
        hscale = max(0.01, float(text_object.get("HScale") or 1.0))
    except ValueError:
        size, hscale = 1.0, 1.0
    styles = _styles(text_object, font, style)
    decoded_text_codes: list[tuple[etree._Element, str]] = []
    position_count = 0
    for text_code in (element for element in text_object if local_name(element.tag) == "TextCode"):
        text = _decode_text_code_element(text_code, budget)
        decoded_text_codes.append((text_code, text))
        if text:
            position_count += len(text)
    glyph_by_position = _glyph_map(text_object, position_count, budget)
    global_position = 0
    inherited_x: float | None = None
    inherited_y: float | None = None
    lines: list[TextLine] = []
    object_id = parse_int(text_object.get("ID"))
    for code_index, (text_code, text) in enumerate(decoded_text_codes):
        if not text:
            continue
        try:
            if text_code.get("X") is not None:
                inherited_x = float(text_code.get("X") or "")
            if text_code.get("Y") is not None:
                inherited_y = float(text_code.get("Y") or "")
        except ValueError:
            continue
        if inherited_x is None or inherited_y is None:
            logger.warning(f"OFD_TEXT_MISSING_ORIGIN: object_id={object_id}, text_code={code_index}")
            global_position += len(text)
            continue
        delta_count = max(0, len(text) - 1)
        delta_x = parse_delta(text_code.get("DeltaX"), delta_count, budget)
        delta_y = parse_delta(text_code.get("DeltaY"), delta_count, budget)
        origins: list[tuple[float, float]] = [(inherited_x, inherited_y)]
        for index in range(delta_count):
            previous = origins[-1]
            origins.append((previous[0] + delta_x[index], previous[1] + delta_y[index]))
        glyph_items: list[GlyphItem] = []
        for index, (character, origin) in enumerate(zip(text, origins, strict=True)):
            glyph_id = glyph_by_position.get(global_position + index)
            resolved_character = font_metrics.resolve_character(font, glyph_id, character)
            advance_hint = None
            if index + 1 < len(origins):
                advance_hint = math.dist(origin, origins[index + 1])
            local_bbox = font_metrics.glyph_bbox(
                font,
                glyph_id,
                resolved_character,
                size=size,
                hscale=hscale,
                advance_hint=advance_hint,
            )
            char_transform = direction_transform.compose(Affine.translation(origin[0], origin[1])).compose(
                Affine.rotation(char_direction)
            )
            quad = transform_quad(rect_quad(local_bbox), char_transform)
            glyph_bbox = quad_bbox(quad)
            if glyph_bbox is None:
                continue
            clipped_glyph_bbox = bbox_intersection(glyph_bbox, object_clip)
            if clipped_glyph_bbox is None:
                continue
            glyph_items.append(
                GlyphItem(
                    text=resolved_character,
                    bbox=clipped_glyph_bbox,
                    quad=quad,
                    origin=char_transform.apply((0.0, 0.0)),
                    glyph_id=glyph_id,
                )
            )
        global_position += len(text)
        if not glyph_items:
            continue
        line_bbox = bbox_union(item.bbox for item in glyph_items)
        if line_bbox is None:
            continue
        if line_bbox[2] <= line_bbox[0] or line_bbox[3] <= line_bbox[1]:
            continue
        lines.append(
            TextLine(
                text="".join(item.text for item in glyph_items),
                bbox=line_bbox,
                glyphs=glyph_items,
                angle=canonical_angle(transform_angle(direction_transform)),
                font_size=size * math.hypot(direction_transform.a, direction_transform.b),
                paint_order=paint_order + code_index,
                object_id=object_id,
                layer_type=layer_type,
                template_id=template_id,
                styles=styles,
            )
        )
    return lines


__all__ = [
    "FontMetricResolver",
    "OfdTextBudget",
    "build_text_lines",
    "decode_text_code",
    "format_line_html",
    "format_line_spans",
    "parse_delta",
]
