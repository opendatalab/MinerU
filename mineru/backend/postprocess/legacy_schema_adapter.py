# Copyright (c) Opendatalab. All rights reserved.
"""MinerU 3.4.5 Middle JSON page 到当前 raw ModelJson 的适配器。

3.4.5 的页面仍使用 ``preproc_blocks``/``para_blocks``、``lines`` 和
``spans``，自然语言行内语义则通过旧字符串标签或旧 span 字段表达。本模块只负责
把单页回推成当前 ``model_json_to_pages`` 能消费的 raw block 列表；文档 envelope
及当前 MiddleJson 的元数据由调用方负责。
"""

from __future__ import annotations

import html
import math
import re
from typing import Any

from ...types import INLINE_STYLE_ORDER, InlineSpan, parse_inline_spans
from ...utils.hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES, sanitize_hyperlink_target
from .inline import join_inline_spans

# 这些父块在 3.4.5 中通过 ``blocks`` 保存 body/caption/footnote；当前 raw
# ModelJson 则要求先展平成同级 block，再由统一后处理重新分组。
_VISUAL_PARENT_TYPES: frozenset[str] = frozenset(
    {
        "image",
        "table",
        "chart",
        "code",
    }
)

_INLINE_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "text",
        "ref_text",
        "doc_title",
        "paragraph_title",
        "header",
        "footer",
        "page_number",
        "aside_text",
        "page_footnote",
        "image_caption",
        "image_footnote",
        "table_caption",
        "table_footnote",
        "chart_caption",
        "chart_footnote",
        "code_caption",
        "code_footnote",
        "caption",
        "footnote",
        "phonetic",
    }
)

_STRING_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "equation",
        "image_body",
        "table_body",
        "chart_body",
        "code_body",
    }
)

_BLOCK_TYPE_ALIASES: dict[str, str] = {
    "abstract": "text",
    "phonetic": "text",
    "interline_equation": "equation",
}

_SUPPORTED_BLOCK_TYPES: frozenset[str] = frozenset(
    _VISUAL_PARENT_TYPES | _INLINE_CONTENT_BLOCK_TYPES | _STRING_CONTENT_BLOCK_TYPES | {"list", "index"}
)

_INLINE_START_RE = re.compile(
    r"<(?P<tag>eq|code|text|hyperlink|sup|sub|strong|b|em|i|s|u)(?P<attrs>\s[^<>]*?)?>",
    re.IGNORECASE,
)
_STYLE_ATTR_RE = re.compile(r"\bstyle\s*=\s*([\"'])(?P<style>.*?)\1", re.IGNORECASE | re.DOTALL)
_URL_RE = re.compile(r"<url>(?P<url>.*?)</url>", re.IGNORECASE | re.DOTALL)
_DIRECT_TAG_STYLES: dict[str, str] = {
    "strong": "bold",
    "b": "bold",
    "em": "italic",
    "i": "italic",
    "s": "strikethrough",
    "u": "underline",
    "sup": "superscript",
    "sub": "subscript",
}

# 3.4.5 PDF 的异常零面积框需要替换后才能通过当前严格 bbox 校验。
_PLACEHOLDER_BBOX: tuple[float, float, float, float] = (0.0, 0.0, 0.001, 0.001)


def legacy_page_to_model_list(page: dict[str, Any]) -> list[dict[str, Any]]:
    """把 MinerU 3.4.5 的单页 Middle JSON 转换为当前 raw model-list。"""
    if not isinstance(page, dict):
        raise ValueError("legacy Middle JSON page must be a dict")

    width, height = _parse_page_size(page.get("page_size"))
    source_blocks = page.get("preproc_blocks")
    if source_blocks is None:
        source_blocks = page.get("para_blocks", [])
    if not isinstance(source_blocks, list):
        raise ValueError("legacy Middle JSON page blocks must be a list")

    discarded_blocks = page.get("discarded_blocks", [])
    if not isinstance(discarded_blocks, list):
        raise ValueError("legacy Middle JSON discarded_blocks must be a list")

    legacy_blocks = [*source_blocks, *discarded_blocks]
    model_list: list[dict[str, Any]] = []
    for group_index, block in enumerate(legacy_blocks):
        if not isinstance(block, dict):
            raise ValueError("legacy Middle JSON block entries must be dicts")
        group_start = len(model_list)
        _collect_blocks(block, width, height, model_list, inherited=None)
        if width <= 0 or height <= 0:
            _assign_synthetic_group_bbox(model_list[group_start:], group_index, len(legacy_blocks))
    return model_list


def _parse_page_size(value: Any) -> tuple[float, float]:
    """读取合法 page_size；缺失、非数值或非正值统一返回零值。"""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return 0.0, 0.0
    width, height = value[:2]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (width, height)):
        return 0.0, 0.0
    parsed_width, parsed_height = float(width), float(height)
    if not all(math.isfinite(item) and item > 0 for item in (parsed_width, parsed_height)):
        return 0.0, 0.0
    return parsed_width, parsed_height


def _assign_synthetic_group_bbox(blocks: list[dict[str, Any]], group_index: int, group_count: int) -> None:
    """为无 page_size 文档分配互不重叠的占位框，并保持原顶层块的子项同组。"""
    safe_count = max(group_count, 1)
    y0 = (group_index + 0.2) / safe_count
    y1 = (group_index + 0.8) / safe_count
    bbox = (0.05, y0, 0.95, y1)
    for block in blocks:
        block["bbox"] = bbox
        lines = block.get("lines")
        if isinstance(lines, list):
            for line in lines:
                if isinstance(line, dict):
                    line["bbox"] = list(bbox)


def _collect_blocks(
    block: dict[str, Any],
    width: float,
    height: float,
    output: list[dict[str, Any]],
    *,
    inherited: dict[str, Any] | None,
) -> None:
    """递归展平视觉父块，并把普通叶子转换为当前 raw block。"""
    block_type = _normalize_block_type(block)
    nested = block.get("blocks")
    if block_type in _VISUAL_PARENT_TYPES and isinstance(nested, list) and nested:
        child_context = _visual_child_context(block, inherited)
        for child in nested:
            if not isinstance(child, dict):
                raise ValueError("legacy visual block children must be dicts")
            _collect_blocks(child, width, height, output, inherited=child_context)
        return

    if block_type == "list" and isinstance(nested, list) and nested:
        output.append(_convert_leaf_block(block, block_type, width, height, inherited))
        output.extend(_flatten_nested_list_items(nested, width, height))
        return
    if block_type == "index" and isinstance(nested, list) and nested:
        output.append(_flatten_nested_index(block, nested, width, height))
        return
    if block_type == "list" and isinstance(block.get("lines"), list) and block["lines"]:
        output.append(_convert_leaf_block(block, block_type, width, height, inherited))
        output.extend(_convert_pdf_list_items(block, width, height))
        return

    raw = _convert_leaf_block(block, block_type, width, height, inherited)
    output.append(raw)


def _visual_child_context(
    block: dict[str, Any],
    inherited: dict[str, Any] | None,
) -> dict[str, Any]:
    """收集视觉父块上需要下沉到 body 的元数据。"""
    context = dict(inherited or {})
    context["bbox"] = block.get("bbox", context.get("bbox"))
    sub_type = block.get("sub_type")
    if not isinstance(sub_type, str) or not sub_type.strip():
        sub_type = None
    if sub_type:
        context["sub_type"] = sub_type
    if isinstance(block.get("guess_lang"), str) and block["guess_lang"].strip():
        context["guess_lang"] = block["guess_lang"].strip()
    if "cell_merge" in block:
        context["cell_merge"] = block.get("cell_merge")
    return context


def _flatten_nested_list_items(nested: list[Any], width: float, height: float) -> list[dict[str, Any]]:
    """把 3.4.5 Office 嵌套列表叶子展平为当前 PDF list 可关联的文本块。"""
    items: list[dict[str, Any]] = []
    for child in nested:
        if not isinstance(child, dict):
            raise ValueError("legacy list children must be dicts")
        child_type = _normalize_block_type(child)
        child_nested = child.get("blocks")
        if child_type == "list" and isinstance(child_nested, list):
            items.extend(_flatten_nested_list_items(child_nested, width, height))
            continue
        if child_type not in {"text", "ref_text"}:
            child_type = "text"
        items.append(_convert_leaf_block(child, child_type, width, height, inherited=None))
    return items


def _flatten_nested_index(
    block: dict[str, Any],
    nested: list[Any],
    width: float,
    height: float,
) -> dict[str, Any]:
    """把 3.4.5 Office 嵌套目录折叠为当前 PDF index 使用的换行 Span。"""
    contents: list[list[dict[str, Any]]] = []

    def collect(children: list[Any]) -> None:
        """按深度优先顺序收集目录叶子的结构化行内内容。"""
        for child in children:
            if not isinstance(child, dict):
                raise ValueError("legacy index children must be dicts")
            child_nested = child.get("blocks")
            if _normalize_block_type(child) == "index" and isinstance(child_nested, list):
                collect(child_nested)
                continue
            converted = _convert_leaf_block(child, "text", width, height, inherited=None)
            content = converted.get("content")
            if isinstance(content, list) and content:
                contents.append([span for span in content if isinstance(span, dict)])

    collect(nested)
    joined: list[dict[str, Any]] = []
    for content_index, content in enumerate(contents):
        if content_index:
            _append_text_span(joined, "\n", ())
        _extend_span_dicts(joined, content)
    return {
        "type": "index",
        "bbox": _normalize_bbox(block.get("bbox"), width, height),
        "content": [span.model_dump(mode="json") for span in parse_inline_spans(joined)],
    }


def _convert_pdf_list_items(block: dict[str, Any], width: float, height: float) -> list[dict[str, Any]]:
    """把 3.4.5 PDF list 的物理行恢复为当前 bbox 关联需要的文本项。"""
    raw_lines = [line for line in block.get("lines", []) if isinstance(line, dict)]
    if not raw_lines:
        return []
    has_item_flags = any(line.get("is_list_start_line") or line.get("is_list_end_line") for line in raw_lines)
    line_groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    for line in raw_lines:
        if current_group and (line.get("is_list_start_line") or not has_item_flags):
            line_groups.append(current_group)
            current_group = []
        current_group.append(line)
        if line.get("is_list_end_line"):
            line_groups.append(current_group)
            current_group = []
    if current_group:
        line_groups.append(current_group)

    item_type = "ref_text" if block.get("sub_type") == "ref_text" else "text"
    items: list[dict[str, Any]] = []
    for lines in line_groups:
        synthetic_block = {
            "type": item_type,
            "bbox": _union_line_bboxes(lines, block.get("bbox")),
            "lines": lines,
        }
        items.append(_convert_leaf_block(synthetic_block, item_type, width, height, inherited=None))
    return items


def _union_line_bboxes(lines: list[dict[str, Any]], fallback: Any) -> Any:
    """合并一个旧列表项的物理行框，缺失时回退列表父框。"""
    bboxes = [line.get("bbox") for line in lines]
    valid = [
        bbox
        for bbox in bboxes
        if isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in bbox)
    ]
    if not valid:
        return fallback
    return [
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    ]


def _convert_leaf_block(
    block: dict[str, Any],
    block_type: str,
    width: float,
    height: float,
    inherited: dict[str, Any] | None,
) -> dict[str, Any]:
    """把一个 3.4.5 叶子块转换为当前 raw block 字典。"""
    inherited = inherited or {}
    raw_bbox = block.get("bbox", inherited.get("bbox"))
    bbox = _normalize_bbox(raw_bbox, width, height)
    lines = block.get("lines")
    legacy_lines = lines if isinstance(lines, list) else []
    effective_sub_type = _effective_sub_type(block, block_type, inherited)

    raw: dict[str, Any] = {"type": block_type, "bbox": bbox}
    if _uses_inline_content(block_type, effective_sub_type):
        raw["content"] = _extract_inline_content(block, legacy_lines, preserve_line_breaks=block_type == "index")
    elif block_type in _STRING_CONTENT_BLOCK_TYPES:
        raw["content"] = _extract_string_content(block, legacy_lines, block_type)

    image_fields = _extract_image_fields(block, legacy_lines)
    raw.update(image_fields)

    normalized_lines = _normalize_lines(legacy_lines, width, height)
    if normalized_lines:
        raw["lines"] = normalized_lines

    _copy_leaf_metadata(raw, block, block_type, effective_sub_type, inherited)
    return raw


def _effective_sub_type(block: dict[str, Any], block_type: str, inherited: dict[str, Any]) -> str | None:
    """解析 code/image/chart body 的有效 sub_type。"""
    sub_type = block.get("sub_type", inherited.get("sub_type"))
    if isinstance(sub_type, str) and sub_type.strip():
        return sub_type.strip()
    if block_type == "code_body":
        return "code"
    return None


def _uses_inline_content(block_type: str, sub_type: str | None) -> bool:
    """判断当前 raw block 是否必须使用结构化 InlineSpan 列表。"""
    return block_type in _INLINE_CONTENT_BLOCK_TYPES or (block_type == "code_body" and sub_type == "algorithm")


def _extract_inline_content(
    block: dict[str, Any],
    lines: list[Any],
    *,
    preserve_line_breaks: bool,
) -> list[dict[str, Any]]:
    """从旧 lines/spans 或直接 content 构造严格 InlineSpan 列表。"""
    line_contents: list[list[InlineSpan]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        converted: list[dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            _append_legacy_span(converted, span)
        parsed = parse_inline_spans(converted)
        if parsed:
            line_contents.append(parsed)

    if not line_contents:
        direct_content = block.get("content")
        converted = _convert_direct_inline_content(direct_content)
        return [span.model_dump(mode="json") for span in parse_inline_spans(converted)]

    if preserve_line_breaks:
        joined: list[dict[str, Any]] = []
        for line_index, content in enumerate(line_contents):
            if line_index:
                _append_text_span(joined, "\n", ())
            _extend_span_dicts(joined, [span.model_dump(mode="json") for span in content])
        return [span.model_dump(mode="json") for span in parse_inline_spans(joined)]

    joined_spans = join_inline_spans(line_contents)
    return [span.model_dump(mode="json") for span in joined_spans]


def _convert_direct_inline_content(content: Any) -> list[dict[str, Any]]:
    """转换没有 lines 包装的旧字符串或 span 列表。"""
    if isinstance(content, str):
        return _parse_legacy_markup(content)
    if not isinstance(content, list):
        return []
    output: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            _append_legacy_span(output, item)
        elif isinstance(item, str):
            _extend_span_dicts(output, _parse_legacy_markup(item))
    return output


def _append_legacy_span(output: list[dict[str, Any]], span: dict[str, Any]) -> None:
    """把一个 3.4.5 span 追加为当前 InlineSpan 字典。"""
    span_type = str(span.get("type", "text") or "text")
    content = span.get("content")
    if span_type in {"inline_equation", "equation_inline", "interline_equation", "equation"}:
        normalized = str(content or "").strip()
        if normalized:
            output.append({"type": "equation_inline", "content": normalized})
        return
    if span_type in {"code", "code_inline"}:
        if isinstance(content, str) and content:
            output.append({"type": "code_inline", "content": content})
        return
    if span_type == "hyperlink":
        _append_legacy_hyperlink(output, span)
        return
    if not isinstance(content, str) or not content:
        return

    styles = _normalize_styles(span.get("styles", span.get("style")))
    if styles:
        _append_text_span(output, content, styles)
    else:
        _extend_span_dicts(output, _parse_legacy_markup(content))


def _append_legacy_hyperlink(output: list[dict[str, Any]], span: dict[str, Any]) -> None:
    """转换结构化旧 hyperlink；不安全目标降级为可见子内容。"""
    children: list[dict[str, Any]] = []
    raw_children = span.get("children")
    if isinstance(raw_children, list):
        for child in raw_children:
            if isinstance(child, dict):
                _append_legacy_span(children, child)
    elif isinstance(span.get("content"), str):
        styles = _normalize_styles(span.get("styles", span.get("style")))
        if styles:
            _append_text_span(children, span["content"], styles)
        else:
            _extend_span_dicts(children, _parse_legacy_markup(span["content"]))

    safe_children = [child for child in children if child.get("type") != "hyperlink"]
    safe_url = sanitize_hyperlink_target(
        html.unescape(str(span.get("url", ""))),
        allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
        allow_relative=True,
        allow_fragment=True,
    )
    if safe_url and safe_children:
        output.append({"type": "hyperlink", "url": safe_url, "content": safe_children})
    else:
        _extend_span_dicts(output, safe_children)


def _parse_legacy_markup(content: str) -> list[dict[str, Any]]:
    """按 3.4.5 的私有字符串标签协议解析行内内容。"""
    output: list[dict[str, Any]] = []
    cursor = 0
    while match := _INLINE_START_RE.search(content, cursor):
        _append_text_span(output, content[cursor : match.start()], ())
        tag = match.group("tag").lower()
        closing = _find_matching_close(content, tag, match.end())
        if closing is None:
            _append_text_span(output, match.group(0), ())
            cursor = match.end()
            continue

        inner = content[match.end() : closing[0]]
        original = content[match.start() : closing[1]]
        parsed = _parse_legacy_element(tag, match.group("attrs") or "", inner)
        if parsed is None:
            _append_text_span(output, original, ())
        else:
            _extend_span_dicts(output, parsed)
        cursor = closing[1]
    _append_text_span(output, content[cursor:], ())
    return output


def _find_matching_close(content: str, tag: str, start: int) -> tuple[int, int] | None:
    """查找旧标签的配对结束位置，并兼容同名嵌套。"""
    token_re = re.compile(rf"<(?P<close>/)?{re.escape(tag)}(?:\s[^<>]*?)?>", re.IGNORECASE)
    depth = 1
    for match in token_re.finditer(content, start):
        depth += -1 if match.group("close") else 1
        if depth == 0:
            return match.start(), match.end()
    return None


def _parse_legacy_element(tag: str, attrs: str, inner: str) -> list[dict[str, Any]] | None:
    """把一个完整旧标签转换为当前 InlineSpan 字典。"""
    if tag == "eq":
        latex = html.unescape(inner).strip()
        return [{"type": "equation_inline", "content": latex}] if latex else []
    if tag == "code":
        code = html.unescape(inner)
        return [{"type": "code_inline", "content": code}] if code else []
    if tag == "hyperlink":
        return _parse_markup_hyperlink(inner)

    children = _parse_legacy_markup(inner)
    styles = _parse_style_attribute(attrs) if tag == "text" else [_DIRECT_TAG_STYLES[tag]]
    return _apply_styles(children, styles)


def _parse_markup_hyperlink(inner: str) -> list[dict[str, Any]] | None:
    """解析旧 hyperlink 标签；危险 URL 只移除链接语义。"""
    match = _URL_RE.search(inner)
    if match is None:
        return None
    label_content = f"{inner[: match.start()]}{inner[match.end() :]}"
    children = [span for span in _parse_legacy_markup(label_content) if span.get("type") != "hyperlink"]
    if not children:
        return []
    safe_url = sanitize_hyperlink_target(
        html.unescape(match.group("url")),
        allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
        allow_relative=True,
        allow_fragment=True,
    )
    if safe_url is None:
        return children
    return [{"type": "hyperlink", "url": safe_url, "content": children}]


def _parse_style_attribute(attrs: str) -> list[str]:
    """读取旧 ``text style`` 属性并过滤为当前白名单样式。"""
    match = _STYLE_ATTR_RE.search(attrs)
    return _normalize_styles(match.group("style") if match else None)


def _normalize_styles(value: Any) -> list[str]:
    """按当前固定顺序去重旧样式，并消解同时上下标的非法组合。"""
    if isinstance(value, str):
        candidates = [item.strip().lower() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(item).strip().lower() for item in value]
    else:
        candidates = []
    style_set = {item for item in candidates if item in INLINE_STYLE_ORDER}
    if "superscript" in style_set and "subscript" in style_set:
        style_set.remove("subscript")
    return [style for style in INLINE_STYLE_ORDER if style in style_set]


def _apply_styles(spans: list[dict[str, Any]], styles: list[str]) -> list[dict[str, Any]]:
    """把旧包装标签的样式递归应用到文字及链接文字。"""
    if not styles:
        return spans
    output: list[dict[str, Any]] = []
    for span in spans:
        span_type = span.get("type")
        if span_type == "text":
            merged_styles = _normalize_styles([*_normalize_styles(span.get("styles")), *styles])
            _append_text_span(output, str(span.get("content", "")), merged_styles)
        elif span_type == "hyperlink" and isinstance(span.get("content"), list):
            children = _apply_styles([child for child in span["content"] if isinstance(child, dict)], styles)
            output.append({**span, "content": children})
        else:
            output.append(dict(span))
    return output


def _append_text_span(output: list[dict[str, Any]], content: str, styles: Any) -> None:
    """追加非空文字，并合并相邻同样式 TextSpan。"""
    if not content:
        return
    normalized_styles = _normalize_styles(styles)
    if output and output[-1].get("type") == "text" and output[-1].get("styles", []) == normalized_styles:
        output[-1]["content"] = f"{output[-1].get('content', '')}{content}"
        return
    span: dict[str, Any] = {"type": "text", "content": content}
    if normalized_styles:
        span["styles"] = normalized_styles
    output.append(span)


def _extend_span_dicts(output: list[dict[str, Any]], spans: list[dict[str, Any]]) -> None:
    """追加一组 Span，并规范化序列边界上的相邻文字。"""
    for span in spans:
        if span.get("type") == "text" and isinstance(span.get("content"), str):
            _append_text_span(output, span["content"], span.get("styles", []))
        else:
            output.append(dict(span))


def _extract_string_content(block: dict[str, Any], lines: list[Any], block_type: str) -> str:
    """从旧 spans 提取公式、视觉主体、表格 HTML 或代码字符串。"""
    parts: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_parts: list[str] = []
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            value = span.get("html") if block_type == "table_body" and span.get("html") is not None else span.get("content")
            normalized = _stringify_visual_content(value)
            if normalized:
                line_parts.append(normalized)
        if line_parts:
            parts.append("".join(line_parts))

    if not parts:
        direct = block.get("html") if block_type == "table_body" and block.get("html") is not None else block.get("content")
        return _stringify_visual_content(direct)
    return "\n".join(parts) if block_type == "code_body" else "".join(parts)


def _stringify_visual_content(value: Any) -> str:
    """把旧视觉识别列表稳定折叠为当前字符串载荷。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    return "" if value is None else str(value)


def _extract_image_fields(block: dict[str, Any], lines: list[Any]) -> dict[str, str]:
    """从 block 或第一个视觉 span 提升当前支持的图片载荷字段。"""
    fields: dict[str, str] = {}
    for field_name in ("image_path", "image_base64", "image_url"):
        value = block.get(field_name)
        if isinstance(value, str) and value:
            fields[field_name] = value
    if fields:
        return fields
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            for field_name in ("image_path", "image_base64", "image_url"):
                value = span.get(field_name)
                if isinstance(value, str) and value and field_name not in fields:
                    fields[field_name] = value
            if fields:
                return fields
    return fields


def _copy_leaf_metadata(
    raw: dict[str, Any],
    block: dict[str, Any],
    block_type: str,
    sub_type: str | None,
    inherited: dict[str, Any],
) -> None:
    """复制当前后处理仍会读取、且不会越过严格模型边界的字段。"""
    if block.get("level") is not None:
        raw["level"] = block["level"]
    if sub_type:
        raw["sub_type"] = sub_type
    guess_lang = block.get("guess_lang", inherited.get("guess_lang"))
    if block_type == "code_body" and isinstance(guess_lang, str) and guess_lang.strip():
        raw["guess_lang"] = guess_lang.strip()
    if block_type == "table_body" and "cell_merge" in inherited:
        raw["cell_merge"] = inherited["cell_merge"]
    if block_type in {"doc_title", "paragraph_title", "page_footnote"}:
        anchor = block.get("anchor")
        if isinstance(anchor, str) and anchor.strip():
            raw["anchor"] = anchor.strip()


def _normalize_block_type(block: dict[str, Any]) -> str:
    """把最终 3.4.5 discriminator 映射到当前 raw 类型并拒绝内部标签。"""
    block_type = str(block.get("type", "") or "")
    if block_type == "title":
        return "doc_title" if block.get("level") == 1 else "paragraph_title"
    normalized = _BLOCK_TYPE_ALIASES.get(block_type, block_type)
    if normalized not in _SUPPORTED_BLOCK_TYPES:
        raise ValueError(f"unsupported MinerU 3.4.5 block type {block_type!r}; source reparse required")
    return normalized


def _normalize_bbox(value: Any, width: float, height: float) -> tuple[float, float, float, float]:
    """把旧 bbox 规范为当前严格的 0-1 非零面积坐标。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return _PLACEHOLDER_BBOX
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return _PLACEHOLDER_BBOX
    coords = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coords):
        return _PLACEHOLDER_BBOX
    if width > 0 and height > 0:
        coords = (coords[0] / width, coords[1] / height, coords[2] / width, coords[3] / height)
    elif not all(0.0 <= item <= 1.0 for item in coords):
        return _PLACEHOLDER_BBOX
    clamped = tuple(min(max(item, 0.0), 1.0) for item in coords)
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return _PLACEHOLDER_BBOX
    return clamped  # type: ignore[return-value]


def _normalize_lines(lines: list[Any], width: float, height: float) -> list[dict[str, list[float]]]:
    """保留跨页文本合并需要的行框，并转换到当前归一化坐标。"""
    result: list[dict[str, list[float]]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        bbox = _normalize_bbox(line.get("bbox"), width, height)
        result.append({"bbox": list(bbox)})
    return result


__all__ = ["legacy_page_to_model_list"]
