# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Optional


VISIBLE_SPACE_STYLES = {"underline", "emphasis", "strikethrough"}


@dataclass(frozen=True)
class OfficeRichTextSegment:
    """表示 Office 行内富文本片段，用于统一样式和超链接输出。"""

    text: str
    style: str | list[str] | tuple[str, ...] | None = None
    hyperlink: Optional[str] = None


def _style_list(style: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """把样式字符串或列表规范为样式列表。"""
    if not style:
        return []
    if isinstance(style, str):
        return [item.strip() for item in style.split(",") if item.strip()]
    return [str(item).strip() for item in style if str(item).strip()]


def _style_str(style: str | list[str] | tuple[str, ...] | None) -> Optional[str]:
    """把样式字符串或列表规范为逗号分隔字符串。"""
    styles = _style_list(style)
    return ",".join(styles) if styles else None


def _script_to_style_name(format_obj: Any) -> Optional[str]:
    """把 DOCX 上下标脚本位置转换为 Office 内部富文本样式名。"""
    script = getattr(format_obj, "script", None)
    script_value = getattr(script, "value", script)
    if script_value == "super":
        return "superscript"
    if script_value == "sub":
        return "subscript"
    return None


def formatting_to_style_str(format_obj: Any) -> Optional[str]:
    """从 Formatting-like 对象提取 Office 内部富文本样式字符串。"""
    if format_obj is None:
        return None
    styles = []
    if getattr(format_obj, "bold", False):
        styles.append("bold")
    if getattr(format_obj, "italic", False):
        styles.append("italic")
    if getattr(format_obj, "underline", False):
        styles.append("underline")
    if getattr(format_obj, "emphasis", False):
        styles.append("emphasis")
    if getattr(format_obj, "strikethrough", False):
        styles.append("strikethrough")
    script_style = _script_to_style_name(format_obj)
    if script_style:
        styles.append(script_style)
    return ",".join(styles) if styles else None


def has_visible_style(format_obj: Any) -> bool:
    """判断格式是否包含让空白文本也可见的样式。"""
    if format_obj is None:
        return False
    return bool(
        getattr(format_obj, "underline", False)
        or getattr(format_obj, "emphasis", False)
        or getattr(format_obj, "strikethrough", False)
    )


def has_non_visible_text_style(format_obj: Any) -> bool:
    """判断格式是否只包含空白文本不可见的字形样式。"""
    if format_obj is None:
        return False
    return bool(
        getattr(format_obj, "bold", False)
        or getattr(format_obj, "italic", False)
    )


def normalize_format_for_text(
    format_obj: Any,
    text: str,
    *,
    preserve_blank_non_visible_style: bool = False,
):
    """按文本内容规范 run 格式，避免空白 run 误把不可见样式带到输出。"""
    if format_obj is None:
        return None
    if text.strip():
        return format_obj

    updates = {}
    if getattr(format_obj, "underline_style", "") == "words":
        updates["underline"] = False
        updates["underline_style"] = ""
    if has_non_visible_text_style(format_obj) and not preserve_blank_non_visible_style:
        updates["bold"] = False
        updates["italic"] = False

    if updates and hasattr(format_obj, "model_copy"):
        format_obj = format_obj.model_copy(update=updates)

    if not has_visible_style(format_obj):
        if preserve_blank_non_visible_style and has_non_visible_text_style(format_obj):
            return format_obj
        return None
    return format_obj


def should_keep_group_text(
    text: str,
    format_obj: Any,
    *,
    preserve_plain_blank: bool = False,
) -> bool:
    """判断当前累积文本是否应输出，保留可见样式或被显式保留的空白。"""
    if not text:
        return False
    if text.strip():
        return True
    if has_visible_style(format_obj):
        return True
    return preserve_plain_blank


def append_rich_text_element(
    paragraph_elements: list[tuple[str, Any, Any]],
    text: str,
    format_obj: Any,
    hyperlink: Any,
) -> None:
    """追加段落元素；相邻同 URL 且同格式的片段合并为一个元素。"""
    if (
        hyperlink is not None
        and paragraph_elements
        and paragraph_elements[-1][2] is not None
        and str(paragraph_elements[-1][2]) == str(hyperlink)
        and paragraph_elements[-1][1] == format_obj
    ):
        previous_text, previous_format, previous_hyperlink = paragraph_elements[-1]
        paragraph_elements[-1] = (
            f"{previous_text}{text}",
            previous_format,
            previous_hyperlink,
        )
        return
    paragraph_elements.append((text, format_obj, hyperlink))


def format_text_tag(
    text: str,
    style_str: Optional[str] = None,
    *,
    force_tag: bool = False,
) -> str:
    """生成 Office 内部富文本 text 标签；无样式普通文本默认不包标签。"""
    if not text:
        return text
    if style_str:
        return f'<text style="{style_str}">{text}</text>'
    if force_tag:
        return f"<text>{text}</text>"
    return text


def is_valid_hyperlink_target(hyperlink: Any) -> bool:
    """判断超链接目标是否可作为真实链接输出。"""
    if hyperlink is None:
        return False
    hyperlink_str = str(hyperlink)
    return bool(hyperlink_str and hyperlink_str.strip() and hyperlink_str != ".")


def format_text_with_hyperlink(
    text: str,
    hyperlink: Any,
    style_str: Optional[str] = None,
) -> str:
    """按 Office 内部约定输出带样式/超链接的文本片段。"""
    if not text:
        return text
    if not is_valid_hyperlink_target(hyperlink):
        return format_text_tag(text, style_str)

    text_tag = format_text_tag(text, style_str, force_tag=True)
    return f"<hyperlink>{text_tag}<url>{hyperlink}</url></hyperlink>"


def _format_hyperlink_segments(group: list[OfficeRichTextSegment]) -> str:
    """将连续同 URL 的多个片段渲染成单个 hyperlink 标签。"""
    if not group:
        return ""
    hyperlink = group[0].hyperlink
    if not is_valid_hyperlink_target(hyperlink):
        return "".join(
            format_text_tag(segment.text, _style_str(segment.style))
            for segment in group
            if segment.text
        )

    text_tags = [
        format_text_tag(segment.text, _style_str(segment.style), force_tag=True)
        for segment in group
        if segment.text
    ]
    return f"<hyperlink>{''.join(text_tags)}<url>{hyperlink}</url></hyperlink>"


def format_hyperlink_group(
    group: list[tuple[str, Any, Any]],
) -> str:
    """将 DOCX paragraph element 分组渲染成单个 hyperlink 标签。"""
    return _format_hyperlink_segments(
        [
            OfficeRichTextSegment(
                text=text,
                style=formatting_to_style_str(format_obj),
                hyperlink=str(hyperlink) if hyperlink is not None else None,
            )
            for text, format_obj, hyperlink in group
        ]
    )


def _style_has_visible_space(style: str | list[str] | tuple[str, ...] | None) -> bool:
    """判断样式列表是否会让空白文本在渲染结果中可见。"""
    return any(style_name in VISIBLE_SPACE_STYLES for style_name in _style_list(style))


def _trim_plain_edge_spaces(
    segments: list[OfficeRichTextSegment],
) -> list[OfficeRichTextSegment]:
    """只裁剪段落首尾普通空白，不裁剪带可见样式的空白。"""
    trimmed_segments = [segment for segment in segments if segment.text is not None]
    if not trimmed_segments:
        return []

    start_idx = 0
    while start_idx < len(trimmed_segments):
        segment = trimmed_segments[start_idx]
        if segment.text.strip() or _style_has_visible_space(segment.style):
            if not _style_has_visible_space(segment.style):
                trimmed_segments[start_idx] = OfficeRichTextSegment(
                    segment.text.lstrip(),
                    segment.style,
                    segment.hyperlink,
                )
            break
        start_idx += 1
    if start_idx == len(trimmed_segments):
        return []

    trimmed_segments = trimmed_segments[start_idx:]
    end_idx = len(trimmed_segments) - 1
    while end_idx >= 0:
        segment = trimmed_segments[end_idx]
        if segment.text.strip() or _style_has_visible_space(segment.style):
            if not _style_has_visible_space(segment.style):
                trimmed_segments[end_idx] = OfficeRichTextSegment(
                    segment.text.rstrip(),
                    segment.style,
                    segment.hyperlink,
                )
            break
        end_idx -= 1
    if end_idx < 0:
        return []
    return trimmed_segments[: end_idx + 1]


def _merge_non_link_segments(
    segments: list[OfficeRichTextSegment],
) -> list[OfficeRichTextSegment]:
    """合并相邻同样式的非超链接片段，避免输出碎片化样式标记。"""
    merged: list[OfficeRichTextSegment] = []
    for segment in segments:
        if (
            merged
            and not is_valid_hyperlink_target(merged[-1].hyperlink)
            and not is_valid_hyperlink_target(segment.hyperlink)
            and _style_str(merged[-1].style) == _style_str(segment.style)
        ):
            previous = merged[-1]
            merged[-1] = OfficeRichTextSegment(
                f"{previous.text}{segment.text}",
                previous.style,
                previous.hyperlink,
            )
            continue
        merged.append(segment)
    return merged


def build_rich_text_from_segments(
    segments: list[OfficeRichTextSegment],
    *,
    trim_plain_edges: bool = False,
) -> str:
    """从 Office 富文本片段构建内部标记，统一处理样式、空白和超链接分组。"""
    normalized_segments = [
        OfficeRichTextSegment(
            segment.text,
            _style_str(segment.style),
            str(segment.hyperlink) if segment.hyperlink is not None else None,
        )
        for segment in segments
        if segment.text is not None and segment.text != ""
    ]
    if trim_plain_edges:
        normalized_segments = _trim_plain_edge_spaces(normalized_segments)
    normalized_segments = _merge_non_link_segments(normalized_segments)

    rendered_parts = []
    index = 0
    while index < len(normalized_segments):
        segment = normalized_segments[index]
        if is_valid_hyperlink_target(segment.hyperlink):
            group = [segment]
            index += 1
            while index < len(normalized_segments):
                next_segment = normalized_segments[index]
                if (
                    not is_valid_hyperlink_target(next_segment.hyperlink)
                    or str(next_segment.hyperlink) != str(segment.hyperlink)
                ):
                    break
                group.append(next_segment)
                index += 1
            rendered_parts.append(_format_hyperlink_segments(group))
            continue

        rendered_parts.append(
            format_text_with_hyperlink(
                segment.text,
                segment.hyperlink,
                _style_str(segment.style),
            )
        )
        index += 1

    return "".join(rendered_parts)


def build_text_mappings_from_elements(
    paragraph_elements: list[tuple[str, Any, Any]],
) -> list[tuple[str, str]]:
    """按连续同 URL hyperlink 分组，生成原文到内部富文本标记的映射。"""
    mappings = []
    index = 0
    while index < len(paragraph_elements):
        text, format_obj, hyperlink = paragraph_elements[index]
        if not text:
            index += 1
            continue

        if is_valid_hyperlink_target(hyperlink):
            group = [(text, format_obj, hyperlink)]
            index += 1
            while index < len(paragraph_elements):
                next_text, next_format, next_hyperlink = paragraph_elements[index]
                if (
                    not next_text
                    or not is_valid_hyperlink_target(next_hyperlink)
                    or str(next_hyperlink) != str(hyperlink)
                ):
                    break
                group.append((next_text, next_format, next_hyperlink))
                index += 1

            original_text = "".join(item_text for item_text, _, _ in group)
            mappings.append((original_text, format_hyperlink_group(group)))
            continue

        style_str = formatting_to_style_str(format_obj)
        mappings.append((text, format_text_with_hyperlink(text, hyperlink, style_str)))
        index += 1
    return mappings
