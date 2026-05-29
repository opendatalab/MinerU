# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass, field
import unicodedata
from html import escape

from mineru.backend.utils.markdown_utils import (
    escape_conservative_markdown_text,
    escape_text_block_markdown_prefix,
)
from mineru.utils.config_reader import get_latex_delimiter_config
from mineru.utils.enum_class import BlockType, ContentType

latex_delimiters_config = get_latex_delimiter_config()

default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}

delimiters = latex_delimiters_config if latex_delimiters_config else default_delimiters

display_left_delimiter = delimiters['display']['left']
display_right_delimiter = delimiters['display']['right']
inline_left_delimiter = delimiters['inline']['left']
inline_right_delimiter = delimiters['inline']['right']

OFFICE_INLINE_SYNTAX_HTML = 'html'
OFFICE_INLINE_SYNTAX_MARKDOWN = 'markdown'
OFFICE_MARKDOWN_STYLE_WRAPPERS = {
    frozenset({'bold'}): '**',
    frozenset({'italic'}): '*',
    frozenset({'strikethrough'}): '~~',
    frozenset({'bold', 'italic'}): '***',
}
OFFICE_COMPLEX_HTML_STYLES = {
    'underline',
    'emphasis',
    'superscript',
    'subscript',
}
OFFICE_EMPHASIS_STYLE = 'text-emphasis: dot; text-emphasis-position: under;'
STYLE_WRAPPER_OPEN = {
    'emphasis': f'<span style="{OFFICE_EMPHASIS_STYLE}">',
    'strikethrough': '<s>',
    'italic': '<em>',
    'bold': '<strong>',
    'underline': '<u>',
    'superscript': '<sup>',
    'subscript': '<sub>',
}
STYLE_WRAPPER_CLOSE = {
    'emphasis': '</span>',
    'strikethrough': '</s>',
    'italic': '</em>',
    'bold': '</strong>',
    'underline': '</u>',
    'superscript': '</sup>',
    'subscript': '</sub>',
}


@dataclass
class RenderedPart:
    """保存一个已渲染行内片段及其原始元数据，供段落拼接阶段使用。"""

    span_type: str
    rendered_content: str
    raw_content: str = ''
    style: list = field(default_factory=list)
    has_markdown_wrapper: bool = False


@dataclass
class StyleRangeToken:
    """保存 HTML 样式范围合并所需的片段内容和样式集合。"""

    content: str
    style: set[str] = field(default_factory=set)


def _apply_markdown_style(content: str, style: list) -> str:
    """按可枚举 Markdown style key 渲染 wrapper，未知组合保持原文。"""
    if not style or not content:
        return content

    style_key = _get_markdown_style_key(style)
    wrapper = OFFICE_MARKDOWN_STYLE_WRAPPERS.get(style_key)
    if wrapper:
        return f'{wrapper}{content}{wrapper}'
    return content


def _apply_html_style(content: str, style: list) -> str:
    """用 HTML 标签渲染 Office 行内样式，适配不宜使用 Markdown wrapper 的场景。"""
    if not style or not content:
        return content

    if 'superscript' in style:
        content = f'<sup>{content}</sup>'
    elif 'subscript' in style:
        content = f'<sub>{content}</sub>'

    if 'underline' in style:
        content = f'<u>{content}</u>'

    if 'bold' in style:
        content = f'<strong>{content}</strong>'

    if 'italic' in style:
        content = f'<em>{content}</em>'

    if 'strikethrough' in style:
        content = f'<s>{content}</s>'

    if 'emphasis' in style:
        content = f'<span style="{OFFICE_EMPHASIS_STYLE}">{content}</span>'

    return content


def _apply_configured_style(content: str, style: list, inline_syntax: str) -> str:
    """按 block 级 auto 选择出来的语法渲染行内样式。"""
    if inline_syntax == OFFICE_INLINE_SYNTAX_MARKDOWN:
        return _apply_markdown_style(content, style)
    return _apply_html_style(content, style)


def _render_link(text: str, url: str, inline_syntax: str) -> str:
    """按 block 级语法渲染链接，复杂 HTML block 内统一使用 <a>。"""
    if inline_syntax == OFFICE_INLINE_SYNTAX_MARKDOWN:
        return f'[{text}]({url})'
    return f'<a href="{escape(url, quote=True)}">{text}</a>'


def _escape_office_inline_text(content: str, inline_syntax: str) -> str:
    """根据当前 block 语法转义普通文本，HTML block 内同时避开标签和 Markdown delimiter。"""
    if not content:
        return content
    if inline_syntax == OFFICE_INLINE_SYNTAX_MARKDOWN:
        return escape_conservative_markdown_text(content)
    return escape_conservative_markdown_text(escape(content, quote=False))


def get_title_level(para_block):
    title_level = para_block.get('level', 2)
    return title_level


def _make_rendered_part(
    span_type,
    rendered_content: str,
    raw_content: str = '',
    style: list | None = None,
    has_markdown_wrapper: bool = False,
) -> RenderedPart:
    """构造段落渲染中间片段，并保留 Markdown 边界补空格所需的原始信息。"""
    return RenderedPart(
        span_type=span_type,
        rendered_content=rendered_content,
        raw_content=raw_content,
        style=style or [],
        has_markdown_wrapper=has_markdown_wrapper,
    )


def _get_first_non_whitespace_char(text: str):
    """返回文本中第一个非空白字符，用于 block 级 Markdown 风险判定。"""
    for ch in text:
        if not ch.isspace():
            return ch
    return None


def _get_last_non_whitespace_char(text: str):
    """返回文本中最后一个非空白字符，用于 block 级 Markdown 风险判定。"""
    for ch in reversed(text):
        if not ch.isspace():
            return ch
    return None


def _is_punctuation_or_symbol(ch: str) -> bool:
    """判断字符是否属于 Unicode 标点或符号类别。"""
    return unicodedata.category(ch).startswith(('P', 'S'))


def _is_boundary_text_char(ch: str) -> bool:
    """判断字符是否是会和 Markdown delimiter 紧贴产生歧义的普通文本字符。"""
    if ch.isspace():
        return False
    return not _is_punctuation_or_symbol(ch)


def _needs_markdown_boundary_space(
    prev_part: RenderedPart,
    next_part: RenderedPart,
) -> bool:
    """判断 Markdown wrapper 后是否需要补空格，避免标点结尾的 wrapper 无法被解析。"""
    if not prev_part.has_markdown_wrapper:
        return False
    if next_part.span_type in {
        ContentType.HYPERLINK,
        ContentType.INLINE_EQUATION,
        ContentType.INTERLINE_EQUATION,
    }:
        return False

    prev_raw = prev_part.raw_content
    next_raw = next_part.raw_content
    if not prev_raw.strip() or not next_raw.strip():
        return False
    if prev_raw[-1].isspace() or next_raw[0].isspace():
        return False

    prev_char = _get_last_non_whitespace_char(prev_raw)
    next_char = _get_first_non_whitespace_char(next_raw)
    if prev_char is None or next_char is None:
        return False
    if not _is_punctuation_or_symbol(prev_char):
        return False
    return _is_boundary_text_char(next_char)


def _join_rendered_parts(parts: list[RenderedPart]) -> str:
    """按 Office 段落规则拼接行内片段，并为行内公式补必要空格。"""
    rendered_parts = []
    prev_part = None

    for i, part in enumerate(parts):
        span_type = part.span_type
        content = part.rendered_content
        is_last = i == len(parts) - 1

        if span_type == ContentType.INLINE_EQUATION:
            if rendered_parts and not rendered_parts[-1].endswith(' '):
                rendered_parts.append(' ')
            rendered_parts.append(content)
            if not is_last:
                rendered_parts.append(' ')
        else:
            if prev_part is not None and _needs_markdown_boundary_space(prev_part, part):
                rendered_parts.append(' ')
            rendered_parts.append(content)

        prev_part = part

    return ''.join(rendered_parts)


def _strip_text_block_markdown_edges(content: str) -> str:
    """去掉 Markdown 文本块首尾普通空白，避免段首缩进被渲染成代码块。"""
    if not content:
        return content
    return content.strip()


def _get_visible_space_marker(style: list) -> str | None:
    """根据可见空格样式选择 Markdown marker，下划线优先于删除线。"""
    if not style:
        return None
    if 'underline' in style:
        return '_'
    if 'strikethrough' in style:
        return '-'
    return None


def _is_ascii_space_only(content: str) -> bool:
    """判断文本是否只由普通 ASCII 空格组成。"""
    return bool(content) and all(char == ' ' for char in content)


def _replace_ascii_spaces_with_marker(
    content: str,
    marker: str,
    inline_syntax: str,
) -> str:
    """将普通 ASCII 空格替换为指定 marker，其他文本按当前 block 语法转义。"""
    rendered_parts = []
    text_buffer = []

    def flush_text_buffer():
        if text_buffer:
            rendered_parts.append(
                _escape_office_inline_text(''.join(text_buffer), inline_syntax)
            )
            text_buffer.clear()

    for char in content:
        if char == ' ':
            flush_text_buffer()
            rendered_parts.append(marker)
        else:
            text_buffer.append(char)

    flush_text_buffer()
    return ''.join(rendered_parts)


def _render_text_with_edge_space_markers(
    content: str,
    marker: str,
    inline_syntax: str,
) -> str:
    """渲染非空文本：只把首尾 ASCII 空格转成 marker，中间普通空格保持原样。"""
    leading_space_count = len(content) - len(content.lstrip(' '))
    trailing_space_count = len(content) - len(content.rstrip(' '))
    text_end = len(content) - trailing_space_count if trailing_space_count else len(content)
    core_text = content[leading_space_count:text_end]
    return (
        marker * leading_space_count
        + _escape_office_inline_text(core_text, inline_syntax)
        + marker * trailing_space_count
    )


def _render_visible_space_marker_text(
    content: str,
    style: list,
    inline_syntax: str,
    render_style: list | None = None,
) -> str:
    """渲染可见样式空格：纯空白用 marker，非空文本只处理边缘空格。"""
    marker = _get_visible_space_marker(style)
    render_style = style if render_style is None else render_style
    if marker is None:
        return _apply_configured_style(
            _escape_office_inline_text(content, inline_syntax),
            render_style or [],
            inline_syntax,
        )

    style = style or []
    if marker == '-' and not _is_ascii_space_only(content):
        return _apply_configured_style(
            _render_text_with_edge_space_markers(content, marker, inline_syntax),
            render_style,
            inline_syntax,
        )

    if _is_ascii_space_only(content):
        rendered_content = _replace_ascii_spaces_with_marker(
            content,
            marker,
            inline_syntax,
        )
        ignored_style = 'underline' if marker == '_' else 'strikethrough'
        render_style = [name for name in (render_style or []) if name != ignored_style]
        return _apply_configured_style(rendered_content, render_style, inline_syntax)

    rendered_content = _render_text_with_edge_space_markers(
        content,
        marker,
        inline_syntax,
    )
    return _apply_configured_style(rendered_content, render_style, inline_syntax)


def _render_styled_inline_text(content: str, style: list, inline_syntax: str) -> str:
    """渲染行内文本内容，统一复用可见空格 marker 规则。"""
    if content and _get_visible_space_marker(style):
        return _render_visible_space_marker_text(content, style, inline_syntax)

    escaped_content = _escape_office_inline_text(content, inline_syntax)
    return _apply_configured_style(escaped_content, style, inline_syntax)


def _escape_standalone_marker_rule(content: str) -> str:
    """独立一行全是 marker 时转义首个字符，避免被 Markdown 当作分隔线。"""
    if content and all(char == '_' for char in content):
        return f'\\{content}'
    if content and all(char == '-' for char in content):
        return f'\\{content}'
    return content


def _append_text_part(
    parts: list[RenderedPart],
    original_content: str,
    span_style: list,
    inline_syntax: str,
    render_style: list | None = None,
):
    render_style = span_style if render_style is None else render_style
    if original_content and _get_visible_space_marker(span_style):
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                _render_visible_space_marker_text(
                    original_content,
                    span_style,
                    inline_syntax,
                    render_style,
                ),
                raw_content=original_content,
                style=render_style,
                has_markdown_wrapper=_has_markdown_wrapper(render_style, inline_syntax),
            )
        )
        return

    escaped_content = _escape_office_inline_text(original_content, inline_syntax)
    content_stripped = escaped_content.strip()
    if content_stripped:
        styled = _apply_configured_style(
            content_stripped,
            render_style,
            inline_syntax,
        )
        leading = escaped_content[
            :len(escaped_content) - len(escaped_content.lstrip())
        ]
        trailing = escaped_content[len(escaped_content.rstrip()):]
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                leading + styled + trailing,
                raw_content=original_content,
                style=render_style,
                has_markdown_wrapper=_has_markdown_wrapper(render_style, inline_syntax),
            )
        )
    elif original_content:
        visible_styles = {'underline', 'strikethrough'}
        rendered_content = original_content
        if span_style and any(s in visible_styles for s in span_style):
            rendered_content = _apply_configured_style(
                _escape_office_inline_text(rendered_content, inline_syntax),
                render_style,
                inline_syntax,
            )
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                rendered_content,
                raw_content=original_content,
                style=render_style,
                has_markdown_wrapper=_has_markdown_wrapper(render_style, inline_syntax),
            )
        )


def _split_plain_blank_edges(
    text_spans: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """拆出首尾纯空白 span，避免把段首 emphasis-only 空格包进 HTML 外层后无法 trim。"""
    start = 0
    end = len(text_spans)
    while start < end and not str(text_spans[start].get('content', '')).strip():
        start += 1
    while end > start and not str(text_spans[end - 1].get('content', '')).strip():
        end -= 1
    return text_spans[:start], text_spans[start:end], text_spans[end:]


def _get_markdown_style_key(
    style: list | tuple | set | None,
) -> frozenset[str] | str | None:
    """返回可安全使用 Markdown 的样式 key；空样式返回 None，不支持返回空串。"""
    style_set = {name for name in (style or []) if name}
    if not style_set:
        return None
    if style_set & OFFICE_COMPLEX_HTML_STYLES:
        return ''
    style_key = frozenset(style_set)
    if style_key in OFFICE_MARKDOWN_STYLE_WRAPPERS:
        return style_key
    return ''


def _is_simple_markdown_style(style: list | tuple | set | None) -> bool:
    """判断样式是否适合保留为简单 Markdown 输出。"""
    return _get_markdown_style_key(style) != ''


def _has_markdown_wrapper(
    style: list | tuple | set | None,
    inline_syntax: str,
) -> bool:
    """判断当前片段是否实际使用 Markdown wrapper，供拼接阶段补空格。"""
    if inline_syntax != OFFICE_INLINE_SYNTAX_MARKDOWN:
        return False
    return _get_markdown_style_key(style) in OFFICE_MARKDOWN_STYLE_WRAPPERS


def _iter_para_inline_spans(para_block):
    """按 block 原始顺序遍历行内 span，供 auto 语法判定复用。"""
    for line in para_block.get('lines', []):
        for span in line.get('spans', []):
            yield span


def _hyperlink_requires_html(span: dict) -> bool:
    """判断 hyperlink 是否存在混合或复杂样式，需要整块切到 HTML。"""
    children = span.get('children') or []
    if not children:
        return not _is_simple_markdown_style(span.get('style', []))

    child_style_keys = set()
    for child in children:
        if child.get('type') != ContentType.TEXT:
            return True
        content = str(child.get('content', ''))
        if not content.strip():
            continue
        child_style = child.get('style', [])
        child_style_key = _get_markdown_style_key(child_style)
        if child_style_key == '':
            return True
        child_style_keys.add(child_style_key)

    return len(child_style_keys) > 1


def _iter_block_inline_units(para_block):
    """把 block 展开为线性文本单元，用于判断 Markdown 边界风险。"""
    if para_block.get('type') == BlockType.TITLE:
        section_number = para_block.get('section_number', '')
        if para_block.get('is_numbered_style', False) and section_number:
            yield {
                'span_type': ContentType.TEXT,
                'content': f'{section_number} ',
                'style': [],
            }

    for span in _iter_para_inline_spans(para_block):
        span_type = span.get('type')
        if span_type == ContentType.TEXT:
            yield {
                'span_type': ContentType.TEXT,
                'content': str(span.get('content', '')),
                'style': span.get('style', []),
            }
        elif span_type == ContentType.HYPERLINK:
            children = span.get('children') or []
            if children:
                for child in children:
                    if child.get('type') != ContentType.TEXT:
                        continue
                    yield {
                        'span_type': ContentType.HYPERLINK,
                        'content': str(child.get('content', '')),
                        'style': child.get('style', []),
                    }
            else:
                yield {
                    'span_type': ContentType.HYPERLINK,
                    'content': str(span.get('content', '')),
                    'style': span.get('style', []),
                }
        elif span_type in {ContentType.INLINE_EQUATION, ContentType.INTERLINE_EQUATION}:
            yield {
                'span_type': span_type,
                'content': str(span.get('content', '')),
                'style': [],
            }


def _select_block_inline_syntax(para_block) -> str:
    """按 block 粒度选择 inline 输出语法：简单样式用 Markdown，复杂样式统一 HTML。"""
    units = list(_iter_block_inline_units(para_block))
    markdown_style_keys = set()

    for span in _iter_para_inline_spans(para_block):
        span_type = span.get('type')
        if span_type == ContentType.HYPERLINK and _hyperlink_requires_html(span):
            return OFFICE_INLINE_SYNTAX_HTML

    for unit in units:
        content = unit.get('content', '')
        if not content:
            continue
        style_key = _get_markdown_style_key(unit.get('style'))
        if style_key == '':
            return OFFICE_INLINE_SYNTAX_HTML
        if style_key is not None:
            markdown_style_keys.add(style_key)
            if len(markdown_style_keys) > 1:
                return OFFICE_INLINE_SYNTAX_HTML

    return OFFICE_INLINE_SYNTAX_MARKDOWN


def _append_style_range_token(
    tokens: list[StyleRangeToken],
    rendered_content: str,
    style: list | set | None,
):
    """追加一个可参与 HTML 范围合并的文本 token，空内容不进入 wrapper diff。"""
    if not rendered_content:
        return
    tokens.append(
        StyleRangeToken(
            content=rendered_content,
            style=set(style or []),
        )
    )


def _extend_style_range_tokens(
    tokens: list[StyleRangeToken],
    original_content: str,
    span_style: list,
    inline_syntax: str,
):
    """把原始 text span 拆成 HTML 合并 token，并保留现有可见空格 marker 规则。"""
    if not original_content:
        return

    marker = _get_visible_space_marker(span_style)
    if marker:
        render_style = list(span_style or [])
        if marker == '-' and not _is_ascii_space_only(original_content):
            rendered_content = _render_text_with_edge_space_markers(
                original_content,
                marker,
                inline_syntax,
            )
        elif _is_ascii_space_only(original_content):
            rendered_content = _replace_ascii_spaces_with_marker(
                original_content,
                marker,
                inline_syntax,
            )
            ignored_style = 'underline' if marker == '_' else 'strikethrough'
            render_style = [
                name for name in render_style
                if name != ignored_style
            ]
        else:
            rendered_content = _render_text_with_edge_space_markers(
                original_content,
                marker,
                inline_syntax,
            )
        _append_style_range_token(
            tokens,
            rendered_content,
            render_style,
        )
        return

    escaped_content = _escape_office_inline_text(original_content, inline_syntax)
    content_stripped = escaped_content.strip()
    if content_stripped:
        leading = escaped_content[:len(escaped_content) - len(escaped_content.lstrip())]
        trailing = escaped_content[len(escaped_content.rstrip()):]
        if leading:
            _append_style_range_token(
                tokens,
                leading,
                [],
            )
        _append_style_range_token(
            tokens,
            content_stripped,
            span_style,
        )
        if trailing:
            _append_style_range_token(
                tokens,
                trailing,
                [],
            )
    else:
        _append_style_range_token(tokens, original_content, [])


def _build_style_range_tokens(
    text_spans: list[dict],
    inline_syntax: str,
) -> list[StyleRangeToken]:
    """将连续文本 span 标准化为 HTML token 序列，供统一 wrapper writer 渲染。"""
    tokens = []
    for span in text_spans:
        _extend_style_range_tokens(
            tokens,
            str(span.get('content', '')),
            span.get('style', []),
            inline_syntax,
        )
    return tokens


def _style_range_stack(token: StyleRangeToken) -> list[str]:
    """按外到内顺序生成 HTML block 需要打开的 wrapper 栈。"""
    style = token.style
    stack = []
    if 'emphasis' in style:
        stack.append('emphasis')
    if 'strikethrough' in style:
        stack.append('strikethrough')
    if 'italic' in style:
        stack.append('italic')
    if 'bold' in style:
        stack.append('bold')
    if 'underline' in style:
        stack.append('underline')
    if 'superscript' in style:
        stack.append('superscript')
    elif 'subscript' in style:
        stack.append('subscript')
    return stack


def _style_wrapper_open(wrapper: str) -> str:
    """返回指定 wrapper 的打开标记。"""
    return STYLE_WRAPPER_OPEN.get(wrapper, '')


def _style_wrapper_close(wrapper: str) -> str:
    """返回指定 wrapper 的关闭标记。"""
    return STYLE_WRAPPER_CLOSE.get(wrapper, '')


def _common_stack_prefix_len(
    current_stack: list[str],
    next_stack: list[str],
) -> int:
    """计算两个 wrapper 栈从外到内的共同前缀长度。"""
    prefix_len = 0
    for current, next_item in zip(current_stack, next_stack):
        if current != next_item:
            break
        prefix_len += 1
    return prefix_len


def _render_style_range_tokens(tokens: list[StyleRangeToken]) -> str:
    """使用 HTML wrapper stack diff 渲染连续文本 token，合并可连续表达的共同样式。"""
    rendered_parts = []
    current_stack = []

    for token in tokens:
        next_stack = _style_range_stack(token)
        prefix_len = _common_stack_prefix_len(current_stack, next_stack)

        for wrapper in reversed(current_stack[prefix_len:]):
            rendered_parts.append(_style_wrapper_close(wrapper))
        for wrapper in next_stack[prefix_len:]:
            rendered_parts.append(_style_wrapper_open(wrapper))

        rendered_parts.append(token.content)
        current_stack = next_stack

    for wrapper in reversed(current_stack):
        rendered_parts.append(_style_wrapper_close(wrapper))

    return ''.join(rendered_parts)


def _append_markdown_grouped_text_parts(
    parts: list[RenderedPart],
    text_spans: list[dict],
):
    """渲染简单 Markdown block，按可枚举 style key 合并相邻等价片段。"""
    pending_content = []
    pending_style = None
    pending_style_key = None
    has_pending = False

    def flush_pending():
        nonlocal pending_content, pending_style, pending_style_key, has_pending
        if pending_content:
            _append_text_part(
                parts,
                ''.join(pending_content),
                list(pending_style or []),
                OFFICE_INLINE_SYNTAX_MARKDOWN,
            )
            pending_content = []
            pending_style = None
            pending_style_key = None
            has_pending = False

    for span in text_spans:
        span_style = tuple(span.get('style', []))
        span_style_key = _get_markdown_style_key(span_style)
        if not has_pending:
            pending_style = span_style
            pending_style_key = span_style_key
            has_pending = True
        if span_style_key != pending_style_key:
            flush_pending()
            pending_style = span_style
            pending_style_key = span_style_key
            has_pending = True
        pending_content.append(str(span.get('content', '')))
    flush_pending()


def _append_style_grouped_text_parts(
    parts: list[RenderedPart],
    text_spans: list[dict],
    inline_syntax: str,
):
    """按 block 语法渲染连续文本 span：简单 Markdown，复杂 HTML。"""
    if inline_syntax == OFFICE_INLINE_SYNTAX_MARKDOWN:
        _append_markdown_grouped_text_parts(parts, text_spans)
        return

    leading_spans, core_spans, trailing_spans = _split_plain_blank_edges(text_spans)
    for span in leading_spans:
        _append_text_part(
            parts,
            span.get('content', ''),
            span.get('style', []),
            inline_syntax,
        )

    tokens = _build_style_range_tokens(core_spans, inline_syntax)
    if tokens:
        rendered_content = _render_style_range_tokens(tokens)
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                rendered_content,
            )
        )

    for span in trailing_spans:
        _append_text_part(
            parts,
            span.get('content', ''),
            span.get('style', []),
            inline_syntax,
        )


def _render_hyperlink_children_label(children: list[dict], inline_syntax: str) -> str:
    """渲染 hyperlink 的子文本片段，保留各自样式后再组成同一个链接 label。"""
    child_parts = []
    child_text_spans = []
    for child in children or []:
        if child.get('type') != ContentType.TEXT:
            continue
        child_text_spans.append({
            'content': child.get('content', ''),
            'style': child.get('style', []),
        })
    _append_style_grouped_text_parts(child_parts, child_text_spans, inline_syntax)
    return _join_rendered_parts(child_parts).strip()


def _append_hyperlink_part(
    parts: list[RenderedPart],
    original_content: str,
    span_style: list,
    inline_syntax: str,
    url: str = '',
    plain_text_only: bool = False,
    children: list[dict] | None = None,
):
    """渲染 hyperlink 片段；HTML block 使用 <a>，Markdown block 使用 []()。"""
    if children:
        styled_text = _render_hyperlink_children_label(children, inline_syntax)
        if not styled_text:
            return
        rendered_content = (
            styled_text
            if plain_text_only
            else _render_link(styled_text, url, inline_syntax)
        )
    else:
        stripped_content = original_content.strip()
        if not stripped_content:
            return

        styled_text = _render_styled_inline_text(
            stripped_content,
            span_style,
            inline_syntax,
        )
        if plain_text_only:
            leading = original_content[:len(original_content) - len(original_content.lstrip())]
            trailing = original_content[len(original_content.rstrip()):]
            rendered_content = leading + styled_text + trailing
        else:
            rendered_content = _render_link(styled_text, url, inline_syntax)

    parts.append(
        _make_rendered_part(
            ContentType.HYPERLINK,
            rendered_content,
        )
    )


def merge_para_with_text(para_block, escape_text_block_prefix=True):
    """将 Office 段落 block 渲染为 Markdown/HTML 混合文本，按 block 自动选择行内语法。"""
    inline_syntax = _select_block_inline_syntax(para_block)
    parts = []
    text_span_buffer = []

    def flush_text_span_buffer():
        """遇到公式、超链接等边界时落盘连续文本 span，避免跨语义边界合并。"""
        if not text_span_buffer:
            return
        _append_style_grouped_text_parts(parts, text_span_buffer, inline_syntax)
        text_span_buffer.clear()

    if para_block['type'] == BlockType.TITLE:
        if para_block.get('is_numbered_style', False):
            section_number = para_block.get('section_number', '')
            if section_number:
                parts.append(
                    _make_rendered_part(
                        ContentType.TEXT,
                        f"{section_number} ",
                    )
                )

    for line in para_block['lines']:
        for span in line['spans']:
            span_type = span['type']
            span_style = span.get('style', [])

            if span_type == ContentType.TEXT:
                text_span_buffer.append({
                    'content': span.get('content', ''),
                    'style': span_style,
                })
            elif span_type == ContentType.INLINE_EQUATION:
                flush_text_span_buffer()
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
                content = content.strip()
                if content:
                    parts.append(
                        _make_rendered_part(
                            span_type,
                            content,
                        )
                    )
            elif span_type == ContentType.INTERLINE_EQUATION:
                flush_text_span_buffer()
                content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                content = content.strip()
                if content:
                    parts.append(
                        _make_rendered_part(
                            span_type,
                            content,
                        )
                    )
            elif span_type == ContentType.HYPERLINK:
                flush_text_span_buffer()
                _append_hyperlink_part(
                    parts,
                    span['content'],
                    span_style,
                    inline_syntax,
                    url=span.get('url', ''),
                    children=span.get('children'),
                )
            else:
                flush_text_span_buffer()

    flush_text_span_buffer()
    para_text = _join_rendered_parts(parts)
    if para_block.get('type') == BlockType.TEXT:
        para_text = _strip_text_block_markdown_edges(para_text)
        para_text = _escape_standalone_marker_rule(para_text)
        if escape_text_block_prefix:
            para_text = escape_text_block_markdown_prefix(para_text)
    return para_text
