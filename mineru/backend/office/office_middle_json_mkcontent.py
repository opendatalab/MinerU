# Copyright (c) Opendatalab. All rights reserved.
import os
import re
import unicodedata
from html import escape

from loguru import logger

from mineru.backend.utils.markdown_utils import (
    escape_conservative_markdown_text,
    escape_text_block_markdown_prefix,
)
from mineru.utils.config_reader import get_latex_delimiter_config
from mineru.utils.enum_class import MakeMode, BlockType, ContentType, ContentTypeV2

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

OFFICE_STYLE_RENDER_MODE_ENV = 'MINERU_OFFICE_STYLE_RENDER_MODE'
OFFICE_STYLE_RENDER_MODE_HTML = 'html'
OFFICE_STYLE_RENDER_MODE_MARKDOWN = 'markdown'
OFFICE_MARKDOWN_WRAPPER_STYLES = {'bold', 'italic', 'strikethrough'}


def _apply_markdown_style(content: str, style: list) -> str:
    """
    按照字体样式列表对文本内容应用 Markdown 格式。

    支持的样式：bold, italic, underline, strikethrough
    组合顺序（由内到外）：
      1. bold/italic（纯 Markdown，最内层，兼容性最广）
      2. strikethrough（~~，中间层，包裹纯 Markdown 符号广泛支持）
      3. underline（HTML <u>，最外层，作为 HTML 容器不干扰内部 Markdown 解析）

    这样可避免 `**~~<u>text</u>~~**` 在部分渲染器中因 HTML 标签打断
    外层 Markdown 标记解析而导致样式失效的问题，
    改为输出 `<u>~~**text**~~</u>`，兼容性更好。

    Args:
        content: 待格式化的文本内容
        style: 样式列表，如 ["bold", "italic"]

    Returns:
        str: 应用 Markdown 格式后的文本
    """
    if not style or not content:
        return content

    # 第一层（最内层）：bold / italic —— 纯 Markdown 符号，放最里面兼容性最好
    if 'bold' in style and 'italic' in style:
        content = f'***{content}***'
    elif 'bold' in style:
        content = f'**{content}**'
    elif 'italic' in style:
        content = f'*{content}*'

    # 第二层：strikethrough —— ~~text~~，包裹纯 Markdown 内容，广泛支持
    if 'strikethrough' in style:
        content = f'~~{content}~~'

    # 第三层（最外层）：underline —— markdown 无原生语法，使用 HTML <u> 标签
    # 作为外层 HTML 容器，不会干扰内部 Markdown 标记的解析
    if 'underline' in style:
        content = f'<u>{content}</u>'

    return content


def _apply_html_style(content: str, style: list) -> str:
    """Apply inline styles with HTML tags for markdown-hostile contexts."""
    if not style or not content:
        return content

    if 'bold' in style and 'italic' in style:
        content = f'<strong><em>{content}</em></strong>'
    elif 'bold' in style:
        content = f'<strong>{content}</strong>'
    elif 'italic' in style:
        content = f'<em>{content}</em>'

    if 'strikethrough' in style:
        content = f'<del>{content}</del>'

    if 'underline' in style:
        content = f'<u>{content}</u>'

    return content


def _get_office_style_render_mode() -> str:
    mode = os.getenv(
        OFFICE_STYLE_RENDER_MODE_ENV,
        OFFICE_STYLE_RENDER_MODE_MARKDOWN,
    ).strip().lower()
    if mode in {
        OFFICE_STYLE_RENDER_MODE_HTML,
        OFFICE_STYLE_RENDER_MODE_MARKDOWN,
    }:
        return mode
    logger.warning(
        f"Invalid {OFFICE_STYLE_RENDER_MODE_ENV}={mode!r}, "
        f"fallback to {OFFICE_STYLE_RENDER_MODE_MARKDOWN!r}"
    )
    return OFFICE_STYLE_RENDER_MODE_MARKDOWN


def _apply_configured_style(content: str, style: list) -> str:
    if _get_office_style_render_mode() == OFFICE_STYLE_RENDER_MODE_MARKDOWN:
        return _apply_markdown_style(content, style)
    return _apply_html_style(content, style)


def _render_link(text: str, url: str) -> str:
    if _get_office_style_render_mode() == OFFICE_STYLE_RENDER_MODE_MARKDOWN:
        return f'[{text}]({url})'
    return f'<a href="{escape(url, quote=True)}">{text}</a>'


def _prefix_table_img_src(html: str, img_buket_path: str) -> str:
    """Prefix local-path img src attributes in table HTML with img_buket_path."""
    if not html or not img_buket_path:
        return html
    return re.sub(
        r'src="(?!data:)([^"]+)"',
        lambda m: f'src="{img_buket_path}/{m.group(1)}"',
        html,
    )


def _replace_eq_tags_in_table_html(html: str) -> str:
    """Replace <eq>...</eq> tags in table HTML with inline math delimiters."""
    if not html:
        return html
    return re.sub(
        r'<eq>(.*?)</eq>',
        lambda m: f' {inline_left_delimiter}{m.group(1)}{inline_right_delimiter} ',
        html,
        flags=re.DOTALL,
    )


def _format_embedded_html(html: str, img_buket_path: str) -> str:
    """Apply image-path prefixing and equation replacement for HTML-like content."""
    return _replace_eq_tags_in_table_html(_prefix_table_img_src(html, img_buket_path))


def _build_media_path(img_buket_path: str, image_path: str) -> str:
    """Build a display path while keeping empty image references empty."""
    if not image_path:
        return ''
    if not img_buket_path:
        return image_path
    return f"{img_buket_path}/{image_path}"


def _escape_office_markdown_text(content: str) -> str:
    """Escape plain-text Office content before applying Markdown wrappers."""
    if not content:
        return content
    if _get_office_style_render_mode() != OFFICE_STYLE_RENDER_MODE_MARKDOWN:
        return content
    return escape_conservative_markdown_text(content)


def get_title_level(para_block):
    title_level = para_block.get('level', 2)
    return title_level


def _make_rendered_part(
    span_type,
    rendered_content: str,
    raw_content: str = '',
    style: list | None = None,
    has_markdown_wrapper: bool = False,
):
    return {
        'span_type': span_type,
        'rendered_content': rendered_content,
        'raw_content': raw_content,
        'style': style or [],
        'has_markdown_wrapper': has_markdown_wrapper,
    }


def _has_markdown_wrapper(style: list) -> bool:
    if _get_office_style_render_mode() != OFFICE_STYLE_RENDER_MODE_MARKDOWN:
        return False
    if not style or 'underline' in style:
        return False
    return any(name in OFFICE_MARKDOWN_WRAPPER_STYLES for name in style)


def _get_first_non_whitespace_char(text: str):
    for ch in text:
        if not ch.isspace():
            return ch
    return None


def _get_last_non_whitespace_char(text: str):
    for ch in reversed(text):
        if not ch.isspace():
            return ch
    return None


def _is_punctuation_or_symbol(ch: str) -> bool:
    return unicodedata.category(ch).startswith(('P', 'S'))


def _is_boundary_text_char(ch: str) -> bool:
    if ch.isspace():
        return False
    return not _is_punctuation_or_symbol(ch)


def _needs_markdown_it_boundary_space(prev_part: dict, next_part: dict) -> bool:
    if _get_office_style_render_mode() != OFFICE_STYLE_RENDER_MODE_MARKDOWN:
        return False
    if not prev_part.get('has_markdown_wrapper', False):
        return False
    if next_part.get('span_type') in {
        ContentType.HYPERLINK,
        ContentType.INLINE_EQUATION,
        ContentType.INTERLINE_EQUATION,
    }:
        return False

    prev_raw = prev_part.get('raw_content', '')
    next_raw = next_part.get('raw_content', '')
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
    if not _is_boundary_text_char(next_char):
        return False
    return True


def _join_rendered_parts(parts: list[dict]) -> str:
    para_text = ''
    prev_part = None

    for i, part in enumerate(parts):
        span_type = part['span_type']
        content = part['rendered_content']
        is_last = i == len(parts) - 1

        if span_type == ContentType.INLINE_EQUATION:
            if para_text and not para_text.endswith(' '):
                para_text += ' '
            para_text += content
            if not is_last:
                para_text += ' '
        else:
            if prev_part is not None and _needs_markdown_it_boundary_space(prev_part, part):
                para_text += ' '
            para_text += content

        prev_part = part

    return para_text


def _append_text_part(parts: list[dict], original_content: str, span_style: list):
    escaped_content = _escape_office_markdown_text(original_content)
    content_stripped = escaped_content.strip()
    if content_stripped:
        styled = _apply_configured_style(content_stripped, span_style)
        leading = escaped_content[:len(escaped_content) - len(escaped_content.lstrip())]
        trailing = escaped_content[len(escaped_content.rstrip()):]
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                leading + styled + trailing,
                raw_content=original_content,
                style=span_style,
                has_markdown_wrapper=_has_markdown_wrapper(span_style),
            )
        )
    elif original_content:
        visible_styles = {'underline', 'strikethrough'}
        if span_style and any(s in visible_styles for s in span_style):
            rendered_content = original_content.replace(" ", "&nbsp;")
            rendered_content = _apply_configured_style(rendered_content, span_style)
        else:
            rendered_content = original_content
        parts.append(
            _make_rendered_part(
                ContentType.TEXT,
                rendered_content,
                raw_content=original_content,
                style=span_style,
            )
        )


def _append_hyperlink_part(
    parts: list[dict],
    original_content: str,
    span_style: list,
    url: str = '',
    plain_text_only: bool = False,
):
    link_text = _escape_office_markdown_text(original_content.strip())
    if not link_text:
        return

    styled_text = _apply_configured_style(link_text, span_style)
    if plain_text_only:
        leading = original_content[:len(original_content) - len(original_content.lstrip())]
        trailing = original_content[len(original_content.rstrip()):]
        rendered_content = leading + styled_text + trailing
        has_markdown_wrapper = _has_markdown_wrapper(span_style)
    else:
        rendered_content = _render_link(styled_text, url)
        has_markdown_wrapper = False

    parts.append(
        _make_rendered_part(
            ContentType.HYPERLINK,
            rendered_content,
            raw_content=original_content,
            style=span_style,
            has_markdown_wrapper=has_markdown_wrapper,
        )
    )


def merge_para_with_text(para_block, escape_text_block_prefix=True):
    # First pass: collect rendered parts with raw boundary metadata.
    parts = []
    if para_block['type'] == BlockType.TITLE:
        if para_block.get('is_numbered_style', False):
            section_number = para_block.get('section_number', '')
            if section_number:
                parts.append(
                    _make_rendered_part(
                        ContentType.TEXT,
                        f"{section_number} ",
                        raw_content=f"{section_number} ",
                    )
                )

    for line in para_block['lines']:
        for span in line['spans']:
            span_type = span['type']
            span_style = span.get('style', [])

            if span_type == ContentType.TEXT:
                _append_text_part(parts, span['content'], span_style)
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
                content = content.strip()
                if content:
                    parts.append(
                        _make_rendered_part(
                            span_type,
                            content,
                            raw_content=span['content'],
                        )
                    )
            elif span_type == ContentType.INTERLINE_EQUATION:
                content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                content = content.strip()
                if content:
                    parts.append(
                        _make_rendered_part(
                            span_type,
                            content,
                            raw_content=span['content'],
                        )
                    )
            elif span_type == ContentType.HYPERLINK:
                _append_hyperlink_part(
                    parts,
                    span['content'],
                    span_style,
                    url=span.get('url', ''),
                )

    para_text = _join_rendered_parts(parts)
    if escape_text_block_prefix and para_block.get('type') == BlockType.TEXT:
        para_text = escape_text_block_markdown_prefix(para_text)
    return para_text


def _flatten_list_items(list_block):
    """Recursively flatten nested list blocks into a list of prefixed item strings."""
    items = []
    ilevel = list_block.get('ilevel', 0)
    attribute = list_block.get('attribute', 'unordered')
    indent = '    ' * ilevel
    ordered_counter = 1

    for block in list_block.get('blocks', []):
        if block['type'] in [BlockType.LIST, BlockType.INDEX]:
            items.extend(_flatten_list_items(block))
        else:
            item_text = merge_para_with_text(block, escape_text_block_prefix=False)
            if item_text.strip():
                if attribute == 'ordered':
                    items.append(f"{indent}{ordered_counter}. {item_text}")
                    ordered_counter += 1
                else:
                    items.append(f"{indent}- {item_text}")

    return items


def _flatten_list_items_v2(list_block):
    """Recursively flatten nested list blocks into v2-structured item dicts."""
    items = []
    ilevel = list_block.get('ilevel', 0)
    attribute = list_block.get('attribute', 'unordered')
    ordered_counter = 1

    for block in list_block.get('blocks', []):
        if block['type'] in [BlockType.LIST, BlockType.INDEX]:
            items.extend(_flatten_list_items_v2(block))
        else:
            item_content = merge_para_with_text_v2(block)
            if item_content:
                if attribute == 'ordered':
                    prefix = f"{'    ' * ilevel}{ordered_counter}."
                    ordered_counter += 1
                else:
                    prefix = f"{'    ' * ilevel}-"
                item = {
                    'item_type': 'text',
                    'ilevel': ilevel,
                    'prefix': prefix,
                    'item_content': item_content,
                }
                anchor = block.get("anchor")
                if isinstance(anchor, str) and anchor.strip():
                    item["anchor"] = anchor.strip()
                items.append(item)

    return items


def merge_list_to_markdown(list_block):
    """Recursively convert a nested list block to markdown text."""
    return '\n'.join(_flatten_list_items(list_block)) + '\n'


def _flatten_index_items(index_block):
    """Recursively flatten index (TOC) blocks into markdown list items.

    Strips the trailing tab+page-number from span content and, when target
    location fields are present on the leaf text block, wraps the text in
    a markdown hyperlink pointing to the body-block anchor.

    Styling (bold, italic, underline, strikethrough) is applied via the
    configured office style render mode. HYPERLINK spans are rendered as
    plain styled text (without the URL) because TOC entries use
    document-internal bookmark links, not external URLs.

    The tab+page-number is stripped from the raw content BEFORE markdown
    style markers are applied, so that closing markers (e.g. ``**``) are
    never inadvertently removed by the tab-stripping step.
    """
    items = []
    ilevel = index_block.get('ilevel', 0)
    indent = '    ' * ilevel

    for child in index_block.get('blocks', []):
        if child.get('type') == BlockType.INDEX:
            items.extend(_flatten_index_items(child))
        elif child.get('type') == BlockType.TEXT:
            span_items = []   # list of (content, span_type, span_style)
            anchor = child.get('anchor')
            if not isinstance(anchor, str) or not anchor.strip():
                anchor = None
            else:
                anchor = anchor.strip()

            for line in child.get('lines', []):
                for span in line.get('spans', []):
                    content = span.get('content', '')
                    span_style = span.get('style', [])
                    span_type = span.get('type')
                    span_items.append((content, span_type, span_style))

            if not span_items:
                continue

            # ----------------------------------------------------------
            # Step 1: Strip the trailing tab+page-number from the raw
            # (unstyled) content BEFORE applying markdown markers.
            #
            # Find the last non-equation span that contains a tab; strip
            # everything after its last tab ONLY when the trailing token
            # actually looks like a page number.
            # Then replace any remaining internal tabs with spaces so that
            # "1.1\t研究对象" → "1.1 研究对象".
            # ----------------------------------------------------------
            def _looks_like_page_token(token: str) -> bool:
                token = token.strip()
                if not token:
                    return False
                # Page tokens are usually short and contain no CJK characters.
                if len(token) > 12:
                    return False
                if re.search(r'[\u4e00-\u9fff]', token):
                    return False
                # Arabic / Roman / single-letter page styles.
                if re.fullmatch(r'\d+', token):
                    return True
                if re.fullmatch(r'[ivxlcdm]+', token.lower()):
                    return True
                if re.fullmatch(r'[a-zA-Z]', token):
                    return True
                return False

            last_tab_span_idx = -1
            for i, (content, span_type, _) in enumerate(span_items):
                if span_type != ContentType.INLINE_EQUATION and '\t' in content:
                    last_tab_span_idx = i

            should_strip_page_tail = False
            if last_tab_span_idx != -1:
                last_tab_content = span_items[last_tab_span_idx][0]
                tab_tail = last_tab_content.rsplit('\t', 1)[1]
                should_strip_page_tail = _looks_like_page_token(tab_tail)

            # Build stripped span_items
            stripped_span_items = []
            for i, (content, span_type, span_style) in enumerate(span_items):
                if span_type != ContentType.INLINE_EQUATION:
                    if i == last_tab_span_idx and should_strip_page_tail:
                        # Strip from last tab onwards (removes tab + page number)
                        content = content.rsplit('\t', 1)[0]
                    # Replace remaining internal tabs with spaces
                    content = content.replace('\t', ' ')
                stripped_span_items.append((content, span_type, span_style))

            # ----------------------------------------------------------
            # Step 2: Apply markdown styles and build the final text.
            #
            # If all non-equation spans share the same non-empty style
            # (common in TOC entries like all-bold), apply style once to
            # the whole item to avoid fragmented markers such as
            # "**foo****bar**".
            # ----------------------------------------------------------
            non_eq_styles = [
                tuple(span_style)
                for content, span_type, span_style in stripped_span_items
                if content and span_type != ContentType.INLINE_EQUATION
            ]
            uniform_style = None
            if non_eq_styles:
                first_style = non_eq_styles[0]
                if first_style and all(s == first_style for s in non_eq_styles):
                    uniform_style = list(first_style)

            if uniform_style:
                raw_parts = []
                for content, span_type, _span_style in stripped_span_items:
                    if not content:
                        continue
                    if span_type == ContentType.INLINE_EQUATION:
                        raw_parts.append(
                            f'{inline_left_delimiter}{content}{inline_right_delimiter}'
                        )
                    else:
                        # For TOC rendering, hyperlink spans output as plain text.
                        raw_parts.append(_escape_office_markdown_text(content))
                item_text = ''.join(raw_parts).strip()
                if item_text:
                    item_text = _apply_configured_style(item_text, uniform_style)
            else:
                rendered_parts = []
                for content, span_type, span_style in stripped_span_items:
                    if not content:
                        continue
                    if span_type == ContentType.INLINE_EQUATION:
                        rendered_parts.append(
                            _make_rendered_part(
                                span_type,
                                f'{inline_left_delimiter}{content}{inline_right_delimiter}',
                                raw_content=content,
                            )
                        )
                    elif span_type == ContentType.HYPERLINK:
                        _append_hyperlink_part(
                            rendered_parts,
                            content,
                            span_style,
                            plain_text_only=True,
                        )
                    else:
                        _append_text_part(rendered_parts, content, span_style)

                item_text = _join_rendered_parts(rendered_parts).strip()
            if not item_text:
                continue

            if anchor is not None:
                item_text = _render_link(item_text, f"#{anchor}")

            items.append(f"{indent}- {item_text}")

    return items


def merge_index_to_markdown(index_block):
    """Convert a nested index (TOC) block to markdown with hyperlinks."""
    return '\n'.join(_flatten_index_items(index_block)) + '\n'


def mk_blocks_to_markdown(para_blocks, make_mode, img_buket_path='', page_idx=None):
    page_markdown = []
    for para_block in para_blocks:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION]:
            para_text = merge_para_with_text(para_block)
            if para_type == BlockType.TEXT:
                bookmark_anchor = para_block.get("anchor")
                if (
                    isinstance(bookmark_anchor, str)
                    and bookmark_anchor.strip()
                    and bookmark_anchor.strip().startswith("_Toc")
                ):
                    para_text = f'<a id="{bookmark_anchor.strip()}"></a>\n{para_text}'
        elif para_type == BlockType.LIST:
            para_text = merge_list_to_markdown(para_block)
        elif para_type == BlockType.INDEX:
            para_text = merge_index_to_markdown(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            title_text = merge_para_with_text(para_block)
            bookmark_anchor = para_block.get("anchor")
            if isinstance(bookmark_anchor, str) and bookmark_anchor.strip():
                para_text = f'<a id="{bookmark_anchor.strip()}"></a>\n{"#" * title_level} {title_text}'
            else:
                para_text = f'{"#" * title_level} {title_text}'
        elif para_type == BlockType.IMAGE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                for block in para_block['blocks']:  # 1st.拼image_body
                    if block['type'] == BlockType.IMAGE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.IMAGE:
                                    if span.get('image_path', ''):
                                        para_text += f"![]({img_buket_path}/{span['image_path']})"
                for block in para_block['blocks']:  # 2nd.拼image_caption
                    if block['type'] == BlockType.IMAGE_CAPTION:
                        para_text += '  \n' + merge_para_with_text(block)

        elif para_type == BlockType.TABLE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                for block in para_block['blocks']:  # 1st.拼table_body
                    if block['type'] == BlockType.TABLE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.TABLE:
                                    para_text += f"\n{_format_embedded_html(span['html'], img_buket_path)}\n"
                for block in para_block['blocks']:  # 2nd.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += '  \n' + merge_para_with_text(block)
        elif para_type == BlockType.CHART:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                image_path, chart_content = get_body_data(para_block)
                if chart_content:
                    para_text += f"\n{_format_embedded_html(chart_content, img_buket_path)}\n"
                elif image_path:
                    para_text += f"![]({_build_media_path(img_buket_path, image_path)})"
                else:
                    continue
                for block in para_block['blocks']:
                    if block['type'] == BlockType.CHART_CAPTION:
                        para_text += '  \n' + merge_para_with_text(block)
        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip())
            page_markdown.append(para_text.strip('\r\n'))

    return page_markdown


def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        para_content = {
            'type': para_type,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.LIST:
        attribute = para_block.get('attribute', 'unordered')
        para_content = {
            'type': para_type,
            'list_items': _flatten_list_items(para_block),
        }
    elif para_type == BlockType.INDEX:
        para_content = {
            'type': para_type,
            'list_items': _flatten_index_items(para_block),
        }
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        para_content = {
            'type': ContentType.EQUATION,
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
        }
    elif para_type == BlockType.IMAGE:
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
    elif para_type == BlockType.TABLE:
        para_content = {'type': ContentType.TABLE, BlockType.TABLE_CAPTION: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = _format_embedded_html(span['html'], img_buket_path)
            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
    elif para_type == BlockType.CHART:
        para_content = {
            'type': ContentType.CHART,
            'img_path': '',
            'content': '',
            BlockType.CHART_CAPTION: [],
        }
        for block in para_block['blocks']:
            if block['type'] == BlockType.CHART_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.CHART:
                            para_content['img_path'] = _build_media_path(
                                img_buket_path,
                                span.get('image_path', ''),
                            )
                            if span.get('content', ''):
                                para_content['content'] = _format_embedded_html(
                                    span['content'],
                                    img_buket_path,
                                )
            if block['type'] == BlockType.CHART_CAPTION:
                para_content[BlockType.CHART_CAPTION].append(merge_para_with_text(block))

    para_content['page_idx'] = page_idx
    anchor = para_block.get("anchor")
    if isinstance(anchor, str) and anchor.strip():
        para_content["anchor"] = anchor.strip()

    return para_content


def make_blocks_to_content_list_v2(para_block, img_buket_path):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
        elif para_type == BlockType.PAGE_FOOTNOTE:
            content_type = ContentTypeV2.PAGE_FOOTNOTE
        else:
            raise ValueError(f"Unknown para_type: {para_type}")
        para_content = {
            'type': content_type,
            'content': {
                f"{content_type}_content": merge_para_with_text_v2(para_block),
            }
        }
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content = {
                'type': ContentTypeV2.TITLE,
                'content': {
                    "title_content": merge_para_with_text_v2(para_block),
                    "level": title_level
                }
            }
        else:
            para_content = {
                'type': ContentTypeV2.PARAGRAPH,
                'content': {
                    "paragraph_content": merge_para_with_text_v2(para_block),
                }
            }
    elif para_type in [
        BlockType.TEXT,
    ]:
        para_content = {
            'type': ContentTypeV2.PARAGRAPH,
            'content': {
                'paragraph_content': merge_para_with_text_v2(para_block),
            }
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        _, math_content = get_body_data(para_block)
        para_content = {
            'type': ContentTypeV2.EQUATION_INTERLINE,
            'content': {
                'math_content': math_content,
                'math_type': 'latex',
            }
        }
    elif para_type == BlockType.IMAGE:
        image_caption = []
        image_path, _ = get_body_data(para_block)
        image_source = {
            'path': f"{img_buket_path}/{image_path}",
        }
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_CAPTION:
                image_caption.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.IMAGE,
            'content': {
                'image_source': image_source,
                'image_caption': image_caption,
            }
        }
    elif para_type == BlockType.TABLE:
        table_caption = []
        _, html = get_body_data(para_block)
        if html.count("<table") > 1:
            table_nest_level = 2
        else:
            table_nest_level = 1
        if (
                "colspan" in html or
                "rowspan" in html or
                table_nest_level > 1
        ):
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE

        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_CAPTION:
                table_caption.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.TABLE,
            'content': {
                'table_caption': table_caption,
                'html': _format_embedded_html(html, img_buket_path),
                'table_type': table_type,
                'table_nest_level': table_nest_level,
            }
        }
    elif para_type == BlockType.CHART:
        chart_caption = []
        image_path, chart_content = get_body_data(para_block)
        for block in para_block['blocks']:
            if block['type'] == BlockType.CHART_CAPTION:
                chart_caption.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.CHART,
            'content': {
                'image_source': {
                    'path': _build_media_path(img_buket_path, image_path),
                },
                'content': _format_embedded_html(chart_content, img_buket_path),
                'chart_caption': chart_caption,
            }
        }
    elif para_type == BlockType.LIST:
        list_type = ContentTypeV2.LIST_TEXT
        attribute = para_block.get('attribute', 'unordered')
        para_content = {
            'type': ContentTypeV2.LIST,
            'content': {
                'list_type': list_type,
                'attribute': attribute,
                'list_items': _flatten_list_items_v2(para_block),
            }
        }
    elif para_type == BlockType.INDEX:
        para_content = {
            'type': ContentTypeV2.INDEX,
            'content': {
                'list_type': ContentTypeV2.LIST_TEXT,
                'list_items': _flatten_list_items_v2(para_block),
            }
        }

    anchor = para_block.get("anchor")
    if isinstance(anchor, str) and anchor.strip():
        para_content["anchor"] = anchor.strip()

    return para_content


def get_body_data(para_block):
    """
    Extract image_path and body content from para_block
    Returns:
        - For IMAGE/INTERLINE_EQUATION: (image_path, '')
        - For TABLE: (image_path, html)
        - For CHART: (image_path, content)
        - Default: ('', '')
    """

    def get_data_from_spans(lines):
        for line in lines:
            for span in line.get('spans', []):
                span_type = span.get('type')
                if span_type == ContentType.TABLE:
                    return span.get('image_path', ''), span.get('html', '')
                elif span_type == ContentType.CHART:
                    return span.get('image_path', ''), span.get('content', '')
                elif span_type == ContentType.IMAGE:
                    return span.get('image_path', ''), ''
                elif span_type == ContentType.INTERLINE_EQUATION:
                    return span.get('image_path', ''), span.get('content', '')
                elif span_type == ContentType.TEXT:
                    return '', span.get('content', '')
        return '', ''

    # 处理嵌套的 blocks 结构
    if 'blocks' in para_block:
        for block in para_block['blocks']:
            block_type = block.get('type')
            if block_type in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.CHART_BODY, BlockType.CODE_BODY]:
                result = get_data_from_spans(block.get('lines', []))
                if result != ('', ''):
                    return result
                if block_type == BlockType.CHART_BODY:
                    return result
        return '', ''

    # 处理直接包含 lines 的结构
    return get_data_from_spans(para_block.get('lines', []))


def merge_para_with_text_v2(para_block):
    _visible_styles = {'underline', 'strikethrough'}
    para_content = []
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            content = span.get("content", '')
            span_style = span.get('style', [])
            has_visible_style = bool(
                span_style and any(s in _visible_styles for s in span_style)
            )
            if content.strip() or (content and has_visible_style):
                if span['type'] == ContentType.INLINE_EQUATION:
                    span['type'] = ContentTypeV2.SPAN_EQUATION_INLINE
                para_content.append(span)
    return para_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               img_buket_path: str = '',
               ):

    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        paras_of_discarded = page_info.get('discarded_blocks')
        page_idx = page_info.get('page_idx')
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            if not paras_of_layout:
                continue
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path,
                                                   page_idx=page_idx)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            if not para_blocks:
                continue
            for para_block in para_blocks:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx)
                output_content.append(para_content)
        elif make_mode == MakeMode.CONTENT_LIST_V2:
            # https://github.com/drunkpig/llm-webkit-mirror/blob/dev6/docs/specification/output_format/content_list_spec.md
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            page_contents = []
            if para_blocks:
                for para_block in para_blocks:
                    para_content = make_blocks_to_content_list_v2(para_block, img_buket_path)
                    page_contents.append(para_content)
            output_content.append(page_contents)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode in [MakeMode.CONTENT_LIST, MakeMode.CONTENT_LIST_V2]:
        return output_content
    return None
