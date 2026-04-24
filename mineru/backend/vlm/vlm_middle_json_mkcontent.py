# Copyright (c) Opendatalab. All rights reserved.
import os
import re
from html import escape, unescape

from loguru import logger

from mineru.utils.char_utils import full_to_half_exclude_marks, is_hyphen_at_line_end
from mineru.utils.config_reader import get_latex_delimiter_config, get_formula_enable, get_table_enable
from mineru.utils.enum_class import MakeMode, BlockType, ContentType, ContentTypeV2
from mineru.utils.language import detect_lang
from mineru.backend.utils.markdown_utils import (
    escape_conservative_markdown_text,
    escape_text_block_markdown_prefix,
)

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


def _prefix_table_img_src(html, img_buket_path):
    """Prefix non-data image sources in table HTML with img_buket_path."""
    if not html or not img_buket_path:
        return html

    return re.sub(
        r'src="(?!data:)([^"]+)"',
        lambda match: f'src="{img_buket_path}/{match.group(1)}"',
        html,
    )


def _replace_eq_tags_in_table_html(html):
    """Replace <eq>...</eq> tags in table HTML with inline math delimiters."""
    if not html:
        return html

    return re.sub(
        r'<eq>(.*?)</eq>',
        lambda match: (
            f" {inline_left_delimiter}{unescape(match.group(1))}{inline_right_delimiter} "
        ),
        html,
        flags=re.DOTALL,
    )


def _format_embedded_html(html, img_buket_path):
    """Normalize embedded table HTML for markdown/content outputs."""
    return _replace_eq_tags_in_table_html(_prefix_table_img_src(html, img_buket_path))


def _build_media_path(img_buket_path, image_path):
    if not image_path:
        return ''
    if not img_buket_path:
        return image_path
    return f"{img_buket_path}/{image_path}"


def _apply_visual_sub_type(para_content, para_block):
    sub_type = para_block.get('sub_type')
    if sub_type:
        para_content['sub_type'] = sub_type


def _build_visual_details_block(content, span_type, summary_override=''):
    if not isinstance(content, str) or not content.strip():
        return ''

    if span_type == ContentType.CHART:
        summary = summary_override or "chart content"
    else:
        summary = summary_override or "image content"

    return (
        "<details>\n"
        f"<summary>{summary}</summary>\n\n"
        f"{content}\n"
        "</details>"
    )


def _build_visual_body_segments(image_path, content, img_buket_path, span_type, summary_override=''):
    body_segments = []
    media_path = _build_media_path(img_buket_path, image_path)
    if media_path:
        body_segments.append((f"![]({media_path})", 'markdown_line'))

    details_block = _build_visual_details_block(
        content,
        span_type,
        summary_override=summary_override,
    )
    if details_block:
        body_segments.append((details_block, 'html_block'))

    return body_segments


def _get_blocks_in_index_order(blocks):
    return [
        block
        for _, block in sorted(
            enumerate(blocks),
            key=lambda item: (item[1].get('index', float('inf')), item[0]),
        )
    ]


def _render_code_block_markdown(block, para_block):
    code_text = merge_para_with_text(block)
    if para_block.get('sub_type') == BlockType.CODE:
        guess_lang = para_block.get('guess_lang', 'txt')
        return f"```{guess_lang}\n{code_text}\n```"
    return code_text


def _render_visual_block_segments(block, para_block, img_buket_path='', table_enable=True):
    block_type = block['type']

    if block_type in [
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.CODE_CAPTION,
        BlockType.CODE_FOOTNOTE,
        BlockType.CHART_CAPTION,
        BlockType.CHART_FOOTNOTE,
    ]:
        block_text = merge_para_with_text(block)
        if block_text.strip():
            return [(block_text, 'markdown_line')]
        return []

    if block_type == BlockType.IMAGE_BODY:
        rendered_segments = []
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span.get('type') != ContentType.IMAGE:
                    continue
                rendered_segments.extend(
                    _build_visual_body_segments(
                        span.get('image_path', ''),
                        span.get('content', ''),
                        img_buket_path,
                        ContentType.IMAGE,
                        summary_override=para_block.get('sub_type', ''),
                    )
                )
        return rendered_segments

    if block_type == BlockType.CHART_BODY:
        rendered_segments = []
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span.get('type') != ContentType.CHART:
                    continue
                rendered_segments.extend(
                    _build_visual_body_segments(
                        span.get('image_path', ''),
                        span.get('content', ''),
                        img_buket_path,
                        ContentType.CHART,
                        summary_override=para_block.get('sub_type', ''),
                    )
                )
        return rendered_segments

    if block_type == BlockType.TABLE_BODY:
        rendered_segments = []
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span.get('type') != ContentType.TABLE:
                    continue
                if table_enable and span.get('html', ''):
                    rendered_segments.append((
                        _format_embedded_html(span['html'], img_buket_path),
                        'html_block',
                    ))
                elif span.get('image_path', ''):
                    rendered_segments.append((
                        f"![]({_build_media_path(img_buket_path, span['image_path'])})",
                        'markdown_line',
                    ))
        return rendered_segments

    if block_type == BlockType.CODE_BODY:
        block_text = _render_code_block_markdown(block, para_block)
        if block_text.strip():
            return [(block_text, 'markdown_line')]
        return []

    return []


def _get_visual_block_separator(prev_segment_kind, current_segment_kind):
    if prev_segment_kind == 'html_block' or current_segment_kind == 'html_block':
        return '\n\n'
    return '  \n'


def _merge_visual_blocks_to_markdown(para_block, img_buket_path='', table_enable=True):
    rendered_segments = []
    for block in _get_blocks_in_index_order(para_block.get('blocks', [])):
        rendered_segments.extend(
            _render_visual_block_segments(
                block,
                para_block,
                img_buket_path,
                table_enable=table_enable,
            )
        )

    para_text = ''
    prev_segment_kind = None
    for segment_text, segment_kind in rendered_segments:
        if para_text:
            para_text += _get_visual_block_separator(prev_segment_kind, segment_kind)
        para_text += segment_text
        prev_segment_kind = segment_kind

    return para_text


def merge_para_with_text(
    para_block,
    formula_enable=True,
    img_buket_path='',
    escape_text_block_prefix=True,
):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.TEXT]:
                span['content'] = full_to_half_exclude_marks(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)
    escape_markdown_text = para_block.get('type') != BlockType.CODE_BODY

    para_text = ''
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content']
                if escape_markdown_text:
                    content = escape_conservative_markdown_text(content)
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                if formula_enable:
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                else:
                    if span.get('image_path', ''):
                        content = f"![]({img_buket_path}/{span['image_path']})"

            content = content.strip()
            if content:

                if span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
                    continue

                # 定义CJK语言集合(中日韩)
                cjk_langs = {'zh', 'ja', 'ko'}
                # logger.info(f'block_lang: {block_lang}, content: {content}')

                # 判断是否为行末span
                is_last_span = j == len(line['spans']) - 1

                if block_lang in cjk_langs:  # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                    if is_last_span and span_type != ContentType.INLINE_EQUATION:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    # 西方文本语境下 每行的最后一个span判断是否要去除连字符
                    if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if (
                                is_last_span
                                and span_type == ContentType.TEXT
                                and is_hyphen_at_line_end(content)
                        ):
                            # 如果下一行的第一个span是小写字母开头，删除连字符
                            if (
                                    i+1 < len(para_block['lines'])
                                    and para_block['lines'][i + 1].get('spans')
                                    and para_block['lines'][i + 1]['spans'][0].get('type') == ContentType.TEXT
                                    and para_block['lines'][i + 1]['spans'][0].get('content', '')
                                    and para_block['lines'][i + 1]['spans'][0]['content'][0].islower()
                            ):
                                para_text += content[:-1]
                            else:  # 如果没有下一行，或者下一行的第一个span不是小写字母开头，则保留连字符但不加空格
                                para_text += content
                        else:  # 西方文本语境下 content间需要空格分隔
                            para_text += f'{content} '
    if escape_text_block_prefix and para_block.get('type') == BlockType.TEXT:
        para_text = escape_text_block_markdown_prefix(para_text)
    return para_text


def mk_blocks_to_markdown(para_blocks, make_mode, formula_enable, table_enable, img_buket_path=''):
    page_markdown = []
    for para_block in para_blocks:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION, BlockType.PHONETIC, BlockType.REF_TEXT]:
            para_text = merge_para_with_text(para_block, formula_enable=formula_enable, img_buket_path=img_buket_path)
        elif para_type == BlockType.LIST:
            for block in para_block['blocks']:
                item_text = merge_para_with_text(
                    block,
                    formula_enable=formula_enable,
                    img_buket_path=img_buket_path,
                    escape_text_block_prefix=False,
                )
                para_text += f"{item_text}  \n"
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.IMAGE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                para_text = _merge_visual_blocks_to_markdown(
                    para_block,
                    img_buket_path,
                    table_enable=table_enable,
                )

        elif para_type == BlockType.TABLE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                para_text = _merge_visual_blocks_to_markdown(
                    para_block,
                    img_buket_path,
                    table_enable=table_enable,
                )
        elif para_type == BlockType.CHART:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                para_text = _merge_visual_blocks_to_markdown(
                    para_block,
                    img_buket_path,
                    table_enable=table_enable,
                )
        elif para_type == BlockType.CODE:
            para_text = _merge_visual_blocks_to_markdown(
                para_block,
                img_buket_path,
                table_enable=table_enable,
            )

        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip() + '  ')
            page_markdown.append(para_text.strip())

    return page_markdown


def make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.PHONETIC,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
    ]:
        para_content = {
            'type': para_type,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.LIST:
        para_content = {
            'type': para_type,
            'sub_type': para_block.get('sub_type', ''),
            'list_items':[],
        }
        for block in para_block['blocks']:
            item_text = merge_para_with_text(block, escape_text_block_prefix=False)
            if item_text.strip():
                para_content['list_items'].append(item_text)
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
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        image_path, image_content = get_body_data(para_block)
        para_content['img_path'] = _build_media_path(img_buket_path, image_path)
        para_content['content'] = image_content if image_content else ''
        _apply_visual_sub_type(para_content, para_block)
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(merge_para_with_text(block))
    elif para_type == BlockType.TABLE:
        para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:

                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = _format_embedded_html(
                                    span['html'],
                                    img_buket_path,
                                )

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))
    elif para_type == BlockType.CHART:
        image_path, chart_content = get_body_data(para_block)
        para_content = {
            'type': ContentType.CHART,
            'img_path': _build_media_path(img_buket_path, image_path),
            'content': chart_content if chart_content else '',
            BlockType.CHART_CAPTION: [],
            BlockType.CHART_FOOTNOTE: [],
        }
        _apply_visual_sub_type(para_content, para_block)
        for block in para_block['blocks']:
            if block['type'] == BlockType.CHART_CAPTION:
                para_content[BlockType.CHART_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.CHART_FOOTNOTE:
                para_content[BlockType.CHART_FOOTNOTE].append(merge_para_with_text(block))
    elif para_type == BlockType.CODE:
        para_content = {'type': BlockType.CODE, 'sub_type': para_block["sub_type"], BlockType.CODE_CAPTION: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.CODE_BODY:
                code_text = merge_para_with_text(block)
                if para_block['sub_type'] == BlockType.CODE:
                    guess_lang = para_block.get("guess_lang", "txt")
                    code_text = f"```{guess_lang}\n{code_text}\n```"
                para_content[BlockType.CODE_BODY] = code_text
            if block['type'] == BlockType.CODE_CAPTION:
                para_content[BlockType.CODE_CAPTION].append(merge_para_with_text(block))

    page_width, page_height = page_size
    para_bbox = para_block.get('bbox')
    if para_bbox:
        x0, y0, x1, y1 = para_bbox
        para_content['bbox'] = [
            int(x0 * 1000 / page_width),
            int(y0 * 1000 / page_height),
            int(x1 * 1000 / page_width),
            int(y1 * 1000 / page_height),
        ]

    para_content['page_idx'] = page_idx

    return para_content


def make_blocks_to_content_list_v2(para_block, img_buket_path, page_size):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
        elif para_type == BlockType.ASIDE_TEXT:
            content_type = ContentTypeV2.PAGE_ASIDE_TEXT
        elif para_type == BlockType.PAGE_NUMBER:
            content_type = ContentTypeV2.PAGE_NUMBER
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
        BlockType.PHONETIC
    ]:
        para_content = {
            'type': ContentTypeV2.PARAGRAPH,
            'content': {
                'paragraph_content': merge_para_with_text_v2(para_block),
            }
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        image_path, math_content = get_body_data(para_block)
        para_content = {
            'type': ContentTypeV2.EQUATION_INTERLINE,
            'content': {
                'math_content': math_content,
                'math_type': 'latex',
                'image_source': {'path': f"{img_buket_path}/{image_path}"},
            }
        }
    elif para_type == BlockType.IMAGE:
        image_caption = []
        image_footnote = []
        image_path, image_content = get_body_data(para_block)
        image_source = {
            'path': _build_media_path(img_buket_path, image_path),
        }
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_CAPTION:
                image_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                image_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.IMAGE,
            'content': {
                'image_source': image_source,
                'content': image_content if image_content else '',
                'image_caption': image_caption,
                'image_footnote': image_footnote,
            }
        }
        _apply_visual_sub_type(para_content, para_block)
    elif para_type == BlockType.TABLE:
        table_caption = []
        table_footnote = []
        image_path, html = get_body_data(para_block)
        table_html = _format_embedded_html(html, img_buket_path)
        image_source = {
            'path': f"{img_buket_path}/{image_path}",
        }
        if table_html.count("<table") > 1:
            table_nest_level = 2
        else:
            table_nest_level = 1
        if (
                "colspan" in table_html or
                "rowspan" in table_html or
                table_nest_level > 1
        ):
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE

        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_CAPTION:
                table_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                table_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.TABLE,
            'content': {
                'image_source': image_source,
                'table_caption': table_caption,
                'table_footnote': table_footnote,
                'html': table_html,
                'table_type': table_type,
                'table_nest_level': table_nest_level,
            }
        }
    elif para_type == BlockType.CHART:
        chart_caption = []
        chart_footnote = []
        image_path, chart_content = get_body_data(para_block)
        for block in para_block['blocks']:
            if block['type'] == BlockType.CHART_CAPTION:
                chart_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.CHART_FOOTNOTE:
                chart_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.CHART,
            'content': {
                'image_source': {
                    'path': _build_media_path(img_buket_path, image_path),
                },
                'content': chart_content if chart_content else '',
                'chart_caption': chart_caption,
                'chart_footnote': chart_footnote,
            }
        }
        _apply_visual_sub_type(para_content, para_block)
    elif para_type == BlockType.CODE:
        code_caption = []
        code_content = []
        for block in para_block['blocks']:
            if block['type'] == BlockType.CODE_CAPTION:
                code_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.CODE_BODY:
                code_content = merge_para_with_text_v2(block)
        sub_type = para_block["sub_type"]
        if sub_type == BlockType.CODE:
            para_content = {
                'type': ContentTypeV2.CODE,
                'content': {
                    'code_caption': code_caption,
                    'code_content': code_content,
                    'code_language': para_block.get('guess_lang', 'txt'),
                }
            }
        elif sub_type == BlockType.ALGORITHM:
            para_content = {
                'type': ContentTypeV2.ALGORITHM,
                'content': {
                    'algorithm_caption': code_caption,
                    'algorithm_content': code_content,
                }
            }
        else:
            raise ValueError(f"Unknown code sub_type: {sub_type}")
    elif para_type == BlockType.REF_TEXT:
        para_content = {
            'type': ContentTypeV2.LIST,
            'content': {
                'list_type': ContentTypeV2.LIST_REF,
                'list_items': [
                    {
                        'item_type': 'text',
                        'item_content': merge_para_with_text_v2(para_block),
                    }
                ],
            }
        }
    elif para_type == BlockType.LIST:
        if 'sub_type' in para_block:
            if para_block['sub_type'] == BlockType.REF_TEXT:
                list_type = ContentTypeV2.LIST_REF
            elif para_block['sub_type'] == BlockType.TEXT:
                list_type = ContentTypeV2.LIST_TEXT
            else:
                raise ValueError(f"Unknown list sub_type: {para_block['sub_type']}")
        else:
            list_type = ContentTypeV2.LIST_TEXT
        list_items = []
        for block in para_block['blocks']:
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append({
                    'item_type': 'text',
                    'item_content': item_content,
                })
        para_content = {
            'type': ContentTypeV2.LIST,
            'content': {
                'list_type': list_type,
                'list_items': list_items,
            }
        }

    page_width, page_height = page_size
    para_bbox = para_block.get('bbox')
    if para_bbox:
        x0, y0, x1, y1 = para_bbox
        para_content['bbox'] = [
            int(x0 * 1000 / page_width),
            int(y0 * 1000 / page_height),
            int(x1 * 1000 / page_width),
            int(y1 * 1000 / page_height),
        ]

    return para_content





def get_body_data(para_block):
    """
    Extract image_path and body content from para_block
    Returns:
        - For IMAGE: (image_path, content)
        - For TABLE: (image_path, html)
        - For CHART: (image_path, content)
        - For INTERLINE_EQUATION: (image_path, content)
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
                    return span.get('image_path', ''), span.get('content', '')
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
        return '', ''

    # 处理直接包含 lines 的结构
    return get_data_from_spans(para_block.get('lines', []))


def merge_para_with_text_v2(para_block):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.TEXT]:
                span['content'] = full_to_half_exclude_marks(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)

    para_content = []
    para_type = para_block['type']
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            if span.get("content", '').strip():
                if span_type == ContentType.TEXT:
                    if para_type == BlockType.PHONETIC:
                        span_type = ContentTypeV2.SPAN_PHONETIC
                    else:
                        span_type = ContentTypeV2.SPAN_TEXT
                if span_type == ContentType.INLINE_EQUATION:
                    span_type = ContentTypeV2.SPAN_EQUATION_INLINE
                if span_type in [
                    ContentTypeV2.SPAN_TEXT,
                ]:
                    # 定义CJK语言集合(中日韩)
                    cjk_langs = {'zh', 'ja', 'ko'}
                    # logger.info(f'block_lang: {block_lang}, content: {content}')

                    # 判断是否为行末span
                    is_last_span = j == len(line['spans']) - 1

                    if block_lang in cjk_langs:  # 中文/日语/韩文语境下，换行不需要空格分隔,但是如果是行内公式结尾，还是要加空格
                        if is_last_span:
                            span_content = span['content']
                        else:
                            span_content = f"{span['content']} "
                    else:
                        # 如果span是line的最后一个且末尾带有-连字符，那么末尾不应该加空格,同时应该把-删除
                        if (
                                is_last_span
                                and is_hyphen_at_line_end(span['content'])
                        ):
                            # 如果下一行的第一个span是小写字母开头，删除连字符
                            if (
                                    i + 1 < len(para_block['lines'])
                                    and para_block['lines'][i + 1].get('spans')
                                    and para_block['lines'][i + 1]['spans'][0].get('type') == ContentType.TEXT
                                    and para_block['lines'][i + 1]['spans'][0].get('content', '')
                                    and para_block['lines'][i + 1]['spans'][0]['content'][0].islower()
                            ):
                                span_content = span['content'][:-1]
                            else:  # 如果没有下一行，或者下一行的第一个span不是小写字母开头，则保留连字符但不加空格
                                span_content = span['content']
                        else:
                            # 西方文本语境下content间需要空格分隔
                            span_content = f"{span['content']} "

                    if para_content and para_content[-1]['type'] == span_type:
                        # 合并相同类型的span
                        para_content[-1]['content'] += span_content
                    else:
                        span_content = {
                            'type': span_type,
                            'content': span_content,
                        }
                        para_content.append(span_content)

                elif span_type in [
                    ContentTypeV2.SPAN_PHONETIC,
                    ContentTypeV2.SPAN_EQUATION_INLINE,
                ]:
                    span_content = {
                        'type': span_type,
                        'content': span['content'],
                    }
                    para_content.append(span_content)
                else:
                    logger.warning(f"Unknown span type in merge_para_with_text_v2: {span_type}")
    return para_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               img_buket_path: str = '',
               ):

    formula_enable = get_formula_enable(os.getenv('MINERU_VLM_FORMULA_ENABLE', 'True').lower() == 'true')
    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')

    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        paras_of_discarded = page_info.get('discarded_blocks')
        page_idx = page_info.get('page_idx')
        page_size = page_info.get('page_size')
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            if not paras_of_layout:
                continue
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, formula_enable, table_enable, img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            if not para_blocks:
                continue
            for para_block in para_blocks:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size)
                output_content.append(para_content)
        elif make_mode == MakeMode.CONTENT_LIST_V2:
            # https://github.com/drunkpig/llm-webkit-mirror/blob/dev6/docs/specification/output_format/content_list_spec.md
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            page_contents = []
            if para_blocks:
                for para_block in para_blocks:
                    para_content = make_blocks_to_content_list_v2(para_block, img_buket_path, page_size)
                    page_contents.append(para_content)
            output_content.append(page_contents)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode in [MakeMode.CONTENT_LIST, MakeMode.CONTENT_LIST_V2]:
        return output_content
    return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level
