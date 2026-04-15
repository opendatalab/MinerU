# Copyright (c) Opendatalab. All rights reserved.
import re
from html import unescape

from loguru import logger

from mineru.utils.char_utils import full_to_half_exclude_marks, is_hyphen_at_line_end
from mineru.utils.config_reader import get_latex_delimiter_config
from mineru.backend.pipeline.para_split import ListLineTag
from mineru.utils.enum_class import BlockType, ContentType, ContentTypeV2, MakeMode
from mineru.utils.language import detect_lang
from mineru.backend.utils.markdown_utils import (
    escape_conservative_markdown_text,
    escape_text_block_markdown_prefix,
)


def make_blocks_to_markdown(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      ):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [
            BlockType.TEXT,
            BlockType.LIST,
            BlockType.INDEX,
            BlockType.ABSTRACT,
            BlockType.REF_TEXT
        ]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.INTERLINE_EQUATION:
            if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
                continue
            if para_block['lines'][0]['spans'][0].get('content', ''):
                para_text = merge_para_with_text(para_block)
            else:
                para_text = f"![]({img_buket_path}/{para_block['lines'][0]['spans'][0]['image_path']})"
        elif para_type == BlockType.SEAL:
            if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
                continue
            para_text = f"![]({img_buket_path}/{para_block['lines'][0]['spans'][0]['image_path']})"
            if para_block['lines'][0]['spans'][0].get('content', []):
                content = " ".join(para_block['lines'][0]['spans'][0]['content'])
                para_text += f"  \n{content}"
        elif para_type == BlockType.IMAGE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                para_text = merge_visual_blocks_to_markdown(para_block, img_buket_path)
        elif para_type == BlockType.CHART:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                para_text = merge_visual_blocks_to_markdown(para_block, img_buket_path)
        elif para_type == BlockType.TABLE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                para_text = merge_visual_blocks_to_markdown(para_block, img_buket_path)
        elif para_type == BlockType.CODE:
            para_text = merge_visual_blocks_to_markdown(para_block)

        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip())

    return page_markdown


def merge_visual_blocks_to_markdown(para_block, img_buket_path=''):
    # 将 image/table/chart/code 这类视觉块的子 block 按阅读顺序拼接成 markdown。
    # 这里不再写死 caption/body/footnote 的优先级，而是先展开成 segment，
    # 再根据 markdown_line / html_block 两类片段决定分隔方式。
    rendered_segments = []

    for block in get_blocks_in_index_order(para_block.get('blocks', [])):
        render_block = _inherit_parent_code_render_metadata(block, para_block)
        rendered_segments.extend(render_visual_block_segments(render_block, img_buket_path))

    para_text = ''
    prev_segment_kind = None
    for segment_text, segment_kind in rendered_segments:
        if para_text:
            para_text += get_visual_block_separator(prev_segment_kind, segment_kind)
        para_text += segment_text
        prev_segment_kind = segment_kind

    return para_text


def get_blocks_in_index_order(blocks):
    # 按 middle json 中的 index 排序子 block；
    # 如果 index 缺失，则退化为保持原始顺序。
    return [
        block
        for _, block in sorted(
            enumerate(blocks),
            key=lambda item: (item[1].get('index', float('inf')), item[0]),
        )
    ]


def _inherit_parent_code_render_metadata(block, parent_block):
    # pipeline_magic_model 会把 code_body 的 sub_type/guess_lang 提升到父 code block。
    # markdown 渲染 code_body 时需要把这两个字段临时透传回来，但不能修改原始输入。
    if block.get('type') != BlockType.CODE_BODY:
        return block
    if parent_block.get('type') != BlockType.CODE:
        return block

    needs_sub_type = 'sub_type' not in block and 'sub_type' in parent_block
    needs_guess_lang = 'guess_lang' not in block and 'guess_lang' in parent_block
    if not needs_sub_type and not needs_guess_lang:
        return block

    render_block = dict(block)
    if needs_sub_type:
        render_block['sub_type'] = parent_block['sub_type']
    if needs_guess_lang:
        render_block['guess_lang'] = parent_block['guess_lang']
    return render_block


def render_visual_block_segments(block, img_buket_path=''):
    # 将单个视觉子 block 渲染成一个或多个 segment。
    # 文本类子块统一输出 markdown_line；
    # table 的 html 输出为 html_block，供后续决定是否需要空行隔开。
    block_type = block['type']

    if block_type in [
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.CODE_BODY,
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
        return [
            (f"![]({img_buket_path}/{span['image_path']})", 'markdown_line')
            for line in block['lines']
            for span in line['spans']
            if span['type'] == ContentType.IMAGE and span.get('image_path', '')
        ]

    if block_type == BlockType.CHART_BODY:
        return [
            (f"![]({img_buket_path}/{span['image_path']})", 'markdown_line')
            for line in block['lines']
            for span in line['spans']
            if span['type'] == ContentType.CHART and span.get('image_path', '')
        ]

    if block_type == BlockType.TABLE_BODY:
        rendered_segments = []
        for line in block['lines']:
            for span in line['spans']:
                if span['type'] != ContentType.TABLE:
                    continue
                if span.get('html', ''):
                    rendered_segments.append((
                        _format_embedded_html(span['html'], img_buket_path),
                        'html_block',
                    ))
                elif span.get('image_path', ''):
                    rendered_segments.append((f"![]({img_buket_path}/{span['image_path']})", 'markdown_line'))
        return rendered_segments

    return []


def get_visual_block_separator(prev_segment_kind, current_segment_kind):
    # 根据前后 segment 类型决定分隔符：
    # 1. 普通 markdown 行之间用 hard break（"  \\n"）
    # 2. 进入 html block 前只换一行
    # 3. html block 后必须留空行，否则后续文本仍会被当作 html 块内容
    if prev_segment_kind == 'html_block':
        # Raw HTML blocks need a blank line after them, otherwise the following
        # markdown text is still treated as part of the HTML block.
        return '\n\n'
    if current_segment_kind == 'html_block':
        return '\n'
    return '  \n'


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

CJK_LANGS = {'zh', 'ja', 'ko'}


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


def merge_para_with_text(para_block):
    if _is_fenced_code_block(para_block):
        code_text = _merge_para_text(
            para_block,
            escape_markdown=False,
            list_line_break='\n',
        )
        if not code_text:
            return ''
        code_text = '\n'.join(line.rstrip() for line in code_text.split('\n'))
        guess_lang = para_block.get('guess_lang', 'txt') or 'txt'
        return f"```{guess_lang}\n{code_text}\n```"

    para_text = _merge_para_text(para_block)
    if para_block.get('type') == BlockType.TEXT:
        para_text = escape_text_block_markdown_prefix(para_text)
    return para_text


def _merge_para_text(para_block, escape_markdown=True, list_line_break='  \n'):
    # 将普通文本段落 block 渲染成 markdown 字符串。
    # 处理流程分为三层：
    # 1. 先收集文本内容做语言检测
    # 2. 再把每个 span 渲染成 markdown 片段
    # 3. 最后按语言和上下文决定 span 之间如何补空格/换行
    block_lang = detect_lang(_collect_text_for_lang_detection(para_block))
    para_parts = []

    for line_idx, line in enumerate(para_block['lines']):
        line_prefix = _line_prefix(line_idx, line, list_line_break)
        if line_prefix:
            para_parts.append(line_prefix)

        for span_idx, span in enumerate(line['spans']):
            rendered_span = _render_span(span, escape_markdown=escape_markdown)
            if rendered_span is None:
                continue

            span_type, content = rendered_span
            content, suffix = _join_rendered_span(
                para_block,
                block_lang,
                line,
                line_idx,
                span_idx,
                span_type,
                content,
            )
            para_parts.append(content)
            if suffix:
                para_parts.append(suffix)

    return ''.join(para_parts).rstrip()


def _is_fenced_code_block(para_block):
    return (
        para_block.get('type') == BlockType.CODE_BODY
        and para_block.get('sub_type') == BlockType.CODE
    )


def _collect_text_for_lang_detection(para_block):
    # 只收集 TEXT span 的内容，用于语言检测。
    # 这里会先做全角转半角，但不会修改原始输入数据。
    block_text_parts = []
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] == ContentType.TEXT:
                block_text_parts.append(_normalize_text_content(span.get('content', '')))
    return ''.join(block_text_parts)


def _normalize_text_content(content):
    # 对原始文本做统一归一化，当前只负责全角转半角。
    # 单独拆出来是为了让语言检测和渲染阶段复用同一套文本预处理。
    return full_to_half_exclude_marks(content or '')


def _render_span(span, escape_markdown=True):
    # 将单个 span 渲染成 markdown 片段。
    # 这里只负责“渲染成什么文本”，不决定后面是否补空格。
    span_type = span['type']
    content = ''

    if span_type == ContentType.TEXT:
        content = _normalize_text_content(span.get('content', ''))
        if escape_markdown:
            content = escape_special_markdown_char(content)
    elif span_type == ContentType.INLINE_EQUATION:
        if span.get('content', ''):
            content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
    elif span_type == ContentType.INTERLINE_EQUATION:
        if span.get('content', ''):
            content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
    else:
        return None

    content = content.strip()
    if not content:
        return None

    return span_type, content


def _join_rendered_span(para_block, block_lang, line, line_idx, span_idx, span_type, content):
    # 根据语言和上下文决定当前 span 后面的分隔符。
    # 这里集中处理：
    # 1. CJK 与西文的空格差异
    # 2. 西文行尾连字符是否需要跨行合并
    # 3. 独立公式是否作为块内容直接插入
    if span_type == ContentType.INTERLINE_EQUATION:
        return content, ''

    is_last_span = span_idx == len(line['spans']) - 1

    if block_lang in CJK_LANGS:
        if is_last_span and span_type != ContentType.INLINE_EQUATION:
            return content, ''
        return content, ' '

    if span_type not in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
        return content, ''

    if (
        is_last_span
        and span_type == ContentType.TEXT
        and is_hyphen_at_line_end(content)
    ):
        if _next_line_starts_with_lowercase_text(para_block, line_idx):
            return content[:-1], ''
        return content, ''

    return content, ' '


def _line_prefix(line_idx, line, list_line_break='  \n'):
    # 处理进入新 list item 前的 block 级换行。
    # 这里保留历史语义：list 起始行前插入一个 hard break。
    if line_idx >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
        return list_line_break
    return ''


def _next_line_starts_with_lowercase_text(para_block, line_idx):
    # 判断下一行是否以小写英文文本开头。
    # 这个条件用于决定西文行尾的连字符是否应与下一行合并。
    if line_idx + 1 >= len(para_block['lines']):
        return False

    next_line_spans = para_block['lines'][line_idx + 1].get('spans')
    if not next_line_spans:
        return False

    next_span = next_line_spans[0]
    if next_span.get('type') != ContentType.TEXT:
        return False

    next_content = _normalize_text_content(next_span.get('content', ''))
    return bool(next_content) and next_content[0].islower()


def merge_adjacent_ref_text_blocks_for_content(para_blocks):
    merged_blocks = []
    ref_group = []

    def flush_ref_group():
        nonlocal ref_group
        if not ref_group:
            return
        if len(ref_group) == 1:
            merged_blocks.append(ref_group[0])
        else:
            merged_blocks.append({
                'type': BlockType.REF_TEXT,
                'blocks': list(ref_group),
                'bbox': ref_group[0].get('bbox'),
            })
        ref_group = []

    for para_block in para_blocks or []:
        if para_block.get('type') == BlockType.REF_TEXT:
            ref_group.append(para_block)
            continue

        flush_ref_group()
        merged_blocks.append(para_block)

    flush_ref_group()
    return merged_blocks


def _build_bbox(para_bbox, page_size):
    if not para_bbox or not page_size:
        return None

    page_width, page_height = page_size
    x0, y0, x1, y1 = para_bbox
    return [
        int(x0 * 1000 / page_width),
        int(y0 * 1000 / page_height),
        int(x1 * 1000 / page_width),
        int(y1 * 1000 / page_height),
    ]


def _get_seal_span(para_block):
    for line in para_block.get('lines', []):
        for span in line.get('spans', []):
            if span.get('type') == ContentType.SEAL:
                return span
    return None


def _get_seal_text(para_block):
    seal_span = _get_seal_span(para_block)
    if not seal_span:
        return ''

    content = seal_span.get('content', '')
    if isinstance(content, list):
        return ' '.join(str(item) for item in content if str(item).strip())
    if isinstance(content, str):
        return content.strip()
    return ''


def _get_ref_text_item_blocks(para_block):
    return para_block.get('blocks') or [para_block]


def _split_list_item_blocks(para_block):
    item_blocks = []
    current_lines = []

    for line_idx, line in enumerate(para_block.get('lines', [])):
        if line_idx > 0 and line.get(ListLineTag.IS_LIST_START_LINE, False) and current_lines:
            item_blocks.append({
                'type': BlockType.TEXT,
                'lines': current_lines,
            })
            current_lines = []
        current_lines.append(line)

    if current_lines:
        item_blocks.append({
            'type': BlockType.TEXT,
            'lines': current_lines,
        })

    return item_blocks


def _get_body_data(para_block):
    def get_data_from_spans(lines):
        for line in lines:
            for span in line.get('spans', []):
                span_type = span.get('type')
                if span_type == ContentType.TABLE:
                    return span.get('image_path', ''), span.get('html', '')
                if span_type == ContentType.CHART:
                    return span.get('image_path', ''), span.get('content', '')
                if span_type == ContentType.IMAGE:
                    return span.get('image_path', ''), ''
                if span_type == ContentType.INTERLINE_EQUATION:
                    return span.get('image_path', ''), span.get('content', '')
        return '', ''

    if 'blocks' in para_block:
        for block in para_block['blocks']:
            block_type = block.get('type')
            if block_type in [
                BlockType.IMAGE_BODY,
                BlockType.TABLE_BODY,
                BlockType.CHART_BODY,
                BlockType.CODE_BODY,
            ]:
                result = get_data_from_spans(block.get('lines', []))
                if result != ('', '') or block_type == BlockType.CHART_BODY:
                    return result
        return '', ''

    return get_data_from_spans(para_block.get('lines', []))


def merge_para_with_text_v2(para_block):
    block_lang = detect_lang(_collect_text_for_lang_detection(para_block))
    para_content = []
    para_type = para_block.get('type')

    for line_idx, line in enumerate(para_block.get('lines', [])):
        for span_idx, span in enumerate(line.get('spans', [])):
            span_type = span.get('type')

            if span_type == ContentType.TEXT:
                content = _normalize_text_content(span.get('content', ''))
                if not content.strip():
                    continue

                output_type = (
                    ContentTypeV2.SPAN_PHONETIC
                    if para_type == BlockType.PHONETIC
                    else ContentTypeV2.SPAN_TEXT
                )
                is_last_span = span_idx == len(line['spans']) - 1

                if block_lang in CJK_LANGS:
                    rendered_content = content if is_last_span else f"{content} "
                else:
                    if (
                        is_last_span
                        and is_hyphen_at_line_end(content)
                        and _next_line_starts_with_lowercase_text(para_block, line_idx)
                    ):
                        rendered_content = content[:-1]
                    elif is_last_span and is_hyphen_at_line_end(content):
                        rendered_content = content
                    else:
                        rendered_content = f"{content} "

                if para_content and para_content[-1]['type'] == output_type:
                    para_content[-1]['content'] += rendered_content
                else:
                    para_content.append({
                        'type': output_type,
                        'content': rendered_content,
                    })
            elif span_type == ContentType.INLINE_EQUATION:
                content = span.get('content', '').strip()
                if content:
                    para_content.append({
                        'type': ContentTypeV2.SPAN_EQUATION_INLINE,
                        'content': content,
                    })

    if para_content and para_content[-1]['type'] in [
        ContentTypeV2.SPAN_TEXT,
        ContentTypeV2.SPAN_PHONETIC,
    ]:
        para_content[-1]['content'] = para_content[-1]['content'].rstrip()

    return para_content


def make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):
    para_type = para_block['type']
    para_content = None
    if para_type in [
        BlockType.TEXT,
        BlockType.INDEX,
        BlockType.LIST,
        BlockType.ABSTRACT,
    ]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
    elif para_type in [
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
    elif para_type == BlockType.REF_TEXT:
        para_content = {
            'type': BlockType.LIST,
            'sub_type': BlockType.REF_TEXT,
            'list_items': [],
        }
        for block in _get_ref_text_item_blocks(para_block):
            item_text = merge_para_with_text(block)
            if item_text.strip():
                para_content['list_items'].append(item_text)
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
            return None
        para_content = {
            'type': ContentType.EQUATION,
            'img_path': f"{img_buket_path}/{para_block['lines'][0]['spans'][0].get('image_path', '')}",
        }
        if para_block['lines'][0]['spans'][0].get('content', ''):
            para_content['text'] = merge_para_with_text(para_block)
            para_content['text_format'] = 'latex'
    elif para_type == BlockType.SEAL:
        seal_span = _get_seal_span(para_block)
        if not seal_span:
            return None
        para_content = {
            'type': ContentType.SEAL,
            'img_path': f"{img_buket_path}/{seal_span.get('image_path', '')}",
            'text': _get_seal_text(para_block),
        }
    elif para_type == BlockType.IMAGE:
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
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
        para_content = {
            'type': ContentType.CHART,
            'img_path': '',
            'content': '',
            BlockType.CHART_CAPTION: [],
            BlockType.CHART_FOOTNOTE: [],
        }
        for block in para_block.get('blocks', []):
            if block['type'] == BlockType.CHART_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.CHART and span.get('image_path', ''):
                            para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.CHART_CAPTION:
                para_content[BlockType.CHART_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.CHART_FOOTNOTE:
                para_content[BlockType.CHART_FOOTNOTE].append(merge_para_with_text(block))
    elif para_type == BlockType.CODE:
        para_content = {
            'type': BlockType.CODE,
            'sub_type': para_block['sub_type'],
            BlockType.CODE_CAPTION: [],
            BlockType.CODE_FOOTNOTE: [],
        }
        for block in para_block.get('blocks', []):
            render_block = _inherit_parent_code_render_metadata(block, para_block)
            if block['type'] == BlockType.CODE_BODY:
                para_content[BlockType.CODE_BODY] = merge_para_with_text(render_block)
            if block['type'] == BlockType.CODE_CAPTION:
                para_content[BlockType.CODE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.CODE_FOOTNOTE:
                para_content[BlockType.CODE_FOOTNOTE].append(merge_para_with_text(block))

    if not para_content:
        return None

    bbox = _build_bbox(para_block.get('bbox'), page_size)
    if bbox:
        para_content['bbox'] = bbox
    para_content['page_idx'] = page_idx

    return para_content


def make_blocks_to_content_list_v2(para_block, img_buket_path, page_size):
    para_type = para_block['type']
    para_content = None

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
            },
        }
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content = {
                'type': ContentTypeV2.TITLE,
                'content': {
                    'title_content': merge_para_with_text_v2(para_block),
                    'level': title_level,
                },
            }
        else:
            para_content = {
                'type': ContentTypeV2.PARAGRAPH,
                'content': {
                    'paragraph_content': merge_para_with_text_v2(para_block),
                },
            }
    elif para_type in [
        BlockType.TEXT,
        BlockType.ABSTRACT,
    ]:
        para_content = {
            'type': ContentTypeV2.PARAGRAPH,
            'content': {
                'paragraph_content': merge_para_with_text_v2(para_block),
            },
        }
    elif para_type == BlockType.INTERLINE_EQUATION:
        image_path, math_content = _get_body_data(para_block)
        para_content = {
            'type': ContentTypeV2.EQUATION_INTERLINE,
            'content': {
                'math_content': math_content,
                'math_type': 'latex',
                'image_source': {'path': f"{img_buket_path}/{image_path}"},
            },
        }
    elif para_type == BlockType.IMAGE:
        image_caption = []
        image_footnote = []
        image_path, _ = _get_body_data(para_block)
        for block in para_block.get('blocks', []):
            if block['type'] == BlockType.IMAGE_CAPTION:
                image_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                image_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.IMAGE,
            'content': {
                'image_source': {'path': f"{img_buket_path}/{image_path}"},
                'image_caption': image_caption,
                'image_footnote': image_footnote,
            },
        }
    elif para_type == BlockType.TABLE:
        table_caption = []
        table_footnote = []
        image_path, html = _get_body_data(para_block)
        table_html = _format_embedded_html(html, img_buket_path)
        table_nest_level = 2 if table_html.count('<table') > 1 else 1
        if 'colspan' in table_html or 'rowspan' in table_html or table_nest_level > 1:
            table_type = ContentTypeV2.TABLE_COMPLEX
        else:
            table_type = ContentTypeV2.TABLE_SIMPLE
        for block in para_block.get('blocks', []):
            if block['type'] == BlockType.TABLE_CAPTION:
                table_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                table_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.TABLE,
            'content': {
                'image_source': {'path': f"{img_buket_path}/{image_path}"},
                'table_caption': table_caption,
                'table_footnote': table_footnote,
                'html': table_html,
                'table_type': table_type,
                'table_nest_level': table_nest_level,
            },
        }
    elif para_type == BlockType.CHART:
        chart_caption = []
        chart_footnote = []
        image_path, _ = _get_body_data(para_block)
        for block in para_block.get('blocks', []):
            if block['type'] == BlockType.CHART_CAPTION:
                chart_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.CHART_FOOTNOTE:
                chart_footnote.extend(merge_para_with_text_v2(block))
        para_content = {
            'type': ContentTypeV2.CHART,
            'content': {
                'image_source': {'path': f"{img_buket_path}/{image_path}"},
                'content': '',
                'chart_caption': chart_caption,
                'chart_footnote': chart_footnote,
            },
        }
    elif para_type == BlockType.CODE:
        code_caption = []
        code_footnote = []
        code_content = []
        for block in para_block.get('blocks', []):
            if block['type'] == BlockType.CODE_CAPTION:
                code_caption.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.CODE_FOOTNOTE:
                code_footnote.extend(merge_para_with_text_v2(block))
            if block['type'] == BlockType.CODE_BODY:
                code_content = merge_para_with_text_v2(block)

        sub_type = para_block['sub_type']
        if sub_type == BlockType.CODE:
            para_content = {
                'type': ContentTypeV2.CODE,
                'content': {
                    'code_caption': code_caption,
                    'code_content': code_content,
                    'code_footnote': code_footnote,
                    'code_language': para_block.get('guess_lang', 'txt'),
                },
            }
        elif sub_type == BlockType.ALGORITHM:
            para_content = {
                'type': ContentTypeV2.ALGORITHM,
                'content': {
                    'algorithm_caption': code_caption,
                    'algorithm_content': code_content,
                    'algorithm_footnote': code_footnote,
                },
            }
        else:
            raise ValueError(f"Unknown code sub_type: {sub_type}")
    elif para_type == BlockType.REF_TEXT:
        list_items = []
        for block in _get_ref_text_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append({
                    'item_type': 'text',
                    'item_content': item_content,
                })
        para_content = {
            'type': ContentTypeV2.LIST,
            'content': {
                'list_type': ContentTypeV2.LIST_REF,
                'list_items': list_items,
            },
        }
    elif para_type == BlockType.LIST:
        list_items = []
        for block in _split_list_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append({
                    'item_type': 'text',
                    'item_content': item_content,
                })
        para_content = {
            'type': ContentTypeV2.LIST,
            'content': {
                'list_type': ContentTypeV2.LIST_TEXT,
                'attribute': para_block.get('attribute', 'unordered'),
                'list_items': list_items,
            },
        }
    elif para_type == BlockType.INDEX:
        list_items = []
        for block in _split_list_item_blocks(para_block):
            item_content = merge_para_with_text_v2(block)
            if item_content:
                list_items.append({
                    'item_type': 'text',
                    'item_content': item_content,
                })
        para_content = {
            'type': ContentTypeV2.INDEX,
            'content': {
                'list_type': ContentTypeV2.LIST_TEXT,
                'list_items': list_items,
            },
        }
    elif para_type == BlockType.SEAL:
        seal_span = _get_seal_span(para_block)
        if not seal_span:
            return None
        seal_text = _get_seal_text(para_block)
        para_content = {
            'type': ContentTypeV2.SEAL,
            'content': {
                'image_source': {
                    'path': f"{img_buket_path}/{seal_span.get('image_path', '')}",
                },
                'seal_content': (
                    [{'type': ContentTypeV2.SPAN_TEXT, 'content': seal_text}]
                    if seal_text else []
                ),
            },
        }

    if not para_content:
        return None

    bbox = _build_bbox(para_block.get('bbox'), page_size)
    if bbox:
        para_content['bbox'] = bbox

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
        page_size = page_info.get('page_size')
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            if not paras_of_layout:
                continue
            page_markdown = make_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            para_blocks = merge_adjacent_ref_text_blocks_for_content(
                (paras_of_layout or []) + (paras_of_discarded or [])
            )
            if not para_blocks:
                continue
            for para_block in para_blocks:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size)
                if para_content:
                    output_content.append(para_content)
        elif make_mode == MakeMode.CONTENT_LIST_V2:
            para_blocks = merge_adjacent_ref_text_blocks_for_content(
                (paras_of_layout or []) + (paras_of_discarded or [])
            )
            page_contents = []
            if para_blocks:
                for para_block in para_blocks:
                    para_content = make_blocks_to_content_list_v2(para_block, img_buket_path, page_size)
                    if para_content:
                        page_contents.append(para_content)
            output_content.append(page_contents)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode in [MakeMode.CONTENT_LIST, MakeMode.CONTENT_LIST_V2]:
        return output_content
    else:
        logger.error(f"Unsupported make mode: {make_mode}")
        return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level < 1:
        title_level = 0
    return title_level


def escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    return escape_conservative_markdown_text(content)
