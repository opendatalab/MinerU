from loguru import logger

from mineru.utils.char_utils import full_to_half_exclude_marks, is_hyphen_at_line_end
from mineru.utils.config_reader import get_latex_delimiter_config
from mineru.backend.pipeline.para_split import ListLineTag
from mineru.utils.enum_class import BlockType, ContentType, MakeMode
from mineru.utils.language import detect_lang


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
                    rendered_segments.append((span['html'], 'html_block'))
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

    return _merge_para_text(para_block)


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


def make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.INDEX,
    ]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.DISCARDED:
        para_content = {
            'type': para_type,
            'text': merge_para_with_text(para_block),
        }
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
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))

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
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            if not para_blocks:
                continue
            for para_block in para_blocks:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size)
                if para_content:
                    output_content.append(para_content)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content
    else:
        logger.error(f"Unsupported make mode: {make_mode}")
        return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level


def escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content
