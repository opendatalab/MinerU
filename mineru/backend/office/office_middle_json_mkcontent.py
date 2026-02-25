import os

from loguru import logger

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


def get_title_level(para_block):
    title_level = para_block.get('level', 2)
    return title_level


def merge_para_with_text(para_block):
    # First pass: collect all non-empty (span_type, content) parts
    parts = []
    if para_block['type'] == BlockType.TITLE:
        if para_block.get('is_numbered_style', False):
            section_number = para_block.get('section_number', '')
            if section_number:
                parts.append((ContentType.TEXT, f"{section_number} "))

    for line in para_block['lines']:
        for span in line['spans']:
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content']
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
            elif span_type == ContentType.HYPERLINK:
                content = f"[{span['content']}]({span.get('url', '')})"

            content = content.strip()
            if content:
                parts.append((span_type, content))

    # Second pass: join parts, keeping one space on each side of inline equations
    para_text = ''
    for i, (span_type, content) in enumerate(parts):
        is_last = i == len(parts) - 1
        if span_type == ContentType.INLINE_EQUATION:
            # Ensure one space before the equation (if there is preceding text)
            if para_text and not para_text.endswith(' '):
                para_text += ' '
            para_text += content
            # Ensure one space after the equation, unless it is the last part
            if not is_last:
                para_text += ' '
        else:
            para_text += content

    return para_text


def _flatten_list_items(list_block):
    """Recursively flatten nested list blocks into a list of prefixed item strings."""
    items = []
    ilevel = list_block.get('ilevel', 0)
    attribute = list_block.get('attribute', 'unordered')
    indent = '    ' * ilevel
    ordered_counter = 1

    for block in list_block.get('blocks', []):
        if block['type'] == BlockType.LIST:
            items.extend(_flatten_list_items(block))
        else:
            item_text = merge_para_with_text(block)
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
        if block['type'] == BlockType.LIST:
            items.extend(_flatten_list_items_v2(block))
        else:
            item_content = merge_para_with_text_v2(block)
            if item_content:
                if attribute == 'ordered':
                    prefix = f"{'    ' * ilevel}{ordered_counter}."
                    ordered_counter += 1
                else:
                    prefix = f"{'    ' * ilevel}-"
                items.append({
                    'item_type': 'text',
                    'ilevel': ilevel,
                    'prefix': prefix,
                    'item_content': item_content,
                })

    return items


def merge_list_to_markdown(list_block):
    """Recursively convert a nested list block to markdown text."""
    return '\n'.join(_flatten_list_items(list_block)) + '\n'


def mk_blocks_to_markdown(para_blocks, make_mode, img_buket_path=''):
    page_markdown = []
    for para_block in para_blocks:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.LIST:
            para_text = merge_list_to_markdown(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
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
                                    para_text += f"\n{span['html']}\n"
                for block in para_block['blocks']:  # 2nd.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += '  \n' + merge_para_with_text(block)
        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip())

    return page_markdown


def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.HEADER,
        BlockType.FOOTER,
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
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"
            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))

    para_content['page_idx'] = page_idx

    return para_content


def make_blocks_to_content_list_v2(para_block, img_buket_path):
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.HEADER,
        BlockType.FOOTER,
    ]:
        if para_type == BlockType.HEADER:
            content_type = ContentTypeV2.PAGE_HEADER
        elif para_type == BlockType.FOOTER:
            content_type = ContentTypeV2.PAGE_FOOTER
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
                'html': html,
                'table_type': table_type,
                'table_nest_level': table_nest_level,
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

    return para_content


def get_body_data(para_block):
    """
    Extract image_path and html from para_block
    Returns:
        - For IMAGE/INTERLINE_EQUATION: (image_path, '')
        - For TABLE: (image_path, html)
        - Default: ('', '')
    """

    def get_data_from_spans(lines):
        for line in lines:
            for span in line.get('spans', []):
                span_type = span.get('type')
                if span_type == ContentType.TABLE:
                    return span.get('image_path', ''), span.get('html', '')
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
            if block_type in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.CODE_BODY]:
                result = get_data_from_spans(block.get('lines', []))
                if result != ('', ''):
                    return result
        return '', ''

    # 处理直接包含 lines 的结构
    return get_data_from_spans(para_block.get('lines', []))


def merge_para_with_text_v2(para_block):
    para_content = []
    for i, line in enumerate(para_block['lines']):
        for j, span in enumerate(line['spans']):
            if span.get("content", '').strip():
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
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
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

