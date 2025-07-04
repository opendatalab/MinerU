import os

from mineru.utils.config_reader import get_latex_delimiter_config, get_formula_enable, get_table_enable
from mineru.utils.enum_class import MakeMode, BlockType, ContentType


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

def merge_para_with_text(para_block, formula_enable=True, img_buket_path=''):
    para_text = ''
    for line in para_block['lines']:
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content']
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                if formula_enable:
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                else:
                    if span.get('image_path', ''):
                        content = f"![]({img_buket_path}/{span['image_path']})"
            # content = content.strip()
            if content:
                if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                    if j == len(line['spans']) - 1:
                        para_text += content
                    else:
                        para_text += f'{content} '
                elif span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
    return para_text

def mk_blocks_to_markdown(para_blocks, make_mode, formula_enable, table_enable, img_buket_path=''):
    page_markdown = []
    for para_block in para_blocks:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX, BlockType.INTERLINE_EQUATION]:
            para_text = merge_para_with_text(para_block, formula_enable=formula_enable, img_buket_path=img_buket_path)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.IMAGE:
            if make_mode == MakeMode.NLP_MD:
                continue
            elif make_mode == MakeMode.MM_MD:
                # 检测是否存在图片脚注
                has_image_footnote = any(block['type'] == BlockType.IMAGE_FOOTNOTE for block in para_block['blocks'])
                # 如果存在图片脚注，则将图片脚注拼接到图片正文后面
                if has_image_footnote:
                    for block in para_block['blocks']:  # 1st.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + '  \n'
                    for block in para_block['blocks']:  # 2nd.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 3rd.拼image_footnote
                        if block['type'] == BlockType.IMAGE_FOOTNOTE:
                            para_text += '  \n' + merge_para_with_text(block)
                else:
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
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TABLE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.TABLE:
                                    # if processed by table model
                                    if table_enable:
                                        if span.get('html', ''):
                                            para_text += f"\n{span['html']}\n"
                                        elif span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                                    else:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TABLE_FOOTNOTE:
                        para_text += '\n' + merge_para_with_text(block) + '  '

        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip() + '  ')
            page_markdown.append(para_text.strip())

    return page_markdown





def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
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

    para_content['page_idx'] = page_idx

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
        page_idx = page_info.get('page_idx')
        if not paras_of_layout:
            continue
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, formula_enable, table_enable, img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            for para_block in paras_of_layout:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx)
                output_content.append(para_content)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content
    return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level
