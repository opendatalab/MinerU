import re

import wordninja
from loguru import logger

from magic_pdf.libs.commons import join_path
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char
from magic_pdf.libs.ocr_content_type import BlockType, ContentType


def __is_hyphen_at_line_end(line):
    """
    Check if a line ends with one or more letters followed by a hyphen.
    
    Args:
    line (str): The line of text to check.
    
    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


def split_long_words(text):
    segments = text.split(' ')
    for i in range(len(segments)):
        words = re.findall(r'\w+|[^\w]', segments[i], re.UNICODE)
        for j in range(len(words)):
            if len(words[j]) > 10:
                words[j] = ' '.join(wordninja.split(words[j]))
        segments[i] = ''.join(words)
    return ' '.join(segments)


def ocr_mk_mm_markdown_with_para(pdf_info_list: list, img_buket_path):
    markdown = []
    for page_info in pdf_info_list:
        paras_of_layout = page_info.get('para_blocks')
        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'mm', img_buket_path)
        markdown.extend(page_markdown)
    return '\n\n'.join(markdown)


def ocr_mk_nlp_markdown_with_para(pdf_info_dict: list):
    markdown = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'nlp')
        markdown.extend(page_markdown)
    return '\n\n'.join(markdown)


def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: list,
                                                img_buket_path):
    markdown_with_para_and_pagination = []
    page_no = 0
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        if not paras_of_layout:
            continue
        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'mm', img_buket_path)
        markdown_with_para_and_pagination.append({
            'page_no':
            page_no,
            'md_content':
            '\n\n'.join(page_markdown)
        })
        page_no += 1
    return markdown_with_para_and_pagination


def ocr_mk_markdown_with_para_core(paras_of_layout, mode, img_buket_path=''):
    page_markdown = []
    for paras in paras_of_layout:
        for para in paras:
            para_text = ''
            for line in para:
                for span in line['spans']:
                    span_type = span.get('type')
                    content = ''
                    language = ''
                    if span_type == ContentType.Text:
                        content = span['content']
                        language = detect_lang(content)
                        if (language == 'en'):  # 只对英文长词进行分词处理，中文分词会丢失文本
                            content = ocr_escape_special_markdown_char(
                                split_long_words(content))
                        else:
                            content = ocr_escape_special_markdown_char(content)
                    elif span_type == ContentType.InlineEquation:
                        content = f"${span['content']}$"
                    elif span_type == ContentType.InterlineEquation:
                        content = f"\n$$\n{span['content']}\n$$\n"
                    elif span_type in [ContentType.Image, ContentType.Table]:
                        if mode == 'mm':
                            content = f"\n![]({join_path(img_buket_path, span['image_path'])})\n"
                        elif mode == 'nlp':
                            pass
                    if content != '':
                        if language == 'en':  # 英文语境下 content间需要空格分隔
                            para_text += content + ' '
                        else:  # 中文语境下，content间不需要空格分隔
                            para_text += content
            if para_text.strip() == '':
                continue
            else:
                page_markdown.append(para_text.strip() + '  ')
    return page_markdown


def ocr_mk_markdown_with_para_core_v2(paras_of_layout,
                                      mode,
                                      img_buket_path=''):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type == BlockType.Text:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Title:
            para_text = f'# {merge_para_with_text(para_block)}'
        elif para_type == BlockType.InterlineEquation:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Image:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼image_body
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 2nd.拼image_caption
                    if block['type'] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block)
                for block in para_block['blocks']:  # 2nd.拼image_caption
                    if block['type'] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block)
        elif para_type == BlockType.Table:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block)
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TableBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Table:
                                    # if processed by table model
                                    if span.get('latex', ''):
                                        para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                    elif span.get('html', ''):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    else:
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block)

        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip() + '  ')

    return page_markdown


def merge_para_with_text(para_block):

    def detect_language(text):
        en_pattern = r'[a-zA-Z]+'
        en_matches = re.findall(en_pattern, text)
        en_length = sum(len(match) for match in en_matches)
        if len(text) > 0:
            if en_length / len(text) >= 0.5:
                return 'en'
            else:
                return 'unknown'
        else:
            return 'empty'

    para_text = ''
    for line in para_block['lines']:
        line_text = ''
        line_lang = ''
        for span in line['spans']:
            span_type = span['type']
            if span_type == ContentType.Text:
                line_text += span['content'].strip()
        if line_text != '':
            line_lang = detect_lang(line_text)
        for span in line['spans']:
            span_type = span['type']
            content = ''
            if span_type == ContentType.Text:
                content = span['content']
                # language = detect_lang(content)
                language = detect_language(content)
                if language == 'en':  # 只对英文长词进行分词处理，中文分词会丢失文本
                    content = ocr_escape_special_markdown_char(
                        split_long_words(content))
                else:
                    content = ocr_escape_special_markdown_char(content)
            elif span_type == ContentType.InlineEquation:
                content = f" ${span['content']}$ "
            elif span_type == ContentType.InterlineEquation:
                content = f"\n$$\n{span['content']}\n$$\n"

            if content != '':
                langs = ['zh', 'ja', 'ko']
                if line_lang in langs:  # 遇到一些一个字一个span的文档，这种单字语言判断不准，需要用整行文本判断
                    para_text += content  # 中文/日语/韩文语境下，content间不需要空格分隔
                elif line_lang == 'en':
                    # 如果是前一行带有-连字符，那么末尾不应该加空格
                    if __is_hyphen_at_line_end(content):
                        para_text += content[:-1]
                    else:
                        para_text += content + ' '
                else:
                    para_text += content + ' '  # 西方文本语境下 content间需要空格分隔
    return para_text


def para_to_standard_format(para, img_buket_path):
    para_content = {}
    if len(para) == 1:
        para_content = line_to_standard_format(para[0], img_buket_path)
    elif len(para) > 1:
        para_text = ''
        inline_equation_num = 0
        for line in para:
            for span in line['spans']:
                language = ''
                span_type = span.get('type')
                content = ''
                if span_type == ContentType.Text:
                    content = span['content']
                    language = detect_lang(content)
                    if language == 'en':  # 只对英文长词进行分词处理，中文分词会丢失文本
                        content = ocr_escape_special_markdown_char(
                            split_long_words(content))
                    else:
                        content = ocr_escape_special_markdown_char(content)
                elif span_type == ContentType.InlineEquation:
                    content = f"${span['content']}$"
                    inline_equation_num += 1
                if language == 'en':  # 英文语境下 content间需要空格分隔
                    para_text += content + ' '
                else:  # 中文语境下，content间不需要空格分隔
                    para_text += content
        para_content = {
            'type': 'text',
            'text': para_text,
            'inline_equation_num': inline_equation_num,
        }
    return para_content


def para_to_standard_format_v2(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    if para_type == BlockType.Text:
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'page_idx': page_idx,
        }
    elif para_type == BlockType.Title:
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'text_level': 1,
            'page_idx': page_idx,
        }
    elif para_type == BlockType.InterlineEquation:
        para_content = {
            'type': 'equation',
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
            'page_idx': page_idx,
        }
    elif para_type == BlockType.Image:
        para_content = {'type': 'image', 'page_idx': page_idx}
        for block in para_block['blocks']:
            if block['type'] == BlockType.ImageBody:
                para_content['img_path'] = join_path(
                    img_buket_path,
                    block['lines'][0]['spans'][0]['image_path'])
            if block['type'] == BlockType.ImageCaption:
                para_content['img_caption'] = merge_para_with_text(block)
            if block['type'] == BlockType.ImageFootnote:
                para_content['img_footnote'] = merge_para_with_text(block)
    elif para_type == BlockType.Table:
        para_content = {'type': 'table', 'page_idx': page_idx}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TableBody:
                if block["lines"][0]["spans"][0].get('latex', ''):
                    para_content['table_body'] = f"\n\n$\n {block['lines'][0]['spans'][0]['latex']}\n$\n\n"
                elif block["lines"][0]["spans"][0].get('html', ''):
                    para_content['table_body'] = f"\n\n{block['lines'][0]['spans'][0]['html']}\n\n"
                para_content['img_path'] = join_path(img_buket_path, block["lines"][0]["spans"][0]['image_path'])
            if block['type'] == BlockType.TableCaption:
                para_content['table_caption'] = merge_para_with_text(block)
            if block['type'] == BlockType.TableFootnote:
                para_content['table_footnote'] = merge_para_with_text(block)

    return para_content


def make_standard_format_with_para(pdf_info_dict: list, img_buket_path: str):
    content_list = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        if not paras_of_layout:
            continue
        for para_block in paras_of_layout:
            para_content = para_to_standard_format_v2(para_block,
                                                      img_buket_path)
            content_list.append(para_content)
    return content_list


def line_to_standard_format(line, img_buket_path):
    line_text = ''
    inline_equation_num = 0
    for span in line['spans']:
        if not span.get('content'):
            if not span.get('image_path'):
                continue
            else:
                if span['type'] == ContentType.Image:
                    content = {
                        'type': 'image',
                        'img_path': join_path(img_buket_path,
                                              span['image_path']),
                    }
                    return content
                elif span['type'] == ContentType.Table:
                    content = {
                        'type': 'table',
                        'img_path': join_path(img_buket_path,
                                              span['image_path']),
                    }
                    return content
        else:
            if span['type'] == ContentType.InterlineEquation:
                interline_equation = span['content']
                content = {
                    'type': 'equation',
                    'latex': f'$$\n{interline_equation}\n$$'
                }
                return content
            elif span['type'] == ContentType.InlineEquation:
                inline_equation = span['content']
                line_text += f'${inline_equation}$'
                inline_equation_num += 1
            elif span['type'] == ContentType.Text:
                text_content = ocr_escape_special_markdown_char(
                    span['content'])  # 转义特殊符号
                line_text += text_content
    content = {
        'type': 'text',
        'text': line_text,
        'inline_equation_num': inline_equation_num,
    }
    return content


def ocr_mk_mm_standard_format(pdf_info_dict: list):
    """content_list type         string
    image/text/table/equation(行间的单独拿出来，行内的和text合并) latex        string
    latex文本字段。 text         string      纯文本格式的文本数据。 md           string
    markdown格式的文本数据。 img_path     string      s3://full/path/to/img.jpg."""
    content_list = []
    for page_info in pdf_info_dict:
        blocks = page_info.get('preproc_blocks')
        if not blocks:
            continue
        for block in blocks:
            for line in block['lines']:
                content = line_to_standard_format(line)
                content_list.append(content)
    return content_list


def union_make(pdf_info_dict: list,
               make_mode: str,
               drop_mode: str,
               img_buket_path: str = ''):
    output_content = []
    for page_info in pdf_info_dict:
        if page_info.get('need_drop', False):
            drop_reason = page_info.get('drop_reason')
            if drop_mode == DropMode.NONE:
                pass
            elif drop_mode == DropMode.WHOLE_PDF:
                raise Exception((f'drop_mode is {DropMode.WHOLE_PDF} ,'
                                 f'drop_reason is {drop_reason}'))
            elif drop_mode == DropMode.SINGLE_PAGE:
                logger.warning((f'drop_mode is {DropMode.SINGLE_PAGE} ,'
                                f'drop_reason is {drop_reason}'))
                continue
            else:
                raise Exception('drop_mode can not be null')

        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx')
        if not paras_of_layout:
            continue
        if make_mode == MakeMode.MM_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'mm', img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.NLP_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'nlp')
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.STANDARD_FORMAT:
            for para_block in paras_of_layout:
                para_content = para_to_standard_format_v2(
                    para_block, img_buket_path, page_idx)
                output_content.append(para_content)
    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.STANDARD_FORMAT:
        return output_content
