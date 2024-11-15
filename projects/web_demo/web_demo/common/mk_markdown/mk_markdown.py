import re
import wordninja
from .libs.language import detect_lang
from .libs.markdown_utils import ocr_escape_special_markdown_char
from .libs.ocr_content_type import BlockType, ContentType


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


def join_path(*args):
    return ''.join(str(s).rstrip('/') for s in args)


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
