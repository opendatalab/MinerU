from magic_pdf.libs.commons import s3_image_save_path, join_path
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char
from magic_pdf.libs.ocr_content_type import ContentType
import wordninja
import re


def split_long_words(text):
    segments = text.split(' ')
    for i in range(len(segments)):
        words = re.findall(r'\w+|[^\w\s]', segments[i], re.UNICODE)
        for j in range(len(words)):
            if len(words[j]) > 15:
                words[j] = ' '.join(wordninja.split(words[j]))
        segments[i] = ''.join(words)
    return ' '.join(segments)


def ocr_mk_nlp_markdown(pdf_info_dict: dict):
    markdown = []

    for _, page_info in pdf_info_dict.items():
        blocks = page_info.get("preproc_blocks")
        if not blocks:
            continue
        for block in blocks:
            for line in block['lines']:
                line_text = ''
                for span in line['spans']:
                    if not span.get('content'):
                        continue
                    content = ocr_escape_special_markdown_char(span['content'])  # 转义特殊符号
                    if span['type'] == ContentType.InlineEquation:
                        content = f"${content}$"
                    elif span['type'] == ContentType.InterlineEquation:
                        content = f"$$\n{content}\n$$"
                    line_text += content + ' '
                # 在行末添加两个空格以强制换行
                markdown.append(line_text.strip() + '  ')
    return '\n'.join(markdown)


def ocr_mk_mm_markdown(pdf_info_dict: dict):
    markdown = []

    for _, page_info in pdf_info_dict.items():
        blocks = page_info.get("preproc_blocks")
        if not blocks:
            continue
        for block in blocks:
            for line in block['lines']:
                line_text = ''
                for span in line['spans']:
                    if not span.get('content'):
                        if not span.get('image_path'):
                            continue
                        else:
                            content = f"![]({join_path(s3_image_save_path, span['image_path'])})"
                    else:
                        content = ocr_escape_special_markdown_char(span['content'])  # 转义特殊符号
                        if span['type'] == ContentType.InlineEquation:
                            content = f"${content}$"
                        elif span['type'] == ContentType.InterlineEquation:
                            content = f"$$\n{content}\n$$"
                    line_text += content + ' '
                # 在行末添加两个空格以强制换行
                markdown.append(line_text.strip() + '  ')
    return '\n'.join(markdown)


def ocr_mk_mm_markdown_with_para(pdf_info_dict: dict):
    markdown = []
    for _, page_info in pdf_info_dict.items():
        paras_of_layout = page_info.get("para_blocks")
        if not paras_of_layout:
            continue
        for paras in paras_of_layout:
            for para in paras:
                para_text = ''
                for line in para:
                    for span in line['spans']:
                        span_type = span.get('type')
                        if span_type == ContentType.Text:
                            content = split_long_words(span['content'])
                            pass
                        elif span_type == ContentType.InlineEquation:
                            content = f" ${span['content']}$ "
                        elif span_type == ContentType.InterlineEquation:
                            content = f"\n$$\n{span['content']}\n$$\n"
                        elif span_type in [ ContentType.Image, ContentType.Table ]:
                            content = f"\n![]({join_path(s3_image_save_path, span['image_path'])})\n"
                        para_text += content + ' '
                markdown.append(para_text.strip() + '  ')

    return '\n\n'.join(markdown)


def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: dict):
    markdown_with_para_and_pagination = []
    for page_no, page_info in pdf_info_dict.items():
        page_markdown = []
        paras = page_info.get("para_blocks")
        if not paras:
            continue
        for para in paras:
            para_text = ''
            for line in para:
                for span in line['spans']:
                    span_type = span.get('type')
                    if span_type == ContentType.Text:
                        content = split_long_words(span['content'])
                        # content = span['content']
                    elif span_type == ContentType.InlineEquation:
                        content = f"${span['content']}$"
                    elif span_type == ContentType.InterlineEquation:
                        content = f"\n$$\n{span['content']}\n$$\n"
                    elif span_type in [ContentType.Image, ContentType.Table]:
                        content = f"\n![]({join_path(s3_image_save_path, span['image_path'])})\n"
                    para_text += content + ' '
            page_markdown.append(para_text.strip() + '  ')
        markdown_with_para_and_pagination.append({
            'page_no': page_no,
            'md': '\n\n'.join(page_markdown)
        })
    return markdown_with_para_and_pagination


def make_standard_format_with_para(pdf_info_dict: dict):
    content_list = []
    for _, page_info in pdf_info_dict.items():
        paras = page_info.get("para_blocks")
        if not paras:
            continue
        for para in paras:
            for line in para:
                content = line_to_standard_format(line)
                content_list.append(content)
    return content_list


def line_to_standard_format(line):
    line_text = ""
    inline_equation_num = 0
    for span in line['spans']:
        if not span.get('content'):
            if not span.get('image_path'):
                continue
            else:
                if span['type'] == ContentType.Image:
                    content = {
                        'type': 'image',
                        'img_path': join_path(s3_image_save_path, span['image_path'])
                    }
                    return content
                elif span['type'] == ContentType.Table:
                    content = {
                        'type': 'table',
                        'img_path': join_path(s3_image_save_path, span['image_path'])
                    }
                    return content
        else:
            if span['type'] == ContentType.InterlineEquation:
                interline_equation = ocr_escape_special_markdown_char(span['content'])  # 转义特殊符号
                content = {
                    'type': 'equation',
                    'latex': f"$$\n{interline_equation}\n$$"
                }
                return content
            elif span['type'] == ContentType.InlineEquation:
                inline_equation = ocr_escape_special_markdown_char(span['content'])  # 转义特殊符号
                line_text += f"${inline_equation}$"
                inline_equation_num += 1
            elif span['type'] == ContentType.Text:
                text_content = ocr_escape_special_markdown_char(span['content'])  # 转义特殊符号
                line_text += text_content
    content = {
        'type': 'text',
        'text': line_text,
        'inline_equation_num': inline_equation_num
    }
    return content


def ocr_mk_mm_standard_format(pdf_info_dict: dict):
    '''
    content_list
    type         string      image/text/table/equation(行间的单独拿出来，行内的和text合并)
    latex        string      latex文本字段。
    text         string      纯文本格式的文本数据。
    md           string      markdown格式的文本数据。
    img_path     string      s3://full/path/to/img.jpg
    '''
    content_list = []
    for _, page_info in pdf_info_dict.items():
        blocks = page_info.get("preproc_blocks")
        if not blocks:
            continue
        for block in blocks:
            for line in block['lines']:
                content = line_to_standard_format(line)
                content_list.append(content)
    return content_list
