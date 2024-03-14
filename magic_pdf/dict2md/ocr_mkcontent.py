from magic_pdf.libs.ocr_content_type import ContentType


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
                    content = span['content'].replace('$', '\$')  # 转义$
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
                            content = f"![]({span['image_path']})"
                    else:
                        content = span['content'].replace('$', '\$')  # 转义$
                        if span['type'] == ContentType.InlineEquation:
                            content = f"${content}$"
                        elif span['type'] == ContentType.InterlineEquation:
                            content = f"$$\n{content}\n$$"
                    line_text += content + ' '
                # 在行末添加两个空格以强制换行
                markdown.append(line_text.strip() + '  ')
    return '\n'.join(markdown)
