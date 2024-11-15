import re


def escape_special_markdown_char(pymu_blocks):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    special_chars = ["*", "`", "~", "$"]
    for blk in pymu_blocks:
        for line in blk['lines']:
            for span in line['spans']:
                for char in special_chars:
                    span_text = span['text']
                    span_type = span.get("_type", None)
                    if span_type in ['inline-equation', 'interline-equation']:
                        continue
                    elif span_text:
                        span['text'] = span['text'].replace(char, "\\" + char)

    return pymu_blocks


def ocr_escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content
