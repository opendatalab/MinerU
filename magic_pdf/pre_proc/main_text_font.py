import collections


def get_main_text_font(pdf_docs):
    font_names = collections.Counter()
    for page in pdf_docs:
        blocks = page.get_text('dict')['blocks']
        if blocks is not None:
            for block in blocks:
                lines = block.get('lines')
                if lines is not None:
                    for line in lines:
                        span_font = [(span['font'], len(span['text'])) for span in line['spans'] if
                                     'font' in span and len(span['text']) > 0]
                        if span_font:
                            # main_text_font应该用基于字数最多的字体而不是span级别的统计
                            # font_names.append(font_name for font_name in span_font)
                            # block_fonts.append(font_name for font_name in span_font)
                            for font, count in span_font:
                                font_names[font] += count
    main_text_font = font_names.most_common(1)[0][0]
    return main_text_font

