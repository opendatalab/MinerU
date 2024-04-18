def ocr_prepare_bboxes_for_layout_split(img_blocks, table_blocks, discarded_blocks, text_blocks,
                                        title_blocks, interline_equation_blocks, page_w, page_h):
    all_bboxes = []

    for image in img_blocks:
        x0, y0, x1, y1 = image['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, 'image_block', None, None, None, None])

    for table in table_blocks:
        x0, y0, x1, y1 = table['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, 'table_block', None, None, None, None])

    for text in text_blocks:
        x0, y0, x1, y1 = text['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, 'text_block', None, None, None, None])

    for title in title_blocks:
        x0, y0, x1, y1 = title['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, 'title_block', None, None, None, None])

    for interline_equation in interline_equation_blocks:
        x0, y0, x1, y1 = interline_equation['bbox']
        all_bboxes.append([x0, y0, x1, y1, None, None, None, 'interline_equation_block', None, None, None, None])

    '''discarded_blocks中只保留宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的（限定footnote）'''
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
            all_bboxes.append([x0, y0, x1, y1, None, None, None, 'footnote', None, None, None, None])

    return all_bboxes

