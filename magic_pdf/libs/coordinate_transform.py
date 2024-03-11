def get_scale_ratio(ocr_page_info, page):
    pix = page.get_pixmap(dpi=72)
    pymu_width = int(pix.w)
    pymu_height = int(pix.h)
    width_from_json = ocr_page_info['page_info']['width']
    height_from_json = ocr_page_info['page_info']['height']
    horizontal_scale_ratio = width_from_json / pymu_width
    vertical_scale_ratio = height_from_json / pymu_height
    return horizontal_scale_ratio, vertical_scale_ratio
