
def ocr_construct_page_component_v2(blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                    images, tables, interline_equations, discarded_blocks, need_drop, drop_reason):
    return_dict = {
        'preproc_blocks': blocks,
        'layout_bboxes': layout_bboxes,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        '_layout_tree': layout_tree,
        'images': images,
        'tables': tables,
        'interline_equations': interline_equations,
        'discarded_blocks': discarded_blocks,
        'need_drop': need_drop,
        'drop_reason': drop_reason,
    }
    return return_dict
