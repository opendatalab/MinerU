def solve_inline_too_large_interval(pdf_info_dict: dict) -> dict:  # text_block -> json中的preproc_block
    """解决行内文本间距过大问题"""
    for i in range(len(pdf_info_dict)):

        text_blocks = pdf_info_dict[f'page_{i}']['preproc_blocks']

        for block in text_blocks:

            x_pre_1, y_pre_1, x_pre_2, y_pre_2 = 0, 0, 0, 0
            
            for line in block['lines']:

                x_cur_1, y_cur_1, x_cur_2, y_cur_2 = line['bbox']
                # line_box = [x1, y1, x2, y2] 
                if int(y_cur_1) == int(y_pre_1) and int(y_cur_2) == int(y_pre_2):
                    # if len(line['spans']) == 1:
                    line['spans'][0]['text'] = ' ' + line['spans'][0]['text']
                
                x_pre_1, y_pre_1, x_pre_2, y_pre_2 = line['bbox'] 

    return pdf_info_dict








