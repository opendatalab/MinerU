from collections import defaultdict

from magic_pdf.libs.boxbase import calculate_iou


def compare_bbox_with_list(bbox, bbox_list, tolerance=1):
    return any(all(abs(a - b) < tolerance for a, b in zip(bbox, common_bbox)) for common_bbox in bbox_list)

def is_single_line_block(block):
    # Determine based on the width and height of the block
    block_width = block["X1"] - block["X0"]
    block_height = block["bbox"][3] - block["bbox"][1]

    # If the height of the block is close to the average character height and the width is large, it is considered a single line
    return block_height <= block["avg_char_height"] * 3 and block_width > block["avg_char_width"] * 3

def get_most_common_bboxes(bboxes, page_height, position="top", threshold=0.25, num_bboxes=3, min_frequency=2):
    """
    This function gets the most common bboxes from the bboxes

    Parameters
    ----------
    bboxes : list
        bboxes
    page_height : float
        height of the page
    position : str, optional
        "top" or "bottom", by default "top"
    threshold : float, optional
        threshold, by default 0.25
    num_bboxes : int, optional
        number of bboxes to return, by default 3
    min_frequency : int, optional
        minimum frequency of the bbox, by default 2

    Returns
    -------
    common_bboxes : list
        common bboxes
    """
    # Filter bbox by position
    if position == "top":
        filtered_bboxes = [bbox for bbox in bboxes if bbox[1] < page_height * threshold]
    else:
        filtered_bboxes = [bbox for bbox in bboxes if bbox[3] > page_height * (1 - threshold)]

    # Find the most common bbox
    bbox_count = defaultdict(int)
    for bbox in filtered_bboxes:
        bbox_count[tuple(bbox)] += 1

    # Get the most frequently occurring bbox, but only consider it when the frequency exceeds min_frequency
    common_bboxes = [
        bbox for bbox, count in sorted(bbox_count.items(), key=lambda item: item[1], reverse=True) if count >= min_frequency
    ][:num_bboxes]
    return common_bboxes

def detect_footer_header2(result_dict, similarity_threshold=0.5):
    """
    This function detects the header and footer of the document.

    Parameters
    ----------
    result_dict : dict
        result dictionary

    Returns
    -------
    result_dict : dict
        result dictionary
    """
    # Traverse all blocks in the document
    single_line_blocks = 0
    total_blocks = 0
    single_line_blocks = 0

    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):
            for block_key, block in blocks.items():
                if block_key.startswith("block_"):
                    total_blocks += 1
                    if is_single_line_block(block):
                        single_line_blocks += 1

    # If there are no blocks, skip the header and footer detection
    if total_blocks == 0:
        print("No blocks found. Skipping header/footer detection.")
        return result_dict

    # If most of the blocks are single-line, skip the header and footer detection
    if single_line_blocks / total_blocks > 0.5:  # 50% of the blocks are single-line
        # print("Skipping header/footer detection for text-dense document.")
        return result_dict

    # Collect the bounding boxes of all blocks
    all_bboxes = []
    all_texts = []

    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):
            for block_key, block in blocks.items():
                if block_key.startswith("block_"):
                    all_bboxes.append(block["bbox"])

    # Get the height of the page
    page_height = max(bbox[3] for bbox in all_bboxes)

    # Get the most common bbox lists for headers and footers
    common_header_bboxes = get_most_common_bboxes(all_bboxes, page_height, position="top") if all_bboxes else []
    common_footer_bboxes = get_most_common_bboxes(all_bboxes, page_height, position="bottom") if all_bboxes else []

    # Detect and mark headers and footers
    for page_id, blocks in result_dict.items():
        if page_id.startswith("page_"):
            for block_key, block in blocks.items():
                if block_key.startswith("block_"):
                    bbox = block["bbox"]
                    text = block["text"]

                    is_header = compare_bbox_with_list(bbox, common_header_bboxes)
                    is_footer = compare_bbox_with_list(bbox, common_footer_bboxes)
                    block["is_header"] = int(is_header)
                    block["is_footer"] = int(is_footer)

    return result_dict


def __get_page_size(page_sizes:list):
    """
    页面大小可能不一样
    """
    w = sum([w for w,h in page_sizes])/len(page_sizes)
    h = sum([h for w,h  in page_sizes])/len(page_sizes)
    return w, h

def __calculate_iou(bbox1, bbox2):
    iou = calculate_iou(bbox1, bbox2)
    return iou

def __is_same_pos(box1, box2, iou_threshold):
    iou = __calculate_iou(box1, box2)
    return iou >= iou_threshold


def get_most_common_bbox(bboxes:list, page_size:list, page_cnt:int,  page_range_threshold=0.2, iou_threshold=0.9):
    """
    common bbox必须大于page_cnt的1/3
    """
    min_occurance_cnt = max(3, page_cnt//4)
    header_det_bbox = []
    footer_det_bbox = []
    
    hdr_same_pos_group = []
    btn_same_pos_group = []
    
    page_w, page_h = __get_page_size(page_size)
    top_y, bottom_y = page_w*page_range_threshold, page_h*(1-page_range_threshold)
    
    top_bbox = [b for b in bboxes if b[3]<top_y]
    bottom_bbox = [b for b in bboxes if b[1]>bottom_y]
    # 然后开始排序，寻找最经常出现的bbox, 寻找的时候如果IOU>iou_threshold就算是一个
    for i in range(0, len(top_bbox)):
        hdr_same_pos_group.append([top_bbox[i]])
        for j in range(i+1, len(top_bbox)):
            if __is_same_pos(top_bbox[i], top_bbox[j], iou_threshold):
                #header_det_bbox = [min(top_bbox[i][0], top_bbox[j][0]), min(top_bbox[i][1], top_bbox[j][1]), max(top_bbox[i][2], top_bbox[j][2]), max(top_bbox[i][3],top_bbox[j][3])]
                hdr_same_pos_group[i].append(top_bbox[j])
                
    for i in range(0, len(bottom_bbox)):
        btn_same_pos_group.append([bottom_bbox[i]])
        for j in range(i+1, len(bottom_bbox)):
            if __is_same_pos(bottom_bbox[i], bottom_bbox[j], iou_threshold):
                #footer_det_bbox = [min(bottom_bbox[i][0], bottom_bbox[j][0]), min(bottom_bbox[i][1], bottom_bbox[j][1]), max(bottom_bbox[i][2], bottom_bbox[j][2]), max(bottom_bbox[i][3],bottom_bbox[j][3])]
                btn_same_pos_group[i].append(bottom_bbox[j])
                
    # 然后看下每一组的bbox，是否符合大于page_cnt一定比例
    hdr_same_pos_group = [g for g in hdr_same_pos_group if len(g)>=min_occurance_cnt]
    btn_same_pos_group = [g for g in btn_same_pos_group if len(g)>=min_occurance_cnt]
    
    # 平铺2个list[list]
    hdr_same_pos_group = [bbox for g in hdr_same_pos_group for bbox in g]
    btn_same_pos_group = [bbox for g in btn_same_pos_group for bbox in g]
    # 寻找hdr_same_pos_group中的box[3]最大值，btn_same_pos_group中的box[1]最小值
    hdr_same_pos_group.sort(key=lambda b:b[3])
    btn_same_pos_group.sort(key=lambda b:b[1])
    
    hdr_y = hdr_same_pos_group[-1][3] if hdr_same_pos_group else 0
    btn_y = btn_same_pos_group[0][1] if btn_same_pos_group else page_h
    
    header_det_bbox = [0, 0, page_w, hdr_y]
    footer_det_bbox = [0, btn_y, page_w, page_h]
    # logger.warning(f"header: {header_det_bbox}, footer: {footer_det_bbox}")
    return header_det_bbox, footer_det_bbox, page_w, page_h
    

def drop_footer_header(pdf_info_dict:dict):
    """
    启用规则探测,在全局的视角上通过统计的方法。
    """
    header = []
    footer = []
    
    all_text_bboxes = [blk['bbox'] for _, val in pdf_info_dict.items() for blk in val['preproc_blocks']]
    image_bboxes = [img['bbox'] for _, val in pdf_info_dict.items() for img in val['images']] + [img['bbox'] for _, val in pdf_info_dict.items() for img in val['image_backup']]
    page_size = [val['page_size'] for _, val in pdf_info_dict.items()]
    page_cnt = len(pdf_info_dict.keys()) # 一共多少页
    header, footer, page_w, page_h = get_most_common_bbox(all_text_bboxes+image_bboxes, page_size, page_cnt)
    
    """"
    把范围扩展到页面水平的整个方向上
    """        
    if header:
        header = [0, 0, page_w, header[3]+1]
        
    if footer:
        footer = [0, footer[1]-1, page_w, page_h]
        
    # 找到footer, header范围之后，针对每一页pdf，从text、图片中删除这些范围内的内容
    # 移除text block
    
    for _, page_info in pdf_info_dict.items():
        header_text_blk = []
        footer_text_blk = []
        for blk in page_info['preproc_blocks']:
            blk_bbox = blk['bbox']
            if header and blk_bbox[3]<=header[3]:
                blk['tag'] = "header"
                header_text_blk.append(blk)
            elif footer and blk_bbox[1]>=footer[1]:
                blk['tag'] = "footer"
                footer_text_blk.append(blk)
                
        # 放入text_block_droped中
        page_info['droped_text_block'].extend(header_text_blk)
        page_info['droped_text_block'].extend(footer_text_blk)
        
        for blk in header_text_blk:
            page_info['preproc_blocks'].remove(blk)
        for blk in footer_text_blk:
            page_info['preproc_blocks'].remove(blk)
            
        """接下来把footer、header上的图片也删除掉。图片包括正常的和backup的"""
        header_image = []
        footer_image = []
        
        for image_info in page_info['images']:
            img_bbox = image_info['bbox']
            if header and img_bbox[3]<=header[3]:
                image_info['tag'] = "header"
                header_image.append(image_info)
            elif footer and img_bbox[1]>=footer[1]:
                image_info['tag'] = "footer"
                footer_image.append(image_info)
                
        page_info['droped_image_block'].extend(header_image)
        page_info['droped_image_block'].extend(footer_image)
        
        for img in header_image:
            page_info['images'].remove(img)
        for img in footer_image:
            page_info['images'].remove(img)
            
        """接下来吧backup的图片也删除掉"""
        header_image = []
        footer_image = []
        
        for image_info in page_info['image_backup']:
            img_bbox = image_info['bbox']
            if header and img_bbox[3]<=header[3]:
                image_info['tag'] = "header"
                header_image.append(image_info)
            elif footer and img_bbox[1]>=footer[1]:
                image_info['tag'] = "footer"
                footer_image.append(image_info)
                
        page_info['droped_image_block'].extend(header_image)
        page_info['droped_image_block'].extend(footer_image)
        
        for img in header_image:
            page_info['image_backup'].remove(img)
        for img in footer_image:
            page_info['image_backup'].remove(img)
            
    return header, footer
