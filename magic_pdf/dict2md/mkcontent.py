import math
from loguru import logger

from magic_pdf.libs.boxbase import find_bottom_nearest_text_bbox, find_top_nearest_text_bbox


def mk_nlp_markdown(para_dict: dict):
    """
    对排序后的bboxes拼接内容
    """
    content_lst = []
    for _, page_info in para_dict.items():
        para_blocks = page_info.get("para_blocks")
        if not para_blocks:
            continue

        for block in para_blocks:
            item = block["paras"]
            for _, p in item.items():
                para_text = p["para_text"]
                is_title = p["is_para_title"]
                title_level = p['para_title_level']
                md_title_prefix = "#"*title_level
                if is_title:
                    content_lst.append(f"{md_title_prefix} {para_text}")
                else:
                    content_lst.append(para_text)

    content_text = "\n\n".join(content_lst)

    return content_text



# 找到目标字符串在段落中的索引
def __find_index(paragraph, target):
    index = paragraph.find(target)
    if index != -1:
        return index
    else:
        return None


def __insert_string(paragraph, target, postion):
    new_paragraph = paragraph[:postion] + target + paragraph[postion:] 
    return new_paragraph


def __insert_after(content, image_content, target):
    """
    在content中找到target，将image_content插入到target后面
    """
    index = content.find(target)
    if index != -1:
        content = content[:index+len(target)] + "\n\n" + image_content + "\n\n" + content[index+len(target):]
    else:
        logger.error(f"Can't find the location of image {image_content} in the markdown file, search target is {target}")
    return content

def __insert_before(content, image_content, target):
    """
    在content中找到target，将image_content插入到target前面
    """
    index = content.find(target)
    if index != -1:
        content = content[:index] + "\n\n" + image_content + "\n\n" + content[index:]
    else:
        logger.error(f"Can't find the location of image {image_content} in the markdown file, search target is {target}")
    return content



def mk_mm_markdown(para_dict: dict):
    """拼装多模态markdown"""
    content_lst = []
    for _, page_info in para_dict.items():
        page_lst = [] # 一个page内的段落列表
        para_blocks = page_info.get("para_blocks")
        pymu_raw_blocks = page_info.get("preproc_blocks")  
        
        all_page_images = []
        all_page_images.extend(page_info.get("images",[]))
        all_page_images.extend(page_info.get("image_backup", []) )
        all_page_images.extend(page_info.get("tables",[]))
        all_page_images.extend(page_info.get("table_backup",[]) )
        
        if not para_blocks or not pymu_raw_blocks: # 只有图片的拼接的场景
            for img in all_page_images:
                page_lst.append(f"![]({img['image_path']})") # TODO 图片顺序
            page_md = "\n\n".join(page_lst)
            
        else:
            for block in para_blocks:
                item = block["paras"]
                for _, p in item.items():
                    para_text = p["para_text"]
                    is_title = p["is_para_title"]
                    title_level = p['para_title_level']
                    md_title_prefix = "#"*title_level
                    if is_title:
                        page_lst.append(f"{md_title_prefix} {para_text}")
                    else:
                        page_lst.append(para_text)
                        
            """拼装成一个页面的文本"""
            page_md = "\n\n".join(page_lst)
            """插入图片"""
            for img in all_page_images:
                imgbox = img['bbox']
                img_content = f"![]({img['image_path']})"
                # 先看在哪个block内
                for block in pymu_raw_blocks:
                    bbox = block['bbox']
                    if bbox[0]-1 <= imgbox[0] < bbox[2]+1 and bbox[1]-1 <= imgbox[1] < bbox[3]+1:# 确定在block内
                        for l in block['lines']:
                            line_box = l['bbox']
                            if line_box[0]-1 <= imgbox[0] < line_box[2]+1 and line_box[1]-1 <= imgbox[1] < line_box[3]+1: # 在line内的，插入line前面
                                line_txt = "".join([s['text'] for s in l['spans']])
                                page_md = __insert_before(page_md, img_content, line_txt)
                                break
                            break
                        else:# 在行与行之间
                            # 找到图片x0,y0与line的x0,y0最近的line
                            min_distance = 100000
                            min_line = None
                            for l in block['lines']:
                                line_box = l['bbox']
                                distance = math.sqrt((line_box[0] - imgbox[0])**2 + (line_box[1] - imgbox[1])**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    min_line = l
                            if min_line:
                                line_txt = "".join([s['text'] for s in min_line['spans']])
                                img_h = imgbox[3] - imgbox[1]
                                if min_distance<img_h: # 文字在图片前面
                                    page_md = __insert_after(page_md, img_content, line_txt)
                                else:
                                    page_md = __insert_before(page_md, img_content, line_txt)
                            else:
                                logger.error(f"Can't find the location of image {img['image_path']} in the markdown file")
                else:# 应当在两个block之间
                    # 找到上方最近的block，如果上方没有就找大下方最近的block
                    top_txt_block = find_top_nearest_text_bbox(pymu_raw_blocks, imgbox)
                    if top_txt_block:
                        line_txt = "".join([s['text'] for s in top_txt_block['lines'][-1]['spans']])
                        page_md = __insert_after(page_md, img_content, line_txt)
                    else:
                        bottom_txt_block = find_bottom_nearest_text_bbox(pymu_raw_blocks, imgbox)
                        if bottom_txt_block:
                            line_txt = "".join([s['text'] for s in bottom_txt_block['lines'][0]['spans']])
                            page_md = __insert_before(page_md, img_content, line_txt)
                        else:
                            logger.error(f"Can't find the location of image {img['image_path']} in the markdown file")
                    
        content_lst.append(page_md)
                    
    """拼装成全部页面的文本"""
    content_text = "\n\n".join(content_lst)

    return content_text
    
    
@DeprecationWarning
def mk_mm_markdown_1(para_dict: dict):
    """
    得到images和tables变量
    """
    image_all_list = []
    
    for _, page_info in para_dict.items():
        images = page_info.get("images",[])
        tables = page_info.get("tables",[])
        image_backup = page_info.get("image_backup", [])  
        table_backup = page_info.get("table_backup",[]) 
        all_page_images = []
        all_page_images.extend(images)
        all_page_images.extend(image_backup)
        all_page_images.extend(tables)
        all_page_images.extend(table_backup)
        
        pymu_raw_blocks = page_info.get("pymu_raw_blocks")  

        # 提取每个图片所在位置
        for image_info in all_page_images:
            x0_image, y0_image, x1_image, y1_image = image_info['bbox'][:4]
            image_path = image_info['image_path']
            
            # 判断图片处于原始PDF中哪个模块之间
            image_internal_dict = {}
            image_external_dict = {}
            between_dict = {}
            for block in pymu_raw_blocks:
                x0, y0, x1, y1 = block['bbox'][:4]

                # 在某个模块内部
                if x0 <= x0_image < x1 and y0 <= y0_image < y1:
                    image_internal_dict['bbox'] = [x0_image, y0_image, x1_image, y1_image]
                    image_internal_dict['path'] = image_path
                    
                    # 确定图片在哪句文本之前
                    y_pre = 0
                    for line in block['lines']:
                        x0, y0, x1, y1 = line['spans'][0]['bbox']
                        if x0 <= x0_image < x1 and y_pre <= y0_image < y0: 
                            text = line['spans']['text']
                            image_internal_dict['text'] = text
                            image_internal_dict['markdown_image'] = f'![image_path]({image_path})'
                            break
                        else:
                            y_pre = y0
                # 在某两个模块之间
                elif x0 <= x0_image < x1:
                    distance = math.sqrt((x1_image - x0)**2 + (y1_image - y0)**2)
                    between_dict[block['number']] = distance
            
            # 找到与定位点距离最小的文本block
            if between_dict:
                min_key = min(between_dict, key=between_dict.get)
                spans_list = []
                for span in pymu_raw_blocks[min_key]['lines']: 
                    for text_piece in span['spans']:
                        # 防止索引定位文本内容过多
                        if len(spans_list) < 60:
                            spans_list.append(text_piece['text'])
                text1 = ''.join(spans_list)
                
                image_external_dict['bbox'] = [x0_image, y0_image, x1_image, y1_image]
                image_external_dict['path'] = image_path 
                image_external_dict['text'] = text1
                image_external_dict['markdown_image'] = f'![image_path]({image_path})'

            # 将内部图片或外部图片存入当页所有图片的列表
            if len(image_internal_dict) != 0:
                image_all_list.append(image_internal_dict)
            elif len(image_external_dict) != 0:
                image_all_list.append(image_external_dict)
            else:
                logger.error(f"Can't find the location of image {image_path} in the markdown file")

    content_text = mk_nlp_markdown(para_dict)

    for image_info_extract in image_all_list:
        loc = __find_index(content_text, image_info_extract['text'])
        if loc is not None:
            content_text = __insert_string(content_text, image_info_extract['markdown_image'], loc)
        else:
            logger.error(f"Can't find the location of image {image_info_extract['path']} in the markdown file")

    return content_text