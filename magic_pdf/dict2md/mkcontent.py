import math
from loguru import logger

from magic_pdf.libs.boxbase import find_bottom_nearest_text_bbox, find_top_nearest_text_bbox
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.ocr_content_type import ContentType

TYPE_INLINE_EQUATION = ContentType.InlineEquation
TYPE_INTERLINE_EQUATION = ContentType.InterlineEquation
UNI_FORMAT_TEXT_TYPE = ['text', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']


@DeprecationWarning
def mk_nlp_markdown_1(para_dict: dict):
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


@DeprecationWarning
def mk_mm_markdown_1(para_dict: dict):
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
                                logger.error(f"Can't find the location of image {img['image_path']} in the markdown file #1")
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
                            logger.error(f"Can't find the location of image {img['image_path']} in the markdown file #2")
                    
        content_lst.append(page_md)
                    
    """拼装成全部页面的文本"""
    content_text = "\n\n".join(content_lst)

    return content_text


def __insert_after_para(text, type, element, content_list):
    """
    在content_list中找到text，将image_path作为一个新的node插入到text后面
    """
    for i, c in enumerate(content_list):
        content_type = c.get("type")
        if content_type in UNI_FORMAT_TEXT_TYPE and text in c.get("text", ''):
            if type == "image":
                content_node = {
                    "type": "image",
                    "img_path": element.get("image_path"),
                    "img_alt": "",
                    "img_title": "",
                    "img_caption": "",
                }
            elif type == "table":
                content_node = {
                    "type": "table",
                    "img_path": element.get("image_path"),
                    "table_latex": element.get("text"),
                    "table_title": "",
                    "table_caption": "",
                    "table_quality": element.get("quality"),
                }
            content_list.insert(i+1, content_node)
            break
    else:
        logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file, search target is {text}")
    


def __insert_before_para(text, type, element, content_list):
    """
    在content_list中找到text，将image_path作为一个新的node插入到text前面
    """
    for i, c in enumerate(content_list):
        content_type = c.get("type")
        if content_type in  UNI_FORMAT_TEXT_TYPE and text in c.get("text", ''):
            if type == "image":
                content_node = {
                    "type": "image",
                    "img_path": element.get("image_path"),
                    "img_alt": "",
                    "img_title": "",
                    "img_caption": "",
                }
            elif type == "table":
                content_node = {
                    "type": "table",
                    "img_path": element.get("image_path"),
                    "table_latex": element.get("text"),
                    "table_title": "",
                    "table_caption": "",
                    "table_quality": element.get("quality"),
                }
            content_list.insert(i, content_node)
            break
    else:
        logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file, search target is {text}")
         

def mk_universal_format(pdf_info_list: list, img_buket_path):
    """
    构造统一格式 https://aicarrier.feishu.cn/wiki/FqmMwcH69iIdCWkkyjvcDwNUnTY
    """
    content_lst = []
    for page_info in pdf_info_list:
        page_lst = [] # 一个page内的段落列表
        para_blocks = page_info.get("para_blocks")
        pymu_raw_blocks = page_info.get("preproc_blocks")
        
        all_page_images = []
        all_page_images.extend(page_info.get("images",[]))
        all_page_images.extend(page_info.get("image_backup", []) )
        # all_page_images.extend(page_info.get("tables",[]))
        # all_page_images.extend(page_info.get("table_backup",[]) )
        all_page_tables = []
        all_page_tables.extend(page_info.get("tables", []))

        if not para_blocks or not pymu_raw_blocks: # 只有图片的拼接的场景
            for img in all_page_images:
                content_node = {
                    "type": "image",
                    "img_path": join_path(img_buket_path, img['image_path']),
                    "img_alt":"",
                    "img_title":"",
                    "img_caption":""
                }
                page_lst.append(content_node) # TODO 图片顺序
            for table in all_page_tables:
                content_node = {
                    "type": "table",
                    "img_path": join_path(img_buket_path, table['image_path']),
                    "table_latex": table.get("text"),
                    "table_title": "",
                    "table_caption": "",
                    "table_quality": table.get("quality"),
                }
                page_lst.append(content_node) # TODO 图片顺序
        else:
            for block in para_blocks:
                item = block["paras"]
                for _, p in item.items():
                    font_type = p['para_font_type']# 对于文本来说，要么是普通文本，要么是个行间公式
                    if font_type == TYPE_INTERLINE_EQUATION:
                        content_node = {
                            "type": "equation",
                            "latex": p["para_text"]
                        }
                        page_lst.append(content_node)
                    else:
                        para_text = p["para_text"]
                        is_title = p["is_para_title"]
                        title_level = p['para_title_level']
                        
                        if is_title:
                            content_node = {
                                "type": f"h{title_level}",
                                "text": para_text
                            }
                            page_lst.append(content_node)
                        else:
                            content_node = {
                                "type": "text",
                                "text": para_text
                            }
                            page_lst.append(content_node)
                            
        content_lst.extend(page_lst)
        
        """插入图片"""
        for img in all_page_images:
            insert_img_or_table("image", img, pymu_raw_blocks, content_lst)

        """插入表格"""
        for table in all_page_tables:
            insert_img_or_table("table", table, pymu_raw_blocks, content_lst)
    # end for
    return content_lst


def insert_img_or_table(type, element, pymu_raw_blocks, content_lst):
    element_bbox = element['bbox']
    # 先看在哪个block内
    for block in pymu_raw_blocks:
        bbox = block['bbox']
        if bbox[0] - 1 <= element_bbox[0] < bbox[2] + 1 and bbox[1] - 1 <= element_bbox[1] < bbox[
            3] + 1:  # 确定在这个大的block内，然后进入逐行比较距离
            for l in block['lines']:
                line_box = l['bbox']
                if line_box[0] - 1 <= element_bbox[0] < line_box[2] + 1 and line_box[1] - 1 <= element_bbox[1] < line_box[
                    3] + 1:  # 在line内的，插入line前面
                    line_txt = "".join([s['text'] for s in l['spans']])
                    __insert_before_para(line_txt, type, element, content_lst)
                    break
                break
            else:  # 在行与行之间
                # 找到图片x0,y0与line的x0,y0最近的line
                min_distance = 100000
                min_line = None
                for l in block['lines']:
                    line_box = l['bbox']
                    distance = math.sqrt((line_box[0] - element_bbox[0]) ** 2 + (line_box[1] - element_bbox[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        min_line = l
                if min_line:
                    line_txt = "".join([s['text'] for s in min_line['spans']])
                    img_h = element_bbox[3] - element_bbox[1]
                    if min_distance < img_h:  # 文字在图片前面
                        __insert_after_para(line_txt, type, element, content_lst)
                    else:
                        __insert_before_para(line_txt, type, element, content_lst)
                    break
                else:
                    logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file #1")
    else:  # 应当在两个block之间
        # 找到上方最近的block，如果上方没有就找大下方最近的block
        top_txt_block = find_top_nearest_text_bbox(pymu_raw_blocks, element_bbox)
        if top_txt_block:
            line_txt = "".join([s['text'] for s in top_txt_block['lines'][-1]['spans']])
            __insert_after_para(line_txt, type, element, content_lst)
        else:
            bottom_txt_block = find_bottom_nearest_text_bbox(pymu_raw_blocks, element_bbox)
            if bottom_txt_block:
                line_txt = "".join([s['text'] for s in bottom_txt_block['lines'][0]['spans']])
                __insert_before_para(line_txt, type, element, content_lst)
            else:  # TODO ，图片可能独占一列，这种情况上下是没有图片的
                logger.error(f"Can't find the location of image {element.get('image_path')} in the markdown file #2")


def mk_mm_markdown(content_list):
    """
    基于同一格式的内容列表，构造markdown，含图片
    """
    content_md = []
    for c in content_list:
        content_type = c.get("type")
        if content_type == "text":
            content_md.append(c.get("text"))
        elif content_type == "equation":
            content = c.get("latex")
            if content.startswith("$$") and content.endswith("$$"):
                content_md.append(content)
            else:
                content_md.append(f"\n$$\n{c.get('latex')}\n$$\n")
        elif content_type in UNI_FORMAT_TEXT_TYPE:
            content_md.append(f"{'#'*int(content_type[1])} {c.get('text')}")
        elif content_type == "image":
            content_md.append(f"![]({c.get('img_path')})")
    return "\n\n".join(content_md)

def mk_nlp_markdown(content_list):
    """
    基于同一格式的内容列表，构造markdown，不含图片
    """
    content_md = []
    for c in content_list:
        content_type = c.get("type")
        if content_type == "text":
            content_md.append(c.get("text"))
        elif content_type == "equation":
            content_md.append(f"$$\n{c.get('latex')}\n$$")
        elif content_type == "table":
            content_md.append(f"$$$\n{c.get('table_latex')}\n$$$")
        elif content_type in UNI_FORMAT_TEXT_TYPE:
            content_md.append(f"{'#'*int(content_type[1])} {c.get('text')}")
    return "\n\n".join(content_md)