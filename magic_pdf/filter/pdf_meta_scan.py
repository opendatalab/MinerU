"""
输入： s3路径，每行一个
输出： pdf文件元信息，包括每一页上的所有图片的长宽高，bbox位置
"""
import sys
import click

from magic_pdf.libs.commons import read_file, mymax, get_top_percent_list
from magic_pdf.libs.commons import fitz
from loguru import logger
from collections import Counter

from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.pdf_check import detect_invalid_chars

scan_max_page = 50
junk_limit_min = 10


def calculate_max_image_area_per_page(result: list, page_width_pts, page_height_pts):
    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in
                               result]
    page_area = int(page_width_pts) * int(page_height_pts)
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.6]
    return max_image_area_per_page


def process_image(page, junk_img_bojids=[]):
    page_result = []  # 存每个页面里的多张图四元组信息
    items = page.get_images()
    dedup = set()
    for img in items:
        # 这里返回的是图片在page上的实际展示的大小。返回一个数组，每个元素第一部分是
        img_bojid = img[0]  # 在pdf文件中是全局唯一的，如果这个图反复出现在pdf里那么就可能是垃圾信息，例如水印、页眉页脚等
        if img_bojid in junk_img_bojids:  # 如果是垃圾图像，就跳过
            continue
        recs = page.get_image_rects(img, transform=True)
        if recs:
            rec = recs[0][0]
            x0, y0, x1, y1 = map(int, rec)
            width = x1 - x0
            height = y1 - y0
            if (x0, y0, x1, y1, img_bojid) in dedup:  # 这里面会出现一些重复的bbox，无需重复出现，需要去掉
                continue
            if not all([width, height]):  # 长和宽任何一个都不能是0，否则这个图片不可见，没有实际意义
                continue
            dedup.add((x0, y0, x1, y1, img_bojid))
            page_result.append([x0, y0, x1, y1, img_bojid])
    return page_result


def get_image_info(doc: fitz.Document, page_width_pts, page_height_pts) -> list:
    """
    返回每个页面里的图片的四元组，每个页面多个图片。
    :param doc:
    :return:
    """
    # 使用 Counter 计数 img_bojid 的出现次数
    img_bojid_counter = Counter(img[0] for page in doc for img in page.get_images())
    # 找出出现次数超过 len(doc) 半数的 img_bojid

    junk_limit = max(len(doc) * 0.5, junk_limit_min)  # 对一些页数比较少的进行豁免

    junk_img_bojids = [img_bojid for img_bojid, count in img_bojid_counter.items() if count >= junk_limit]

    #todo 加个判断，用前十页就行，这些垃圾图片需要满足两个条件，不止出现的次数要足够多，而且图片占书页面积的比例要足够大，且图与图大小都差不多
    #有两种扫描版，一种文字版，这里可能会有误判
    #扫描版1：每页都有所有扫描页图片，特点是图占比大，每页展示1张
    #扫描版2，每页存储的扫描页图片数量递增，特点是图占比大，每页展示1张，需要清空junklist跑前50页图片信息用于分类判断
    #文字版1.每页存储所有图片，特点是图片占页面比例不大，每页展示可能为0也可能不止1张 这种pdf需要拿前10页抽样检测img大小和个数，如果符合需要清空junklist
    imgs_len_list = [len(page.get_images()) for page in doc]

    special_limit_pages = 10

    # 统一用前十页结果做判断
    result = []
    break_loop = False
    for i, page in enumerate(doc):
        if break_loop:
            break
        if i >= special_limit_pages:
            break
        page_result = process_image(page)  # 这里不传junk_img_bojids，拿前十页所有图片信息用于后续分析
        result.append(page_result)
        for item in result:
            if not any(item):  # 如果任何一页没有图片，说明是个文字版，需要判断是否为特殊文字版
                if max(imgs_len_list) == min(imgs_len_list) and max(
                        imgs_len_list) >= junk_limit_min:  # 如果是特殊文字版，就把junklist置空并break
                    junk_img_bojids = []
                else:  # 不是特殊文字版，是个普通文字版，但是存在垃圾图片，不置空junklist
                    pass
                break_loop = True
                break
    if not break_loop:
        # 获取前80%的元素
        top_eighty_percent = get_top_percent_list(imgs_len_list, 0.8)
        # 检查前80%的元素是否都相等
        if len(set(top_eighty_percent)) == 1 and max(imgs_len_list) >= junk_limit_min:

            # # 如果前10页跑完都有图，根据每页图片数量是否相等判断是否需要清除junklist
            # if max(imgs_len_list) == min(imgs_len_list) and max(imgs_len_list) >= junk_limit_min:

            #前10页都有图，且每页数量一致，需要检测图片大小占页面的比例判断是否需要清除junklist
            max_image_area_per_page = calculate_max_image_area_per_page(result, page_width_pts, page_height_pts)
            if len(max_image_area_per_page) < 0.8 * special_limit_pages:  # 前10页不全是大图，说明可能是个文字版pdf，把垃圾图片list置空
                junk_img_bojids = []
            else:  # 前10页都有图，而且80%都是大图，且每页图片数量一致并都很多，说明是扫描版1，不需要清空junklist
                pass
        else:  # 每页图片数量不一致，需要清掉junklist全量跑前50页图片
            junk_img_bojids = []

    #正式进入取前50页图片的信息流程
    result = []
    for i, page in enumerate(doc):
        if i >= scan_max_page:
            break
        page_result = process_image(page, junk_img_bojids)
        # logger.info(f"page {i} img_len: {len(page_result)}")
        result.append(page_result)

    return result, junk_img_bojids


def get_pdf_page_size_pts(doc: fitz.Document):
    page_cnt = len(doc)
    l: int = min(page_cnt, 50)
    #把所有宽度和高度塞到两个list 分别取中位数（中间遇到了个在纵页里塞横页的pdf，导致宽高互换了）
    page_width_list = []
    page_height_list = []
    for i in range(l):
        page = doc[i]
        page_rect = page.rect
        page_width_list.append(page_rect.width)
        page_height_list.append(page_rect.height)

    page_width_list.sort()
    page_height_list.sort()

    median_width = page_width_list[len(page_width_list) // 2]
    median_height = page_height_list[len(page_height_list) // 2]

    return median_width, median_height


def get_pdf_textlen_per_page(doc: fitz.Document):
    text_len_lst = []
    for page in doc:
        # 拿包含img和text的所有blocks
        # text_block = page.get_text("blocks")
        # 拿所有text的blocks
        # text_block = page.get_text("words")
        # text_block_len = sum([len(t[4]) for t in text_block])
        #拿所有text的str
        text_block = page.get_text("text")
        text_block_len = len(text_block)
        # logger.info(f"page {page.number} text_block_len: {text_block_len}")
        text_len_lst.append(text_block_len)

    return text_len_lst


def get_pdf_text_layout_per_page(doc: fitz.Document):
    """
    根据PDF文档的每一页文本布局，判断该页的文本布局是横向、纵向还是未知。

    Args:
        doc (fitz.Document): PDF文档对象。

    Returns:
        List[str]: 每一页的文本布局（横向、纵向、未知）。

    """
    text_layout_list = []

    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break
        # 创建每一页的纵向和横向的文本行数计数器
        vertical_count = 0
        horizontal_count = 0
        text_dict = page.get_text("dict")
        if "blocks" in text_dict:
            for block in text_dict["blocks"]:
                if 'lines' in block:
                    for line in block["lines"]:
                        # 获取line的bbox顶点坐标
                        x0, y0, x1, y1 = line['bbox']
                        # 计算bbox的宽高
                        width = x1 - x0
                        height = y1 - y0
                        # 计算bbox的面积
                        area = width * height
                        font_sizes = []
                        for span in line['spans']:
                            if 'size' in span:
                                font_sizes.append(span['size'])
                        if len(font_sizes) > 0:
                            average_font_size = sum(font_sizes) / len(font_sizes)
                        else:
                            average_font_size = 10  # 有的line拿不到font_size，先定一个阈值100
                        if area <= average_font_size ** 2:  # 判断bbox的面积是否小于平均字体大小的平方,单字无法计算是横向还是纵向
                            continue
                        else:
                            if 'wmode' in line:  # 通过wmode判断文本方向
                                if line['wmode'] == 1:  # 判断是否为竖向文本
                                    vertical_count += 1
                                elif line['wmode'] == 0:  # 判断是否为横向文本
                                    horizontal_count += 1
                        #     if 'dir' in line:  # 通过旋转角度计算判断文本方向
                        #         # 获取行的 "dir" 值
                        #         dir_value = line['dir']
                        #         cosine, sine = dir_value
                        #         # 计算角度
                        #         angle = math.degrees(math.acos(cosine))
                        #
                        #         # 判断是否为横向文本
                        #         if abs(angle - 0) < 0.01 or abs(angle - 180) < 0.01:
                        #             # line_text = ' '.join(span['text'] for span in line['spans'])
                        #             # print('This line is horizontal:', line_text)
                        #             horizontal_count += 1
                        #         # 判断是否为纵向文本
                        #         elif abs(angle - 90) < 0.01 or abs(angle - 270) < 0.01:
                        #             # line_text = ' '.join(span['text'] for span in line['spans'])
                        #             # print('This line is vertical:', line_text)
                        #             vertical_count += 1
        # print(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")
        # 判断每一页的文本布局
        if vertical_count == 0 and horizontal_count == 0:  # 该页没有文本，无法判断
            text_layout_list.append("unknow")
            continue
        else:
            if vertical_count > horizontal_count:  # 该页的文本纵向行数大于横向的
                text_layout_list.append("vertical")
            else:  # 该页的文本横向行数大于纵向的
                text_layout_list.append("horizontal")
        # logger.info(f"page_id: {page_id}, vertical_count: {vertical_count}, horizontal_count: {horizontal_count}")
    return text_layout_list


'''定义一个自定义异常用来抛出单页svg太多的pdf'''


class PageSvgsTooManyError(Exception):
    def __init__(self, message="Page SVGs are too many"):
        self.message = message
        super().__init__(self.message)


def get_svgs_per_page(doc: fitz.Document):
    svgs_len_list = []
    for page_id, page in enumerate(doc):
        # svgs = page.get_drawings()
        svgs = page.get_cdrawings()  # 切换成get_cdrawings，效率更高
        len_svgs = len(svgs)
        if len_svgs >= 3000:
            raise PageSvgsTooManyError()
        else:
            svgs_len_list.append(len_svgs)
        # logger.info(f"page_id: {page_id}, svgs_len: {len(svgs)}")
    return svgs_len_list


def get_imgs_per_page(doc: fitz.Document):
    imgs_len_list = []
    for page_id, page in enumerate(doc):
        imgs = page.get_images()
        imgs_len_list.append(len(imgs))
        # logger.info(f"page_id: {page}, imgs_len: {len(imgs)}")

    return imgs_len_list


def get_language(doc: fitz.Document):
    """
    获取PDF文档的语言。
    Args:
        doc (fitz.Document): PDF文档对象。
    Returns:
        str: 文档语言，如 "en-US"。
    """
    language_lst = []
    for page_id, page in enumerate(doc):
        if page_id >= scan_max_page:
            break
        # 拿所有text的str
        text_block = page.get_text("text")
        page_language = detect_lang(text_block)
        language_lst.append(page_language)

        # logger.info(f"page_id: {page_id}, page_language: {page_language}")

    # 统计text_language_list中每种语言的个数
    count_dict = Counter(language_lst)
    # 输出text_language_list中出现的次数最多的语言
    language = max(count_dict, key=count_dict.get)
    return language


def check_invalid_chars(pdf_bytes):
    """
    乱码检测
    """
    return detect_invalid_chars(pdf_bytes)


def pdf_meta_scan(pdf_bytes: bytes):
    """
    :param s3_pdf_path:
    :param pdf_bytes: pdf文件的二进制数据
    几个维度来评价：是否加密，是否需要密码，纸张大小，总页数，是否文字可提取
    """
    doc = fitz.open("pdf", pdf_bytes)
    is_needs_password = doc.needs_pass
    is_encrypted = doc.is_encrypted
    total_page = len(doc)
    if total_page == 0:
        logger.warning(f"drop this pdf, drop_reason: {DropReason.EMPTY_PDF}")
        result = {"_need_drop": True, "_drop_reason": DropReason.EMPTY_PDF}
        return result
    else:
        page_width_pts, page_height_pts = get_pdf_page_size_pts(doc)
        # logger.info(f"page_width_pts: {page_width_pts}, page_height_pts: {page_height_pts}")

        # svgs_per_page = get_svgs_per_page(doc)
        # logger.info(f"svgs_per_page: {svgs_per_page}")
        imgs_per_page = get_imgs_per_page(doc)
        # logger.info(f"imgs_per_page: {imgs_per_page}")

        image_info_per_page, junk_img_bojids = get_image_info(doc, page_width_pts, page_height_pts)
        # logger.info(f"image_info_per_page: {image_info_per_page}, junk_img_bojids: {junk_img_bojids}")
        text_len_per_page = get_pdf_textlen_per_page(doc)
        # logger.info(f"text_len_per_page: {text_len_per_page}")
        text_layout_per_page = get_pdf_text_layout_per_page(doc)
        # logger.info(f"text_layout_per_page: {text_layout_per_page}")
        text_language = get_language(doc)
        # logger.info(f"text_language: {text_language}")
        invalid_chars = check_invalid_chars(pdf_bytes)
        # logger.info(f"invalid_chars: {invalid_chars}")

        # 最后输出一条json
        res = {
            "is_needs_password": is_needs_password,
            "is_encrypted": is_encrypted,
            "total_page": total_page,
            "page_width_pts": int(page_width_pts),
            "page_height_pts": int(page_height_pts),
            "image_info_per_page": image_info_per_page,
            "text_len_per_page": text_len_per_page,
            "text_layout_per_page": text_layout_per_page,
            "text_language": text_language,
            # "svgs_per_page": svgs_per_page,
            "imgs_per_page": imgs_per_page,  # 增加每页img数量list
            "junk_img_bojids": junk_img_bojids,  # 增加垃圾图片的bojid list
            "invalid_chars": invalid_chars,
            "metadata": doc.metadata
        }
        # logger.info(json.dumps(res, ensure_ascii=False))
        return res


@click.command()
@click.option('--s3-pdf-path', help='s3上pdf文件的路径')
@click.option('--s3-profile', help='s3上的profile')
def main(s3_pdf_path: str, s3_profile: str):
    """

    """
    try:
        file_content = read_file(s3_pdf_path, s3_profile)
        pdf_meta_scan(file_content)
    except Exception as e:
        print(f"ERROR: {s3_pdf_path}, {e}", file=sys.stderr)
        logger.exception(e)


if __name__ == '__main__':
    main()
    # "D:\project/20231108code-clean\pdf_cost_time\竖排例子\净空法师-大乘无量寿.pdf"
    # "D:\project/20231108code-clean\pdf_cost_time\竖排例子\三国演义_繁体竖排版.pdf"
    # "D:\project/20231108code-clean\pdf_cost_time\scihub\scihub_86800000\libgen.scimag86880000-86880999.zip_10.1021/acsami.1c03109.s002.pdf"
    # "D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_18600000/libgen.scimag18645000-18645999.zip_10.1021/om3006239.pdf"
    # file_content = read_file("D:/project/20231108code-clean/pdf_cost_time/scihub/scihub_31000000/libgen.scimag31098000-31098999.zip_10.1109/isit.2006.261791.pdf","")
    # file_content = read_file("D:\project/20231108code-clean\pdf_cost_time\竖排例子\净空法师_大乘无量寿.pdf","")
    # doc = fitz.open("pdf", file_content)
    # text_layout_lst = get_pdf_text_layout_per_page(doc)
    # print(text_layout_lst)
