import os

import pytest

from magic_pdf.filter.pdf_classify_by_type import classify_by_area, classify_by_text_len, classify_by_avg_words, \
    classify_by_img_num, classify_by_text_layout, classify_by_img_narrow_strips
from magic_pdf.filter.pdf_meta_scan import get_pdf_page_size_pts, get_pdf_textlen_per_page, get_imgs_per_page
from test_commons import get_docs_from_test_pdf, get_test_json_data

# 获取当前目录
current_directory = os.path.dirname(os.path.abspath(__file__))

'''
根据图片尺寸占页面面积的比例，判断是否为扫描版
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_area",
                         [
                             ("the_eye/the_eye_cdn_00391653", True),  # 特殊文字版1.每页存储所有图片，特点是图片占页面比例不大，每页展示可能为0也可能不止1张
                             ("scihub/scihub_08400000/libgen.scimag08489000-08489999.zip_10.1016/0370-1573(90)90070-i", False),  # 特殊扫描版2，每页存储的扫描页图片数量递增，特点是图占比大，每页展示1张
                             ("zlib/zlib_17216416", False),  # 特殊扫描版3，有的页面是一整张大图，有的页面是通过一条条小图拼起来的，检测图片占比之前需要先按规则把小图拼成大图
                             ("the_eye/the_eye_wtl_00023799", False),  # 特殊扫描版4，每一页都是一张张小图拼出来的，检测图片占比之前需要先按规则把小图拼成大图
                             ("the_eye/the_eye_cdn_00328381", False),  # 特殊扫描版5，每一页都是一张张小图拼出来的，存在多个小图多次重复使用情况，检测图片占比之前需要先按规则把小图拼成大图
                             ("scihub/scihub_25800000/libgen.scimag25889000-25889999.zip_10.2307/4153991", False),  # 特殊扫描版6，只有三页，其中两页是扫描版
                             ("scanned_detection/llm-raw-scihub-o.O-0584-8539%2891%2980165-f", False),  # 特殊扫描版7，只有一页且由小图拼成大图
                             ("scanned_detection/llm-raw-scihub-o.O-bf01427123", False),  # 特殊扫描版8，只有3页且全是大图扫描版
                             ("scihub/scihub_41200000/libgen.scimag41253000-41253999.zip_10.1080/00222938709460256", False),  # 特殊扫描版12，头两页文字版且有一页没图片，后面扫描版11页
                             ("scihub/scihub_37000000/libgen.scimag37068000-37068999.zip_10.1080/0015587X.1936.9718622", False)  # 特殊扫描版13，头两页文字版且有一页没图片，后面扫描版3页
                         ])
def test_classify_by_area(book_name, expected_bool_classify_by_area):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    docs = get_docs_from_test_pdf(book_name)
    median_width, median_height = get_pdf_page_size_pts(docs)
    page_width = int(median_width)
    page_height = int(median_height)
    img_sz_list = test_data[book_name]["expected_image_info"]
    total_page = len(docs)
    text_len_list = get_pdf_textlen_per_page(docs)
    bool_classify_by_area = classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list)
    # assert bool_classify_by_area == expected_bool_classify_by_area


'''
广义上的文字版检测，任何一页大于100字，都认为为文字版
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_text_len",
                         [
                             ("scihub/scihub_67200000/libgen.scimag67237000-67237999.zip_10.1515/crpm-2017-0020", True),  # 文字版，少于50页
                             ("scihub/scihub_83300000/libgen.scimag83306000-83306999.zip_10.1007/978-3-658-30153-8", True),  # 文字版，多于50页
                             ("zhongwenzaixian/zhongwenzaixian_65771414", False),  # 完全无字的宣传册
                         ])
def test_classify_by_text_len(book_name, expected_bool_classify_by_text_len):
    docs = get_docs_from_test_pdf(book_name)
    text_len_list = get_pdf_textlen_per_page(docs)
    total_page = len(docs)
    bool_classify_by_text_len = classify_by_text_len(text_len_list, total_page)
    # assert bool_classify_by_text_len == expected_bool_classify_by_text_len


'''
狭义上的文字版检测，需要平均每页字数大于200字
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_avg_words",
                         [
                             ("zlib/zlib_21207669", False),  # 扫描版，书末尾几页有大纲文字
                             ("zlib/zlib_19012845", False),  # 扫描版，好几本扫描书的集合，每本书末尾有一页文字页
                             ("scihub/scihub_67200000/libgen.scimag67237000-67237999.zip_10.1515/crpm-2017-0020", True),# 正常文字版
                             ("zhongwenzaixian/zhongwenzaixian_65771414", False),  # 宣传册
                             ("zhongwenzaixian/zhongwenzaixian_351879", False),  # 图解书/无字or少字
                             ("zhongwenzaixian/zhongwenzaixian_61357496_pdfvector", False),  # 书法集
                             ("zhongwenzaixian/zhongwenzaixian_63684541", False),  # 设计图
                             ("zhongwenzaixian/zhongwenzaixian_61525978", False),  # 绘本
                             ("zhongwenzaixian/zhongwenzaixian_63679729", False),  # 摄影集

                         ])
def test_classify_by_avg_words(book_name, expected_bool_classify_by_avg_words):
    docs = get_docs_from_test_pdf(book_name)
    text_len_list = get_pdf_textlen_per_page(docs)
    bool_classify_by_avg_words = classify_by_avg_words(text_len_list)
    # assert bool_classify_by_avg_words == expected_bool_classify_by_avg_words


'''
这个规则只针对特殊扫描版1，因为扫描版1的图片信息都由于junk_list的原因被舍弃了，只能通过图片数量来判断
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_img_num",
                         [
                             ("zlib/zlib_21370453", False),  # 特殊扫描版1，每页都有所有扫描页图片，特点是图占比大，每页展示1至n张
                             ("zlib/zlib_22115997", False),  # 特殊扫描版2，类似特1，但是每页数量不完全相等
                             ("zlib/zlib_21814957", False),  # 特殊扫描版3，类似特1，但是每页数量不完全相等
                             ("zlib/zlib_21814955", False),  # 特殊扫描版4，类似特1，但是每页数量不完全相等
                         ])
def test_classify_by_img_num(book_name, expected_bool_classify_by_img_num):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    docs = get_docs_from_test_pdf(book_name)
    img_num_list = get_imgs_per_page(docs)
    img_sz_list = test_data[book_name]["expected_image_info"]
    bool_classify_by_img_num = classify_by_img_num(img_sz_list, img_num_list)
    # assert bool_classify_by_img_num == expected_bool_classify_by_img_num


'''
排除纵向排版的pdf
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_text_layout",
                         [
                             ("vertical_detection/三国演义_繁体竖排版", False),  # 竖排版本1
                             ("vertical_detection/净空法师_大乘无量寿", False),  # 竖排版本2
                             ("vertical_detection/om3006239", True),  # 横排版本1
                             ("vertical_detection/isit.2006.261791", True),  # 横排版本2
                         ])
def test_classify_by_text_layout(book_name, expected_bool_classify_by_text_layout):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    text_layout_per_page = test_data[book_name]["expected_text_layout"]
    bool_classify_by_text_layout = classify_by_text_layout(text_layout_per_page)
    # assert bool_classify_by_text_layout == expected_bool_classify_by_text_layout


'''
通过检测页面是否由多个窄长条图像组成，来过滤特殊的扫描版
这个规则只对窄长条组成的pdf进行识别，而不会识别常规的大图扫描pdf
'''
@pytest.mark.parametrize("book_name, expected_bool_classify_by_img_narrow_strips",
                         [
                             ("scihub/scihub_25900000/libgen.scimag25991000-25991999.zip_10.2307/40066695", False),  # 特殊扫描版
                             ("the_eye/the_eye_wtl_00023799", False),  # 特殊扫描版4，每一页都是一张张小图拼出来的，检测图片占比之前需要先按规则把小图拼成大图
                             ("the_eye/the_eye_cdn_00328381", False),  # 特殊扫描版5，每一页都是一张张小图拼出来的，存在多个小图多次重复使用情况，检测图片占比之前需要先按规则把小图拼成大图
                             ("scanned_detection/llm-raw-scihub-o.O-0584-8539%2891%2980165-f", False),  # 特殊扫描版7，只有一页且由小图拼成大图
                             ("scihub/scihub_25800000/libgen.scimag25889000-25889999.zip_10.2307/4153991", True),  # 特殊扫描版6，只有三页，其中两页是扫描版
                             ("scanned_detection/llm-raw-scihub-o.O-bf01427123", True),  # 特殊扫描版8，只有3页且全是大图扫描版
                             ("scihub/scihub_53700000/libgen.scimag53724000-53724999.zip_10.1097/00129191-200509000-00018", True),  # 特殊文本版，有一长条，但是只有一条
                         ])
def test_classify_by_img_narrow_strips(book_name, expected_bool_classify_by_img_narrow_strips):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    img_sz_list = test_data[book_name]["expected_image_info"]
    docs = get_docs_from_test_pdf(book_name)
    median_width, median_height = get_pdf_page_size_pts(docs)
    page_width = int(median_width)
    page_height = int(median_height)
    bool_classify_by_img_narrow_strips = classify_by_img_narrow_strips(page_width, page_height, img_sz_list)
    # assert bool_classify_by_img_narrow_strips == expected_bool_classify_by_img_narrow_strips