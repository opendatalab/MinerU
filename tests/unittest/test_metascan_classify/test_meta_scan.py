import os

import pytest
from magic_pdf.filter.pdf_meta_scan import get_pdf_page_size_pts, get_image_info, get_pdf_text_layout_per_page, get_language
from tests.test_commons import get_docs_from_test_pdf, get_test_json_data

# 获取当前目录
current_directory = os.path.dirname(os.path.abspath(__file__))

'''
获取pdf的宽与高，宽和高各用一个list，分别取中位数
'''
@pytest.mark.parametrize("book_name, expected_width, expected_height",
                         [
                             ("zlib/zlib_17058115", 795, 1002),  # pdf中最大页与最小页差异极大个例
                             ("the_eye/the_eye_wtl_00023799", 616, 785)  # 采样的前50页存在中位数大小页面横竖旋转情况
                         ])
def test_get_pdf_page_size_pts(book_name, expected_width, expected_height):
    docs = get_docs_from_test_pdf(book_name)
    median_width, median_height = get_pdf_page_size_pts(docs)

    assert int(median_width) == expected_width
    assert int(median_height) == expected_height


'''
获取pdf前50页的图片信息，为了提速，对特殊扫描版1的情况做了过滤，其余情况都正常取图片信息
'''
@pytest.mark.parametrize("book_name",
                         [
                            "zlib/zlib_21370453",  # 特殊扫描版1，每页都有所有扫描页图片，特点是图占比大，每页展示1至n张
                            "the_eye/the_eye_cdn_00391653",  # 特殊文字版1.每页存储所有图片，特点是图片占页面比例不大，每页展示可能为0也可能不止1张，这种pdf需要拿前10页抽样检测img大小和个数，如果符合需要清空junklist
                            "scihub/scihub_08400000/libgen.scimag08489000-08489999.zip_10.1016/0370-1573(90)90070-i",  # 扫描版2，每页存储的扫描页图片数量递增，特点是图占比大，每页展示1张，需要清空junklist跑前50页图片信息用于分类判断
                            "zlib/zlib_17216416",  # 特殊扫描版3，有的页面是一整张大图，有的页面是通过一条条小图拼起来的
                            "the_eye/the_eye_wtl_00023799",  # 特殊扫描版4，每一页都是一张张小图拼出来的
                            "the_eye/the_eye_cdn_00328381",  # 特殊扫描版5，每一页都是一张张小图拼出来的，但是存在多个小图多次重复使用情况
                            "scihub/scihub_25800000/libgen.scimag25889000-25889999.zip_10.2307/4153991",  # 特殊扫描版6，只有3页且其中两页是扫描页
                            "scanned_detection/llm-raw-scihub-o.O-0584-8539%2891%2980165-f",  # 特殊扫描版7，只有一页，且是一张张小图拼出来的
                            "scanned_detection/llm-raw-scihub-o.O-bf01427123",  # 特殊扫描版8，只有3页且全是大图扫描版
                            "zlib/zlib_22115997",  # 特殊扫描版9，类似特1，但是每页数量不完全相等
                            "zlib/zlib_21814957",  # 特殊扫描版10，类似特1，但是每页数量不完全相等
                            "zlib/zlib_21814955",  # 特殊扫描版11，类似特1，但是每页数量不完全相等
                            "scihub/scihub_41200000/libgen.scimag41253000-41253999.zip_10.1080/00222938709460256",  # 特殊扫描版12，头两页文字版且有一页没图片，后面扫描版11页
                            "scihub/scihub_37000000/libgen.scimag37068000-37068999.zip_10.1080/0015587X.1936.9718622"  # 特殊扫描版13，头两页文字版且有一页没图片，后面扫描版3页
                         ])
def test_get_image_info(book_name):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")
    docs = get_docs_from_test_pdf(book_name)
    page_width_pts, page_height_pts = get_pdf_page_size_pts(docs)
    image_info, junk_img_bojids = get_image_info(docs, page_width_pts, page_height_pts)

    assert image_info == test_data[book_name]["expected_image_info"]
    assert junk_img_bojids == test_data[book_name]["expected_junk_img_bojids"]


'''
获取pdf前50页的文本布局信息，输出list，每个元素为一个页面的横竖排信息
'''
@pytest.mark.parametrize("book_name",
                         [
                            "vertical_detection/三国演义_繁体竖排版",  # 竖排版本1
                            "vertical_detection/净空法师_大乘无量寿",  # 竖排版本2
                            "vertical_detection/om3006239",  # 横排版本1
                            "vertical_detection/isit.2006.261791"  # 横排版本2
                         ])
def test_get_text_layout_info(book_name):
    test_data = get_test_json_data(current_directory, "test_metascan_classify_data.json")

    docs = get_docs_from_test_pdf(book_name)
    text_layout_info = get_pdf_text_layout_per_page(docs)
    assert text_layout_info == test_data[book_name]["expected_text_layout"]


'''
获取pdf的语言信息
'''
@pytest.mark.parametrize("book_name, expected_language",
                         [
                             ("scihub/scihub_05000000/libgen.scimag05023000-05023999.zip_10.1034/j.1601-0825.2003.02933.x", "en"),  # 英文论文
                         ])
def test_get_text_language_info(book_name, expected_language):
    docs = get_docs_from_test_pdf(book_name)
    text_language = get_language(docs)
    assert text_language == expected_language
