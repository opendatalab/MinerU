# coding=utf8
import sys
import time
from urllib.parse import quote

from magic_pdf.libs.commons import read_file, join_path, parse_bucket_key, formatted_time
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.dict2md.mkcontent import mk_nlp_markdown
from magic_pdf.pdf_parse_by_model import parse_pdf_by_model
from magic_pdf.filter.pdf_classify_by_type import classify
from magic_pdf.filter.pdf_meta_scan import pdf_meta_scan
from loguru import logger

from app.common.s3 import get_s3_config, get_s3_client


def exception_handler(jso: dict, e):
    logger.exception(e)
    jso['need_drop'] = True
    jso['drop_reason'] = DropReason.Exception
    jso['exception'] = f"ERROR: {e}"
    return jso


def get_data_type(jso: dict):
    data_type = jso.get('data_type')
    if data_type is None:
        data_type = jso.get('file_type')
    return data_type


def get_bookid(jso: dict):
    book_id = jso.get('bookid')
    if book_id is None:
        book_id = jso.get('original_file_id')
    return book_id


def get_data_source(jso: dict):
    data_source = jso.get('data_source')
    if data_source is None:
        data_source = jso.get('file_source')
    return data_source


def meta_scan(jso: dict, doc_layout_check=True) -> dict:
    s3_pdf_path = jso.get('file_location')
    s3_config = get_s3_config(s3_pdf_path)
    if doc_layout_check:
        if 'doc_layout_result' not in jso:  # 检测json中是存在模型数据，如果没有则需要跳过该pdf
            jso['need_drop'] = True
            jso['drop_reason'] = DropReason.MISS_DOC_LAYOUT_RESULT
            return jso
    try:
        data_source = get_data_source(jso)
        file_id = jso.get('file_id')
        book_name = data_source + "/" + file_id

        # 首页存在超量drawing问题
        # special_pdf_list = ['zlib/zlib_21822650']
        # if book_name in special_pdf_list:
        #     jso['need_drop'] = True
        #     jso['drop_reason'] = DropReason.SPECIAL_PDF
        #     return jso

        start_time = time.time()  # 记录开始时间
        logger.info(f"book_name is:{book_name},start_time is:{formatted_time(start_time)}", file=sys.stderr)
        file_content = read_file(s3_pdf_path, s3_config)
        read_file_time = int(time.time() - start_time)  # 计算执行时间

        start_time = time.time()  # 记录开始时间
        res = pdf_meta_scan(s3_pdf_path, file_content)
        if res.get('need_drop', False):  # 如果返回的字典里有need_drop，则提取drop_reason并跳过本次解析
            jso['need_drop'] = True
            jso['drop_reason'] = res["drop_reason"]
        else:  # 正常返回
            jso['pdf_meta'] = res
            jso['content'] = ""
            jso['remark'] = ""
            jso['data_url'] = ""
        end_time = time.time()  # 记录结束时间
        meta_scan_time = int(end_time - start_time)  # 计算执行时间
        logger.info(f"book_name is:{book_name},end_time is:{formatted_time(end_time)},read_file_time is:{read_file_time},meta_scan_time is:{meta_scan_time}", file=sys.stderr)
        jso['read_file_time'] = read_file_time
        jso['meta_scan_time'] = meta_scan_time
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def classify_by_type(jso: dict, debug_mode=False) -> dict:
    #检测debug开关
    if debug_mode:
        pass
    else:# 如果debug没开，则检测是否有needdrop字段
        if jso.get('need_drop', False):
            return jso
    # 开始正式逻辑
    try:
        pdf_meta = jso.get('pdf_meta')
        data_source = get_data_source(jso)
        file_id = jso.get('file_id')
        book_name = data_source + "/" + file_id
        total_page = pdf_meta["total_page"]
        page_width = pdf_meta["page_width_pts"]
        page_height = pdf_meta["page_height_pts"]
        img_sz_list = pdf_meta["image_info_per_page"]
        img_num_list = pdf_meta['imgs_per_page']
        text_len_list = pdf_meta['text_len_per_page']
        text_layout_list = pdf_meta['text_layout_per_page']
        text_language = pdf_meta['text_language']
        # allow_language = ['zh', 'en']  # 允许的语言,目前只允许简中和英文的

        # if text_language not in allow_language:  # 如果语言不在允许的语言中，则drop
        #     jso['need_drop'] = True
        #     jso['drop_reason'] = DropReason.NOT_ALLOW_LANGUAGE
        #     return jso
        pdf_path = pdf_meta['pdf_path']
        is_encrypted = pdf_meta['is_encrypted']
        is_needs_password = pdf_meta['is_needs_password']
        if is_encrypted or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
            jso['need_drop'] = True
            jso['drop_reason'] = DropReason.ENCRYPTED
        else:
            start_time = time.time()  # 记录开始时间
            is_text_pdf, results = classify(pdf_path, total_page, page_width, page_height, img_sz_list, text_len_list, img_num_list, text_layout_list)
            classify_time = int(time.time() - start_time)  # 计算执行时间
            if is_text_pdf:
                pdf_meta['is_text_pdf'] = is_text_pdf
                jso['pdf_meta'] = pdf_meta
                jso['classify_time'] = classify_time
                # print(json.dumps(pdf_meta, ensure_ascii=False))

                allow_language = ['zh', 'en']  # 允许的语言,目前只允许简中和英文的
                if text_language not in allow_language:  # 如果语言不在允许的语言中，则drop
                    jso['need_drop'] = True
                    jso['drop_reason'] = DropReason.NOT_ALLOW_LANGUAGE
                    return jso
            else:
                # 先不drop
                pdf_meta['is_text_pdf'] = is_text_pdf
                jso['pdf_meta'] = pdf_meta
                jso['classify_time'] = classify_time
                jso['need_drop'] = True
                jso['drop_reason'] = DropReason.NOT_IS_TEXT_PDF
                extra_info = {"classify_rules": []}
                for condition, result in results.items():
                    if not result:
                        extra_info["classify_rules"].append(condition)
                jso['extra_info'] = extra_info

    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def save_tables_to_s3(jso: dict, debug_mode=False) -> dict:

    if debug_mode:
        pass
    else:# 如果debug没开，则检测是否有needdrop字段
        if jso.get('need_drop', False):
            logger.info(f"book_name is:{get_data_source(jso)}/{jso['file_id']} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        data_source = get_data_source(jso)
        file_id = jso.get('file_id')
        book_name = data_source + "/" + file_id
        title = jso.get('title')
        url_encode_title = quote(title, safe='')
        if data_source != 'scihub':
            return jso
        pdf_intermediate_dict = jso['pdf_intermediate_dict']
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        i = 0
        for page in pdf_intermediate_dict.values():
            if page.get('tables'):
                if len(page['tables']) > 0:
                    j = 0
                    for table in page['tables']:
                        if debug_mode:
                            image_path = join_path("s3://mllm-raw-media/pdf2md_img/", book_name, table['image_path'])
                        else:
                            image_path = join_path("s3://mllm-raw-media/pdf2md_img/", table['image_path'])

                        if image_path.endswith('.jpg'):
                            j += 1
                            s3_client = get_s3_client(image_path)
                            bucket_name, bucket_key = parse_bucket_key(image_path)
                            # 通过s3_client获取图片到内存
                            image_bytes = s3_client.get_object(Bucket=bucket_name, Key=bucket_key)['Body'].read()
                            # 保存图片到新的位置
                            if debug_mode:
                                new_image_path = join_path("s3://mllm-raw-media/pdf2md_img/table_new/", url_encode_title + "_" + table['image_path'].lstrip('tables/'))
                            else:
                                new_image_path = join_path("s3://mllm-raw-media/pdf2md_img/table_new/", url_encode_title + f"_page{i}_{j}.jpg")

                            logger.info(new_image_path, file=sys.stderr)
                            bucket_name, bucket_key = parse_bucket_key(new_image_path)
                            s3_client.put_object(Bucket=bucket_name, Key=bucket_key, Body=image_bytes)
                        else:
                            continue
            i += 1

        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def drop_needdrop_pdf(jso: dict) -> dict:
    if jso.get('need_drop', False):
        logger.info(f"book_name is:{get_data_source(jso)}/{jso['file_id']} need drop", file=sys.stderr)
        jso["dropped"] = True
    return jso


def pdf_intermediate_dict_to_markdown(jso: dict, debug_mode=False) -> dict:

    if debug_mode:
        pass
    else:# 如果debug没开，则检测是否有needdrop字段
        if jso.get('need_drop', False):
            book_name = join_path(get_data_source(jso), jso['file_id'])
            logger.info(f"book_name is:{book_name} need drop", file=sys.stderr)
            jso["dropped"] = True
            return jso
    try:
        pdf_intermediate_dict = jso['pdf_intermediate_dict']
        # 将 pdf_intermediate_dict 解压
        pdf_intermediate_dict = JsonCompressor.decompress_json(pdf_intermediate_dict)
        markdown_content = mk_nlp_markdown(pdf_intermediate_dict)
        jso["content"] = markdown_content
        logger.info(f"book_name is:{get_data_source(jso)}/{jso['file_id']},markdown content length is {len(markdown_content)}", file=sys.stderr)
        # 把无用的信息清空
        jso["doc_layout_result"] = ""
        jso["pdf_intermediate_dict"] = ""
        jso["pdf_meta"] = ""
    except Exception as e:
        jso = exception_handler(jso, e)
    return jso


def parse_pdf(jso: dict, start_page_id=0, debug_mode=False) -> dict:
    #检测debug开关
    if debug_mode:
        pass
    else:# 如果debug没开，则检测是否有needdrop字段
        if jso.get('need_drop', False):
            return jso
    # 开始正式逻辑
    s3_pdf_path = jso.get('file_location')
    s3_config = get_s3_config(s3_pdf_path)
    model_output_json_list = jso.get('doc_layout_result')
    data_source = get_data_source(jso)
    file_id = jso.get('file_id')
    book_name = data_source + "/" + file_id

    # 1.23.22已修复
    # if debug_mode:
    #     pass
    # else:
    #     if book_name == "zlib/zlib_21929367":
    #         jso['need_drop'] = True
    #         jso['drop_reason'] = DropReason.SPECIAL_PDF
    #         return jso

    junk_img_bojids = jso['pdf_meta']['junk_img_bojids']
    # total_page = jso['pdf_meta']['total_page']

    # 增加检测 max_svgs 数量的检测逻辑，如果 max_svgs 超过3000则drop
    svgs_per_page_list = jso['pdf_meta']['svgs_per_page']
    max_svgs = max(svgs_per_page_list)
    if max_svgs > 3000:
        jso['need_drop'] = True
        jso['drop_reason'] = DropReason.HIGH_COMPUTATIONAL_lOAD_BY_SVGS
    # elif total_page > 1000:
    #     jso['need_drop'] = True
    #     jso['drop_reason'] = DropReason.HIGH_COMPUTATIONAL_lOAD_BY_TOTAL_PAGES
    else:
        try:
            save_path = "s3://mllm-raw-media/pdf2md_img/"
            image_s3_config = get_s3_config(save_path)
            start_time = time.time()  # 记录开始时间
            # 先打印一下book_name和解析开始的时间
            logger.info(f"book_name is:{book_name},start_time is:{formatted_time(start_time)}", file=sys.stderr)
            pdf_info_dict = parse_pdf_by_model(s3_pdf_path, s3_config, model_output_json_list, save_path,
                                                  book_name, pdf_model_profile=None,
                                                  image_s3_config=image_s3_config,
                                                  start_page_id=start_page_id, junk_img_bojids=junk_img_bojids,
                                                  debug_mode=debug_mode)
            if pdf_info_dict.get('need_drop', False):  # 如果返回的字典里有need_drop，则提取drop_reason并跳过本次解析
                jso['need_drop'] = True
                jso['drop_reason'] = pdf_info_dict["drop_reason"]
            else:  # 正常返回，将 pdf_info_dict 压缩并存储
                pdf_info_dict = JsonCompressor.compress_json(pdf_info_dict)
                jso['pdf_intermediate_dict'] = pdf_info_dict
            end_time = time.time()  # 记录完成时间
            parse_time = int(end_time - start_time)  # 计算执行时间
            # 解析完成后打印一下book_name和耗时
            logger.info(f"book_name is:{book_name},end_time is:{formatted_time(end_time)},cost_time is:{parse_time}", file=sys.stderr)
            jso['parse_time'] = parse_time
        except Exception as e:
            jso = exception_handler(jso, e)
    return jso


def ocr_parse_pdf(jso: dict, start_page_id=0, debug_mode=False) -> dict:
    pass


if __name__ == "__main__":
    pass
