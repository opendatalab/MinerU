#!/usr/bin/env python3
"""修复的MinerU进程池 - 简化版本 (完整修改版)"""

import os
import time
import json
import zipfile
import traceback
import shutil # 新增导入，用于目录操作

from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
# ----------------------------------------------------------------------------------

def get_pdf_page_count(pdf_path):
    """使用fitz获取PDF页数"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        # print(f"Error getting page count for {pdf_path}: {e}")
        return 0

# --- 移植第一个脚本中的 _process_output 函数 ---
def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=False
):
    """处理输出文件 - VLM模式"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    # 这里的 image_dir 应该是相对路径，用于在 markdown 中引用
    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )
# --- _process_output 函数结束 ---


def infer_one_pdf(pdf_file_path, lang="en", backend="vlm-engine", gpu_memory_utilization=0.5, mm_processor_cache_gb=0, split_pdf_chunk_size=0):
    """PDF推理处理 - VLM后端"""
    temp_root_dir = None
    try:
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
        from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
        # 尝试导入 vlm_doc_analyze_split, 如果没有则忽略
        try:
            from mineru.backend.vlm.vlm_analyze import doc_analyze_split as vlm_doc_analyze_split
        except ImportError:
            vlm_doc_analyze_split = None

        # --- 设置临时目录 ---
        pdf_name = os.path.basename(pdf_file_path).replace(".pdf", "")
        # 临时根目录，使用进程ID和时间戳确保唯一性
        temp_root_dir = f"/tmp/mineru_process_output/{os.getpid()}_{pdf_name}_{int(time.time() * 1000)}"
        os.makedirs(temp_root_dir, exist_ok=True)
        # 临时图片目录 (子目录)
        local_image_dir = os.path.join(temp_root_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        # 临时md/json目录 (使用临时根目录)
        local_md_dir = temp_root_dir

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        with open(pdf_file_path, 'rb') as fi:
            pdf_bytes = fi.read()

        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)

        # VLM 后端处理

        if split_pdf_chunk_size and vlm_doc_analyze_split:
            middle_json, infer_result = vlm_doc_analyze_split(new_pdf_bytes,
                                                              image_writer=image_writer,
                                                              backend=backend,
                                                              gpu_memory_utilization=gpu_memory_utilization,
                                                              mm_processor_cache_gb=mm_processor_cache_gb,
                                                              chunk_size=split_pdf_chunk_size)
        else:
            middle_json, infer_result = vlm_doc_analyze(new_pdf_bytes,
                                                        image_writer=image_writer,
                                                        backend=backend,
                                                        gpu_memory_utilization=gpu_memory_utilization,
                                                        mm_processor_cache_gb=mm_processor_cache_gb)

        pdf_info = middle_json["pdf_info"]

        # --- 调用 _process_output 生成文件到临时目录 ---
        if middle_json and pdf_info:
            # 使用VLM模式参数
            _process_output(
                pdf_info, pdf_bytes, pdf_name, local_md_dir, local_image_dir,
                md_writer,
                f_draw_layout_bbox=True,
                f_draw_span_bbox=False, # VLM不画span
                f_dump_orig_pdf=True,
                f_dump_md=True,
                f_dump_content_list=True,
                f_dump_middle_json=True,
                f_dump_model_output=True,
                f_make_md_mode=MakeMode.MM_MD,
                middle_json=middle_json,
                model_output=infer_result,
                is_pipeline=False
            )
        else:
            raise Exception("Parsing failed: middle_json or pdf_info is None.")

        # 返回临时目录路径和页数
        return {
            "temp_dir": temp_root_dir,
            "success": True
        }

    except Exception as e:
        # 清理临时目录
        if temp_root_dir and os.path.exists(temp_root_dir):
            shutil.rmtree(temp_root_dir, ignore_errors=True)

        print(f"Error in infer_one_pdf for {pdf_file_path}: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            "temp_dir": None,
            "success": False
        }


def process_one_pdf_file(pdf_path, save_dir, lang="en", max_pages_per_pdf=None , backend="vlm-engine", gpu_memory_utilization=0.5, mm_processor_cache_gb=0, split_pdf_chunk_size=0):
    """处理单个PDF文件 - 修改版本：打包所有输出文件"""
    fitz_page_count = 0
    pdf_file_name = None
    temp_root_dir = None
    result = {}

    print(f"process_one_pdf_file: {pdf_path}")

    try:
        pdf_file_name = os.path.basename(pdf_path).replace(".pdf", "")
        target_file = f"{save_dir}/{pdf_file_name}.json.zip"

        if not os.path.exists(f"{save_dir}/"):
            os.makedirs(save_dir, exist_ok=True)

        # 提前使用fitz检查页数
        if max_pages_per_pdf is not None:
            fitz_page_count = get_pdf_page_count(pdf_path)
            if fitz_page_count is not None and fitz_page_count > max_pages_per_pdf:
                print(f"PDF {pdf_path} has {fitz_page_count} pages (limit: {max_pages_per_pdf}), skipped")
                return {
                    'input_path': pdf_path,
                    'output_path': None,
                    'page_count': fitz_page_count,
                    'file_size': 0,
                    'skipped': True,
                    'reason': f'Exceeded page limit: {fitz_page_count} > {max_pages_per_pdf}'
                }
            elif fitz_page_count is None:
                print(f"Warning: Could not get page count using fitz for {pdf_path}, will proceed and use mineru page count")

        # 执行OCR推理，获取临时目录
        infer_result = infer_one_pdf(pdf_path, lang=lang, backend=backend,
                                    gpu_memory_utilization=gpu_memory_utilization,
                                    mm_processor_cache_gb=mm_processor_cache_gb,
                                    split_pdf_chunk_size=split_pdf_chunk_size)

        # 处理错误情况
        if not infer_result.get('success', False):
            # infer_one_pdf 中已经包含了 error 和 traceback
            infer_result['input_path'] = pdf_path
            infer_result['output_path'] = None
            infer_result['success'] = False
            return infer_result

        temp_root_dir = infer_result.get("temp_dir")
        mineru_page_count = infer_result.get('page_count', fitz_page_count)
        
        # --- 核心修改：打包临时目录中的所有文件 ---
        if not os.path.exists(temp_root_dir):
            raise FileNotFoundError(f"Temporary output directory not found: {temp_root_dir}")

        # 使用 zipfile 将临时目录中的所有文件打包
        with zipfile.ZipFile(target_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算文件在zip中的相对路径 (去除 temp_root_dir)
                    arcname = os.path.relpath(file_path, temp_root_dir)
                    zf.write(file_path, arcname)
        
        # --- 清理临时目录 ---
        shutil.rmtree(temp_root_dir, ignore_errors=True)

        file_size = os.path.getsize(target_file)
        print(f"Finished processing: {pdf_path} -> {target_file} ({file_size} bytes)")
        print(f"Page count: {mineru_page_count}")

        result = {
            'input_path': pdf_path,
            'output_path': target_file,
            'page_count': mineru_page_count,
            'file_size': file_size,
            'success': True
        }
    
    except Exception as e:
        # 清理临时目录
        if temp_root_dir and os.path.exists(temp_root_dir):
            shutil.rmtree(temp_root_dir, ignore_errors=True)
            
        print(f"Error processing {pdf_path}: {e}")
        traceback.print_exc()
        result = {
            'input_path': pdf_path,
            'output_path': None,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

    return result


