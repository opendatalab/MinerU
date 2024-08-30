# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import time
import zipfile
from pathlib import Path
import re

import gradio as gr
from loguru import logger

from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, prepare_env


def read_fn(path):
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)


def parse_pdf(doc_path, output_dir, end_page_id):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)
        parse_method = "auto"
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages):
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1)
    archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info("压缩成功")
    else:
        logger.error("压缩失败")
    md_path = os.path.join(local_md_dir, file_name + ".md")
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")

    return md_content, txt_content, archive_zip_path, show_pdf(new_pdf_path)


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                  f'width="100%" height="1000" type="application/pdf">'
    return pdf_display


latex_delimiters = [{"left": "$$", "right": "$$", "display": True},
                    {"left": '$', "right": '$', "display": False}]

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                file = gr.File(label="请上传pdf", file_types=[".pdf"])
                max_pages = gr.Slider(1, 10, 5, step=1, label="最大转换页数")
                with gr.Row() as bu_flow:
                    change_bu = gr.Button("转换")
                    clear_bu = gr.ClearButton([file, max_pages], value="清除")
                gr.Markdown(value="### PDF预览")
                pdf_show = gr.HTML(label="PDF预览")

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label="Markdown识别结果文件", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Markdown渲染"):
                        md = gr.Markdown(label="Markdown渲染", height=1100, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    with gr.Tab("Markdown文本"):
                        md_text = gr.TextArea(lines=55, show_copy_button=True)
        file.upload(fn=show_pdf, inputs=file, outputs=pdf_show)
        change_bu.click(fn=to_markdown, inputs=[file, max_pages], outputs=[md, md_text, output_file, pdf_show])
        clear_bu.add([md, pdf_show, md_text, output_file])

    demo.launch()
