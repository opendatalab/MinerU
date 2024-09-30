## 以下代码是可以对目录中pdf文件进行循环处理，方便使用。修改前，由于文件打开后没有关闭，造成内存报错，感谢i12345的帮助，修改了代码，提交PR，请作者完善，造福大家。

import os
import json
import copy
import glob
from loguru import logger

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True

def json_md_dump(
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content,
):
    # 写入模型结果到 model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # 写入中间结果到 middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # text文本结果写入到 conent_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    # 写入结果到 .md 文件中
    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )

def pdf_parse_main(
        pdf_path: str,
        parse_method: str = 'ocr',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
        output_dir: str = None
):
    try:
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        pdf_path_parent = os.path.dirname(pdf_path)

        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            output_path = os.path.join(pdf_path_parent, pdf_name)

        output_image_path = os.path.join(output_path, 'images')

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        # 使用 with open 自动处理文件的打开和关闭
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        if model_json_path:
            with open(model_json_path, "r", encoding="utf-8") as model_file:
                model_json = json.load(model_file)
        else:
            model_json = []

        # 执行解析步骤
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("unknown parse method, only auto, ocr, txt allowed")
            exit(1)

        pipe.pipe_classify()

        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()
            else:
                logger.error("need model list input")
                exit(1)

        pipe.pipe_parse()

        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    pdf_directory = r"E:/MinerU/pdf_to_go"
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    output_dir = r"E:/MinerU/pdfoutput"

    # 打印PDF文件列表
    print("本次处理的PDF 文件列表:")
    for pdf_path in pdf_files:
        print(pdf_path)

    # 循环处理每个PDF文件
    for pdf_path in pdf_files:
        pdf_parse_main(pdf_path, output_dir=output_dir)
