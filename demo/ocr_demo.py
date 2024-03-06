import json
import os

from loguru import logger

from magic_pdf.dict2md.ocr_mkcontent import mk_nlp_markdown
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr


def save_markdown(markdown_text, input_filepath):
    # 获取输入文件的目录
    directory = os.path.dirname(input_filepath)
    # 获取输入文件的文件名（不带扩展名）
    base_name = os.path.basename(input_filepath)
    file_name_without_ext = os.path.splitext(base_name)[0]
    # 定义输出文件的路径
    output_filepath = os.path.join(directory, f"{file_name_without_ext}.md")

    # 将Markdown文本写入.md文件
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(markdown_text)


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    ocr_json_file_path = r"D:\project\20231108code-clean\ocr\new\demo_4\ocr_0.json"
    ocr_pdf_info = read_json_file(ocr_json_file_path)
    pdf_info_dict = parse_pdf_by_ocr(ocr_pdf_info)
    markdown_text = mk_nlp_markdown(pdf_info_dict)
    logger.info(markdown_text)
    save_markdown(markdown_text, ocr_json_file_path)

