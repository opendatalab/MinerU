import json
import os
import sys
from pathlib import Path

import click
from loguru import logger

from magic_pdf.libs.commons import join_path, read_file
from magic_pdf.dict2md.mkcontent import mk_mm_markdown, mk_universal_format
from magic_pdf.pdf_parse_by_txt import parse_pdf_by_txt



def main(s3_pdf_path: str, s3_pdf_profile: str, pdf_model_path: str, pdf_model_profile: str, start_page_num=0, debug_mode=True):
    """ """
    pth = Path(s3_pdf_path)
    book_name = pth.name
    # book_name = "".join(os.path.basename(s3_pdf_path).split(".")[0:-1])
    save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "..", "tmp", "unittest")
    save_path = join_path(save_tmp_path, "md")
    text_content_save_path = f"{save_path}/{book_name}/book.md"
    # metadata_save_path = f"{save_path}/{book_name}/metadata.json"

    pdf_bytes = read_file(s3_pdf_path, s3_pdf_profile)

    try:
        paras_dict = parse_pdf_by_txt(
            pdf_bytes, pdf_model_path, save_path, book_name, pdf_model_profile, start_page_num, debug_mode=debug_mode
        )
        parent_dir = os.path.dirname(text_content_save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
                
        if not paras_dict.get('need_drop'):
            content_list = mk_universal_format(paras_dict)
            markdown_content = mk_mm_markdown(content_list)
        else:
            markdown_content = paras_dict['drop_reason']
            
        with open(text_content_save_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    except Exception as e:
        print(f"ERROR: {s3_pdf_path}, {e}", file=sys.stderr)
        logger.exception(e)


@click.command()
@click.option("--pdf-file-path", help="s3上pdf文件的路径")
@click.option("--save-path", help="解析出来的图片，文本的保存父目录")
def main_shell(pdf_file_path: str, save_path: str):
    # pdf_bin_file_path = "s3://llm-raw-snew/llm-raw-scihub/scimag07865000-07865999/10.1007/"
    pdf_bin_file_parent_path = "s3://llm-raw-snew/llm-raw-scihub/"
    pdf_bin_file_profile = "s2"
    pdf_model_parent_dir = "s3://llm-pdf-text/eval_1k/layout_res/"
    pdf_model_profile = "langchao"

    p = Path(pdf_file_path)
    pdf_parent_path = p.parent
    pdf_file_name = p.name  # pdf文件名字，含后缀
    pdf_bin_file_path = join_path(pdf_bin_file_parent_path, pdf_parent_path)
    pdf_model_dir = join_path(pdf_model_parent_dir, pdf_parent_path)

    main(
        join_path(pdf_bin_file_path, pdf_file_name),
        pdf_bin_file_profile,
        join_path(pdf_model_dir, pdf_file_name),
        pdf_model_profile,
        save_path,
    )


@click.command()
@click.option("--pdf-dir", help="本地pdf文件的路径")
@click.option("--model-dir", help="本地模型文件的路径")
@click.option("--start-page-num", default=0, help="从第几页开始解析")
def main_shell2(pdf_dir: str, model_dir: str,start_page_num: int):
    # 先扫描所有的pdf目录里的文件名字
    pdf_dir = Path(pdf_dir)
    model_dir = Path(model_dir)

    if pdf_dir.is_file():
        pdf_file_names = [pdf_dir.name]
        pdf_dir = pdf_dir.parent
    else:
        pdf_file_names = [f.name for f in pdf_dir.glob("*.pdf")]

    for pdf_file in pdf_file_names:
        pdf_file_path = os.path.join(pdf_dir, pdf_file)
        model_file_path = os.path.join(model_dir, pdf_file).rstrip(".pdf") + ".json"
        with open(model_file_path, "r") as json_file:
            model_list = json.load(json_file)
        main(pdf_file_path, None, model_list, None, start_page_num)



if __name__ == "__main__":
    main_shell2()
