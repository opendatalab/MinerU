# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import re
import time
import zipfile
from pathlib import Path

import gradio as gr
from gradio_pdf import PDF
from loguru import logger

from mineru.cli.common import prepare_env, read_fn, aio_do_parse
from mineru.utils.hash_utils import str_sha256


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = 'ocr'
        else:
            parse_method = 'auto'

        if backend.startswith("vlm"):
            parse_method = "vlm"
            url = None
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=parse_method,
            end_page_id=end_page_id,
            p_formula_enable=formula_enable,
            p_table_enable=table_enable,
            backend=backend,
            server_url=url,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

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
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None):
    file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_pdf(file_path, './output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url)
    archive_zip_path = os.path.join('./output', str_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('压缩成功')
    else:
        logger.error('压缩失败')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')

    return md_content, txt_content, archive_zip_path, new_pdf_path


latex_delimiters = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, 'r') as file:
    header = file.read()


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',  # noqa: E126
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',  # noqa: E126
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',  # noqa: E126
        'sa', 'bgc'
]
other_lang = ['ch', 'ch_lite', 'ch_server', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']
add_lang = ['latin', 'arabic', 'cyrillic', 'devanagari']

# all_lang = ['', 'auto']
all_lang = []
# all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])
all_lang.extend([*other_lang, *add_lang])


def safe_stem(file_path):
    stem = Path(file_path).stem
    # 只保留字母、数字、下划线和点，其他字符替换为下划线
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # 构建完整的文件路径
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # 将字节数据写入文件
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


def main():
    example_enable = False

    # try:
    #     print("Start init SgLang engine...")
    #     from mineru.backend.vlm.vlm_analyze import ModelSingleton
    #     modelsingleton = ModelSingleton()
    #     predictor = modelsingleton.get_model(
    #         "sglang-engine",
    #         None,
    #         None,
    #         mem_fraction_static=0.5,
    #         enable_torch_compile=True,
    #     )
    #     print("SgLang engine init successfully.")
    # except Exception as e:
    #     logger.exception(e)

    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    file = gr.File(label='Please upload a PDF or image', file_types=['.pdf', '.png', '.jpeg', '.jpg'])
                with gr.Row():
                    max_pages = gr.Slider(1, 20, 10, step=1, label='Max convert pages')
                with gr.Row():
                    backend = gr.Dropdown(["pipeline", "vlm-transformers", "vlm-sglang-client"], label="Backend", value="pipeline")
                with gr.Row(visible=True) as ocr_options:
                    language = gr.Dropdown(all_lang, label='Language', value='ch')
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(label='Server URL', value='http://localhost:30000', placeholder='http://localhost:30000')
                with gr.Row(visible=True) as pipeline_options:
                    is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                    formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                    table_enable = gr.Checkbox(label='Enable table recognition(test)', value=True)
                with gr.Row():
                    change_bu = gr.Button('Convert')
                    clear_bu = gr.ClearButton(value='Clear')
                pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
                if example_enable:
                    example_root = os.path.join(os.path.dirname(__file__), 'examples')
                    if os.path.exists(example_root):
                        with gr.Accordion('Examples:'):
                            gr.Examples(
                                examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                          _.endswith('pdf')],
                                inputs=file
                            )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label='convert result', interactive=False)
                with gr.Tabs():
                    with gr.Tab('Markdown rendering'):
                        md = gr.Markdown(label='Markdown rendering', height=1100, show_copy_button=True,
                                         latex_delimiters=latex_delimiters,
                                         line_breaks=True)
                    with gr.Tab('Markdown text'):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)


        # 更新界面函数
        def update_interface(backend_choice):
            if backend_choice in ["vlm-transformers", "vlm-sglang-engine"]:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif backend_choice in ["vlm-sglang-client"]:  # pipeline
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif backend_choice in ["pipeline"]:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            else:
                pass


        # 添加事件处理
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, pipeline_options]
        )

        file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
        change_bu.click(fn=to_markdown, inputs=[file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
                        outputs=[md, md_text, output_file, pdf_show])
        clear_bu.add([file, md, pdf_show, md_text, output_file, is_ocr])

    demo.launch(server_name='localhost')


if __name__ == '__main__':
    main()