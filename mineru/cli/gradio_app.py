# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level=log_level)  # 添加新handler

from mineru.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.hash_utils import str_sha256


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        # 根据 backend 确定 parse_method
        if backend.startswith("vlm"):
            parse_method = "vlm"
        else:
            parse_method = 'ocr' if is_ocr else 'auto'

        # 根据 backend 类型准备环境目录
        if backend.startswith("hybrid"):
            env_name = f"hybrid_{parse_method}"
        else:
            env_name = parse_method

        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, env_name)

        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=parse_method,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
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
        # 只处理以.jpg结尾的图片
        if relative_path.endswith('.jpg'):
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'
        else:
            # 其他格式的图片保持原样
            return match.group(0)
    # 应用替换
    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None):
    # 如果language包含()，则提取括号前的内容作为实际语言
    if '(' in language and ')' in language:
        language = language.split('(')[0].strip()
    file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_pdf(file_path, './output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url)
    archive_zip_path = os.path.join('./output', str_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')

    return md_content, txt_content, archive_zip_path, new_pdf_path


latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, mode='r', encoding='utf-8') as header_file:
    header = header_file.read()

other_lang = [
    'ch (Chinese, English, Chinese Traditional)',
    'ch_lite (Chinese, English, Chinese Traditional, Japanese)',
    'ch_server (Chinese, English, Chinese Traditional, Japanese)',
    'en (English)',
    'korean (Korean, English)',
    'japan (Chinese, English, Chinese Traditional, Japanese)',
    'chinese_cht (Chinese, English, Chinese Traditional, Japanese)',
    'ta (Tamil, English)',
    'te (Telugu, English)',
    'ka (Kannada)',
    'el (Greek, English)',
    'th (Thai, English)'
]
add_lang = [
    'latin (French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua)',
    'arabic (Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English)',
    'east_slavic (Russian, Belarusian, Ukrainian, English)',
    'cyrillic (Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English)',
    'devanagari (Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English)'
]
all_lang = [*other_lang, *add_lang]


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


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-http-client',
    'http_client_enable',
    type=bool,
    help="Enable http-client backend to link openai-compatible servers.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
def main(ctx,
        example_enable,
        http_client_enable,
        api_enable, max_convert_pages,
        server_name, server_port, latex_delimiters_type, **kwargs
):

    # 创建 i18n 实例，支持中英文
    i18n = gr.I18n(
        en={
            "upload_file": "Please upload a PDF or image",
            "max_pages": "Max convert pages",
            "backend": "Backend",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL for http-client backend.",
            "recognition_options": "**Recognition Options:**",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_pipeline": "If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for image-based PDFs and images.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Requires correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "pdf_preview": "PDF preview",
            "examples": "Examples:",
            "convert_result": "Convert result",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "backend_info_vlm": "High-precision parsing via VLM, supports Chinese and English documents only.",
            "backend_info_pipeline": "Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.",
            "backend_info_hybrid": "High-precision hybrid parsing, supports multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
        zh={
            "upload_file": "请上传 PDF 或图片",
            "max_pages": "最大转换页数",
            "backend": "解析后端",
            "server_url": "服务器地址",
            "server_url_info": "http-client 后端的 OpenAI 兼容服务器地址。",
            "recognition_options": "**识别选项：**",
            "table_enable": "启用表格识别",
            "table_info": "禁用后，表格将显示为图片。",
            "formula_label_vlm": "启用行间公式识别",
            "formula_label_pipeline": "启用公式识别",
            "formula_label_hybrid": "启用行内公式识别",
            "formula_info_vlm": "禁用后，行间公式将显示为图片。",
            "formula_info_pipeline": "禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。",
            "formula_info_hybrid": "禁用后，行内公式将不会被检测或解析。",
            "ocr_language": "OCR 语言",
            "ocr_language_info": "为扫描版 PDF 和图片选择 OCR 语言。",
            "force_ocr": "强制启用 OCR",
            "force_ocr_info": "仅在识别效果极差时启用，需选择正确的 OCR 语言。",
            "convert": "转换",
            "clear": "清除",
            "pdf_preview": "PDF 预览",
            "examples": "示例：",
            "convert_result": "转换结果",
            "md_rendering": "Markdown 渲染",
            "md_text": "Markdown 文本",
            "backend_info_vlm": "多模态大模型高精度解析，仅支持中英文文档。",
            "backend_info_pipeline": "传统多模型管道解析，支持多语言，无幻觉。",
            "backend_info_hybrid": "高精度混合解析，支持多语言。",
            "backend_info_default": "选择文档解析的后端引擎。",
        },
    )

    # 根据后端类型获取公式识别标签（闭包函数以支持 i18n）
    def get_formula_label(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_label_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_label_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_label_hybrid")
        else:
            return i18n("formula_label_pipeline")

    def get_formula_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_info_hybrid")
        else:
            return ""

    def get_backend_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("backend_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("backend_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("backend_info_hybrid")
        else:
            return i18n("backend_info_default")

    # 更新界面函数
    def update_interface(backend_choice):
        formula_label_update = gr.update(label=get_formula_label(backend_choice), info=get_formula_info(backend_choice))
        backend_info_update = gr.update(info=get_backend_info(backend_choice))
        if "http-client" in backend_choice:
            client_options_update = gr.update(visible=True)
        else:
            client_options_update = gr.update(visible=False)
        if "vlm" in backend_choice:
            ocr_options_update = gr.update(visible=False)
        else:
            ocr_options_update = gr.update(visible=True)

        return client_options_update, ocr_options_update, formula_label_update, backend_info_update


    kwargs.update(arg_parse(ctx))

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")

    vlm_engine = get_vlm_engine("auto", is_async=True)
    if vlm_engine in ["transformers", "mlx-engine"]:
        http_client_enable = True
    else:
        try:
            logger.info(f"Start init {vlm_engine}...")
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
            model_singleton = ModelSingleton()
            predictor = model_singleton.get_model(
                vlm_engine,
                None,
                None,
                **kwargs
            )
            logger.info(f"{vlm_engine} init successfully.")
        except Exception as e:
            logger.exception(e)

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes]
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label=i18n("upload_file"), file_types=suffixes)
                with gr.Row():
                    max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                with gr.Row():
                    drop_list = ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]
                    preferred_option = "hybrid-auto-engine"
                    if http_client_enable:
                        drop_list.extend(["vlm-http-client", "hybrid-http-client"])
                    backend = gr.Dropdown(drop_list, label=i18n("backend"), value=preferred_option, info=get_backend_info(preferred_option))
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(label=i18n("server_url"), value='http://localhost:30000', placeholder='http://localhost:30000', info=i18n("server_url_info"))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown(i18n("recognition_options"))
                        table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                        formula_enable = gr.Checkbox(label=get_formula_label(preferred_option), value=True, info=get_formula_info(preferred_option))
                    with gr.Column(visible=False) as ocr_options:
                        language = gr.Dropdown(all_lang, label=i18n("ocr_language"), value='ch (Chinese, English, Chinese Traditional)', info=i18n("ocr_language_info"))
                        is_ocr = gr.Checkbox(label=i18n("force_ocr"), value=False, info=i18n("force_ocr_info"))
                with gr.Row():
                    change_bu = gr.Button(i18n("convert"))
                    clear_bu = gr.ClearButton(value=i18n("clear"))
                pdf_show = PDF(label=i18n("pdf_preview"), interactive=False, visible=True, height=800)
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        gr.Examples(
                            label=i18n("examples"),
                            examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                      _.endswith(tuple(suffixes))],
                            inputs=input_file
                        )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label=i18n("convert_result"), interactive=False)
                with gr.Tabs():
                    with gr.Tab(i18n("md_rendering")):
                        md = gr.Markdown(
                            label=i18n("md_rendering"),
                            height=1200,
                            # buttons=["copy"],  # gradio 6 以上版本使用
                            show_copy_button=True,  # gradio 6 以下版本使用
                            latex_delimiters=latex_delimiters,
                            line_breaks=True
                        )
                    with gr.Tab(i18n("md_text")):
                        md_text = gr.TextArea(
                            lines=45,
                            # buttons=["copy"],  # gradio 6 以上版本使用
                            show_copy_button=True,  # gradio 6 以下版本使用
                            label=i18n("md_text")
                        )

        # 添加事件处理
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            # api_visibility="private"  # gradio 6 以上版本使用
            api_name=False  # gradio 6 以下版本使用
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            # api_visibility="private"  # gradio 6 以上版本使用
            api_name=False  # gradio 6 以下版本使用
        )
        clear_bu.add([input_file, md, pdf_show, md_text, output_file, is_ocr])

        input_file.change(
            fn=to_pdf,
            inputs=input_file,
            outputs=pdf_show,
            api_name="to_pdf" if api_enable else False,  # gradio 6 以下版本使用
            # api_visibility="public" if api_enable else "private"  # gradio 6 以上版本使用
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[md, md_text, output_file, pdf_show],
            api_name="to_markdown" if api_enable else False,  # gradio 6 以下版本使用
            # api_visibility="public" if api_enable else "private"  # gradio 6 以上版本使用
        )

    footer_links = ["gradio", "settings"]
    if api_enable:
        footer_links.append("api")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        # footer_links=footer_links,  # gradio 6 以上版本使用
        show_api=api_enable,  # gradio 6 以下版本使用
        i18n=i18n
    )


if __name__ == '__main__':
    main()
