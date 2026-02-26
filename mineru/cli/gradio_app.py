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

from mineru.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes, office_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.hash_utils import str_sha256


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)

        # 检测 office 文件类型
        file_suffix = Path(doc_path).suffix.lower().lstrip('.')
        if file_suffix in office_suffixes:
            # office 文件使用固定的 env_name，parse_method 设为默认值（aio_do_parse 内部会处理）
            env_name = "office"
            parse_method = "auto"
        elif backend.startswith("vlm"):
            # 根据 backend 确定 parse_method
            parse_method = "vlm"
            env_name = parse_method
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
    # MIME类型映射
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }

    def _path_to_data_uri(relative_path):
        file_ext = os.path.splitext(relative_path)[1].lower()
        if file_ext not in mime_types:
            return None
        try:
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'data:{mime_types[file_ext]};base64,{base64_image}'
        except Exception as e:
            logger.warning(f"Failed to convert image {relative_path} to base64: {e}")
            return None

    # 匹配Markdown中的图片标签 ![...](path)
    def replace_md(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'![{relative_path}]({data_uri})'
        return match.group(0)

    result = re.sub(r'\!\[(?:[^\]]*)\]\(([^)]+)\)', replace_md, markdown_text)

    # 匹配HTML表格中的 <img src="path"> (跳过已有的data: URI)
    def replace_html_src(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'src="{data_uri}"'
        return match.group(0)

    result = re.sub(r'src="(?!data:)([^"]+)"', replace_html_src, result)

    return result


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None):
    # 如果language包含()，则提取括号前的内容作为实际语言
    if '(' in language and ')' in language:
        language = language.split('(')[0].strip()

    # office 文件不需要转换为 PDF，直接使用原始文件路径
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes
    if is_office:
        parse_file_path = file_path
    else:
        parse_file_path = to_pdf(file_path)

    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_pdf(parse_file_path, './output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url)
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
    # office 文件没有 layout PDF，返回 None；其他格式返回转换后的 PDF 路径
    if is_office:
        new_pdf_path = None
    else:
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


def to_pdf_preview(file_path):
    """用于 PDF 预览的转换函数，office 文件不支持预览，返回 None。"""
    if file_path is None:
        return None
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return None
    return to_pdf(file_path)


def update_file_preview(file_path, request: gr.Request):
    """处理文件上传：根据文件类型显示/隐藏元素并返回预览内容。
    - office 文件：隐藏参数选项，显示 Microsoft 在线预览 iframe
    - PDF/图片文件：显示参数选项，显示 PDF 预览组件
    """
    if file_path is None:
        return (
            gr.update(visible=True),        # options_group - 恢复显示
            gr.update(value=None, visible=True),  # doc_show (PDF) - 清空并显示
            gr.update(value="", visible=False),   # office_html - 隐藏
        )

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes

    if is_office:
        # 构建可公开访问的文件 URL，供 Microsoft 在线预览使用
        host = (request.headers.get('x-forwarded-host')
                or request.headers.get('host', 'localhost:7860'))
        proto = request.headers.get('x-forwarded-proto', 'http')
        base_url = f"{proto}://{host}"
        public_url = f"{base_url}/gradio_api/file={file_path}"
        viewer_url = f"https://view.officeapps.live.com/op/embed.aspx?src={public_url}"
        html_content = (
            f'<iframe src="{viewer_url}" '
            f'width="100%" height="800px" frameborder="0" '
            f'style="border: none;"></iframe>'
        )
        return (
            gr.update(visible=False),                        # options_group - 隐藏
            gr.update(value=None, visible=False),            # doc_show - 隐藏
            gr.update(value=html_content, visible=True),     # office_html - 显示
        )
    else:
        pdf_path = to_pdf_preview(file_path)
        return (
            gr.update(visible=True),                         # options_group - 显示
            gr.update(value=pdf_path, visible=True),         # doc_show - 显示 PDF 预览
            gr.update(value="", visible=False),              # office_html - 隐藏
        )


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
            "doc_preview": "Document preview",
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
            "doc_preview": "文档预览",
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

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes + office_suffixes]
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label=i18n("upload_file"), file_types=suffixes)
                # 下面这些选项在上传 office 文件时会被自动隐藏
                with gr.Group() as options_group:
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
                doc_show = PDF(label=i18n("doc_preview"), interactive=False, visible=True, height=800)
                office_html = gr.HTML(value="", visible=False)
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
        clear_bu.add([input_file, md, doc_show, md_text, output_file, is_ocr, office_html])

        # 清除按钮额外重置 UI 可见性（ClearButton 不一定触发 input_file.change）
        clear_bu.click(
            fn=lambda: (gr.update(visible=True), gr.update(value=None, visible=True), gr.update(value="", visible=False)),
            inputs=[],
            outputs=[options_group, doc_show, office_html],
            api_name=False,  # gradio 6 以下版本使用
            # api_visibility="private"  # gradio 6 以上版本使用
        )

        input_file.change(
            fn=update_file_preview,
            inputs=input_file,
            outputs=[options_group, doc_show, office_html],
            api_name=False,  # gradio 6 以下版本使用
            # api_visibility="private"  # gradio 6 以上版本使用
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[md, md_text, output_file, doc_show],
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
