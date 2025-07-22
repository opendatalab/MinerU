import os
import uuid
from base64 import b64encode
from glob import glob
from pathlib import Path
from typing import List, Optional

import click
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from mineru.cli.common import aio_do_parse, image_suffixes, pdf_suffixes, read_fn
from mineru.utils.cli_parser import arg_parse
from mineru.version import __version__

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(
    file_suffix_identifier: str, pdf_name: str, parse_dir: str
) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


# 在文件顶部添加默认值变量
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_LANG_LIST = ["ch"]
DEFAULT_BACKEND = "pipeline"
DEFAULT_PARSE_METHOD = "auto"
DEFAULT_FORMULA_ENABLE = True
DEFAULT_TABLE_ENABLE = True
DEFAULT_RETURN_MD = True
DEFAULT_RETURN_MIDDLE_JSON = False
DEFAULT_RETURN_MODEL_OUTPUT = False
DEFAULT_RETURN_CONTENT_LIST = False
DEFAULT_RETURN_IMAGES = False
DEFAULT_START_PAGE_ID = 0
DEFAULT_END_PAGE_ID = 99999

# 定义表单默认值
FORM_DEFAULT_OUTPUT_DIR = Form(DEFAULT_OUTPUT_DIR)
FORM_DEFAULT_LANG_LIST = Form(DEFAULT_LANG_LIST)
FORM_DEFAULT_BACKEND = Form(DEFAULT_BACKEND)
FORM_DEFAULT_PARSE_METHOD = Form(DEFAULT_PARSE_METHOD)
FORM_DEFAULT_FORMULA_ENABLE = Form(DEFAULT_FORMULA_ENABLE)
FORM_DEFAULT_TABLE_ENABLE = Form(DEFAULT_TABLE_ENABLE)
FORM_DEFAULT_SERVER_URL = Form(None)
FORM_DEFAULT_RETURN_MD = Form(DEFAULT_RETURN_MD)
FORM_DEFAULT_RETURN_MIDDLE_JSON = Form(DEFAULT_RETURN_MIDDLE_JSON)
FORM_DEFAULT_RETURN_MODEL_OUTPUT = Form(DEFAULT_RETURN_MODEL_OUTPUT)
FORM_DEFAULT_RETURN_CONTENT_LIST = Form(DEFAULT_RETURN_CONTENT_LIST)
FORM_DEFAULT_RETURN_IMAGES = Form(DEFAULT_RETURN_IMAGES)
FORM_DEFAULT_START_PAGE_ID = Form(DEFAULT_START_PAGE_ID)
FORM_DEFAULT_END_PAGE_ID = Form(DEFAULT_END_PAGE_ID)


@app.post(
    path="/file_parse",
)
async def parse_pdf(
    files: List[UploadFile] = File(...),
    output_dir: str = FORM_DEFAULT_OUTPUT_DIR,
    lang_list: List[str] = FORM_DEFAULT_LANG_LIST,
    backend: str = FORM_DEFAULT_BACKEND,
    parse_method: str = FORM_DEFAULT_PARSE_METHOD,
    formula_enable: bool = FORM_DEFAULT_FORMULA_ENABLE,
    table_enable: bool = FORM_DEFAULT_TABLE_ENABLE,
    server_url: Optional[str] = FORM_DEFAULT_SERVER_URL,
    return_md: bool = FORM_DEFAULT_RETURN_MD,
    return_middle_json: bool = FORM_DEFAULT_RETURN_MIDDLE_JSON,
    return_model_output: bool = FORM_DEFAULT_RETURN_MODEL_OUTPUT,
    return_content_list: bool = FORM_DEFAULT_RETURN_CONTENT_LIST,
    return_images: bool = FORM_DEFAULT_RETURN_IMAGES,
    start_page_id: int = FORM_DEFAULT_START_PAGE_ID,
    end_page_id: int = FORM_DEFAULT_END_PAGE_ID,
):
    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 如果是图像文件或PDF，使用read_fn处理
            if file_path.suffix.lower() in pdf_suffixes + image_suffixes:
                # 创建临时文件以便使用read_fn
                temp_path = Path(unique_dir) / file_path.name
                with open(temp_path, "wb") as f:
                    f.write(content)

                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"},
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_path.suffix}"},
                )

        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [
                actual_lang_list[0] if actual_lang_list else "ch"
            ] * len(pdf_file_names)

        # 调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config,
        )

        # 构建结果路径
        result_dict = {}
        for pdf_name in pdf_file_names:
            result_dict[pdf_name] = {}
            data = result_dict[pdf_name]

            if backend.startswith("pipeline"):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

            if os.path.exists(parse_dir):
                if return_md:
                    data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                if return_middle_json:
                    data["middle_json"] = get_infer_result(
                        "_middle.json", pdf_name, parse_dir
                    )
                if return_model_output:
                    if backend.startswith("pipeline"):
                        data["model_output"] = get_infer_result(
                            "_model.json", pdf_name, parse_dir
                        )
                    else:
                        data["model_output"] = get_infer_result(
                            "_model_output.txt", pdf_name, parse_dir
                        )
                if return_content_list:
                    data["content_list"] = get_infer_result(
                        "_content_list.json", pdf_name, parse_dir
                    )
                if return_images:
                    image_paths = glob(f"{parse_dir}/images/*.jpg")
                    data["images"] = {
                        os.path.basename(
                            image_path
                        ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                        for image_path in image_paths
                    }
        return JSONResponse(
            status_code=200,
            content={
                "backend": backend,
                "version": __version__,
                "results": result_dict,
            },
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to process file: {str(e)}"}
        )


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
@click.option("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="Server port (default: 8000)")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
def main(ctx, host, port, reload, **kwargs):
    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run("mineru.cli.fast_api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
