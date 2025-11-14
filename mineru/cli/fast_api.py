import uuid
import os
import re
import tempfile
import uvicorn
import click
import zipfile
from pathlib import Path
import glob
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from typing import List, Optional
from loguru import logger
from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path, guess_suffix_by_bytes
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.version import __version__
from mineru.data.io.s3 import S3Reader
from mineru.data.utils.path_utils import parse_s3path

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._- 
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'

def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


def read_file_with_s3_support(
    file_path: str,
    s3_ak: Optional[str] = None,
    s3_sk: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_addressing_style: str = "auto"
) -> bytes:
    """
    读取文件，支持本地路径和 S3 URI
    
    Args:
        file_path: 文件路径（本地路径或 s3://bucket/key 格式）
        s3_ak: S3 Access Key（可选，默认从环境变量读取）
        s3_sk: S3 Secret Key（可选，默认从环境变量读取）
        s3_endpoint: S3 Endpoint URL（可选，默认从环境变量读取）
        s3_addressing_style: S3 addressing style（auto/path/virtual）
    
    Returns:
        bytes: 文件内容（PDF 或转换后的 PDF）
    """
    # S3 路径
    if file_path.startswith("s3://") or file_path.startswith("s3a://"):
        # 获取 S3 配置（优先使用参数，其次环境变量）
        ak = s3_ak or os.getenv("AWS_ACCESS_KEY_ID")
        sk = s3_sk or os.getenv("AWS_SECRET_ACCESS_KEY")
        endpoint = s3_endpoint or os.getenv("S3_ENDPOINT_URL")
        
        if not all([ak, sk, endpoint]):
            raise ValueError(
                "S3 credentials not provided. Please provide s3_ak, s3_sk, s3_endpoint "
                "or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL environment variables."
            )
        
        # 解析 S3 路径
        bucket, key = parse_s3path(file_path)
        logger.info(f"Reading from S3: bucket={bucket}, key={key}")
        
        # 创建 S3Reader 并读取
        s3_reader = S3Reader(
            bucket=bucket,
            ak=ak,
            sk=sk,
            endpoint_url=endpoint,
            addressing_style=s3_addressing_style
        )
        file_bytes = s3_reader.read(key)
        
        # 根据文件类型处理
        file_suffix = guess_suffix_by_bytes(file_bytes, Path(key))
        if file_suffix in image_suffixes:
            logger.info(f"Converting image to PDF: {file_suffix}")
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes:
            return file_bytes
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")
    
    # 本地路径
    else:
        return read_fn(Path(file_path))


@app.post(path="/file_parse",)
async def parse_pdf(
        files: List[UploadFile] = File(...),
        output_dir: str = Form("./output"),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("pipeline"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        response_format_zip: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
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

            # 创建临时文件
            temp_path = Path(unique_dir) / file_path.name
            with open(temp_path, "wb") as f:
                f.write(content)

            # 如果是图像文件或PDF，使用read_fn处理
            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

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
            **config
        )

        # 根据 response_format_zip 决定返回类型
        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
            os.close(zip_fd) 
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_name in pdf_file_names:
                    safe_pdf_name = sanitize_filename(pdf_name)
                    if backend.startswith("pipeline"):
                        parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    else:
                        parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                    if not os.path.exists(parse_dir):
                        continue

                    # 写入文本类结果
                    if return_md:
                        path = os.path.join(parse_dir, f"{pdf_name}.md")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))

                    if return_middle_json:
                        path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_middle.json"))

                    if return_model_output:
                        path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                        if os.path.exists(path): 
                            zf.write(path, arcname=os.path.join(safe_pdf_name, os.path.basename(path)))

                    if return_content_list:
                        path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_content_list.json"))

                    # 写入图片
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                        for image_path in image_paths:
                            zf.write(image_path, arcname=os.path.join(safe_pdf_name, "images", os.path.basename(image_path)))

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="results.zip",
                background=BackgroundTask(cleanup_file, zip_path)
            )
        else:
            # 构建 JSON 结果
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
                        data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                    if return_model_output:
                        data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                    if return_content_list:
                        data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                        image_paths = glob.glob(safe_pattern)
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
                    "results": result_dict
                }
            )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )


@app.post(path="/file_parse_by_path")
async def parse_pdf_by_path(
        file_path: str = Form(..., description="文件路径（本地路径或 s3://bucket/key）"),
        output_dir: Optional[str] = Form(None, description="输出目录路径（为空则在源文件同目录下生成）"),
        lang: str = Form("ch", description="语言设置"),
        backend: str = Form("pipeline", description="后端类型: pipeline, vlm-transformers, vlm-vllm-engine等"),
        parse_method: str = Form("auto", description="解析方法: auto, txt, ocr"),
        formula_enable: bool = Form(True, description="是否启用公式识别"),
        table_enable: bool = Form(True, description="是否启用表格识别"),
        server_url: Optional[str] = Form(None, description="VLM服务器URL（backend为vlm-http-client时需要）"),
        return_md: bool = Form(True, description="是否生成 Markdown 文件"),
        return_middle_json: bool = Form(True, description="是否生成中间 JSON 文件"),
        return_model_output: bool = Form(False, description="是否生成模型输出文件"),
        return_content_list: bool = Form(False, description="是否生成内容列表文件"),
        return_images: bool = Form(True, description="是否提取图片"),
        return_content: bool = Form(False, description="是否在响应中返回文件内容（默认只返回路径）"),
        start_page_id: int = Form(0, description="起始页码"),
        end_page_id: int = Form(99999, description="结束页码"),
        s3_ak: Optional[str] = Form(None, description="S3 Access Key（可选，默认从环境变量读取）"),
        s3_sk: Optional[str] = Form(None, description="S3 Secret Key（可选，默认从环境变量读取）"),
        s3_endpoint: Optional[str] = Form(None, description="S3 Endpoint URL（可选，默认从环境变量读取）"),
        s3_addressing_style: str = Form("auto", description="S3 addressing style: auto, path, virtual"),
):
    """
    基于路径的文件解析接口（支持本地路径和 S3）
    
    支持的文件路径格式：
    1. 本地路径: /data/docs/report.pdf
    2. S3 URI: s3://bucket-name/path/to/file.pdf
    
    S3 配置说明：
    - 方式1（推荐）：在环境变量中配置
      * AWS_ACCESS_KEY_ID
      * AWS_SECRET_ACCESS_KEY
      * S3_ENDPOINT_URL
    - 方式2：通过 API 参数传递（s3_ak, s3_sk, s3_endpoint）
    
    使用示例：
    - 本地文件: /data/docs/report.pdf
    - S3 文件: s3://my-bucket/docs/report.pdf
    
    参数说明：
    - return_content=False (默认): 只返回文件路径和大小，客户端自行读取
    - return_content=True: 在响应中返回文件内容，会增加网络传输
    - output_dir: 为空时，本地文件在源文件同目录生成，S3文件必须指定
    """
    
    # 获取命令行配置参数
    config = getattr(app.state, "config", {})
    
    try:
        logger.info(f"开始处理文件: {file_path}")
        
        # 读取文件（支持本地和 S3）
        pdf_bytes = read_file_with_s3_support(
            file_path=file_path,
            s3_ak=s3_ak,
            s3_sk=s3_sk,
            s3_endpoint=s3_endpoint,
            s3_addressing_style=s3_addressing_style
        )
        
        # 提取文件名
        if file_path.startswith("s3://") or file_path.startswith("s3a://"):
            # S3 路径: 从 key 中提取文件名
            _, key = parse_s3path(file_path)
            file_name = Path(key).stem
            
            # S3 文件必须指定 output_dir
            if not output_dir:
                raise ValueError("output_dir must be specified for S3 files")
        else:
            # 本地路径
            file_path_obj = Path(file_path)
            file_name = file_path_obj.stem
            
            # 确定输出目录
            if not output_dir:
                output_dir = str(file_path_obj.parent / "mineru_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用解析函数
        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[lang],
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
            **config
        )
        
        # 确定解析结果目录
        if backend.startswith("pipeline"):
            parse_dir = os.path.join(output_dir, file_name, parse_method)
        else:
            parse_dir = os.path.join(output_dir, file_name, "vlm")
        
        # 构建响应数据
        result_dict = {
            "output_dir": output_dir,
            "parse_dir": parse_dir
        }
        
        # 默认：只返回文件路径和大小（轻量级）
        if os.path.exists(parse_dir):
            # 定义文件配置：(return参数, 文件后缀, 结果key前缀)
            file_configs = [
                (return_md, ".md", "markdown"),
                (return_middle_json, "_middle.json", "middle_json"),
                (return_model_output, "_model.json", "model_output"),
                (return_content_list, "_content_list.json", "content_list"),
            ]
            
            # 统一处理所有文件
            for should_return, suffix, key_prefix in file_configs:
                if should_return:
                    file_path = os.path.join(parse_dir, f"{file_name}{suffix}")
                    if os.path.exists(file_path):
                        result_dict[f"{key_prefix}_path"] = file_path
                        result_dict[f"{key_prefix}_size"] = os.path.getsize(file_path)
            
            # images 特殊处理（目录而非文件）
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                if os.path.exists(images_dir):
                    image_files = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                    result_dict["images_dir"] = images_dir
                    result_dict["images_count"] = len(image_files)
        
        # 可选：返回文件内容（return_content=True 时）
        if return_content and os.path.exists(parse_dir):
            result_dict["content"] = {}
            
            if return_md:
                result_dict["content"]["md_content"] = get_infer_result(".md", file_name, parse_dir)
            if return_middle_json:
                result_dict["content"]["middle_json"] = get_infer_result("_middle.json", file_name, parse_dir)
            if return_model_output:
                result_dict["content"]["model_output"] = get_infer_result("_model.json", file_name, parse_dir)
            if return_content_list:
                result_dict["content"]["content_list"] = get_infer_result("_content_list.json", file_name, parse_dir)
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                result_dict["content"]["images"] = {
                    os.path.basename(image_path): f"data:image/jpeg;base64,{encode_image(image_path)}"
                    for image_path in image_paths
                }
        
        logger.info(f"处理完成: {parse_dir}")
        
        return JSONResponse(
            status_code=200,
            content={
                "backend": backend,
                "version": __version__,
                "results": {file_name: result_dict}
            }
        )
        
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }
        )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "mineru.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()