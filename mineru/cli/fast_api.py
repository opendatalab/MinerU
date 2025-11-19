import uuid
import os
import re
import tempfile
import shutil
import uvicorn
import click
import zipfile
from pathlib import Path
import glob
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from typing import List, Optional, Dict
from loguru import logger
from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path, guess_suffix_by_bytes
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.version import __version__
from mineru.data.io.s3 import S3Reader
from mineru.data.data_reader_writer.s3 import S3DataWriter
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


def get_s3_config() -> Dict[str, str]:
    """从环境变量获取 S3 配置"""
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = os.getenv("S3_ENDPOINT_URL")
    
    if not all([ak, sk, endpoint]):
        raise ValueError(
            "S3 credentials not configured. Please set environment variables: "
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL"
        )
    
    return {"ak": ak, "sk": sk, "endpoint": endpoint, "addressing_style": "auto"}


def upload_directory_to_s3(local_dir: str, s3_bucket: str, s3_prefix: str, s3_config: Dict[str, str]) -> None:
    """上传整个目录到 S3"""
    logger.info(f"Uploading directory {local_dir} to s3://{s3_bucket}/{s3_prefix}")
    
    s3_writer = S3DataWriter(
        default_prefix_without_bucket=s3_prefix,
        bucket=s3_bucket,
        ak=s3_config["ak"],
        sk=s3_config["sk"],
        endpoint_url=s3_config["endpoint"],
        addressing_style=s3_config["addressing_style"]
    )
    
    upload_count = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = relative_path.replace("\\", "/")
            
            with open(local_path, "rb") as f:
                s3_writer.write(s3_key, f.read())
                upload_count += 1
    
    logger.info(f"Uploaded {upload_count} files")


def read_file_with_s3_support(file_path: str) -> bytes:
    """读取文件，支持本地路径和 S3 URI"""
    if file_path.startswith("s3://") or file_path.startswith("s3a://"):
        s3_config = get_s3_config()
        bucket, key = parse_s3path(file_path)
        logger.info(f"Reading from S3: bucket={bucket}, key={key}")
        
        s3_reader = S3Reader(
            bucket=bucket,
            ak=s3_config["ak"],
            sk=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            addressing_style=s3_config["addressing_style"]
        )
        file_bytes = s3_reader.read(key)
        
        file_suffix = guess_suffix_by_bytes(file_bytes, Path(key))
        if file_suffix in image_suffixes:
            logger.info(f"Converting image to PDF: {file_suffix}")
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes:
            return file_bytes
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")
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
        output_dir: Optional[str] = Form(None, description="输出目录（本地路径或 s3://bucket/prefix，为空则自动推断）"),
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
        return_content: bool = Form(False, description="是否在响应中返回文件内容"),
        start_page_id: int = Form(0, description="起始页码"),
        end_page_id: int = Form(99999, description="结束页码"),
):
    """
    基于路径的文件解析接口（支持本地路径和 S3，支持交叉输入输出）
    
    支持的场景：
    1. 本地 → 本地
    2. S3 → S3
    3. S3 → 本地
    4. 本地 → S3
    
    S3 配置：必须在服务端配置环境变量（AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL）
    """
    
    config = getattr(app.state, "config", {})
    
    try:
        logger.info(f"开始处理文件: {file_path}")
        if not file_path:
            raise ValueError("file_path is required")
        
        # 判断输入输出类型
        is_s3_input = file_path.startswith(("s3://", "s3a://"))
        file_name = Path(parse_s3path(file_path)[1] if is_s3_input else file_path).stem
        
        # 推断输出路径
        if output_dir is None:
            if is_s3_input:
                bucket, key = parse_s3path(file_path)
                parent = str(Path(key).parent)
                output_dir = f"s3://{bucket}/{parent}/mineru_output/" if parent != "." else f"s3://{bucket}/mineru_output/"
            else:
                output_dir = str(Path(file_path).parent / "mineru_output")
            logger.info(f"Auto-inferred output: {output_dir}")
        
        is_s3_output = output_dir.startswith(("s3://", "s3a://"))
        
        # 读取输入文件
        pdf_bytes = read_file_with_s3_support(file_path)
        
        # 确定处理目录
        temp_dir = None
        if is_s3_output:
            temp_dir = tempfile.mkdtemp(prefix="mineru_")
            actual_output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            actual_output_dir = output_dir
        
        # 执行解析
        await aio_do_parse(
            output_dir=actual_output_dir,
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
        subdir = parse_method if backend.startswith("pipeline") else "vlm"
        local_parse_dir = os.path.join(actual_output_dir, file_name, subdir)
        
        # 构建响应（在上传/清理之前）
        storage_type = "s3" if is_s3_output else "local"
        
        # 计算最终路径
        if is_s3_output:
            bucket, prefix = parse_s3path(output_dir)
            full_prefix = f"{prefix.rstrip('/')}/{file_name}/{subdir}".lstrip("/")
            final_parse_dir = f"s3://{bucket}/{full_prefix}/"
        else:
            final_parse_dir = local_parse_dir
        
        result_dict = {
            "storage_type": storage_type,
            "output_dir": output_dir,
            "parse_dir": final_parse_dir
        }
        
        # 添加文件路径和大小（统一使用 _path 字段，可能是 S3 URI 或本地路径）
        file_configs = [
            (return_md, ".md", "markdown"),
            (return_middle_json, "_middle.json", "middle_json"),
            (return_model_output, "_model.json", "model_output"),
            (return_content_list, "_content_list.json", "content_list"),
        ]
        
        for should_return, suffix, key_prefix in file_configs:
            if should_return:
                local_file = os.path.join(local_parse_dir, f"{file_name}{suffix}")
                if os.path.exists(local_file):
                    file_location = f"{final_parse_dir.rstrip('/')}/{file_name}{suffix}"
                    result_dict[f"{key_prefix}_path"] = file_location
                    result_dict[f"{key_prefix}_size"] = os.path.getsize(local_file)
        
        # 添加图片信息（统一使用 _path 字段）
        if return_images:
            local_images_dir = os.path.join(local_parse_dir, "images")
            if os.path.exists(local_images_dir):
                images_location = f"{final_parse_dir.rstrip('/')}/images/"
                result_dict["images_path"] = images_location
                result_dict["images_count"] = len(glob.glob(os.path.join(glob.escape(local_images_dir), "*.jpg")))
        
        # 读取内容（在上传/清理之前）
        if return_content:
            result_dict["content"] = {}
            if return_md:
                result_dict["content"]["md_content"] = get_infer_result(".md", file_name, local_parse_dir)
            if return_middle_json:
                result_dict["content"]["middle_json"] = get_infer_result("_middle.json", file_name, local_parse_dir)
            if return_model_output:
                result_dict["content"]["model_output"] = get_infer_result("_model.json", file_name, local_parse_dir)
            if return_content_list:
                result_dict["content"]["content_list"] = get_infer_result("_content_list.json", file_name, local_parse_dir)
            if return_images:
                images_dir = os.path.join(local_parse_dir, "images")
                if os.path.exists(images_dir):
                    image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                    result_dict["content"]["images"] = {
                        os.path.basename(img): f"data:image/jpeg;base64,{encode_image(img)}"
                        for img in image_paths
                    }
        
        # 上传到 S3 并清理（在读取内容之后）
        if is_s3_output:
            s3_config = get_s3_config()
            upload_directory_to_s3(local_parse_dir, bucket, full_prefix, s3_config)
            shutil.rmtree(temp_dir)
        
        logger.info(f"处理完成: {final_parse_dir}")
        
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