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
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

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
        file_path: str = Form(..., description="服务器本地文件路径"),
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
        return_content: bool = Form(False, description="是否在响应中返回文件内容（会增加网络传输）"),
        start_page_id: int = Form(0, description="起始页码"),
        end_page_id: int = Form(99999, description="结束页码"),
):
    """
    基于路径的文件解析接口（仅支持服务器本地路径）
    
    特点：
    1. 无需上传文件，直接读取服务器本地文件系统
    2. 结果写入共享存储路径，客户端可直接访问
    3. 灵活控制生成哪些文件（Markdown、JSON、图片等）
    4. 默认只返回元数据（路径、文件信息），最小化网络传输
    5. 如果不指定output_dir，则在源文件同目录下生成结果
    
    使用示例：
    - 输入文件: /data/docs/report.pdf
    - 不指定output_dir: 结果在 /data/docs/mineru_output/report/auto/
    - 指定output_dir=/shared/results: 结果在 /shared/results/report/auto/
    
    参数说明：
    - return_md/return_middle_json/return_model_output/return_content_list/return_images: 
      控制是否生成对应的文件
    - return_content: 控制是否在 HTTP 响应中返回文件内容
      * False（默认）: 只返回文件路径和大小（轻量级）
      * True: 在响应中包含文件内容（会增加网络传输）
    
    注：auto 是解析方法目录（auto/txt/ocr），由 MinerU 内部结构决定
    """
    
    # 获取命令行配置参数
    config = getattr(app.state, "config", {})
    
    try:
        # 验证本地文件路径
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return JSONResponse(
                status_code=400,
                content={"error": f"文件不存在: {file_path}"}
            )
        
        if not file_path_obj.is_file():
            return JSONResponse(
                status_code=400,
                content={"error": f"路径不是文件: {file_path}"}
            )
        
        # 检查文件类型
        file_suffix = guess_suffix_by_path(file_path_obj)
        if file_suffix not in pdf_suffixes + image_suffixes:
            return JSONResponse(
                status_code=400,
                content={"error": f"不支持的文件类型: {file_suffix}，仅支持PDF和图片格式"}
            )
        
        # 读取文件
        logger.info(f"开始处理文件: {file_path}")
        file_name = str(file_path_obj.stem)
        pdf_bytes = read_fn(file_path_obj)
        
        # 确定输出目录
        user_specified_output = output_dir is not None and output_dir.strip() != ""
        default_output_dir = str(file_path_obj.parent / "mineru_output")
        
        if not user_specified_output:
            # 未指定，使用默认路径
            output_dir = default_output_dir
            logger.info(f"未指定输出目录，使用默认路径: {output_dir}")
        
        # 检查输出目录的可写性
        def check_writable(path: str) -> bool:
            """检查目录是否可写"""
            try:
                # 如果目录不存在，尝试创建
                os.makedirs(path, exist_ok=True)
                # 尝试创建测试文件
                test_file = Path(path) / f".mineru_write_test_{uuid.uuid4().hex[:8]}"
                test_file.touch()
                test_file.unlink()
                return True
            except (OSError, PermissionError) as e:
                logger.warning(f"目录 {path} 不可写: {e}")
                return False
        
        # 如果用户指定的目录不可写，尝试回退到默认路径
        if user_specified_output and not check_writable(output_dir):
            logger.warning(f"指定的输出目录 {output_dir} 无写权限，回退到默认路径")
            output_dir = default_output_dir
            
            # 检查默认路径是否可写
            if not check_writable(output_dir):
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"输出目录无写权限，包括默认路径 {output_dir}",
                        "suggestion": "请检查文件系统权限或指定其他输出目录"
                    }
                )
            logger.info(f"使用默认输出路径: {output_dir}")
        
        logger.info(f"最终输出目录: {output_dir}")
        
        # 调用解析函数（结果直接写入output_dir）
        await aio_do_parse(
            output_dir=output_dir,  # 用户指定的路径，不添加UUID
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
        
        # 9. 确定输出路径
        if backend.startswith("pipeline"):
            parse_dir = os.path.join(output_dir, file_name, parse_method)
        else:
            parse_dir = os.path.join(output_dir, file_name, "vlm")
        
        # 构建轻量级响应（只包含元数据）
        response_data = {
            "status": "success",
            "backend": backend,
            "version": __version__,
            "file_info": {
                "input_file": str(file_path),
                "file_name": file_name,
                "file_size": file_path_obj.stat().st_size,
            },
            "output_info": {
                "output_dir": output_dir,
                "parse_dir": parse_dir,
                "files": {}  # 只列出生成的文件路径，不包含内容
            }
        }
        
        # 列出生成的文件（不读取内容）
        if os.path.exists(parse_dir):
            # Markdown 文件
            if return_md:
                md_file = os.path.join(parse_dir, f"{file_name}.md")
                if os.path.exists(md_file):
                    response_data["output_info"]["files"]["markdown"] = md_file
                    response_data["output_info"]["files"]["markdown_size"] = os.path.getsize(md_file)
            
            # 中间 JSON 文件
            if return_middle_json:
                json_file = os.path.join(parse_dir, f"{file_name}_middle.json")
                if os.path.exists(json_file):
                    response_data["output_info"]["files"]["middle_json"] = json_file
                    response_data["output_info"]["files"]["middle_json_size"] = os.path.getsize(json_file)
            
            # 模型输出文件
            if return_model_output:
                model_file = os.path.join(parse_dir, f"{file_name}_model.json")
                if os.path.exists(model_file):
                    response_data["output_info"]["files"]["model_output"] = model_file
                    response_data["output_info"]["files"]["model_output_size"] = os.path.getsize(model_file)
            
            # 内容列表文件
            if return_content_list:
                content_list_file = os.path.join(parse_dir, f"{file_name}_content_list.json")
                if os.path.exists(content_list_file):
                    response_data["output_info"]["files"]["content_list"] = content_list_file
                    response_data["output_info"]["files"]["content_list_size"] = os.path.getsize(content_list_file)
            
            # 图片目录
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                if os.path.exists(images_dir):
                    image_files = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                    response_data["output_info"]["files"]["images_dir"] = images_dir
                    response_data["output_info"]["files"]["images_count"] = len(image_files)
        
        # 可选：如果用户明确要求，才返回文件内容（在响应中包含实际内容）
        if return_content:
            logger.warning("return_content=True，将增加网络传输量")
            response_data["content"] = {}
            
            # 返回 Markdown 内容
            if return_md:
                md_file = response_data["output_info"]["files"].get("markdown")
                if md_file and os.path.exists(md_file):
                    with open(md_file, 'r', encoding='utf-8') as f:
                        response_data["content"]["markdown"] = f.read()
            
            # 返回中间 JSON 内容
            if return_middle_json:
                json_file = response_data["output_info"]["files"].get("middle_json")
                if json_file and os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        response_data["content"]["middle_json"] = f.read()
            
            # 返回模型输出内容
            if return_model_output:
                model_file = response_data["output_info"]["files"].get("model_output")
                if model_file and os.path.exists(model_file):
                    with open(model_file, 'r', encoding='utf-8') as f:
                        response_data["content"]["model_output"] = f.read()
            
            # 返回内容列表
            if return_content_list:
                content_list_file = response_data["output_info"]["files"].get("content_list")
                if content_list_file and os.path.exists(content_list_file):
                    with open(content_list_file, 'r', encoding='utf-8') as f:
                        response_data["content"]["content_list"] = f.read()
            
            # 返回图片（Base64 编码）
            if return_images:
                images_dir = response_data["output_info"]["files"].get("images_dir")
                if images_dir and os.path.exists(images_dir):
                    image_files = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                    response_data["content"]["images"] = {
                        os.path.basename(img_path): f"data:image/jpeg;base64,{encode_image(img_path)}"
                        for img_path in image_files
                    }
        
        logger.info(f"处理完成: {parse_dir}")
        
        return JSONResponse(
            status_code=200,
            content=response_data
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