"""MinerU File转Markdown转换的FastMCP服务器实现。"""

import json
import re
import traceback
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import aiohttp
import uvicorn
from fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from . import config
from .api import MinerUClient
from .language import get_language_list

# 初始化 FastMCP 服务器
mcp = FastMCP(
    name="MinerU File to Markdown Conversion",
    instructions="""
    一个将文档转化工具，可以将文档转化成Markdown、Json等格式，支持多种文件格式，包括
    PDF、Word、PPT以及图片格式（JPG、PNG、JPEG）。

    系统工具:
    parse_documents: 解析文档（支持本地文件和URL，自动读取内容）
    get_ocr_languages: 获取OCR支持的语言列表
    """,
)

# 全局客户端实例
_client_instance: Optional[MinerUClient] = None


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """创建用于SSE传输的Starlette应用。

    Args:
        mcp_server: MCP服务器实例
        debug: 是否启用调试模式

    Returns:
        Starlette: 配置好的Starlette应用实例
    """
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        """处理SSE连接请求。"""
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def run_server(mode=None, port=8001, host="127.0.0.1"):
    """运行 FastMCP 服务器。

    Args:
        mode: 运行模式，支持stdio、sse、streamable-http
        port: 服务器端口，默认为8001，仅在HTTP模式下有效
        host: 服务器主机地址，默认为127.0.0.1，仅在HTTP模式下有效
    """
    # 确保输出目录存在
    config.ensure_output_dir(output_dir)

    # 检查是否设置了 API 密钥
    if not config.MINERU_API_KEY:
        config.logger.warning("警告: MINERU_API_KEY 环境变量未设置。")
        config.logger.warning("使用以下命令设置: export MINERU_API_KEY=your_api_key")

    # 获取MCP服务器实例
    mcp_server = mcp._mcp_server

    try:
        # 运行服务器
        if mode == "sse":
            config.logger.info(f"启动SSE服务器: {host}:{port}")
            starlette_app = create_starlette_app(mcp_server, debug=True)
            uvicorn.run(starlette_app, host=host, port=port)
        elif mode == "streamable-http":
            config.logger.info(f"启动Streamable HTTP服务器: {host}:{port}")
            # 在HTTP模式下传递端口参数
            mcp.run(mode, port=port)
        else:
            # 默认stdio模式
            config.logger.info("启动STDIO服务器")
            mcp.run(mode or "stdio")
    except Exception as e:
        config.logger.error(f"\n❌ 服务异常退出: {str(e)}")
        traceback.print_exc()
    finally:
        # 清理资源
        cleanup_resources()


def cleanup_resources():
    """清理全局资源。"""
    global _client_instance
    if _client_instance is not None:
        try:
            # 如果客户端有close方法，调用它
            if hasattr(_client_instance, "close"):
                _client_instance.close()
        except Exception as e:
            config.logger.error(f"清理客户端资源时出错: {str(e)}")
        finally:
            _client_instance = None
    config.logger.info("资源清理完成")


def get_client() -> MinerUClient:
    """获取 MinerUClient 的单例实例。如果尚未初始化，则进行初始化。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = MinerUClient()  # Initialization happens here
    return _client_instance


# Markdown 文件的输出目录
output_dir = config.DEFAULT_OUTPUT_DIR


def set_output_dir(dir_path: str):
    """设置转换后文件的输出目录。"""
    global output_dir
    output_dir = dir_path
    config.ensure_output_dir(output_dir)
    return output_dir


def parse_list_input(input_str: str) -> List[str]:
    """
    解析可能包含由逗号或换行符分隔的多个项目的字符串输入。

    Args:
        input_str: 可能包含多个项目的字符串

    Returns:
        解析出的项目列表
    """
    if not input_str:
        return []

    # 按逗号、换行符或空格分割
    items = re.split(r"[,\n\s]+", input_str)

    # 移除空项目并处理带引号的项目
    result = []
    for item in items:
        item = item.strip()
        # 如果存在引号，则移除
        if (item.startswith('"') and item.endswith('"')) or (
            item.startswith("'") and item.endswith("'")
        ):
            item = item[1:-1]

        if item:
            result.append(item)

    return result


async def convert_file_url(
    url: str,
    enable_ocr: bool = False,
    language: str = "ch",
    page_ranges: str | None = None,
) -> Dict[str, Any]:
    """
    从URL转换文件到Markdown格式。支持单个或多个URL处理。

    返回:
        成功: {"status": "success", "result_path": "输出目录路径"}
        失败: {"status": "error", "error": "错误信息"}
    """
    urls_to_process = None

    # 检查是否为字典或字典列表格式的URL配置
    if isinstance(url, dict):
        # 单个URL配置字典
        urls_to_process = url
    elif isinstance(url, list) and len(url) > 0 and isinstance(url[0], dict):
        # URL配置字典列表
        urls_to_process = url
    elif isinstance(url, str):
        # 检查是否为 JSON 字符串格式的多URL配置
        if url.strip().startswith("[") and url.strip().endswith("]"):
            try:
                # 尝试解析 JSON 字符串为URL配置列表
                url_configs = json.loads(url)
                if not isinstance(url_configs, list):
                    raise ValueError("JSON URL配置必须是列表格式")

                urls_to_process = url_configs
            except json.JSONDecodeError:
                # 不是有效的 JSON，继续使用字符串解析方式
                pass

    if urls_to_process is None:
        # 解析普通URL列表
        urls = parse_list_input(url)

        if not urls:
            raise ValueError("未提供有效的 URL")

        if len(urls) == 1:
            # 单个URL处理
            urls_to_process = {"url": urls[0], "is_ocr": enable_ocr}
        else:
            # 多个URL，转换为URL配置列表
            urls_to_process = []
            for url_item in urls:
                urls_to_process.append(
                    {
                        "url": url_item,
                        "is_ocr": enable_ocr,
                    }
                )

    # 使用submit_file_url_task处理URLs
    try:
        result_path = await get_client().process_file_to_markdown(
            lambda urls, o: get_client().submit_file_url_task(
                urls,
                o,
                language=language,
                page_ranges=page_ranges,
            ),
            urls_to_process,
            enable_ocr,
            output_dir,
        )
        return {"status": "success", "result_path": result_path}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def convert_file_path(
    file_path: str,
    enable_ocr: bool = False,
    language: str = "ch",
    page_ranges: str | None = None,
) -> Dict[str, Any]:
    """
    将本地文件转换为Markdown格式。支持单个或多个文件批量处理。

    返回:
        成功: {"status": "success", "result_path": "输出目录路径"}
        失败: {"status": "error", "error": "错误信息"}
    """

    files_to_process = None

    # 检查是否为字典或字典列表格式的文件配置
    if isinstance(file_path, dict):
        # 单个文件配置字典
        files_to_process = file_path
    elif (
        isinstance(file_path, list)
        and len(file_path) > 0
        and isinstance(file_path[0], dict)
    ):
        # 文件配置字典列表
        files_to_process = file_path
    elif isinstance(file_path, str):
        # 检查是否为 JSON 字符串格式的多文件配置
        if file_path.strip().startswith("[") and file_path.strip().endswith("]"):
            try:
                # 尝试解析 JSON 字符串为文件配置列表
                file_configs = json.loads(file_path)
                if not isinstance(file_configs, list):
                    raise ValueError("JSON 文件配置必须是列表格式")

                files_to_process = file_configs
            except json.JSONDecodeError:
                # 不是有效的 JSON，继续使用字符串解析方式
                pass

    if files_to_process is None:
        # 解析普通文件路径列表
        file_paths = parse_list_input(file_path)

        if not file_paths:
            raise ValueError("未提供有效的文件路径")

        if len(file_paths) == 1:
            # 单个文件处理
            files_to_process = {
                "path": file_paths[0],
                "is_ocr": enable_ocr,
            }
        else:
            # 多个文件路径，转换为文件配置列表
            files_to_process = []
            for i, path in enumerate(file_paths):
                files_to_process.append(
                    {
                        "path": path,
                        "is_ocr": enable_ocr,
                    }
                )

    # 使用submit_file_task处理文件
    try:
        result_path = await get_client().process_file_to_markdown(
            lambda files, o: get_client().submit_file_task(
                files,
                o,
                language=language,
                page_ranges=page_ranges,
            ),
            files_to_process,
            enable_ocr,
            output_dir,
        )
        return {"status": "success", "result_path": result_path}
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "params": {
                "file_path": file_path,
                "enable_ocr": enable_ocr,
                "language": language,
            },
        }


async def local_parse_file(
    file_path: str,
    parse_method: str = "auto",
) -> Dict[str, Any]:
    """
    根据环境变量设置使用本地或远程API解析文件。

    返回:
        成功: {"status": "success", "result": 处理结果} 或 {"status": "success", "result_path": "输出目录路径"}
        失败: {"status": "error", "error": "错误信息"}
    """
    file_path = Path(file_path)

    # 检查文件是否存在
    if not file_path.exists():
        return {"status": "error", "error": f"文件不存在: {file_path}"}

    try:
        # 根据环境变量决定使用本地API还是远程API
        if config.USE_LOCAL_API:
            config.logger.debug(f"使用本地API: {config.LOCAL_MINERU_API_BASE}")
            return await _parse_file_local(
                file_path=str(file_path),
                parse_method=parse_method,
            )
        else:
            return {"status": "error", "error": "远程API未配置"}
    except Exception as e:
        config.logger.error(f"解析文件时出错: {str(e)}")
        return {"status": "error", "error": str(e)}


async def read_converted_file(
    file_path: str,
) -> Dict[str, Any]:
    """
    读取解析后的文件内容。主要支持Markdown和其他文本文件格式。

    返回:
        成功: {"status": "success", "content": "文件内容"}
        失败: {"status": "error", "error": "错误信息"}
    """
    try:
        target_file = Path(file_path)
        parent_dir = target_file.parent
        suffix = target_file.suffix.lower()

        # 支持的文本文件格式
        text_extensions = [".md", ".txt", ".json", ".html", ".tex", ".latex"]

        if suffix not in text_extensions:
            return {
                "status": "error",
                "error": f"不支持的文件格式: {suffix}。目前仅支持以下格式: {', '.join(text_extensions)}",
            }

        if not target_file.exists():
            if not parent_dir.exists():
                return {"status": "error", "error": f"目录 {parent_dir} 不存在"}

            # 递归搜索所有子目录下的同后缀文件
            similar_files_paths = [
                str(f) for f in parent_dir.rglob(f"*{suffix}") if f.is_file()
            ]

            if similar_files_paths:
                if len(similar_files_paths) == 1:
                    # 如果只找到一个文件，直接读取并返回内容
                    alternative_file = similar_files_paths[0]
                    try:
                        with open(alternative_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        return {
                            "status": "success",
                            "content": content,
                            "message": f"未找到文件 {target_file.name}，但找到了 {Path(alternative_file).name}，已返回其内容",
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "error": f"尝试读取替代文件时出错: {str(e)}",
                        }
                else:
                    # 如果找到多个文件，提供建议列表
                    suggestion = f"你是否在找: {', '.join(similar_files_paths)}?"
                    return {
                        "status": "error",
                        "error": f"文件 {target_file.name} 不存在。在 {parent_dir} 及其子目录下找到以下同类型文件。{suggestion}",
                    }
            else:
                return {
                    "status": "error",
                    "error": f"文件 {target_file.name} 不存在，且在目录 {parent_dir} 及其子目录下未找到其他 {suffix} 文件。",
                }

        # 以文本模式读取
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        return {"status": "success", "content": content}

    except Exception as e:
        config.logger.error(f"读取文件时出错: {str(e)}")
        return {"status": "error", "error": str(e)}


async def find_and_read_markdown_content(result_path: str) -> Dict[str, Any]:
    """
    在给定的路径中寻找并读取Markdown文件内容。
    查找所有可能的文件位置，返回所有找到的有效内容。

    Args:
        result_path: 结果目录路径

    Returns:
        Dict[str, Any]: 包含所有文件内容或错误信息的字典
    """
    if not result_path:
        return {"status": "warning", "message": "未提供有效的结果路径"}

    base_path = Path(result_path)
    if not base_path.exists():
        return {"status": "warning", "message": f"结果路径不存在: {result_path}"}

    # 使用集合来存储文件路径，确保唯一性
    unique_files = set()

    # 添加常见文件名
    common_files = [
        base_path / "full.md",
        base_path / "full.txt",
        base_path / "output.md",
        base_path / "result.md",
    ]
    for f in common_files:
        if f.exists():
            unique_files.add(str(f))

    # 添加子目录中的常见文件名
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            subdir_files = [
                subdir / "full.md",
                subdir / "full.txt",
                subdir / "output.md",
                subdir / "result.md",
            ]
            for f in subdir_files:
                if f.exists():
                    unique_files.add(str(f))

    # 查找所有的.md和.txt文件
    for md_file in base_path.glob("**/*.md"):
        unique_files.add(str(md_file))
    for txt_file in base_path.glob("**/*.txt"):
        unique_files.add(str(txt_file))

    # 将集合转换回Path对象列表
    possible_files = [Path(f) for f in unique_files]

    config.logger.debug(f"找到 {len(possible_files)} 个可能的文件")

    # 收集所有找到的有效文件内容
    found_contents = []

    # 尝试读取每个可能的文件
    for file_path in possible_files:
        if file_path.exists():
            result = await read_converted_file(str(file_path))
            if result["status"] == "success":
                config.logger.debug(f"成功读取文件内容: {file_path}")
                found_contents.append(
                    {"file_path": str(file_path), "content": result["content"]}
                )

    # 如果找到了文件内容
    if found_contents:
        config.logger.debug(f"在结果目录中找到了 {len(found_contents)} 个可读取的文件")
        # 如果只找到一个文件，保持向后兼容的返回格式
        if len(found_contents) == 1:
            return {
                "status": "success",
                "content": found_contents[0]["content"],
                "file_path": found_contents[0]["file_path"],
            }
        # 如果找到多个文件，返回内容列表
        else:
            return {"status": "success", "contents": found_contents}

    # 如果没有找到任何有效的文件
    return {
        "status": "warning",
        "message": f"无法在结果目录中找到可读取的Markdown文件: {result_path}",
    }


async def _process_conversion_result(
    result: Dict[str, Any], source: str, is_url: bool = False
) -> Dict[str, Any]:
    """
    处理转换结果，统一格式化输出。

    Args:
        result: 转换函数返回的结果
        source: 源文件路径或URL
        is_url: 是否为URL

    Returns:
        格式化后的结果字典
    """
    filename = source.split("/")[-1]
    if is_url and "?" in filename:
        filename = filename.split("?")[0]
    elif not is_url:
        filename = Path(source).name

    base_result = {
        "filename": filename,
        "source_url" if is_url else "source_path": source,
    }

    if result["status"] == "success":
        # 获取result_path，可能是字符串或字典
        result_path = result.get("result_path")

        # 记录调试信息
        config.logger.debug(f"处理结果 result_path 类型: {type(result_path)}")

        if result_path:
            # 情况1: result_path是字典且包含results字段（批量处理结果）
            if isinstance(result_path, dict) and "results" in result_path:
                config.logger.debug("检测到批量处理结果格式")

                # 查找与当前源文件匹配的结果
                for item in result_path.get("results", []):
                    if item.get("filename") == filename or (
                        not is_url and Path(source).name == item.get("filename")
                    ):
                        # 直接返回匹配项的状态，无论是success还是error
                        if item.get("status") == "success" and "content" in item:
                            base_result.update(
                                {
                                    "status": "success",
                                    "content": item.get("content", ""),
                                }
                            )
                            # 如果有extract_path，也添加进去
                            if "extract_path" in item:
                                base_result["extract_path"] = item["extract_path"]
                            return base_result
                        elif item.get("status") == "error":
                            # 处理失败的文件，直接返回error状态
                            base_result.update(
                                {
                                    "status": "error",
                                    "error_message": item.get(
                                        "error_message", "文件处理失败"
                                    ),
                                }
                            )
                            return base_result

                # 如果没有找到匹配的结果，但有extract_dir，尝试从那里读取
                if "extract_dir" in result_path:
                    config.logger.debug(
                        f"尝试从extract_dir读取: {result_path['extract_dir']}"
                    )
                    try:
                        content_result = await find_and_read_markdown_content(
                            result_path["extract_dir"]
                        )
                        if content_result.get("status") == "success":
                            base_result.update(
                                {
                                    "status": "success",
                                    "content": content_result.get("content", ""),
                                    "extract_path": result_path["extract_dir"],
                                }
                            )
                            return base_result
                    except Exception as e:
                        config.logger.error(f"从extract_dir读取内容时出错: {str(e)}")

                # 如果上述方法都失败，返回错误
                base_result.update(
                    {
                        "status": "error",
                        "error_message": "未能在批量处理结果中找到匹配的内容",
                    }
                )

            # 情况2: result_path是字符串（传统格式）
            elif isinstance(result_path, str):
                config.logger.debug(f"处理传统格式结果路径: {result_path}")
                content_result = await find_and_read_markdown_content(result_path)
                if content_result.get("status") == "success":
                    base_result.update(
                        {
                            "status": "success",
                            "content": content_result.get("content", ""),
                            "extract_path": result_path,
                        }
                    )
                else:
                    base_result.update(
                        {
                            "status": "error",
                            "error_message": f"无法读取转换结果: {content_result.get('message', '')}",
                        }
                    )

            # 情况3: result_path是其他类型的字典（尝试处理）
            elif isinstance(result_path, dict):
                config.logger.debug(f"处理其他字典格式: {result_path}")
                # 尝试从字典中提取可能的路径
                extract_path = (
                    result_path.get("extract_dir")
                    or result_path.get("path")
                    or result_path.get("dir")
                )
                if extract_path and isinstance(extract_path, str):
                    try:
                        content_result = await find_and_read_markdown_content(
                            extract_path
                        )
                        if content_result.get("status") == "success":
                            base_result.update(
                                {
                                    "status": "success",
                                    "content": content_result.get("content", ""),
                                    "extract_path": extract_path,
                                }
                            )
                            return base_result
                    except Exception as e:
                        config.logger.error(f"从extract_path读取内容时出错: {str(e)}")

                # 如果没有找到有效路径，返回错误
                base_result.update(
                    {"status": "error", "error_message": "转换结果格式无法识别"}
                )
            else:
                # 情况4: result_path是其他类型（错误）
                base_result.update(
                    {
                        "status": "error",
                        "error_message": f"无法识别的result_path类型: {type(result_path)}",
                    }
                )
        else:
            base_result.update(
                {"status": "error", "error_message": "转换成功但未返回结果路径"}
            )
    else:
        base_result.update(
            {"status": "error", "error_message": result.get("error", "未知错误")}
        )

    return base_result


@mcp.tool()
async def parse_documents(
    file_sources: Annotated[
        str,
        Field(
            description="""文件路径或URL，支持以下格式:
            - 单个路径或URL: "/path/to/file.pdf" 或 "https://example.com/document.pdf"
            - 多个路径或URL(逗号分隔): "/path/to/file1.pdf, /path/to/file2.pdf" 或
              "https://example.com/doc1.pdf, https://example.com/doc2.pdf"
            - 混合路径和URL: "/path/to/file.pdf, https://example.com/document.pdf"
            (支持pdf、ppt、pptx、doc、docx以及图片格式jpg、jpeg、png)"""
        ),
    ],
    # 通用参数
    enable_ocr: Annotated[bool, Field(description="启用OCR识别,默认False")] = False,
    language: Annotated[
        str, Field(description='文档语言，默认"ch"中文，可选"en"英文等')
    ] = "ch",
    # 远程API参数
    page_ranges: Annotated[
        str | None,
        Field(
            description='指定页码范围，格式为逗号分隔的字符串。例如："2,4-6"：表示选取第2页、第4页至第6页；"2--2"：表示从第2页一直选取到倒数第二页。（远程API）,默认None'
        ),
    ] = None,
) -> Dict[str, Any]:
    """
    统一接口，将文件转换为Markdown格式。支持本地文件和URL，会根据USE_LOCAL_API配置自动选择合适的处理方式。

    当USE_LOCAL_API=true时:
    - 会过滤掉http/https开头的URL路径
    - 对本地文件使用本地API进行解析

    当USE_LOCAL_API=false时:
    - 将http/https开头的路径使用convert_file_url处理
    - 将其他路径使用convert_file_path处理

    处理完成后，会自动尝试读取转换后的文件内容并返回。

    返回:
        成功: {"status": "success", "content": "文件内容"} 或 {"status": "success", "results": [处理结果列表]}
        失败: {"status": "error", "error": "错误信息"}
    """
    # 解析路径列表
    sources = parse_list_input(file_sources)
    if not sources:
        return {"status": "error", "error": "未提供有效的文件路径或URL"}

    # 去重处理，使用字典来保持原始顺序
    sources = list(dict.fromkeys(sources))

    config.logger.debug(f"去重后的文件路径: {sources}")

    # 记录去重信息
    original_count = len(parse_list_input(file_sources))
    unique_count = len(sources)
    if original_count > unique_count:
        config.logger.debug(
            f"检测到重复路径，已自动去重: {original_count} -> {unique_count}"
        )

    # 将路径分类
    url_paths = []
    file_paths = []

    for source in sources:
        if source.lower().startswith(("http://", "https://")):
            url_paths.append(source)
        else:
            file_paths.append(source)

    results = []

    # 根据USE_LOCAL_API决定处理方式
    if config.USE_LOCAL_API:
        # 在本地API模式下，只处理本地文件路径
        if not file_paths:
            return {
                "status": "warning",
                "message": "在本地API模式下，无法处理URL，且未提供有效的本地文件路径",
            }

        config.logger.info(f"使用本地API处理 {len(file_paths)} 个文件")

        # 逐个处理本地文件
        for path in file_paths:
            try:
                # 跳过不存在的文件
                if not Path(path).exists():
                    results.append(
                        {
                            "filename": Path(path).name,
                            "source_path": path,
                            "status": "error",
                            "error_message": f"文件不存在: {path}",
                        }
                    )
                    continue

                result = await local_parse_file(
                    file_path=path,
                    parse_method=(
                        "ocr" if enable_ocr else "txt"
                    ),  # 如果启用OCR，使用ocr，否则使用txt
                )

                # 添加文件名信息
                result_with_filename = {
                    "filename": Path(path).name,
                    "source_path": path,
                    **result,
                }
                results.append(result_with_filename)

            except Exception as e:
                # 处理文件时出现异常，记录错误但继续处理下一个文件
                config.logger.error(f"处理文件 {path} 时出现错误: {str(e)}")
                results.append(
                    {
                        "filename": Path(path).name,
                        "source_path": path,
                        "status": "error",
                        "error_message": f"处理文件时出现异常: {str(e)}",
                    }
                )

    else:
        # 在远程API模式下，分别处理URL和本地文件路径
        if url_paths:
            config.logger.info(f"使用远程API处理 {len(url_paths)} 个文件URL")

            try:
                # 调用convert_file_url处理URLs
                url_result = await convert_file_url(
                    url=",".join(url_paths),
                    enable_ocr=enable_ocr,
                    language=language,
                    page_ranges=page_ranges,
                )

                if url_result["status"] == "success":
                    # 为每个URL生成对应的结果
                    for url in url_paths:
                        result_item = await _process_conversion_result(
                            url_result, url, is_url=True
                        )
                        results.append(result_item)
                else:
                    # 转换失败，为所有URL添加错误结果
                    for url in url_paths:
                        results.append(
                            {
                                "filename": url.split("/")[-1].split("?")[0],
                                "source_url": url,
                                "status": "error",
                                "error_message": url_result.get("error", "URL处理失败"),
                            }
                        )

            except Exception as e:
                config.logger.error(f"处理URL时出现错误: {str(e)}")
                for url in url_paths:
                    results.append(
                        {
                            "filename": url.split("/")[-1].split("?")[0],
                            "source_url": url,
                            "status": "error",
                            "error_message": f"处理URL时出现异常: {str(e)}",
                        }
                    )

        if file_paths:
            config.logger.info(f"使用远程API处理 {len(file_paths)} 个本地文件")

            # 过滤出存在的文件
            existing_files = []
            for file_path in file_paths:
                if not Path(file_path).exists():
                    results.append(
                        {
                            "filename": Path(file_path).name,
                            "source_path": file_path,
                            "status": "error",
                            "error_message": f"文件不存在: {file_path}",
                        }
                    )
                else:
                    existing_files.append(file_path)

            if existing_files:
                try:
                    # 调用convert_file_path处理本地文件
                    file_result = await convert_file_path(
                        file_path=",".join(existing_files),
                        enable_ocr=enable_ocr,
                        language=language,
                        page_ranges=page_ranges,
                    )

                    config.logger.debug(f"file_result: {file_result}")

                    if file_result["status"] == "success":
                        # 为每个文件生成对应的结果
                        for file_path in existing_files:
                            result_item = await _process_conversion_result(
                                file_result, file_path, is_url=False
                            )
                            results.append(result_item)
                    else:
                        # 转换失败，为所有文件添加错误结果
                        for file_path in existing_files:
                            results.append(
                                {
                                    "filename": Path(file_path).name,
                                    "source_path": file_path,
                                    "status": "error",
                                    "error_message": file_result.get(
                                        "error", "文件处理失败"
                                    ),
                                }
                            )

                except Exception as e:
                    config.logger.error(f"处理本地文件时出现错误: {str(e)}")
                    for file_path in existing_files:
                        results.append(
                            {
                                "filename": Path(file_path).name,
                                "source_path": file_path,
                                "status": "error",
                                "error_message": f"处理文件时出现异常: {str(e)}",
                            }
                        )

    # 处理结果为空的情况
    if not results:
        return {"status": "error", "error": "未处理任何文件"}

    # 计算成功和失败的统计信息
    success_count = len([r for r in results if r.get("status") == "success"])
    error_count = len([r for r in results if r.get("status") == "error"])
    total_count = len(results)

    # 只有一个结果时，直接返回该结果（保持向后兼容）
    if len(results) == 1:
        result = results[0].copy()
        # 为了向后兼容，移除新增的字段
        if "filename" in result:
            del result["filename"]
        if "source_path" in result:
            del result["source_path"]
        if "source_url" in result:
            del result["source_url"]
        return result

    # 多个结果时，返回详细的结果列表
    # 根据成功/失败情况决定整体状态
    overall_status = "success"
    if success_count == 0:
        # 所有文件都失败
        overall_status = "error"
    elif error_count > 0:
        # 有部分文件失败，但不是全部
        overall_status = "partial_success"

    return {
        "status": overall_status,
        "results": results,
        "summary": {
            "total_files": total_count,
            "success_count": success_count,
            "error_count": error_count,
        },
    }


@mcp.tool()
async def get_ocr_languages() -> Dict[str, Any]:
    """
    获取 OCR 支持的语言列表。

    Returns:
        Dict[str, Any]: 包含所有支持的OCR语言列表的字典
    """
    try:
        # 从language模块获取语言列表
        languages = get_language_list()
        return {"status": "success", "languages": languages}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _parse_file_local(
    file_path: str,
    parse_method: str = "auto",
) -> Dict[str, Any]:
    """
    使用本地API解析文件。

    Args:
        file_path: 要解析的文件路径
        parse_method: 解析方法
        output_dir: 输出目录

    Returns:
        Dict[str, Any]: 包含解析结果的字典
    """
    # API URL路径
    api_url = f"{config.LOCAL_MINERU_API_BASE}/file_parse"

    # 使用Path对象确保文件路径正确
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取文件二进制数据
    with open(file_path_obj, "rb") as f:
        file_data = f.read()

    # 准备用于上传文件的表单数据
    file_type = file_path_obj.suffix.lower()
    form_data = aiohttp.FormData()
    form_data.add_field(
        "file", file_data, filename=file_path_obj.name, content_type=file_type
    )
    form_data.add_field("parse_method", parse_method)

    config.logger.debug(f"发送本地API请求到: {api_url}")
    config.logger.debug(f"上传文件: {file_path_obj.name} (大小: {len(file_data)} 字节)")

    # 发送请求
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    config.logger.error(
                        f"API返回错误状态码: {response.status}, 错误信息: {error_text}"
                    )
                    raise RuntimeError(f"API返回错误: {response.status}, {error_text}")

                result = await response.json()

                config.logger.debug(f"本地API响应: {result}")

                # 处理响应
                if "error" in result:
                    return {"status": "error", "error": result["error"]}

                return {"status": "success", "result": result}
    except aiohttp.ClientError as e:
        error_msg = f"与本地API通信时出错: {str(e)}"
        config.logger.error(error_msg)
        raise RuntimeError(error_msg)
