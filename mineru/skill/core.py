# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 核心解析逻辑"""

import asyncio
import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from mineru.cli.backend_options import normalize_backend, validate_effort
from mineru.cli.common import aio_do_parse, do_parse, read_fn
from mineru.cli.output_paths import build_parse_dir
from mineru.skill.config import (
    DEFAULT_BACKEND,
    DEFAULT_EFFORT,
    DEFAULT_FORMULA_ENABLE,
    DEFAULT_IMAGE_ANALYSIS,
    DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARSE_METHOD,
    DEFAULT_PARSE_TIMEOUT,
    DEFAULT_TABLE_ENABLE,
    SUPPORTED_SUFFIXES,
)
from mineru.skill.result import ParseResult
from mineru.utils.enum_class import MakeMode


@dataclass
class ParseOptions:
    """OCR/解析配置选项"""

    backend: str = DEFAULT_BACKEND
    parse_method: str = DEFAULT_PARSE_METHOD
    language: str = DEFAULT_LANGUAGE
    formula_enable: bool = DEFAULT_FORMULA_ENABLE
    table_enable: bool = DEFAULT_TABLE_ENABLE
    image_analysis: bool = DEFAULT_IMAGE_ANALYSIS
    effort: str = DEFAULT_EFFORT
    server_url: Optional[str] = None
    start_page_id: int = 0
    end_page_id: Optional[int] = None
    output_dir: Optional[Union[str, Path]] = None


async def parse_file(
    input_path: Union[str, Path],
    options: Optional[ParseOptions] = None,
    timeout: Optional[float] = None,
) -> ParseResult:
    """异步解析单个 PDF/图片/Office 文件

    Args:
        input_path: 输入文件路径
        options: 解析配置选项
        timeout: 解析超时时间（秒），默认 600 秒

    Returns:
        ParseResult 解析结果

    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 不支持的文件类型或非法参数
        TimeoutError: 解析超时
        RuntimeError: 解析过程中发生其他错误
    """
    options = options or ParseOptions()
    timeout = timeout or DEFAULT_PARSE_TIMEOUT

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    suffix = input_path.suffix.lstrip(".").lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文件类型: {suffix}。"
            f"支持的后缀: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    # 校验后端与 effort
    backend = normalize_backend(options.backend)
    effort = validate_effort(options.effort)

    # 准备输出目录
    output_dir = Path(options.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取文件 bytes（read_fn 会自动将图片转为 PDF bytes）
    file_bytes = read_fn(input_path)
    file_name = input_path.stem

    # 执行解析，带超时控制
    try:
        await asyncio.wait_for(
            _do_parse_impl(
                output_dir=output_dir,
                file_name=file_name,
                file_bytes=file_bytes,
                options=options,
                backend=backend,
                effort=effort,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"解析超时（{timeout} 秒）。文档可能过大或后端负载过高。"
        ) from exc

    # 加载并返回结果
    return _load_result(output_dir, file_name, backend, options.parse_method)


async def _do_parse_impl(
    output_dir: Path,
    file_name: str,
    file_bytes: bytes,
    options: ParseOptions,
    backend: str,
    effort: str,
) -> None:
    """调用 MinerU 核心解析逻辑"""
    await aio_do_parse(
        output_dir=str(output_dir),
        pdf_file_names=[file_name],
        pdf_bytes_list=[file_bytes],
        p_lang_list=[options.language],
        backend=backend,
        parse_method=options.parse_method,
        formula_enable=options.formula_enable,
        table_enable=options.table_enable,
        server_url=options.server_url,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=options.start_page_id,
        end_page_id=options.end_page_id,
        image_analysis=options.image_analysis,
        client_side_output_generation=False,
        effort=effort,
    )


def _load_result(
    output_dir: Path,
    file_name: str,
    backend: str,
    parse_method: str,
) -> ParseResult:
    """从输出目录加载解析结果"""
    parse_dir = build_parse_dir(output_dir, file_name, backend, parse_method)

    if not parse_dir.exists():
        raise RuntimeError(f"解析结果目录不存在: {parse_dir}")

    # 读取 markdown
    md_path = parse_dir / f"{file_name}.md"
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    # 读取 content_list
    cl_path = parse_dir / f"{file_name}_content_list.json"
    content_list = _load_json(cl_path, default=[])

    # 读取 content_list_v2
    cl_v2_path = parse_dir / f"{file_name}_content_list_v2.json"
    content_list_v2 = _load_json(cl_v2_path, default=[])

    # 读取 middle_json
    mj_path = parse_dir / f"{file_name}_middle.json"
    middle_json = _load_json(mj_path, default={})

    # 读取 model_output
    mo_path = parse_dir / f"{file_name}_model.json"
    model_output = _load_json(mo_path, default=None)

    # 收集图片
    images_dir = parse_dir / "images"
    images: dict[str, str] = {}
    image_paths: list[Path] = []
    if images_dir.exists():
        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file():
                continue
            mime, _ = mimetypes.guess_type(str(img_path))
            mime = mime or "image/jpeg"
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            images[img_path.name] = f"data:{mime};base64,{b64}"
            image_paths.append(img_path)

    return ParseResult(
        markdown=markdown,
        content_list=content_list,
        content_list_v2=content_list_v2,
        middle_json=middle_json,
        model_output=model_output,
        images=images,
        image_paths=image_paths,
        output_dir=parse_dir,
        file_name=file_name,
    )


def _load_json(path: Path, default: Any) -> Any:
    """安全读取 JSON 文件"""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"JSON 文件解析失败: {path}")
        return default


def parse_file_sync(
    input_path: Union[str, Path],
    options: Optional[ParseOptions] = None,
    timeout: Optional[float] = None,
) -> ParseResult:
    """同步解析单个 PDF/图片/Office 文件

    内部通过 asyncio.run() 调用异步实现。
    """
    return asyncio.run(parse_file(input_path, options, timeout))
