# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from fastapi import File, Form, HTTPException, Request, UploadFile

from mineru.cli.backend_options import (
    BACKEND_SCHEMA_EXTRA,
    DEFAULT_BACKEND,
    DEFAULT_HYBRID_EFFORT,
    HYBRID_EFFORT_SCHEMA_EXTRA,
    validate_backend as validate_public_backend,
    validate_effort as validate_public_effort,
)
from mineru.cli.public_http_client_policy import validate_public_http_client_request
from mineru.utils.ocr_language import (
    PUBLIC_OCR_LANGUAGE_SCHEMA_EXTRA,
    format_public_ocr_lang_description,
    validate_public_ocr_lang_list,
)

ALLOWED_PARSE_METHODS = {"auto", "txt", "ocr"}
SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA = {
    # Swagger UI 5 currently fails to render a usable multi-file picker when
    # FastAPI emits OpenAPI 3.1 byte arrays with contentMediaType.
    "items": {"type": "string", "format": "binary"}
}

# 表单参数的 query string 回退规格：参数名 -> (类型, 默认值)。
# 默认值与 parse_request_form 的表单默认值保持同源，防止两处漂移。
_QUERY_FALLBACK_SPECS: dict[str, tuple[str, Any]] = {
    "lang_list": ("str_list", ["ch"]),
    "backend": ("str", DEFAULT_BACKEND),
    "effort": ("str", DEFAULT_HYBRID_EFFORT),
    "parse_method": ("str", "auto"),
    "formula_enable": ("bool", True),
    "table_enable": ("bool", True),
    "image_analysis": ("bool", True),
    "server_url": ("str", None),
    "return_md": ("bool", True),
    "return_middle_json": ("bool", False),
    "return_model_output": ("bool", False),
    "return_content_list": ("bool", False),
    "return_images": ("bool", False),
    "response_format_zip": ("bool", False),
    "return_original_file": ("bool", False),
    "client_side_output_generation": ("bool", False),
    "start_page_id": ("int", 0),
    "end_page_id": ("int", 99999),
}


@dataclass
class ParseRequestOptions:
    """保存公开解析接口共用的表单参数，供 API 与 Router 复用。"""

    files: list[UploadFile]
    lang_list: list[str]
    backend: str
    effort: str
    parse_method: str
    formula_enable: bool
    table_enable: bool
    image_analysis: bool
    server_url: Optional[str]
    return_md: bool
    return_middle_json: bool
    return_model_output: bool
    return_content_list: bool
    return_images: bool
    response_format_zip: bool
    return_original_file: bool
    client_side_output_generation: bool
    start_page_id: int
    end_page_id: int


def _coerce_query_bool(raw: str, name: str) -> bool:
    """把 query string 的布尔文本矫正为 bool，非法值返回 422。"""
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(
        status_code=422,
        detail=f"Invalid boolean value for query parameter '{name}': {raw!r}",
    )


def _resolve_request_option(name: str, form_value: Any, request: Request) -> Any:
    """裁决单个参数的最终取值：表单值优先，请求体缺失时回退 query string，再回落默认值。"""
    if form_value is not None:
        return form_value
    kind, default = _QUERY_FALLBACK_SPECS[name]
    if name not in request.query_params:
        return list(default) if kind == "str_list" else default
    if kind == "str_list":
        # 兼容 lang_list=a&lang_list=b 与 lang_list=a,b 两种 query 写法。
        return [
            part
            for raw in request.query_params.getlist(name)
            for part in (chunk.strip() for chunk in raw.split(","))
            if part
        ]
    raw = request.query_params[name]
    if kind == "bool":
        return _coerce_query_bool(raw, name)
    if kind == "int":
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid integer value for query parameter '{name}': {raw!r}",
            ) from exc
    return raw if raw != "" else default


def validate_parse_method(parse_method: str) -> str:
    """校验公开 API 允许的 PDF 解析方式，避免各入口维护不同规则。"""
    if parse_method not in ALLOWED_PARSE_METHODS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid parse_method. Allowed values: "
                + ", ".join(sorted(ALLOWED_PARSE_METHODS))
            ),
        )
    return parse_method


def validate_parse_backend(backend: str) -> str:
    """校验公开 API 允许的解析后端，避免旧入口名进入下游执行链路。"""
    try:
        return validate_public_backend(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_parse_effort(effort: str) -> str:
    """校验公开 API 允许的 hybrid effort，避免非法值进入解析链路。"""
    try:
        return validate_public_effort(effort)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_parse_lang_list(lang_list: list[str]) -> list[str]:
    """校验公开 API 允许的 OCR 语言列表，避免旧语言入口进入解析链路。"""
    try:
        return validate_public_ocr_lang_list(lang_list)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def parse_request_form(
    request: Request,
    files: Annotated[
        list[UploadFile],
        File(
            description="Upload PDF, image, DOCX, PPTX, or XLSX files for parsing",
            json_schema_extra=SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA,
        ),
    ],
    lang_list: Annotated[
        Optional[list[str]],
        Form(
            description=format_public_ocr_lang_description(),
            json_schema_extra=PUBLIC_OCR_LANGUAGE_SCHEMA_EXTRA,
        ),
    ] = None,
    backend: Annotated[
        Optional[str],
        Form(
            description="""The backend for parsing:
- pipeline: More general, supports multiple languages, hallucination-free.
- vlm-engine: High accuracy via local computing power, supports Chinese and English documents only.
- vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers), supports Chinese and English documents only.
- hybrid-engine: Hybrid parsing via local computing power, supports multiple languages. Use effort to switch medium/high behavior.
- hybrid-http-client: Hybrid parsing via remote computing power but requires a little local computing power(client suitable for openai-compatible servers), supports multiple languages. Use effort to switch medium/high behavior.""",
            json_schema_extra=BACKEND_SCHEMA_EXTRA,
        ),
    ] = None,
    effort: Annotated[
        Optional[str],
        Form(
            description="""(Adapted only for hybrid backend) Hybrid parsing effort:
- medium: Faster parsing for most documents, balancing accuracy and efficiency. Image/chart analysis is disabled.
- high: Higher-accuracy parsing with image/chart analysis support, which may take longer.""",
            json_schema_extra=HYBRID_EFFORT_SCHEMA_EXTRA,
        ),
    ] = None,
    parse_method: Annotated[
        Optional[str],
        Form(
            description="""(Adapted only for pipeline and hybrid backend)The method for parsing PDF:
- auto: Automatically determine the method based on the file type
- txt: Use text extraction method
- ocr: Use OCR method for image-based PDFs
""",
        ),
    ] = None,
    formula_enable: Annotated[
        Optional[bool],
        Form(description="Enable formula parsing."),
    ] = None,
    table_enable: Annotated[
        Optional[bool],
        Form(description="Enable table parsing."),
    ] = None,
    image_analysis: Annotated[
        Optional[bool],
        Form(
            description=(
                "Enable image/chart analysis for VLM and hybrid backends. "
                "Hybrid medium effort automatically disables image/chart analysis."
            ),
        ),
    ] = None,
    server_url: Annotated[
        Optional[str],
        Form(
            description="(Adapted only for <vlm/hybrid>-http-client backend)openai compatible server url, e.g., http://127.0.0.1:30000",
        ),
    ] = None,
    return_md: Annotated[
        Optional[bool],
        Form(description="Return markdown content in response"),
    ] = None,
    return_middle_json: Annotated[
        Optional[bool],
        Form(description="Return middle JSON in response"),
    ] = None,
    return_model_output: Annotated[
        Optional[bool],
        Form(description="Return model output JSON in response"),
    ] = None,
    return_content_list: Annotated[
        Optional[bool],
        Form(description="Return content list JSON in response"),
    ] = None,
    return_images: Annotated[
        Optional[bool],
        Form(description="Return extracted images in response"),
    ] = None,
    response_format_zip: Annotated[
        Optional[bool],
        Form(description="Return results as a ZIP file instead of JSON"),
    ] = None,
    return_original_file: Annotated[
        Optional[bool],
        Form(
            description=(
                "Include the processed original input file in the ZIP result; "
                "ignored unless response_format_zip=true"
            ),
        ),
    ] = None,
    client_side_output_generation: Annotated[
        Optional[bool],
        Form(
            description=(
                "Defer final markdown/content-list generation to the client. "
                "When enabled, the server returns staged middle JSON, model output, and images."
            ),
        ),
    ] = None,
    start_page_id: Annotated[
        Optional[int],
        Form(description="The starting page for PDF parsing, beginning from 0"),
    ] = None,
    end_page_id: Annotated[
        Optional[int],
        Form(description="The ending page for PDF parsing, beginning from 0"),
    ] = None,
) -> ParseRequestOptions:
    """解析 API/Router 共用的 multipart 表单，并保持 Swagger 参数同源。

    FastAPI 的 Form() 只读取请求体；为兼容把参数放在 query string 的旧插件
    调用方式，这里对请求体未提供的参数做 query 回退：表单值严格优先于
    query 值，两者都缺失时回落到同一份默认值。
    """
    lang_list = _resolve_request_option("lang_list", lang_list, request)
    backend = _resolve_request_option("backend", backend, request)
    effort = _resolve_request_option("effort", effort, request)
    parse_method = _resolve_request_option("parse_method", parse_method, request)
    formula_enable = _resolve_request_option("formula_enable", formula_enable, request)
    table_enable = _resolve_request_option("table_enable", table_enable, request)
    image_analysis = _resolve_request_option("image_analysis", image_analysis, request)
    server_url = _resolve_request_option("server_url", server_url, request)
    return_md = _resolve_request_option("return_md", return_md, request)
    return_middle_json = _resolve_request_option("return_middle_json", return_middle_json, request)
    return_model_output = _resolve_request_option("return_model_output", return_model_output, request)
    return_content_list = _resolve_request_option("return_content_list", return_content_list, request)
    return_images = _resolve_request_option("return_images", return_images, request)
    response_format_zip = _resolve_request_option("response_format_zip", response_format_zip, request)
    return_original_file = _resolve_request_option("return_original_file", return_original_file, request)
    client_side_output_generation = _resolve_request_option(
        "client_side_output_generation", client_side_output_generation, request
    )
    start_page_id = _resolve_request_option("start_page_id", start_page_id, request)
    end_page_id = _resolve_request_option("end_page_id", end_page_id, request)

    backend = validate_parse_backend(backend)
    effort = validate_parse_effort(effort)
    validate_public_http_client_request(
        public_bind_exposed=bool(
            getattr(request.app.state, "public_bind_exposed", False)
        ),
        allow_public_http_client=bool(
            getattr(request.app.state, "allow_public_http_client", False)
        ),
        backend=backend,
        server_url=server_url,
    )
    if client_side_output_generation:
        return_md = False
        return_middle_json = True
        return_model_output = True
        return_content_list = False
        return_images = True

    effective_return_original_file = return_original_file and response_format_zip
    return ParseRequestOptions(
        files=files,
        lang_list=validate_parse_lang_list(lang_list),
        backend=backend,
        effort=effort,
        parse_method=validate_parse_method(parse_method),
        formula_enable=formula_enable,
        table_enable=table_enable,
        image_analysis=image_analysis,
        server_url=server_url,
        return_md=return_md,
        return_middle_json=return_middle_json,
        return_model_output=return_model_output,
        return_content_list=return_content_list,
        return_images=return_images,
        response_format_zip=response_format_zip,
        return_original_file=effective_return_original_file,
        client_side_output_generation=client_side_output_generation,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
