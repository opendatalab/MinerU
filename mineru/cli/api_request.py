# Copyright (c) Opendatalab. All rights reserved.
import json
from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import File, Form, HTTPException, Request, UploadFile

from mineru.cli.public_http_client_policy import validate_public_http_client_request

ALLOWED_PARSE_METHODS = {"auto", "txt", "ocr"}
SWAGGER_UI_FILE_ARRAY_SCHEMA_EXTRA = {
    # Swagger UI 5 currently fails to render a usable multi-file picker when
    # FastAPI emits OpenAPI 3.1 byte arrays with contentMediaType.
    "items": {"type": "string", "format": "binary"}
}


@dataclass
class ParseRequestOptions:
    """保存公开解析接口共用的表单参数，供 API 与 Router 复用。"""

    files: list[UploadFile]
    lang_list: list[str]
    backend: str
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
    server_headers: Optional[dict[str, str]]


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
        list[str],
        Form(
            description="""(Adapted only for pipeline and hybrid backend)Input the languages in the pdf to improve OCR accuracy.Options:
- ch: Chinese, English, Chinese Traditional.
- ch_lite: Chinese, English, Chinese Traditional, Japanese.
- ch_server: Chinese, English, Chinese Traditional, Japanese.
- en: English.
- korean: Korean, English.
- japan: Chinese, English, Chinese Traditional, Japanese.
- chinese_cht: Chinese, English, Chinese Traditional, Japanese.
- ta: Tamil, English.
- te: Telugu, English.
- ka: Kannada.
- th: Thai, English.
- el: Greek, English.
- latin: French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua.
- arabic: Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English.
- east_slavic: Russian, Belarusian, Ukrainian, English.
- cyrillic: Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English.
- devanagari: Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English.
""",
        ),
    ] = ["ch"],
    backend: Annotated[
        str,
        Form(
            description="""The backend for parsing:
- pipeline: More general, supports multiple languages, hallucination-free.
- vlm-auto-engine: High accuracy via local computing power, supports Chinese and English documents only.
- vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers), supports Chinese and English documents only.
- hybrid-auto-engine: Next-generation high accuracy solution via local computing power, supports multiple languages.
- hybrid-http-client: High accuracy via remote computing power but requires a little local computing power(client suitable for openai-compatible servers), supports multiple languages.""",
        ),
    ] = "hybrid-auto-engine",
    parse_method: Annotated[
        str,
        Form(
            description="""(Adapted only for pipeline and hybrid backend)The method for parsing PDF:
- auto: Automatically determine the method based on the file type
- txt: Use text extraction method
- ocr: Use OCR method for image-based PDFs
""",
        ),
    ] = "auto",
    formula_enable: Annotated[
        bool,
        Form(description="Enable formula parsing."),
    ] = True,
    table_enable: Annotated[
        bool,
        Form(description="Enable table parsing."),
    ] = True,
    image_analysis: Annotated[
        bool,
        Form(description="Enable image/chart analysis for VLM and hybrid backends."),
    ] = True,
    server_url: Annotated[
        Optional[str],
        Form(
            description="(Adapted only for <vlm/hybrid>-http-client backend)openai compatible server url, e.g., http://127.0.0.1:30000",
        ),
    ] = None,
    return_md: Annotated[
        bool,
        Form(description="Return markdown content in response"),
    ] = True,
    return_middle_json: Annotated[
        bool,
        Form(description="Return middle JSON in response"),
    ] = False,
    return_model_output: Annotated[
        bool,
        Form(description="Return model output JSON in response"),
    ] = False,
    return_content_list: Annotated[
        bool,
        Form(description="Return content list JSON in response"),
    ] = False,
    return_images: Annotated[
        bool,
        Form(description="Return extracted images in response"),
    ] = False,
    response_format_zip: Annotated[
        bool,
        Form(description="Return results as a ZIP file instead of JSON"),
    ] = False,
    return_original_file: Annotated[
        bool,
        Form(
            description=(
                "Include the processed original input file in the ZIP result; "
                "ignored unless response_format_zip=true"
            ),
        ),
    ] = False,
    client_side_output_generation: Annotated[
        bool,
        Form(
            description=(
                "Defer final markdown/content-list generation to the client. "
                "When enabled, the server returns staged middle JSON and images."
            ),
        ),
    ] = False,
    start_page_id: Annotated[
        int,
        Form(description="The starting page for PDF parsing, beginning from 0"),
    ] = 0,
    end_page_id: Annotated[
        int,
        Form(description="The ending page for PDF parsing, beginning from 0"),
    ] = 99999,
    server_headers: Annotated[
        Optional[str],
        Form(
            description=(
                "(Adapted only for <vlm/hybrid>-http-client backend) "
                "JSON object specifying custom HTTP headers to include when "
                "connecting to the remote OpenAI-compatible server, e.g. "
                '{\"Authorization\": \"Bearer token\"}'
            ),
        ),
    ] = None,
) -> ParseRequestOptions:
    """解析 API/Router 共用的 multipart 表单，并保持 Swagger 参数同源。"""
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
        return_model_output = False
        return_content_list = False
        return_images = True

    effective_return_original_file = return_original_file and response_format_zip
    parsed_server_headers: Optional[dict[str, str]] = None
    if server_headers is not None:
        try:
            parsed_server_headers = json.loads(server_headers)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid server_headers JSON: {exc}",
            )
        if not isinstance(parsed_server_headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in parsed_server_headers.items()
        ):
            raise HTTPException(
                status_code=400,
                detail="server_headers must be a JSON object with string keys and string values",
            )
    return ParseRequestOptions(
        files=files,
        lang_list=lang_list,
        backend=backend,
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
        server_headers=parsed_server_headers,
    )
