# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 多格式渲染的统一公共入口。"""

from __future__ import annotations

from typing import Any, Literal, overload

from ..types import MiddleJson

from .contracts import (
    ContentListRenderOptions,
    ContentListV2RenderOptions,
    DocxRenderOptions,
    EpubRenderOptions,
    HtmlRenderOptions,
    LatexRenderOptions,
    MarkdownRenderOptions,
    PdfRenderOptions,
    RenderFormat,
    RenderOptions,
    RenderOutput,
    StructuredContentRenderOptions,
)
from .content_list import render_content_list
from .content_list_v2 import render_content_list_v2
from .docx import render_docx
from .epub import render_epub
from .html import render_html
from .latex import render_latex
from .markdown import render_markdown
from .pdf import render_pdf
from .structured_content import render_structured_content


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.MARKDOWN],
    *,
    options: MarkdownRenderOptions | None = None,
) -> str:
    """声明 Markdown 目标对应的字符串返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.HTML],
    *,
    options: HtmlRenderOptions | None = None,
) -> str:
    """声明 HTML 目标对应的字符串返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.LATEX],
    *,
    options: LatexRenderOptions | None = None,
) -> str:
    """声明 LaTeX 目标对应的字符串返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.DOCX],
    *,
    options: DocxRenderOptions | None = None,
) -> bytes:
    """声明 DOCX 目标对应的字节返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.EPUB],
    *,
    options: EpubRenderOptions | None = None,
) -> bytes:
    """声明 EPUB 目标对应的字节返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.PDF],
    *,
    options: PdfRenderOptions | None = None,
) -> bytes:
    """声明 PDF 目标对应的字节返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.STRUCTURED_CONTENT],
    *,
    options: StructuredContentRenderOptions | None = None,
) -> dict[str, Any]:
    """声明 Structured Content 目标对应的字典返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.CONTENT_LIST],
    *,
    options: ContentListRenderOptions | None = None,
) -> list[dict[str, Any]]:
    """声明 Content List V1 目标对应的扁平列表返回类型。"""
    ...


@overload
def render(
    middle_json: MiddleJson,
    output_format: Literal[RenderFormat.CONTENT_LIST_V2],
    *,
    options: ContentListV2RenderOptions | None = None,
) -> list[list[dict[str, Any]]]:
    """声明 Content List V2 目标对应的按页列表返回类型。"""
    ...


def render(
    middle_json: MiddleJson,
    output_format: RenderFormat,
    *,
    options: RenderOptions | None = None,
) -> RenderOutput:
    """按严格目标格式和对应选项把 MiddleJson 渲染为原生结果。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("render expects a MiddleJson instance")
    if not isinstance(output_format, RenderFormat):
        raise TypeError("output_format must be a RenderFormat value")

    if output_format is RenderFormat.MARKDOWN:
        resolved_options = options if options is not None else MarkdownRenderOptions()
        if not isinstance(resolved_options, MarkdownRenderOptions):
            raise TypeError("MARKDOWN output requires MarkdownRenderOptions")
        return render_markdown(
            middle_json,
            mode=resolved_options.mode,
            asset_base_url=resolved_options.asset_base_url,
            image_renderer=resolved_options.image_renderer,
        )

    if output_format is RenderFormat.HTML:
        resolved_options = options if options is not None else HtmlRenderOptions()
        if not isinstance(resolved_options, HtmlRenderOptions):
            raise TypeError("HTML output requires HtmlRenderOptions")
        return render_html(
            middle_json,
            mode=resolved_options.mode,
            asset_base_url=resolved_options.asset_base_url,
            standalone=resolved_options.standalone,
            document_title=resolved_options.document_title,
        )

    if output_format is RenderFormat.LATEX:
        resolved_options = options if options is not None else LatexRenderOptions()
        if not isinstance(resolved_options, LatexRenderOptions):
            raise TypeError("LATEX output requires LatexRenderOptions")
        return render_latex(
            middle_json,
            asset_base_path=resolved_options.asset_base_path,
            document_title=resolved_options.document_title,
        )

    if output_format is RenderFormat.DOCX:
        resolved_options = options if options is not None else DocxRenderOptions()
        if not isinstance(resolved_options, DocxRenderOptions):
            raise TypeError("DOCX output requires DocxRenderOptions")
        return render_docx(
            middle_json,
            asset_resolver=resolved_options.asset_resolver,
        )

    if output_format is RenderFormat.EPUB:
        resolved_options = options if options is not None else EpubRenderOptions()
        if not isinstance(resolved_options, EpubRenderOptions):
            raise TypeError("EPUB output requires EpubRenderOptions")
        return render_epub(
            middle_json,
            title=resolved_options.title,
            authors=resolved_options.authors,
            language=resolved_options.language,
            identifier=resolved_options.identifier,
            modified_at=resolved_options.modified_at,
            asset_resolver=resolved_options.asset_resolver,
        )

    if output_format is RenderFormat.PDF:
        resolved_options = options if options is not None else PdfRenderOptions()
        if not isinstance(resolved_options, PdfRenderOptions):
            raise TypeError("PDF output requires PdfRenderOptions")
        return render_pdf(
            middle_json,
            asset_resolver=resolved_options.asset_resolver,
            document_title=resolved_options.document_title,
        )

    if output_format is RenderFormat.STRUCTURED_CONTENT:
        resolved_options = options if options is not None else StructuredContentRenderOptions()
        if not isinstance(resolved_options, StructuredContentRenderOptions):
            raise TypeError("STRUCTURED_CONTENT output requires StructuredContentRenderOptions")
        return render_structured_content(
            middle_json,
            asset_base_url=resolved_options.asset_base_url,
        )

    if output_format is RenderFormat.CONTENT_LIST:
        resolved_options = options if options is not None else ContentListRenderOptions()
        if not isinstance(resolved_options, ContentListRenderOptions):
            raise TypeError("CONTENT_LIST output requires ContentListRenderOptions")
        return render_content_list(
            middle_json,
            asset_base_url=resolved_options.asset_base_url,
        )

    if output_format is RenderFormat.CONTENT_LIST_V2:
        resolved_options = options if options is not None else ContentListV2RenderOptions()
        if not isinstance(resolved_options, ContentListV2RenderOptions):
            raise TypeError("CONTENT_LIST_V2 output requires ContentListV2RenderOptions")
        return render_content_list_v2(
            middle_json,
            asset_base_url=resolved_options.asset_base_url,
        )

    raise ValueError(f"Unsupported RenderFormat: {output_format}")


__all__ = ["render"]
