"""Doclib 内容读取服务：定位计划、渐进 Markdown、源图像和文件导出。"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from docvortex.assets import parse_image_data_uri_strict, transcode_image
from docvortex.content.tree import iter_image_payloads
from docvortex.document.pdf import PDFDocument
from docvortex.render.markdown import build_markdown_image, render_single_block

from docvortex.options import LatexDelimitersConfig
from ...errors import InvalidRequestError, NotFoundError
from ...filetypes import IMAGE_EXTENSIONS, TEXT_FILE_TYPES, is_page_range_parse_extension
from ...parser.page_range import (
    count_pages_in_range,
    expand_page_range as _expand_page_range,
    format_page_range as _page_numbers_to_range_str,
    normalize_page_range_input,
    parse_page_range_set,
)
from ...types import TIERS, BlockBase, BlockType, PageBlock, PageInfo, Tier, select_highest_cached_tier
from ..core.db import DatabaseManager
from ..core.parse_artifacts import (
    accessible_file_for_sha256,
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    parse_batch_is_readable,
)
from ..locators import ContentCursor, block_char_ref, block_ref, page_ref, parse_content_cursor
from ..rows import DocRow, ParseBatchRow, ParseRow
from ..types import (
    PARSE_STATUS_DONE,
    ContentAsset,
    ContentFormat,
    ContentNextRequest,
    ContentRange,
    ContentRequestScope,
    DocContentExportRequest,
    DocContentExportResponse,
    DocContentResponse,
    ImageFormat,
)


@dataclass(frozen=True)
class _RenderedContent:
    content: str
    content_ranges: list[ContentRange]
    truncated: bool
    last_page_no: int
    next_page_no: int | None = None
    cut_inside_page: bool = False


@dataclass(frozen=True)
class _ReadPlan:
    sha256: str
    short_id: str
    tier: Tier
    page_range: str | None
    after: str | None
    locator: str | None
    context: int
    limit: int
    format: ContentFormat
    no_marker: bool
    image_format: ImageFormat = "jpeg"
    target: ContentCursor | None = None
    next_mode: str = "parse"


@dataclass(frozen=True)
class _BlockImageSource:
    kind: str
    data_uri: str | None = None


@dataclass(frozen=True)
class _LocatorParts:
    short_id: str
    tier: Tier | None = None
    page_no: int | None = None
    block_no: int | None = None
    char_offset: int | None = None
    page_range: str | None = None
    range_selection: bool = False


_VISUAL_BLOCK_LABELS = {
    BlockType.IMAGE: "Image block",
    BlockType.TABLE: "Table block",
    BlockType.CHART: "Chart block",
    BlockType.EQUATION: "Formula block",
}


class ContentReadService:
    """协调缓存内容读取，不持有 HTTP 状态或后台任务生命周期。"""

    def __init__(self, db: DatabaseManager, *, data_dir: str, latex_delimiters: LatexDelimitersConfig) -> None:
        """绑定数据库、产物目录和公式配置快照，支持不同应用独立读取。"""
        self.latex_delimiters = latex_delimiters.model_copy(deep=True)
        self.db = db
        self.data_dir = os.path.expanduser(data_dir)

    async def read_document(
        self,
        doc_ref: str,
        *,
        tier: Tier,
        page_range: str | None = None,
        after: str | None = None,
        limit: int = 30000,
        format: str = "markdown",
        no_marker: bool = False,
    ) -> DocContentResponse:
        """读取文档解析结果，并保留页范围、截断及续读语义。"""
        plan = await self._build_read_plan_from_parse(
            doc_ref,
            tier=tier,
            page_range=page_range,
            after=after,
            limit=limit,
            format=format,
            no_marker=no_marker,
        )
        return await self._execute_read_plan(plan)

    async def read_locator(
        self,
        locator: str,
        *,
        after: str | None = None,
        context: int = 0,
        limit: int = 30000,
        format: ContentFormat = "markdown",
        image_format: ImageFormat = "jpeg",
        no_marker: bool = False,
    ) -> DocContentResponse:
        """读取文档、页、块或字符定位符，可返回文本或图片素材。"""
        plan = await self._build_read_plan_from_locator(
            locator,
            after=after,
            context=context,
            limit=limit,
            format=format,
            image_format=image_format,
            no_marker=no_marker,
        )
        return await self._execute_read_plan(plan)

    async def export_document(self, doc_ref: str, request: DocContentExportRequest) -> DocContentExportResponse:
        """从缓存生成完整内容并写入用户指定的导出文件。"""
        doc = await self.get_document(doc_ref, param="doc_ref")
        content = await self._render_doc_content(
            doc["sha256"],
            tier=request.tier,
            page_range=request.page_range,
            format=request.format,
            no_marker=request.no_marker,
        )
        output_path = os.path.abspath(request.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        return DocContentExportResponse(sha256=doc["sha256"], short_id=doc["short_id"], tier=request.tier, output=output_path)

    async def get_document(self, doc_ref: str, *, param: str) -> DocRow:
        """按完整或短文档标识查找记录，缺失时返回统一业务错误。"""
        row = await self.find_document(doc_ref)
        if row is None:
            raise NotFoundError("doc_not_found", f"Document {doc_ref} not found.", param)
        return row

    async def find_document(self, doc_ref: str) -> DocRow | None:
        """按完整或短文档标识查询记录，允许调用者处理不存在的文档。"""
        row = cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE sha256=?", (doc_ref,)))
        if row is not None:
            return row
        return cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE short_id=?", (doc_ref,)))

    async def _build_read_plan_from_parse(
        self,
        doc_ref: str,
        *,
        tier: Tier,
        page_range: str | None,
        after: str | None,
        limit: int,
        format: str,
        no_marker: bool,
    ) -> _ReadPlan:
        """校验解析内容请求并生成统一读取计划。"""
        if format != "markdown":
            raise InvalidRequestError("invalid_request", "Only markdown is currently implemented for parse content.", "format")
        limit = max(1, limit)
        doc = await self.get_document(doc_ref, param="doc_ref")

        # 非 PDF 的省略范围表示整本，逻辑分节数不应触发 PDF 的前十页默认限制。
        full_document = not is_page_range_parse_extension(doc["file_type"]) and not normalize_page_range_input(page_range)
        normalized_page_range = None if full_document else _normalize_content_page_range(page_range, after, doc)
        after_cursor = _parse_after_cursor(after)
        if after_cursor:
            _validate_cursor_for_doc(after_cursor, doc, tier, normalized_page_range)
        return _ReadPlan(
            sha256=doc["sha256"],
            tier=tier,
            short_id=doc["short_id"],
            page_range=normalized_page_range,
            after=after,
            limit=limit,
            format="markdown",
            no_marker=no_marker,
            locator=None,
            context=0,
            target=None,
            next_mode="parse",
        )

    async def _build_read_plan_from_locator(
        self,
        locator: str,
        *,
        context: int,
        limit: int,
        format: ContentFormat,
        image_format: ImageFormat,
        no_marker: bool,
        after: str | None = None,
    ) -> _ReadPlan:
        """将页选择解析为实际页范围，同时保持单点块和图片的定位约束。"""
        limit = max(1, limit)
        context = max(0, context)
        cursor = _parse_doc_locator(locator)
        doc = cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE short_id=?", (cursor.short_id,)))
        if doc is None:
            raise NotFoundError("doc_not_found", f"Document {cursor.short_id} not found.", "locator")
        if doc["file_type"] in TEXT_FILE_TYPES:
            raise InvalidRequestError(
                "parse_not_required",
                "Text files do not require MinerU parsing. Read the file directly.",
                "locator",
            )
        cursor = _resolve_locator_pages(cursor, doc)

        tier = cursor.tier or await self._default_read_tier(doc["sha256"])
        if tier is None:
            raise NotFoundError(
                "tier_not_cached",
                f"No current parsed tier is cached for document {cursor.short_id}; reparse the source document.",
                "locator",
            )
        if format == "image" and context:
            raise InvalidRequestError("context_not_applicable", "context is not supported for image reads.", "context")
        if cursor.page_range and count_pages_in_range(cursor.page_range) > 1:
            if cursor.block_no is not None:
                raise InvalidRequestError("invalid_locator", "Block and char locators require a single page.", "locator")
            if format == "image":
                raise InvalidRequestError("multi_page_image_not_supported", "image format supports only one page.", "locator")
            if context:
                raise InvalidRequestError("context_not_applicable", "context requires a single page.", "context")
        if cursor.page_no is None and context:
            raise InvalidRequestError("context_not_applicable", "context requires a page, block, or char locator.", "context")

        page_range = _locator_page_range(cursor, doc, context)
        if after is not None:
            if cursor.page_range is None or cursor.block_no is not None or context or format != "markdown":
                raise InvalidRequestError(
                    "invalid_request", "after requires a Markdown page selection without context.", "after"
                )
            after_cursor = _parse_after_cursor(after)
            if after_cursor is None:
                raise InvalidRequestError("invalid_locator", "after must be a non-empty content cursor.", "after")
            _validate_cursor_for_doc(after_cursor, doc, tier, page_range)
        return _ReadPlan(
            sha256=doc["sha256"],
            short_id=doc["short_id"],
            tier=tier,
            page_range=page_range,
            after=after if after is not None else _locator_after(cursor, tier),
            locator=_canonical_locator(doc["short_id"], tier, cursor),
            context=context,
            limit=limit,
            format=format,
            image_format=image_format,
            no_marker=no_marker,
            target=ContentCursor(
                short_id=doc["short_id"],
                tier=tier,
                page_no=cursor.page_no or 1,
                block_no=cursor.block_no,
                char_offset=cursor.char_offset,
            )
            if cursor.page_no is not None
            else None,
            next_mode="range"
            if (cursor.range_selection and cursor.block_no is None and not context) or after is not None
            else "read",
        )

    async def _render_doc_content(
        self,
        sha256: str,
        *,
        tier: Tier,
        page_range: str | None,
        format: str,
        no_marker: bool,
    ) -> str:
        """读取有效缓存页并生成完整导出 Markdown。"""
        if format != "markdown":
            raise InvalidRequestError("invalid_request", "Only markdown is currently implemented.", "format")
        data_dir = self.data_dir
        rows = await self.db.fetchall(
            "SELECT page_range, done_at FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
            (sha256, tier, PARSE_STATUS_DONE),
        )
        doc = cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,)))
        if doc is None:
            raise NotFoundError("doc_not_found", f"Document {sha256} not found.", "doc_ref")
        page_range = normalize_page_range_input(page_range)
        if page_range and doc["page_count"] is not None:
            page_range = _expand_page_range(page_range, doc["page_count"])
        requested = parse_page_range_set(page_range) if page_range and page_range != "all" else None
        loaded_pages = load_pages_from_done_batches(
            data_dir,
            sha256,
            tier,
            cast(list[ParseBatchRow], rows),
            requested_page_numbers=requested,
        )
        if requested is not None:
            loaded_pages = [page for page in loaded_pages if page.page_idx + 1 in requested]
        if not loaded_pages:
            raise NotFoundError("not_cached", "Requested parsed content is not cached.", "page_range")
        output: list[str] = []
        page_count = doc["page_count"]
        for page in loaded_pages:
            if not no_marker:
                output.append(_page_marker(page.page_idx + 1, page_count))
            output.extend(
                text
                for _, text in _page_markdown_blocks(
                    page,
                    latex_delimiters=self.latex_delimiters,
                    short_id=doc["short_id"],
                    tier=tier,
                )
            )
        return "\n\n".join(output)

    async def _execute_read_plan(self, plan: _ReadPlan) -> DocContentResponse:
        """执行读取计划并构造分页、续读和素材响应。"""
        data_dir = self.data_dir
        rows = await self.db.fetchall(
            "SELECT page_range, done_at FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
            (plan.sha256, plan.tier, PARSE_STATUS_DONE),
        )
        loaded_pages = load_pages_from_done_batches(
            data_dir,
            plan.sha256,
            plan.tier,
            cast(list[ParseBatchRow], rows),
            requested_page_numbers=parse_page_range_set(plan.page_range) if plan.page_range else None,
        )
        if plan.page_range:
            loaded_pages = filter_pages_by_user_range(loaded_pages, plan.page_range)
        if not loaded_pages:
            raise NotFoundError("not_cached", "Requested parsed content is not cached.", "page_range")

        doc = cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE sha256=?", (plan.sha256,)))
        if plan.format == "image":
            return await self._render_image_response(plan, loaded_pages)

        pages_for_render = _select_context_pages(loaded_pages, plan.target, plan.context)
        rendered = _render_progressive_markdown(
            pages_for_render,
            latex_delimiters=self.latex_delimiters,
            short_id=plan.short_id,
            tier=plan.tier,
            after=_parse_after_cursor(plan.after),
            limit=plan.limit,
            add_markers=not plan.no_marker,
            target=plan.target,
            context=plan.context,
            page_count=doc["page_count"] if doc else None,
        )
        if not rendered.content_ranges:
            raise NotFoundError("not_cached", "Requested parsed content is not cached after cursor.", "after")

        page_count = doc["page_count"] if doc else None
        paginated = _is_paginated_doc(doc) if doc else True
        if (
            plan.next_mode == "parse"
            and plan.page_range is None
            and doc
            and not is_page_range_parse_extension(doc["file_type"])
        ):
            # 整本格式仅用内容游标续读，避免生成它不接受的 --pages 解析参数。
            paginated = False
        if plan.next_mode == "range":
            next_request = _next_range_read_request(rendered, plan.locator)
        elif plan.next_mode == "read":
            next_request = _next_read_request(rendered, plan.short_id, plan.tier, page_count)
        else:
            next_request = _next_content_request(
                rendered=rendered,
                request_page_range=plan.page_range,
                after=plan.after,
                page_count=page_count,
                paginated=paginated,
            )
        return DocContentResponse(
            sha256=plan.sha256,
            short_id=plan.short_id,
            tier=plan.tier,
            format="markdown",
            content=rendered.content,
            request_scope=ContentRequestScope(
                page_range=plan.page_range,
                after=plan.after,
                limit=plan.limit,
                locator=plan.locator,
                context=plan.context,
            ),
            content_ranges=rendered.content_ranges,
            truncated=rendered.truncated,
            next_request=next_request,
        )

    async def _render_image_response(
        self,
        plan: _ReadPlan,
        loaded_pages: list[PageInfo],
    ) -> DocContentResponse:
        """解析页或块的图片来源并构造素材响应。"""
        if plan.target is None or plan.target.page_no is None:
            raise InvalidRequestError("format_not_supported", "image format requires a page or block locator.", "format")
        if plan.page_range and len(parse_page_range_set(plan.page_range)) != 1:
            raise InvalidRequestError("multi_page_image_not_supported", "image format supports only one page.", "locator")
        page = _find_page(loaded_pages, plan.target.page_no)
        if page is None:
            raise NotFoundError("page_not_cached", f"Page {plan.target.page_no} is not cached.", "locator")

        if plan.target.block_no is None:
            asset = await self._render_source_image_asset(plan, page)
        else:
            block = _find_block_by_no(page, plan.target.block_no)
            if block is None:
                raise NotFoundError("block_not_found", f"Block {plan.target.block_no} not found.", "locator")
            source = _resolve_block_image_source(block)
            if source is None:
                raise NotFoundError("asset_not_available", "Block image asset is not available.", "locator")
            if source.kind == "source_bbox":
                asset = await self._render_source_image_asset(plan, page)
            else:
                assert source.data_uri is not None
                asset = self._render_base64_image_asset(plan, source.data_uri)

        target_ref = plan.locator or page_ref(plan.short_id, plan.tier, plan.target.page_no)
        return DocContentResponse(
            sha256=plan.sha256,
            short_id=plan.short_id,
            tier=plan.tier,
            format="image",
            content="",
            request_scope=ContentRequestScope(
                page_range=plan.page_range,
                after=plan.after,
                limit=plan.limit,
                locator=plan.locator,
                context=plan.context,
                image_format=plan.image_format,
            ),
            content_ranges=[ContentRange(page_range=str(plan.target.page_no), start=target_ref, end=target_ref)],
            truncated=False,
            next_request=None,
            asset=asset,
        )

    async def _default_read_tier(self, sha256: str) -> Tier | None:
        """从可读取的已完成缓存中选择最高质量档位。"""
        rows = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND status=? ORDER BY done_at DESC",
                (sha256, PARSE_STATUS_DONE),
            ),
        )
        tiers: set[Tier] = {
            row["tier"]
            for row in rows
            if row["tier"] in TIERS
            and row["tier"] != "flash"
            and parse_batch_is_readable(self.data_dir, sha256, row["tier"], row)
        }
        if not tiers:
            return None
        return select_highest_cached_tier(tiers)

    async def _render_source_image_asset(self, plan: _ReadPlan, page: PageInfo) -> ContentAsset:
        """查找源文件并调用公共图像输出，保留 Doclib 定位与错误协议。"""
        if plan.target is None:
            raise InvalidRequestError("format_not_supported", "image format requires a page or block locator.", "format")
        file_row = await accessible_file_for_sha256(self.db, plan.sha256)
        if file_row is None:
            raise NotFoundError("no_accessible_file", "No active source file found for this document.", "locator")
        source_path = file_row["path"]
        source_ext = str(file_row.get("ext") or Path(source_path).suffix).lower().lstrip(".")
        source_bytes = Path(source_path).read_bytes()
        if source_ext == "pdf":
            doc = PDFDocument(source_bytes)
        elif source_ext in IMAGE_EXTENSIONS:
            doc = PDFDocument.from_image(source_bytes)
        else:
            raise InvalidRequestError(
                "format_not_supported",
                "Page and bbox image output is only supported for PDF and image sources.",
                "format",
            )
        with doc:
            bbox = None
            if plan.target.block_no is not None:
                block = _find_block_by_no(page, plan.target.block_no)
                if block is None:
                    raise NotFoundError("block_not_found", f"Block {plan.target.block_no} not found.", "locator")
                if block.bbox is None or _is_empty_bbox(block.bbox):
                    raise InvalidRequestError("bbox_not_available", "Block bbox is not available for image output.", "locator")
                bbox = block.bbox
            artifact = doc.render_image(plan.target.page_no - 1, bbox=bbox, image_format=plan.image_format)
        return _write_temp_asset(
            self.data_dir,
            plan.short_id,
            artifact.extension,
            artifact.data,
            mime_type=artifact.mime_type,
            width=artifact.width,
            height=artifact.height,
        )

    def _render_base64_image_asset(self, plan: _ReadPlan, data_uri: str) -> ContentAsset:
        """通过公共素材接口解码并转码，Doclib 只负责临时资产写出。"""
        image_bytes, _extension = parse_image_data_uri_strict(data_uri)
        artifact = transcode_image(image_bytes, image_format=plan.image_format)
        return _write_temp_asset(
            self.data_dir,
            plan.short_id,
            artifact.extension,
            artifact.data,
            mime_type=artifact.mime_type,
            width=artifact.width,
            height=artifact.height,
        )


_READ_LOCATOR_RE = re.compile(
    r"^doc:(?P<short_id>[0-9a-fA-F]+)"
    r"(?:/tier:(?P<tier>flash|basic|standard|advanced)"
    r"(?:/page:(?P<page_range>[^/]+)"
    r"(?:/block:(?P<block_no>[1-9][0-9]*)(?:/char:(?P<char_offset>0|[1-9][0-9]*))?)?)?)?$"
)


def _parse_doc_locator(locator: str) -> _LocatorParts:
    """只解析输入语法，倒数页和范围在获取文档页数后统一求值。"""
    match = _READ_LOCATOR_RE.match(locator)
    if match is None:
        raise InvalidRequestError("invalid_locator", f"Invalid doclib locator: {locator}", "locator")
    page_range = match.group("page_range")
    if page_range is not None:
        if not page_range.strip():
            raise InvalidRequestError("invalid_locator", "Page selection cannot be empty.", "locator")
        try:
            normalized_range = normalize_page_range_input(page_range)
        except InvalidRequestError as error:
            raise InvalidRequestError("invalid_locator", error.message, "locator") from error
    else:
        normalized_range = None
    block_no = match.group("block_no")
    char_offset = match.group("char_offset")
    return _LocatorParts(
        short_id=match.group("short_id"),
        tier=cast(Tier | None, match.group("tier")),
        page_no=int(normalized_range) if normalized_range and normalized_range.isdecimal() else None,
        block_no=int(block_no) if block_no is not None else None,
        char_offset=int(char_offset) if char_offset is not None else None,
        page_range=normalized_range,
        range_selection=page_range is not None and (page_range.strip() == "all" or "," in page_range or "-" in page_range),
    )


def _resolve_locator_pages(locator: _LocatorParts, doc: DocRow) -> _LocatorParts:
    """复用公共页范围算法求值；倒数页和 all 禁止用部分缓存猜测总页数。"""
    if locator.page_range is None:
        return locator
    page_count = doc.get("page_count")
    if (page_count is None or page_count < 1) and ("r" in locator.page_range or locator.page_range == "all"):
        raise InvalidRequestError(
            "invalid_locator", "Cannot resolve all/rN because the document page count is unavailable.", "locator"
        )
    resolved_range = _expand_page_range(locator.page_range, page_count or 1)
    single_page = count_pages_in_range(resolved_range) == 1
    return replace(locator, page_range=resolved_range, page_no=int(resolved_range) if single_page else None)


def _canonical_locator(short_id: str, tier: Tier, locator: _LocatorParts) -> str:
    """将已解析的定位信息编码为统一的文档引用。"""
    if locator.range_selection and locator.page_range is not None and locator.block_no is None:
        return f"doc:{short_id}/tier:{tier}/page:{locator.page_range}"
    if locator.page_no is None:
        return f"doc:{short_id}/tier:{tier}"
    if locator.block_no is None:
        return page_ref(short_id, tier, locator.page_no)
    if locator.char_offset is None:
        return block_ref(short_id, tier, locator.page_no, locator.block_no)
    return block_char_ref(short_id, tier, locator.page_no, locator.block_no, locator.char_offset)


def _locator_after(locator: _LocatorParts, tier: Tier) -> str | None:
    """仅为字符级定位构造块内续读游标。"""
    if locator.page_no is None:
        return None
    if locator.block_no is None:
        return None
    return _canonical_locator(locator.short_id, tier, locator) if locator.char_offset is not None else None


def _locator_page_range(locator: _LocatorParts, doc: DocRow, context: int) -> str | None:
    """按定位页和上下文范围确定实际读取页码。"""
    page_count = doc.get("page_count") or 1
    if locator.page_range is not None and not context:
        return locator.page_range
    if locator.page_no is None:
        return _normalize_content_page_range(None, None, doc)
    if locator.page_no > page_count:
        return _normalize_page_range(str(locator.page_no), page_count)
    start = max(1, locator.page_no - context)
    end = min(page_count, locator.page_no + context)
    return _range_str(start, end)


def _parse_after_cursor(after: str | None) -> ContentCursor | None:
    """解析续读游标并转换为 Doclib 的参数错误。"""
    if not after:
        return None
    try:
        return parse_content_cursor(after)
    except ValueError as exc:
        raise InvalidRequestError("invalid_locator", str(exc), "after") from None


def _validate_cursor_for_doc(cursor: ContentCursor, doc: DocRow, tier: Tier, page_range: str | None) -> None:
    """验证续读游标属于指定文档、档位和页范围。"""
    if cursor.short_id != doc["short_id"]:
        raise InvalidRequestError("invalid_request", "after cursor does not belong to this document.", "after")
    if cursor.tier != tier:
        raise InvalidRequestError("invalid_request", "after cursor tier does not match request tier.", "after")
    if page_range and cursor.page_no not in parse_page_range_set(page_range):
        raise InvalidRequestError("invalid_request", "after cursor is outside requested page_range.", "after")


def _is_paginated_doc(doc: DocRow) -> bool:
    """根据已知页面总数判断文档是否需要分页读取。"""
    page_count = doc.get("page_count")
    return page_count is not None and page_count > 1


def _normalize_content_page_range(page_range: str | None, after: str | None, doc: DocRow) -> str | None:
    """统一内容页范围并保留默认前十页与游标续读规则。"""
    page_range = normalize_page_range_input(page_range)
    page_count = doc.get("page_count") or 1
    if _is_paginated_doc(doc):
        if page_range and page_range.strip() == "all":
            return _expand_page_range("all", page_count)
        if page_range:
            return _normalize_page_range(page_range, page_count)
        if after:
            cursor = _parse_after_cursor(after)
            assert cursor is not None
            end = min(page_count, cursor.page_no + 9)
            return _range_str(cursor.page_no, end)
        end = min(page_count, 10)
        return _range_str(1, end)
    if page_range and page_range.strip() == "all":
        return None
    return _normalize_page_range(page_range, page_count) if page_range else None


def _normalize_page_range(page_range: str, page_count: int) -> str:
    """复用公共页范围算法获得规范化的一基页码范围。"""
    return _expand_page_range(page_range, page_count)


def _select_context_pages(pages: list[PageInfo], target: ContentCursor | None, context: int) -> list[PageInfo]:
    """按目标页与上下文数量筛选已加载页面。"""
    if target is None or target.page_no is None or context <= 0:
        return pages
    start = target.page_no - context
    end = target.page_no + context
    return [page for page in pages if start <= page.page_idx + 1 <= end]


def _find_page(pages: list[PageInfo], page_no: int) -> PageInfo | None:
    """从缓存页面中查找指定的一基页码。"""
    return next((page for page in pages if page.page_idx + 1 == page_no), None)


def _find_block_by_no(page: PageInfo, block_no: int) -> PageBlock | None:
    """将一基块号映射为页面中的顶层块。"""
    target_index = block_no - 1
    return next((block for block in page.blocks if block.index == target_index), None)


def _is_empty_bbox(bbox: object) -> bool:
    """验证边框是否为有限数值组成的有效矩形。"""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return True
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return True
    if not all(math.isfinite(value) for value in values):
        return True
    return values[0] >= values[2] or values[1] >= values[3]


def _resolve_block_image_source(block: BlockBase) -> _BlockImageSource | None:
    """优先选择原文矩形裁图，其次使用已物化的内嵌图片。"""
    if not _is_empty_bbox(block.bbox):
        return _BlockImageSource(kind="source_bbox")
    for payload in iter_image_payloads(block):
        if payload.image_base64:
            return _BlockImageSource(kind="base64", data_uri=payload.image_base64)
    return None


def _make_doclib_image_renderer(
    *,
    short_id: str,
    tier: Tier,
    page_no: int,
) -> Callable[[BlockBase], str]:
    """构造输出 Doclib 素材定位符的 Markdown 图片渲染器。"""

    def _render(block: BlockBase) -> str:
        """将可读取图片转换为块定位符，并保留远端图片链接。"""
        label = _VISUAL_BLOCK_LABELS.get(block.type, "Image block")
        source = _resolve_block_image_source(block)
        if source is not None:
            block_no = block.index + 1 if block.index is not None else 0  # TODO: change after block.index's type changed.
            return build_markdown_image(block_ref(short_id, tier, page_no, block_no), label)
        remote_url = next((payload.image_url for payload in iter_image_payloads(block) if payload.image_url), None)
        return build_markdown_image(remote_url, label) if remote_url is not None else f"![{label}]()"

    return _render


def _write_temp_asset(
    data_dir: str,
    short_id: str,
    ext: str,
    image_bytes: bytes,
    *,
    mime_type: str,
    width: int | None,
    height: int | None,
) -> ContentAsset:
    """写出临时图片文件并返回包含尺寸与媒体类型的素材信息。"""
    safe_ext = ext.lower().lstrip(".") or "png"
    output_dir = Path(os.path.expanduser(data_dir)) / "temp" / "read-assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{short_id}-{int(time.time() * 1000)}.{safe_ext}"
    path.write_bytes(image_bytes)
    return ContentAsset(
        path=str(path),
        mime_type=mime_type,
        size_bytes=len(image_bytes),
        width=width,
        height=height,
    )


def _render_progressive_markdown(
    pages: list[PageInfo],
    *,
    latex_delimiters: LatexDelimitersConfig,
    short_id: str,
    tier: Tier,
    after: ContentCursor | None,
    limit: int,
    add_markers: bool,
    target: ContentCursor | None = None,
    context: int = 0,
    page_count: int | None = None,
) -> _RenderedContent:
    """在页面与块边界渐进生成 Markdown，并记录截断位置。"""
    output: list[str] = []
    ranges: list[ContentRange] = []
    started = after is None
    last_page_no = 0
    next_page_no: int | None = None
    cut_inside_page = False

    for page in pages:
        page_no = page.page_idx + 1
        if after and page_no < after.page_no:
            continue
        page_blocks = _page_markdown_blocks(
            page,
            latex_delimiters=latex_delimiters,
            short_id=short_id,
            tier=tier,
        )
        if target and target.block_no is not None and page_no == target.page_no:
            start_block_no = max(1, target.block_no - context)
            end_block_no = target.block_no + context
            page_blocks = [(index, text) for index, text in page_blocks if start_block_no <= index + 1 <= end_block_no]
        page_output: list[str] = []
        page_start_cursor = page_ref(short_id, tier, page_no)
        page_end_cursor = page_ref(short_id, tier, page_no)
        page_started = False
        empty_page_selected = False

        for block_index, block_text in page_blocks:
            block_no = block_index + 1
            if after and page_no == after.page_no:
                if after.block_no is None:
                    continue
                if block_no < after.block_no:
                    continue
                if block_no == after.block_no and after.char_offset is None:
                    continue
            started = True
            text = block_text
            block_start_offset = 0
            if after and page_no == after.page_no and after.block_no == block_no and after.char_offset is not None:
                block_start_offset = after.char_offset
                text = text[block_start_offset:]
            if not text:
                continue
            block_start_cursor = (
                block_char_ref(short_id, tier, page_no, block_no, block_start_offset)
                if block_start_offset
                else block_ref(short_id, tier, page_no, block_no)
            )
            if not page_started:
                page_start_cursor = block_start_cursor if after and after.block_no is not None else page_start_cursor
                page_started = True
            candidate_page = [*page_output, text]
            candidate_content = _join_markdown([*output, *candidate_page])
            if len(candidate_content) <= limit:
                page_output.append(text)
                page_end_cursor = block_ref(short_id, tier, page_no, block_no)
                continue

            if page_output and not output:
                cut_inside_page = True
                break
            if output:
                next_page_no = page_no
                return _RenderedContent(
                    content=_join_markdown(output),
                    content_ranges=_merge_content_ranges(ranges),
                    truncated=True,
                    last_page_no=last_page_no,
                    next_page_no=next_page_no,
                    cut_inside_page=False,
                )

            cut = _soft_cut(text, limit)
            if cut <= 0:
                cut = min(len(text), limit)
            page_output.append(text[:cut])
            end_offset = block_start_offset + cut
            page_end_cursor = block_char_ref(short_id, tier, page_no, block_no, end_offset)
            cut_inside_page = True
            break

        if not page_output and after and page_no == after.page_no:
            continue
        if not page_output and (target is None or target.block_no is None):
            started = True
            empty_page_selected = True
            if add_markers:
                page_output.append(_page_marker(page_no, page_count))

        if not started or (not page_output and not empty_page_selected):
            continue
        if add_markers:
            marker = _page_marker(page_no, page_count)
            if not page_output or page_output[0] != marker:
                page_output.insert(0, marker)
        if page_output:
            output.extend(page_output)
        last_page_no = page_no
        ranges.append(ContentRange(page_range=str(page_no), start=page_start_cursor, end=page_end_cursor))
        if cut_inside_page:
            next_page_no = page_no
            return _RenderedContent(
                content=_join_markdown(output),
                content_ranges=_merge_content_ranges(ranges),
                truncated=True,
                last_page_no=last_page_no,
                next_page_no=next_page_no,
                cut_inside_page=True,
            )
        # Keep whole pages as the normal truncation boundary for paginated docs.
        if len(_join_markdown(output)) >= limit:
            next_page_no = page_no + 1
            return _RenderedContent(
                content=_join_markdown(output),
                content_ranges=_merge_content_ranges(ranges),
                truncated=next_page_no <= (pages[-1].page_idx + 1),
                last_page_no=last_page_no,
                next_page_no=next_page_no,
                cut_inside_page=False,
            )

    return _RenderedContent(
        content=_join_markdown(output),
        content_ranges=_merge_content_ranges(ranges),
        truncated=False,
        last_page_no=last_page_no,
        next_page_no=None,
        cut_inside_page=False,
    )


def _page_marker(page_no: int, page_count: int | None) -> str:
    """按已知总页数生成页标记。"""
    if page_count is None:
        return f"<!-- page {page_no} -->"
    return f"<!-- page {page_no} of {page_count} -->"


def _page_markdown_blocks(
    page: PageInfo,
    *,
    latex_delimiters: LatexDelimitersConfig,
    short_id: str = "",
    tier: Tier = "standard",
) -> list[tuple[int, str]]:
    """使用共享渲染器和宿主公式配置生成顶层块内容。"""
    result: list[tuple[int, str]] = []
    image_renderer = _make_doclib_image_renderer(
        short_id=short_id,
        tier=tier,
        page_no=page.page_idx + 1,
    )
    delimiters = latex_delimiters
    for block in page.blocks:
        rendered = render_single_block(
            block,
            delimiters=delimiters,
            asset_base_url="",
            image_renderer=image_renderer,
        )
        text = rendered.strip() if rendered else ""
        if text:
            # TODO: simplify after BlockBase.index type narrows to int (top-level blocks always have index)
            block_index = block.index if block.index is not None else -1
            result.append((block_index, text))
    return result


def _join_markdown(parts: list[str]) -> str:
    """以段落分隔符连接非空 Markdown 片段。"""
    return "\n\n".join(part for part in parts if part)


def _soft_cut(text: str, limit: int) -> int:
    """在字符预算内优先选择自然文本分隔位置。"""
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    for pattern in ("\n\n", "\n", "。", ". ", " "):
        pos = window.rfind(pattern)
        if pos > max(0, limit // 2):
            return pos + len(pattern)
    return limit


def _merge_content_ranges(ranges: list[ContentRange]) -> list[ContentRange]:
    """合并相邻页面的内容引用范围。"""
    if not ranges:
        return []
    merged: list[ContentRange] = []
    current = ranges[0]
    for item in ranges[1:]:
        if current.page_range and item.page_range:
            current_page_numbers = parse_page_range_set(current.page_range)
            item_page_numbers = parse_page_range_set(item.page_range)
            if max(current_page_numbers) + 1 == min(item_page_numbers):
                current = ContentRange(
                    page_range=_page_numbers_to_range_str(current_page_numbers | item_page_numbers),
                    start=current.start,
                    end=item.end,
                )
                continue
        merged.append(current)
        current = item
    merged.append(current)
    return merged


def _next_content_request(
    *,
    rendered: _RenderedContent,
    request_page_range: str | None,
    after: str | None,
    page_count: int | None,
    paginated: bool,
) -> ContentNextRequest | None:
    """结合页范围与截断位置生成解析内容的续读请求。"""
    if not rendered.content_ranges:
        return None
    if not paginated:
        if rendered.truncated:
            return ContentNextRequest(after=rendered.content_ranges[-1].end)
        return None

    total_page_count = page_count or rendered.last_page_no
    if rendered.truncated:
        if rendered.cut_inside_page:
            start = rendered.next_page_no or rendered.last_page_no
            end = _last_requested_page(request_page_range) or start
            return ContentNextRequest(page_range=_range_str(start, end), after=rendered.content_ranges[-1].end)
        start = rendered.next_page_no or (rendered.last_page_no + 1)
        end = _last_requested_page(request_page_range) or min(total_page_count, start + 9)
        if start <= end:
            return ContentNextRequest(page_range=_range_str(start, end))
        if after:
            return ContentNextRequest(page_range=request_page_range, after=rendered.content_ranges[-1].end)
        return None

    start = rendered.last_page_no + 1
    if start > total_page_count:
        return None
    end = min(total_page_count, start + 9)
    return ContentNextRequest(page_range=_range_str(start, end))


def _next_read_request(
    rendered: _RenderedContent,
    short_id: str,
    tier: Tier,
    page_count: int | None,
) -> ContentNextRequest | None:
    """为定位符读取生成后续块或下一页请求。"""
    if not rendered.content_ranges:
        return None
    if rendered.truncated:
        return ContentNextRequest(locator=rendered.content_ranges[-1].end)
    total_page_count = page_count or rendered.last_page_no
    next_page_no = rendered.last_page_no + 1
    if next_page_no > total_page_count:
        return None
    return ContentNextRequest(locator=page_ref(short_id, tier, next_page_no))


def _next_range_read_request(rendered: _RenderedContent, locator: str | None) -> ContentNextRequest | None:
    """保留范围定位器并推进实际页码游标，避免续读扩入未选择的页面。"""
    if not rendered.truncated or not rendered.content_ranges or locator is None:
        return None
    return ContentNextRequest(locator=locator, after=rendered.content_ranges[-1].end)


def _last_requested_page(page_range: str | None) -> int | None:
    """返回请求范围中的最后一页。"""
    if not page_range:
        return None
    page_numbers = parse_page_range_set(page_range)
    return max(page_numbers) if page_numbers else None


def _range_str(start: int, end: int) -> str:
    """将起止页码转换为规范化范围字符串。"""
    return _page_numbers_to_range_str(range(start, end + 1))


__all__ = ["ContentReadService"]
