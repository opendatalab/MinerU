"""Gradio 解析结果的持久化、图片物化与按需多格式渲染。"""

from __future__ import annotations

import base64
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast
from urllib.parse import quote

from loguru import logger

from ...filetypes import IMAGE_EXTENSIONS, PDF_EXTENSIONS
from ...model.flash.pdf.document import PDFDocument
from ...model.flash.pdf.pdfium import safe_rewrite_pdf_bytes_with_pdfium_result
from ...parser.base import ParseResult
from ...parser.page_range import parse_page_range
from ...render import (
    DocxRenderOptions,
    EpubRenderOptions,
    HtmlRenderOptions,
    LatexRenderOptions,
    MarkdownRenderOptions,
    PdfRenderOptions,
    RenderFormat,
    RenderMode,
    StructuredContentRenderOptions,
    render,
)
from ...types import BlockBase, ImagePayloadBlock, MiddleJson
from ...utils.image_payload import parse_image_data_uri_strict, validate_image_sidecar_path

DownloadFormat = Literal["zip", "html", "docx", "latex", "epub", "pdf"]

_IMAGE_DATA_URI_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_HTML_IMAGE_RE = re.compile(
    r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>[\"'])(?P<src>[^\"']+)(?P=quote)",
    re.IGNORECASE,
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}


@dataclass
class RunArtifacts:
    """保存一次 Gradio 解析所产生的稳定文件路径。"""

    root: Path
    stem: str
    source_path: Path
    middle_json_path: Path
    markdown_path: Path
    structured_content_path: Path
    bundle_zip_path: Path
    downloads_dir: Path
    file_suffix: str
    origin_pdf_path: Path | None = None
    layout_pdf_path: Path | None = None
    page_indices: tuple[int, ...] = ()
    generated_downloads: dict[str, Path] = field(default_factory=dict)

    def as_state(self) -> dict[str, Any]:
        """转换为可安全放入 Gradio State 的纯字符串字典。"""
        return {
            "root": str(self.root),
            "stem": self.stem,
            "source_path": str(self.source_path),
            "middle_json_path": str(self.middle_json_path),
            "markdown_path": str(self.markdown_path),
            "structured_content_path": str(self.structured_content_path),
            "bundle_zip_path": str(self.bundle_zip_path),
            "downloads_dir": str(self.downloads_dir),
            "file_suffix": self.file_suffix,
            "origin_pdf_path": str(self.origin_pdf_path) if self.origin_pdf_path else "",
            "layout_pdf_path": str(self.layout_pdf_path) if self.layout_pdf_path else "",
            "page_indices": list(self.page_indices),
            "generated_downloads": {key: str(value) for key, value in self.generated_downloads.items()},
        }

    @classmethod
    def from_state(cls, state: object) -> "RunArtifacts":
        """从 Gradio State 恢复路径，并拒绝缺失或类型错误的字段。"""
        if not isinstance(state, dict):
            raise ValueError("No parsed document is available")
        required = (
            "root",
            "stem",
            "source_path",
            "middle_json_path",
            "markdown_path",
            "structured_content_path",
            "bundle_zip_path",
            "downloads_dir",
            "file_suffix",
        )
        if any(not isinstance(state.get(key), str) for key in required):
            raise ValueError("Invalid Gradio artifact state")
        stem = cast(str, state["stem"])
        if stem != _safe_stem(stem):
            raise ValueError("Invalid Gradio artifact stem")
        generated = state.get("generated_downloads")
        generated_downloads: dict[str, Path] = {}
        if isinstance(generated, dict):
            root = Path(cast(str, state["root"])).resolve()
            for key, value in generated.items():
                if not isinstance(value, str):
                    continue
                candidate = Path(value).resolve()
                _ensure_path_inside(root, candidate)
                generated_downloads[str(key)] = candidate
        raw_indices = state.get("page_indices", [])
        page_indices = tuple(int(value) for value in raw_indices) if isinstance(raw_indices, list) else ()
        root = Path(cast(str, state["root"])).resolve()
        core_paths = [
            Path(cast(str, state[key])).resolve()
            for key in (
                "source_path",
                "middle_json_path",
                "markdown_path",
                "structured_content_path",
                "bundle_zip_path",
                "downloads_dir",
            )
        ]
        for path in core_paths:
            _ensure_path_inside(root, path)
        optional_paths = [_optional_path(state.get(key)) for key in ("origin_pdf_path", "layout_pdf_path")]
        for path in optional_paths:
            if path is not None:
                _ensure_path_inside(root, path)
        return cls(
            root=root,
            stem=stem,
            source_path=Path(cast(str, state["source_path"])).resolve(),
            middle_json_path=Path(cast(str, state["middle_json_path"])).resolve(),
            markdown_path=Path(cast(str, state["markdown_path"])).resolve(),
            structured_content_path=Path(cast(str, state["structured_content_path"])).resolve(),
            bundle_zip_path=Path(cast(str, state["bundle_zip_path"])).resolve(),
            downloads_dir=Path(cast(str, state["downloads_dir"])).resolve(),
            file_suffix=cast(str, state["file_suffix"]),
            origin_pdf_path=optional_paths[0],
            layout_pdf_path=optional_paths[1],
            page_indices=page_indices,
            generated_downloads=generated_downloads,
        )


@dataclass
class _ImageContext:
    """为一次渲染保存源 PDF、页面映射和已写出的图片资源。"""

    source_pdf: PDFDocument | None
    source_page_by_middle_page: dict[int, int]
    output_dir: Path
    asset_root: Path
    image_paths: dict[tuple[int, str, int | None, tuple[float, ...]], str] = field(default_factory=dict)

    def crop_for_block(self, block: BlockBase, *, middle_page_idx: int) -> tuple[str, bytes] | None:
        """按严格 block bbox 从源 PDF 裁剪图片并写入任务目录。"""
        if self.source_pdf is None or block.bbox is None:
            return None
        source_page_idx = self.source_page_by_middle_page.get(middle_page_idx, middle_page_idx)
        if source_page_idx < 0 or source_page_idx >= self.source_pdf.page_count:
            return None
        bbox = tuple(float(item) for item in block.bbox)
        key = (source_page_idx, str(block.type), block.index, bbox)
        existing = self.image_paths.get(key)
        if existing is not None:
            path = self.output_dir / existing
            return existing, path.read_bytes() if path.is_file() else b""
        try:
            data = self.source_pdf.crop_image(block.bbox, source_page_idx)
        except Exception as exc:
            logger.warning("Failed to crop Gradio PDF image page={} bbox={}: {}", source_page_idx, bbox, exc)
            return None
        relative = f"images/page_{source_page_idx}_{_safe_type_name(str(block.type))}_{block.index or 0}.jpg"
        relative = _unique_relative_image_name(relative, self.image_paths.values())
        target = self.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.image_paths[key] = relative
        return relative, data


def create_run_artifacts(source_path: Path, output_root: Path) -> RunArtifacts:
    """创建一次解析专属目录，并返回全部基础产物路径。"""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source_path.stem)
    run_id = f"{stem}_{uuid.uuid4().hex[:12]}"
    root = output_root / "gradio" / run_id
    root.mkdir(parents=True, exist_ok=False)
    downloads_dir = root / "downloads"
    downloads_dir.mkdir()
    source_suffix = source_path.suffix.lower() or ".bin"
    source_copy = root / f"source{source_suffix}"
    return RunArtifacts(
        root=root,
        stem=stem,
        source_path=source_copy,
        middle_json_path=root / "middle_json.json",
        markdown_path=root / "markdown.md",
        structured_content_path=root / "structured_content.json",
        bundle_zip_path=root / f"{stem}.zip",
        downloads_dir=downloads_dir,
        file_suffix=source_suffix.removeprefix("."),
    )


def persist_parse_result(
    result: ParseResult,
    source_path: Path,
    *,
    output_root: Path,
    page_range: str,
) -> RunArtifacts:
    """保存 V1 ParseResult，并生成旧 UI 所需的 Markdown、JSON 和预览文件。"""
    artifacts = create_run_artifacts(source_path, output_root)
    shutil.copyfile(source_path, artifacts.source_path)
    artifacts.middle_json_path.write_text(result.to_json(), encoding="utf-8")
    if result._model_output is not None:
        model_output = (
            result._model_output.to_dict(skip_defaults=False)
            if hasattr(result._model_output, "to_dict")
            else result._model_output
        )
        (artifacts.root / "model_output.json").write_text(
            json.dumps(model_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    origin_pdf, page_indices = _prepare_origin_pdf(
        artifacts.source_path,
        page_range=page_range,
        output_path=artifacts.root / "origin.pdf",
    )
    artifacts.origin_pdf_path = origin_pdf
    artifacts.page_indices = tuple(page_indices)

    image_context = _build_image_context(
        result.middle_json,
        origin_pdf,
        artifacts.root,
        page_indices=artifacts.page_indices,
    )
    try:
        markdown = cast(
            str,
            render(
                result.middle_json,
                RenderFormat.MARKDOWN,
                options=MarkdownRenderOptions(
                    mode=RenderMode.DEFAULT,
                    image_renderer=(lambda block: _markdown_image_for_block(image_context, block, result.middle_json))
                    if image_context.source_pdf is not None
                    else None,
                ),
            ),
        )
        artifacts.markdown_path.write_text(markdown, encoding="utf-8")
        structured = cast(
            dict[str, Any],
            render(
                result.middle_json,
                RenderFormat.STRUCTURED_CONTENT,
                options=StructuredContentRenderOptions(),
            ),
        )
        artifacts.structured_content_path.write_text(
            json.dumps(structured, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        _close_image_context(image_context)

    if origin_pdf is not None:
        from .preview import draw_layout_overlay

        layout_path = artifacts.root / "layout.pdf"
        try:
            draw_layout_overlay(
                result.middle_json,
                origin_pdf,
                layout_path,
                page_indices=artifacts.page_indices,
            )
        except Exception as exc:
            logger.warning("Skipping Gradio layout overlay for {}: {}", artifacts.stem, exc)
        else:
            artifacts.layout_pdf_path = layout_path

    build_bundle_zip(artifacts)
    return artifacts


def build_bundle_zip(artifacts: RunArtifacts) -> Path:
    """把当前任务的源文件和基础产物压缩为用户可下载的 ZIP。"""
    _ensure_path_inside(artifacts.root, artifacts.bundle_zip_path)
    with zipfile.ZipFile(artifacts.bundle_zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(artifacts.root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Artifact tree must not contain symlinks: {path}")
            if not path.is_file() or path == artifacts.bundle_zip_path or artifacts.downloads_dir in path.parents:
                continue
            relative = path.relative_to(artifacts.root).as_posix()
            validate_image_sidecar_path(relative)
            archive.write(path, arcname=relative)
    return artifacts.bundle_zip_path


def markdown_for_gradio(markdown: str, artifacts: RunArtifacts) -> str:
    """把任务目录中的相对图片路径转换为 Gradio 可访问的绝对文件 URL。"""
    if not isinstance(markdown, str):
        return ""

    def _resolve(src: str) -> str | None:
        """解析并校验任务目录内的图片引用。"""
        normalized = src.strip()
        if not normalized or _is_external_source(normalized):
            return None
        try:
            safe = validate_image_sidecar_path(normalized)
        except ValueError:
            return None
        candidate = (artifacts.root / safe).resolve()
        try:
            candidate.relative_to(artifacts.root.resolve())
        except ValueError:
            return None
        if not candidate.is_file() or candidate.suffix.lower() not in _IMAGE_SUFFIXES:
            return None
        return f"/gradio_api/file={quote(str(candidate), safe='/:')}"

    def replace_markdown(match: re.Match[str]) -> str:
        """替换 Markdown 图片链接，保留无法安全解析的原文。"""
        public = _resolve(match.group("src"))
        return f"![{match.group('alt')}]({public})" if public else match.group(0)

    result = _MARKDOWN_IMAGE_RE.sub(replace_markdown, markdown)

    def replace_html(match: re.Match[str]) -> str:
        """替换 HTML img 的本地图片链接。"""
        public = _resolve(match.group("src"))
        if public is None:
            return match.group(0)
        return f"{match.group('prefix')}{match.group('quote')}{public}{match.group('quote')}"

    return _HTML_IMAGE_RE.sub(replace_html, result)


def render_download(artifacts_state: object, format_name: str, *, allowed_root: Path | None = None) -> str:
    """从已保存 Middle JSON 按需生成指定下载文件，并返回绝对路径。"""
    artifacts = RunArtifacts.from_state(artifacts_state)
    if allowed_root is not None:
        _ensure_path_inside(allowed_root.resolve(), artifacts.root)
        for path in (
            artifacts.source_path,
            artifacts.middle_json_path,
            artifacts.markdown_path,
            artifacts.structured_content_path,
            artifacts.bundle_zip_path,
            artifacts.downloads_dir,
        ):
            _ensure_path_inside(artifacts.root, path)
    if format_name not in {"zip", "html", "docx", "latex", "epub", "pdf"}:
        raise ValueError(f"Unsupported download format: {format_name}")
    download_format = cast(DownloadFormat, format_name)
    cached = artifacts.generated_downloads.get(download_format)
    if cached is not None and cached.is_file():
        return str(cached)
    inferred_cache = _inferred_download_path(artifacts, download_format)
    if inferred_cache is not None and inferred_cache.is_file():
        return str(inferred_cache)
    if download_format == "zip":
        return str(build_bundle_zip(artifacts))

    result = ParseResult.from_json(artifacts.middle_json_path.read_text(encoding="utf-8"))
    middle_json = result.middle_json
    source_pdf = artifacts.origin_pdf_path if artifacts.origin_pdf_path and artifacts.origin_pdf_path.is_file() else None
    page_indices = artifacts.page_indices
    if source_pdf is None:
        source_pdf, resolved_page_indices = _prepare_origin_pdf(
            artifacts.source_path,
            page_range="",
            output_path=artifacts.downloads_dir / f".{artifacts.stem}_render_source.pdf",
            keep_existing=True,
        )
        page_indices = tuple(resolved_page_indices)
    image_context = _build_image_context(
        middle_json,
        source_pdf,
        artifacts.root,
        page_indices=page_indices,
    )
    try:
        if download_format == "html":
            target = artifacts.downloads_dir / f"{artifacts.stem}.html"
            _ensure_path_inside(artifacts.root, target)
            html_middle = _materialize_middle_json(middle_json, image_context, mode="inline")
            value = cast(
                str,
                render(
                    html_middle,
                    RenderFormat.HTML,
                    options=HtmlRenderOptions(standalone=True, document_title=artifacts.stem),
                ),
            )
            target.write_text(value, encoding="utf-8")
        elif download_format == "docx":
            target = artifacts.downloads_dir / f"{artifacts.stem}.docx"
            _ensure_path_inside(artifacts.root, target)
            docx_middle = _materialize_middle_json(middle_json, image_context, mode="path")
            value = cast(
                bytes,
                render(
                    docx_middle,
                    RenderFormat.DOCX,
                    options=DocxRenderOptions(asset_resolver=_asset_resolver(artifacts.root)),
                ),
            )
            target.write_bytes(value)
        elif download_format == "epub":
            target = artifacts.downloads_dir / f"{artifacts.stem}.epub"
            _ensure_path_inside(artifacts.root, target)
            epub_middle = _materialize_middle_json(middle_json, image_context, mode="path")
            value = cast(
                bytes,
                render(
                    epub_middle,
                    RenderFormat.EPUB,
                    options=EpubRenderOptions(title=artifacts.stem, asset_resolver=_asset_resolver(artifacts.root)),
                ),
            )
            target.write_bytes(value)
        elif download_format == "pdf":
            target = artifacts.downloads_dir / f"{artifacts.stem}_rendered.pdf"
            _ensure_path_inside(artifacts.root, target)
            pdf_middle = _materialize_middle_json(middle_json, image_context, mode="path")
            value = cast(
                bytes,
                render(
                    pdf_middle,
                    RenderFormat.PDF,
                    options=PdfRenderOptions(
                        document_title=artifacts.stem,
                        asset_resolver=_asset_resolver(artifacts.root),
                    ),
                ),
            )
            target.write_bytes(value)
        else:
            target = artifacts.downloads_dir / f"{artifacts.stem}_latex.zip"
            latex_root = artifacts.downloads_dir / "latex"
            _ensure_path_inside(artifacts.root, target)
            _ensure_path_inside(artifacts.root, latex_root)
            latex_root.mkdir(parents=True, exist_ok=True)
            latex_context = _build_image_context(
                middle_json,
                source_pdf,
                latex_root,
                page_indices=page_indices,
                asset_root=artifacts.root,
            )
            try:
                latex_middle = _materialize_middle_json(middle_json, latex_context, mode="path")
                latex_text = cast(
                    str,
                    render(
                        latex_middle,
                        RenderFormat.LATEX,
                        options=LatexRenderOptions(document_title=artifacts.stem),
                    ),
                )
                tex_path = latex_root / f"{artifacts.stem}.tex"
                tex_path.write_text(latex_text, encoding="utf-8")
                _zip_directory(latex_root, target)
            finally:
                _close_image_context(latex_context)
        _ensure_path_inside(artifacts.root, target)
        build_bundle_zip(artifacts)
        artifacts.generated_downloads[download_format] = target
        return str(target)
    finally:
        _close_image_context(image_context)


def _prepare_origin_pdf(
    source_path: Path,
    *,
    page_range: str,
    output_path: Path,
    keep_existing: bool = False,
) -> tuple[Path | None, list[int]]:
    """将 PDF/image 输入转换为与解析范围一致的 origin PDF。"""
    if keep_existing and output_path.is_file():
        try:
            with PDFDocument(output_path.read_bytes()) as doc:
                return output_path, list(range(doc.page_count))
        except Exception:
            output_path.unlink(missing_ok=True)
    suffix = source_path.suffix.lower().lstrip(".")
    source_bytes = source_path.read_bytes()
    if suffix in IMAGE_EXTENSIONS:
        pdf_bytes = PDFDocument.from_image(source_bytes).bytes
        output_path.write_bytes(pdf_bytes)
        return output_path, [0]
    if suffix not in PDF_EXTENSIONS:
        return None, []
    with PDFDocument(source_bytes) as doc:
        page_count = doc.page_count
    page_indices = parse_page_range(page_range, page_count)
    if page_range.strip() and not page_indices:
        raise ValueError(f"Page range does not select any pages: {page_range}")
    if page_indices == list(range(page_count)):
        output_path.write_bytes(source_bytes)
        return output_path, page_indices
    rewrite = safe_rewrite_pdf_bytes_with_pdfium_result(source_bytes, page_indices=page_indices)
    output_path.write_bytes(rewrite.pdf_bytes or source_bytes)
    return output_path, rewrite.retained_page_indices or page_indices


def _build_image_context(
    middle_json: MiddleJson,
    origin_pdf: Path | None,
    output_dir: Path,
    *,
    page_indices: tuple[int, ...] = (),
    asset_root: Path | None = None,
) -> _ImageContext:
    """创建页面映射和源 PDF 上下文，供 Markdown 与多格式渲染共享。"""
    resolved_asset_root = (asset_root or output_dir).resolve()
    if origin_pdf is None:
        return _ImageContext(
            source_pdf=None,
            source_page_by_middle_page={},
            output_dir=output_dir,
            asset_root=resolved_asset_root,
        )
    source_pdf = PDFDocument(origin_pdf.read_bytes())
    page_map = _source_page_map(middle_json, source_pdf.page_count, page_indices)
    return _ImageContext(
        source_pdf=source_pdf,
        source_page_by_middle_page=page_map,
        output_dir=output_dir,
        asset_root=resolved_asset_root,
    )


def _source_page_map(middle_json: MiddleJson, source_page_count: int, page_indices: tuple[int, ...]) -> dict[int, int]:
    """将 Middle JSON 原始页号映射到当前 origin PDF 的顺序页号。"""
    original_to_output = {
        original_page_idx: output_page_idx
        for output_page_idx, original_page_idx in enumerate(page_indices)
        if output_page_idx < source_page_count
    }
    page_map: dict[int, int] = {}
    for position, page in enumerate(middle_json.pages):
        mapped = original_to_output.get(page.page_idx)
        if mapped is not None:
            page_map[page.page_idx] = mapped
        elif position < source_page_count:
            page_map[page.page_idx] = position
        elif page.page_idx < source_page_count:
            page_map[page.page_idx] = page.page_idx
    return page_map


def _close_image_context(context: _ImageContext) -> None:
    """释放上下文持有的 PDFium 文档。"""
    if context.source_pdf is not None:
        context.source_pdf.close()


def _markdown_image_for_block(context: _ImageContext, block: BlockBase, middle_json: MiddleJson) -> str:
    """为 Markdown renderer 提供当前视觉 block 的相对图片引用。"""
    page_idx = _find_block_page_idx(middle_json, block)
    if page_idx is None:
        return ""
    cropped = context.crop_for_block(block, middle_page_idx=page_idx)
    if cropped is None:
        return ""
    relative, _data = cropped
    return f"![{str(block.type)}]({relative})"


def _materialize_middle_json(middle_json: MiddleJson, context: _ImageContext, *, mode: Literal["path", "inline"]) -> MiddleJson:
    """复制 Middle JSON，并为 PDF bbox 或 inline image 准备目标 renderer 所需的图片载荷。"""
    copied = middle_json.model_copy(deep=True)
    for page in copied.pages:
        page.blocks = [_materialize_block(block, context, middle_page_idx=page.page_idx, mode=mode) for block in page.blocks]
    return copied


def _materialize_block(
    block: BlockBase,
    context: _ImageContext,
    *,
    middle_page_idx: int,
    mode: Literal["path", "inline"],
) -> BlockBase:
    """递归复制 block 树，并统一处理图片 data URI、sidecar 和 PDF bbox。"""
    updates: dict[str, Any] = {}
    if isinstance(block, ImagePayloadBlock):
        data: bytes | None = None
        extension = "jpg"
        if block.image_base64:
            try:
                data, extension = parse_image_data_uri_strict(block.image_base64)
            except ValueError:
                data = None
        elif block.image_path:
            safe_path = validate_image_sidecar_path(block.image_path)
            candidate = (context.asset_root / safe_path).resolve()
            _ensure_path_inside(context.asset_root, candidate)
            if candidate.is_file():
                data = candidate.read_bytes()
                extension = candidate.suffix.lstrip(".") or extension
        elif block.bbox is not None:
            cropped = context.crop_for_block(block, middle_page_idx=middle_page_idx)
            if cropped is not None:
                relative, data = cropped
                extension = Path(relative).suffix.lstrip(".") or extension
        if data:
            if mode == "inline":
                updates["image_base64"] = _data_uri(data, extension)
                updates["image_path"] = None
            else:
                relative = _write_materialized_asset(context.output_dir, data, extension, block, middle_page_idx)
                updates["image_path"] = relative
                updates["image_base64"] = None

    content = getattr(block, "content", None)
    if isinstance(content, list):
        updates["content"] = [
            _materialize_block(child, context, middle_page_idx=middle_page_idx, mode=mode) for child in content
        ]
    elif isinstance(content, str) and "<img" in content.lower():
        if mode == "path":
            updates["content"] = _replace_inline_images_in_markup(content, context.output_dir, block, middle_page_idx)
        else:
            updates["content"] = _inline_relative_images_in_markup(content, context.asset_root)
    return block.model_copy(update=updates, deep=True) if updates else block


def _write_materialized_asset(
    output_dir: Path,
    data: bytes,
    extension: str,
    block: BlockBase,
    page_idx: int,
    *,
    ordinal: int = 0,
) -> str:
    """把图片写入受控目录并返回安全相对路径。"""
    safe_extension = extension.lower().lstrip(".") or "jpg"
    suffix = f"_{ordinal}" if ordinal > 0 else ""
    relative = f"images/page_{page_idx}_{_safe_type_name(str(block.type))}_{block.index or 0}{suffix}.{safe_extension}"
    relative = validate_image_sidecar_path(relative)
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    return relative


def _replace_inline_images_in_markup(content: str, output_dir: Path, block: BlockBase, page_idx: int) -> str:
    """将 HTML 字符串中的 data URI 图片外置为安全相对路径。"""
    ordinal = 0

    def replace(match: re.Match[str]) -> str:
        """写出匹配到的内嵌图片并返回相对 sidecar 路径。"""
        nonlocal ordinal
        ordinal += 1
        try:
            data, extension = parse_image_data_uri_strict(match.group(0))
        except ValueError:
            return match.group(0)
        relative = _write_materialized_asset(output_dir, data, extension, block, page_idx, ordinal=ordinal)
        return relative

    return _IMAGE_DATA_URI_RE.sub(replace, content)


def _inline_relative_images_in_markup(content: str, root: Path) -> str:
    """将已有 HTML sidecar 引用读取为 data URI，保证 HTML 下载单文件可用。"""

    def replace(match: re.Match[str]) -> str:
        """读取一个安全相对图片并替换 src 属性。"""
        source = match.group("src").strip()
        if _is_external_source(source):
            return match.group(0)
        try:
            safe_path = validate_image_sidecar_path(source)
        except ValueError:
            return match.group(0)
        candidate = (root / safe_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return match.group(0)
        if not candidate.is_file():
            return match.group(0)
        uri = _data_uri(candidate.read_bytes(), candidate.suffix.lstrip(".") or "png")
        return f"{match.group('prefix')}{match.group('quote')}{uri}{match.group('quote')}"

    return _HTML_IMAGE_RE.sub(replace, content)


def _asset_resolver(root: Path) -> Callable[[str], bytes]:
    """创建只允许读取当前任务目录内安全相对路径的 renderer asset resolver。"""
    resolved_root = root.resolve()

    def resolve(relative_path: str) -> bytes:
        """读取一个已验证的任务图片 sidecar。"""
        safe_path = validate_image_sidecar_path(relative_path)
        candidate = (resolved_root / safe_path).resolve()
        _ensure_path_inside(resolved_root, candidate)
        if not candidate.is_file():
            raise FileNotFoundError(safe_path)
        return candidate.read_bytes()

    return resolve


def _find_block_page_idx(middle_json: MiddleJson, target: BlockBase) -> int | None:
    """按 block 类型、index 和 bbox 查找视觉 block 所属页面。"""
    target_key = _block_lookup_key(target)
    for page in middle_json.pages:
        pending: list[BlockBase] = list(page.blocks)
        while pending:
            block = pending.pop()
            if block is target or _block_lookup_key(block) == target_key:
                return page.page_idx
            content = getattr(block, "content", None)
            if isinstance(content, list):
                pending.extend(child for child in content if isinstance(child, BlockBase))
    return None


def _block_lookup_key(block: BlockBase) -> tuple[str, int | None, tuple[float, ...] | None]:
    """生成跨 renderer 深拷贝仍稳定的 block 查找键。"""
    block_type = getattr(block.type, "value", block.type)
    bbox = tuple(float(item) for item in block.bbox) if block.bbox is not None else None
    return str(block_type), block.index, bbox


def _zip_directory(directory: Path, target: Path) -> None:
    """将目录内容压缩为稳定的相对路径 ZIP。"""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"LaTeX artifact tree must not contain symlinks: {path}")
            if path.is_file() and path != target:
                relative = path.relative_to(directory).as_posix()
                validate_image_sidecar_path(relative)
                archive.write(path, arcname=relative)


def _inferred_download_path(artifacts: RunArtifacts, format_name: DownloadFormat) -> Path | None:
    """根据稳定命名规则定位已生成的下载文件，避免 State 未更新时重复渲染。"""
    if format_name == "zip":
        return artifacts.bundle_zip_path
    names = {
        "html": f"{artifacts.stem}.html",
        "docx": f"{artifacts.stem}.docx",
        "latex": f"{artifacts.stem}_latex.zip",
        "epub": f"{artifacts.stem}.epub",
        "pdf": f"{artifacts.stem}_rendered.pdf",
    }
    name = names.get(format_name)
    return artifacts.downloads_dir / name if name is not None else None


def _ensure_path_inside(root: Path, target: Path) -> None:
    """确保文件路径解析后仍位于任务根目录内。"""
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes output root: {target}") from exc


def _optional_path(value: object) -> Path | None:
    """把 State 中的可选路径字段转换为绝对 Path。"""
    if not isinstance(value, str) or not value:
        return None
    return Path(value).resolve()


def _safe_stem(value: str) -> str:
    """生成文件系统安全且长度受限的任务名称。"""
    normalized = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")
    if not normalized:
        normalized = "document"
    encoded = normalized.encode("utf-8")
    return encoded[:120].decode("utf-8", errors="ignore") or "document"


def _safe_type_name(value: str) -> str:
    """把 block 类型转换为稳定的文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "image"


def _unique_relative_image_name(relative: str, existing: Any) -> str:
    """在同一任务中为重复图片路径追加稳定序号。"""
    existing_set = set(existing)
    if relative not in existing_set:
        return relative
    path = Path(relative)
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}").as_posix()
        if candidate not in existing_set:
            return candidate
    raise ValueError("Too many duplicate image assets")


def _data_uri(data: bytes, extension: str) -> str:
    """按图片扩展名生成严格的 image data URI。"""
    subtype = {"jpg": "jpeg", "jpeg": "jpeg", "svg": "svg+xml"}.get(extension.lower(), extension.lower())
    return f"data:image/{subtype};base64,{base64.b64encode(data).decode('ascii')}"


def _is_external_source(value: str) -> bool:
    """判断图片引用是否已经是 scheme URL 或 data URI。"""
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value))


__all__ = [
    "DownloadFormat",
    "RunArtifacts",
    "build_bundle_zip",
    "create_run_artifacts",
    "markdown_for_gradio",
    "persist_parse_result",
    "render_download",
]
