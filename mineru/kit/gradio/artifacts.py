"""Gradio 解析结果的持久化、图片物化与按需多格式渲染。"""

from __future__ import annotations

import html
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, cast
from urllib.parse import quote, unquote, urlsplit

from bs4 import BeautifulSoup
from docvortex.assets import parse_image_data_uri_strict, validate_image_sidecar_path
from docvortex.document.pdf import PDFDocument
from docvortex.document.pdf.pdfium import safe_rewrite_pdf_bytes_with_pdfium_result
from loguru import logger

from ...filetypes import IMAGE_EXTENSIONS, PDF_EXTENSIONS
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
    StructuredContentRenderOptions,
    render,
)
from ...types import (
    AlgorithmBodyBlock,
    BlockBase,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    ImageBlock,
    ImageBodyBlock,
    ImagePayloadBlock,
    MiddleJson,
    TableBlock,
    TableBodyBlock,
)

DownloadFormat = Literal["markdown", "json", "html", "docx", "latex", "epub", "pdf"]

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
    crop_cache: dict[tuple[int, tuple[float, ...]], bytes] = field(default_factory=dict)

    def crop_for_block(self, block: BlockBase, *, middle_page_idx: int) -> bytes | None:
        """按 bbox 读取裁后 PDF 页面，仅缓存图片字节，不写出临时图片文件。"""
        if self.source_pdf is None or block.bbox is None:
            return None
        source_page_idx = self.source_page_by_middle_page.get(middle_page_idx, middle_page_idx)
        if source_page_idx < 0 or source_page_idx >= self.source_pdf.page_count:
            return None
        bbox = tuple(float(item) for item in block.bbox)
        key = (middle_page_idx, bbox)
        existing = self.crop_cache.get(key)
        if existing is not None:
            return existing
        try:
            data = self.source_pdf.crop_image(block.bbox, source_page_idx)
        except Exception as exc:
            logger.warning("Failed to crop Gradio PDF image page={} bbox={}: {}", source_page_idx, bbox, exc)
            return None
        self.crop_cache[key] = data
        return data


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
    """保存图片已物化的语义副本，并生成 Markdown、Structured Content 和预览文件。"""
    artifacts = create_run_artifacts(source_path, output_root)
    shutil.copyfile(source_path, artifacts.source_path)
    if result._model_output is not None:
        model_output = result._model_output.to_dict(skip_defaults=False)
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
        asset_root=source_path.parent,
    )
    try:
        materialized = _materialize_middle_json(result.middle_json, image_context)
        artifacts.middle_json_path.write_text(ParseResult(middle_json=materialized).to_json(), encoding="utf-8")
        markdown = cast(
            str,
            render(
                materialized,
                RenderFormat.MARKDOWN,
                options=MarkdownRenderOptions(),
            ),
        )
        artifacts.markdown_path.write_text(markdown, encoding="utf-8")
        structured = cast(
            dict[str, Any],
            render(
                materialized,
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

    return artifacts


def render_html_preview(artifacts: RunArtifacts, *, public_base_url: str) -> str:
    """用独立 iframe 展示公共 HTML renderer 的完整输出，保留样式及公式脚本。"""
    value = _render_html(artifacts, public_base_url=public_base_url)
    # srcdoc 默认继承外层页面的基准地址，显式指定自身才能让章节锚点在预览内跳转。
    value = value.replace("<head>", '<head>\n<base href="about:srcdoc">', 1)
    value = _prepare_preview_links(value)
    return (
        '<iframe class="mineru-rendered-html-frame" title="Markdown preview" '
        'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" '
        f'srcdoc="{html.escape(value, quote=True)}"></iframe>'
    )


def _prepare_preview_links(document: str) -> str:
    """仅为预览中的网页外链指定新标签页，保留内部锚点及邮件、电话等链接行为。"""
    soup = BeautifulSoup(document, "html.parser")
    for link in soup.find_all("a", href=True):
        try:
            scheme = urlsplit(str(link["href"])).scheme.lower()
        except ValueError:
            continue
        if scheme not in {"http", "https"}:
            continue
        link["target"] = "_blank"
        link["rel"] = list(dict.fromkeys([*link.get("rel", []), "noopener", "noreferrer"]))
    return str(soup)


def _render_html(artifacts: RunArtifacts, *, public_base_url: str) -> str:
    """把已物化语义树渲染为使用当前 Gradio 静态图片地址的独立 HTML。"""
    if urlsplit(public_base_url).scheme not in {"http", "https"} or not urlsplit(public_base_url).netloc:
        raise ValueError("HTML rendering requires the current Gradio HTTP base URL")
    result = ParseResult.from_json(artifacts.middle_json_path.read_text(encoding="utf-8"))
    asset_base_url = f"{public_base_url.rstrip('/')}/gradio_api/file={quote(artifacts.root.as_posix(), safe='/:')}"
    return cast(
        str,
        render(
            result.middle_json,
            RenderFormat.HTML,
            options=HtmlRenderOptions(
                standalone=True,
                document_title=artifacts.stem,
                asset_base_url=asset_base_url,
            ),
        ),
    )


def _build_text_download(artifacts: RunArtifacts, format_name: Literal["markdown", "json"]) -> Path:
    """仅打包所选正文文件和物化图片，所有 ZIP 成员使用相对路径。"""
    source = artifacts.markdown_path if format_name == "markdown" else artifacts.structured_content_path
    extension = "md" if format_name == "markdown" else "json"
    target = artifacts.downloads_dir / f"{artifacts.stem}_{format_name}.zip"
    _ensure_path_inside(artifacts.root, target)
    images = artifacts.root / "images"
    _ensure_path_inside(artifacts.root, images)
    files = [(source, f"{artifacts.stem}.{extension}")]
    for path in sorted(images.rglob("*")):
        _ensure_path_inside(images, path)
        if path.is_symlink():
            raise ValueError(f"Image assets must not contain symlinks: {path}")
        if path.is_file():
            files.append((path, path.relative_to(artifacts.root).as_posix()))
    _write_archive(files, target)
    return target


def render_download(
    artifacts_state: object,
    format_name: str,
    *,
    allowed_root: Path | None = None,
    public_base_url: str = "",
) -> str:
    """从已保存 Middle JSON 按需生成指定下载文件，并返回绝对路径。"""
    artifacts = RunArtifacts.from_state(artifacts_state)
    if allowed_root is not None:
        _ensure_path_inside(allowed_root.resolve(), artifacts.root)
        for path in (
            artifacts.source_path,
            artifacts.middle_json_path,
            artifacts.markdown_path,
            artifacts.structured_content_path,
            artifacts.downloads_dir,
        ):
            _ensure_path_inside(artifacts.root, path)
    if format_name not in {"markdown", "json", "html", "docx", "latex", "epub", "pdf"}:
        raise ValueError(f"Unsupported download format: {format_name}")
    download_format = cast(DownloadFormat, format_name)
    # HTML 总是使用本次请求的站点地址，不能复用其他域名下生成的文件。
    if download_format == "html":
        target = artifacts.downloads_dir / f"{artifacts.stem}.html"
        _ensure_path_inside(artifacts.root, target)
        target.write_text(_render_html(artifacts, public_base_url=public_base_url), encoding="utf-8")
        return str(target)
    cached = artifacts.generated_downloads.get(download_format)
    if cached is not None and cached.is_file():
        return str(cached)
    inferred_cache = _inferred_download_path(artifacts, download_format)
    if inferred_cache is not None and inferred_cache.is_file():
        return str(inferred_cache)
    if download_format in {"markdown", "json"}:
        return str(_build_text_download(artifacts, cast(Literal["markdown", "json"], download_format)))

    result = ParseResult.from_json(artifacts.middle_json_path.read_text(encoding="utf-8"))
    middle_json = result.middle_json
    if download_format == "docx":
        target = artifacts.downloads_dir / f"{artifacts.stem}.docx"
        _ensure_path_inside(artifacts.root, target)
        value = cast(
            bytes,
            render(
                middle_json,
                RenderFormat.DOCX,
                options=DocxRenderOptions(asset_resolver=_asset_resolver(artifacts.root)),
            ),
        )
        target.write_bytes(value)
    elif download_format == "epub":
        target = artifacts.downloads_dir / f"{artifacts.stem}.epub"
        _ensure_path_inside(artifacts.root, target)
        value = cast(
            bytes,
            render(
                middle_json,
                RenderFormat.EPUB,
                options=EpubRenderOptions(title=artifacts.stem, asset_resolver=_asset_resolver(artifacts.root)),
            ),
        )
        target.write_bytes(value)
    elif download_format == "pdf":
        target = artifacts.downloads_dir / f"{artifacts.stem}_rendered.pdf"
        _ensure_path_inside(artifacts.root, target)
        value = cast(
            bytes,
            render(
                middle_json,
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
        _copy_materialized_images(artifacts.root, latex_root)
        latex_text = cast(
            str,
            render(middle_json, RenderFormat.LATEX, options=LatexRenderOptions(document_title=artifacts.stem)),
        )
        tex_path = latex_root / f"{artifacts.stem}.tex"
        tex_path.write_text(latex_text, encoding="utf-8")
        _zip_directory(latex_root, target)
    _ensure_path_inside(artifacts.root, target)
    artifacts.generated_downloads[download_format] = target
    return str(target)


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


def _materialize_middle_json(middle_json: MiddleJson, context: _ImageContext) -> MiddleJson:
    """复制 Middle JSON，并为 PDF bbox 或 inline image 准备目标 renderer 所需的图片载荷。"""
    copied = middle_json.model_copy(deep=True)
    for page in copied.pages:
        page.blocks = [_materialize_block(block, context, middle_page_idx=page.page_idx, owner=block) for block in page.blocks]
    return copied


def _materialize_block(
    block: BlockBase,
    context: _ImageContext,
    *,
    middle_page_idx: int,
    owner: BlockBase,
) -> BlockBase:
    """沿语义树传递所属视觉父块，仅为图片载荷和富 HTML 正文物化素材。"""
    if isinstance(block, (CodeBlock, CodeBodyBlock, AlgorithmBodyBlock)):
        return block
    if isinstance(block, (ImageBlock, TableBlock, ChartBlock)):
        owner = block
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
            data = context.crop_for_block(block, middle_page_idx=middle_page_idx)
        if data:
            updates["image_path"] = _write_materialized_asset(
                context.output_dir,
                data,
                extension,
                page_idx=middle_page_idx,
                owner=owner,
            )
            updates["image_base64"] = None
            updates["image_url"] = None

    content = getattr(block, "content", None)
    if isinstance(content, list):
        updates["content"] = [
            _materialize_block(child, context, middle_page_idx=middle_page_idx, owner=owner)
            if isinstance(child, BlockBase)
            else child
            for child in content
        ]
    elif isinstance(block, (ImageBodyBlock, TableBodyBlock, ChartBodyBlock)) and "<img" in content.lower():
        updates["content"] = _materialize_markup_images(content, context, page_idx=middle_page_idx, owner=owner)

    return block.model_copy(update=updates, deep=True) if updates else block


def _write_materialized_asset(
    output_dir: Path,
    data: bytes,
    extension: str,
    *,
    page_idx: int,
    owner: BlockBase,
    ordinal: int | None = None,
) -> str:
    """使用原始页索引和所属块命名，表内多图区分序号，冲突文件禁止覆盖。"""
    safe_extension = extension.lower().lstrip(".") or "jpg"
    if f".{safe_extension}" not in _IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {extension}")
    if owner.index is None:
        raise ValueError("Image owner must have a block index")
    kind = str(owner.type)
    if ordinal is not None and kind != "image":
        kind += "_image"
    suffix = f"_{ordinal}" if ordinal is not None else ""
    stem = f"page_{page_idx}_{kind}_{owner.index}{suffix}"
    # 冲突后缀独立于表内图片序号，避免占用另一张内嵌图片的正式名称。
    for attempt in range(10000):
        duplicate = f"_duplicate_{attempt}" if attempt else ""
        relative = validate_image_sidecar_path(f"images/{stem}{duplicate}.{safe_extension}")
        target = output_dir / relative
        _ensure_path_inside(output_dir, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
            return relative
        if target.is_file() and target.read_bytes() == data:
            return relative
    raise ValueError("Too many conflicting image assets")


def _materialize_markup_images(
    content: str,
    context: _ImageContext,
    *,
    page_idx: int,
    owner: BlockBase,
) -> str:
    """按富 HTML 中的出现顺序，为所属视觉块的内嵌图片分配语义名称。"""
    ordinal = 0

    def replace(match: re.Match[str]) -> str:
        """解析一个 img 的图片载荷并保留原有 HTML 属性及引号。"""
        nonlocal ordinal
        ordinal += 1
        source = html.unescape(match.group("src")).strip()
        if source.startswith("data:"):
            data, extension = parse_image_data_uri_strict(source)
        elif _is_external_source(source):
            return match.group(0)
        else:
            safe_path = validate_image_sidecar_path(unquote(source))
            candidate = context.asset_root / safe_path
            _ensure_path_inside(context.asset_root, candidate)
            data = candidate.read_bytes()
            extension = candidate.suffix.lstrip(".")
        relative = _write_materialized_asset(
            context.output_dir,
            data,
            extension,
            page_idx=page_idx,
            owner=owner,
            ordinal=ordinal,
        )
        return f"{match.group('prefix')}{match.group('quote')}{relative}{match.group('quote')}"

    return _HTML_IMAGE_RE.sub(replace, content)


def _copy_materialized_images(asset_root: Path, output_dir: Path) -> None:
    """把已有素材按原始相对路径复制到下载包，不重新物化或改变图片名称。"""
    images = asset_root / "images"
    _ensure_path_inside(asset_root, images)
    for source in sorted(images.rglob("*")):
        _ensure_path_inside(images, source)
        if source.is_symlink():
            raise ValueError(f"Image assets must not contain symlinks: {source}")
        if source.is_file():
            target = output_dir / source.relative_to(asset_root)
            _ensure_path_inside(output_dir, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


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


def _zip_directory(directory: Path, target: Path) -> None:
    """将目录内容压缩为稳定的相对路径 ZIP。"""
    files: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        _ensure_path_inside(directory, path)
        if path.is_symlink():
            raise ValueError(f"Artifact tree must not contain symlinks: {path}")
        if path.is_file() and path != target:
            files.append((path, path.relative_to(directory).as_posix()))
    _write_archive(files, target)


def _write_archive(files: list[tuple[Path, str]], target: Path) -> None:
    """完整写出后再发布 ZIP，失败时不留下会被下次请求误用的半成品缓存。"""
    with TemporaryDirectory(prefix=".bundle-", dir=target.parent) as directory:
        pending = Path(directory) / target.name
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, relative in files:
                validate_image_sidecar_path(relative)
                archive.write(path, arcname=relative)
        pending.replace(target)


def _inferred_download_path(artifacts: RunArtifacts, format_name: DownloadFormat) -> Path | None:
    """根据稳定命名规则定位已生成的下载文件，避免 State 未更新时重复渲染。"""
    names = {
        "markdown": f"{artifacts.stem}_markdown.zip",
        "json": f"{artifacts.stem}_json.zip",
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


def _is_external_source(value: str) -> bool:
    """判断图片引用是否已经是 scheme URL 或 data URI。"""
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value))


__all__ = [
    "DownloadFormat",
    "RunArtifacts",
    "create_run_artifacts",
    "render_html_preview",
    "persist_parse_result",
    "render_download",
]
