"""Gradio 解析结果的持久化、图片物化与按需多格式渲染。"""

from __future__ import annotations

import html
import json
import re
import shutil
import time
import uuid
import zipfile
from copy import deepcopy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, cast
from urllib.parse import quote, urlsplit

from bs4 import BeautifulSoup
from docvortex.assets import AssetStore, parse_image_data_uri_strict, validate_image_sidecar_path
from docvortex.export import materialize_middle, validate_materialized_assets
from loguru import logger

from ...filetypes import IMAGE_EXTENSIONS, PDF_EXTENSIONS
from ...parser.base import ParseResult
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
from ...types import ImagePayloadBlock, MiddleJson
from .sources import PreparedSource

DownloadFormat = Literal["markdown", "json", "html", "docx", "latex", "epub", "pdf"]


@contextmanager
def _output_stage(stage: str, artifacts: RunArtifacts) -> Iterator[None]:
    """按素材目录标识记录输出子阶段耗时，便于与转换及浏览器回执关联。"""
    started = time.monotonic()
    try:
        yield
    finally:
        logger.debug(
            "WebUI output stage={} artifacts={} elapsed={:.3f}s", stage, artifacts.root.name, time.monotonic() - started
        )


@dataclass
class RunArtifacts:
    """保存一次 Gradio 解析所产生的稳定文件路径。"""

    root: Path
    stem: str
    middle_json_path: Path
    markdown_path: Path
    structured_content_path: Path
    downloads_dir: Path
    file_suffix: str
    prepared_source: PreparedSource | None = None
    initial_page_idx: int | None = None
    generated_downloads: dict[str, Path] = field(default_factory=dict)

    def as_state(self) -> dict[str, Any]:
        """转换为可安全放入 Gradio State 的纯字符串字典。"""
        return {
            "root": str(self.root),
            "stem": self.stem,
            "middle_json_path": str(self.middle_json_path),
            "markdown_path": str(self.markdown_path),
            "structured_content_path": str(self.structured_content_path),
            "downloads_dir": str(self.downloads_dir),
            "file_suffix": self.file_suffix,
            "prepared_source": self.prepared_source.as_state() if self.prepared_source else None,
            "initial_page_idx": self.initial_page_idx,
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
        root = Path(cast(str, state["root"])).resolve()
        core_paths = [
            Path(cast(str, state[key])).resolve()
            for key in (
                "middle_json_path",
                "markdown_path",
                "structured_content_path",
                "downloads_dir",
            )
        ]
        for path in core_paths:
            _ensure_path_inside(root, path)
        prepared = state.get("prepared_source")
        prepared_source = PreparedSource.from_state(prepared, source_root=root.parent / "sources") if prepared else None
        initial_page_idx = state.get("initial_page_idx")
        if initial_page_idx is not None and (type(initial_page_idx) is not int or initial_page_idx < 0):
            raise ValueError("Invalid initial preview page")
        return cls(
            root=root,
            stem=stem,
            middle_json_path=Path(cast(str, state["middle_json_path"])).resolve(),
            markdown_path=Path(cast(str, state["markdown_path"])).resolve(),
            structured_content_path=Path(cast(str, state["structured_content_path"])).resolve(),
            downloads_dir=Path(cast(str, state["downloads_dir"])).resolve(),
            file_suffix=cast(str, state["file_suffix"]),
            prepared_source=prepared_source,
            initial_page_idx=initial_page_idx,
            generated_downloads=generated_downloads,
        )


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
    return RunArtifacts(
        root=root,
        stem=stem,
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
    artifacts.initial_page_idx = result.pages[0].page_idx if result.pages else None
    with _output_stage("materialize", artifacts):
        materialized = _materialize_middle_json(result.middle_json, artifacts.root, asset_root=source_path.parent)
        artifacts.middle_json_path.write_text(ParseResult(middle_json=materialized).to_json(), encoding="utf-8")
    with _output_stage("markdown", artifacts):
        markdown = cast(
            str,
            render(
                materialized,
                RenderFormat.MARKDOWN,
                options=MarkdownRenderOptions(),
            ),
        )
        artifacts.markdown_path.write_text(markdown, encoding="utf-8")
    with _output_stage("structured_json", artifacts):
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

    return artifacts


def render_html_preview(artifacts: RunArtifacts, *, public_base_url: str) -> str:
    """用独立 iframe 展示公共 HTML renderer 的完整输出，保留样式及公式脚本。"""
    with _output_stage("html_preview", artifacts):
        if artifacts.file_suffix in PDF_EXTENSIONS | IMAGE_EXTENSIONS:
            value = _render_html(artifacts, public_base_url=public_base_url, include_source_map=True)
        else:
            value = _render_html(artifacts, public_base_url=public_base_url)
        # srcdoc 默认继承外层页面的基准地址，显式指定自身才能让章节锚点在预览内跳转。
        value = value.replace("<head>", '<head>\n<base href="about:srcdoc">', 1)
        value = _prepare_preview_links(value, include_outline=True)
        return (
            '<iframe class="mineru-rendered-html-frame" title="Markdown preview" '
            'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" '
            f'srcdoc="{html.escape(value, quote=True)}"></iframe>'
        )


def _add_trace_preview(document: str, artifacts: RunArtifacts, *, middle_json: MiddleJson, public_base_url: str) -> str:
    """向隔离结果页注入溯源配置，并由公共接口提供合并前的原始布局。"""
    from docvortex.visualization import collect_layout_regions

    from .i18n import MESSAGES

    base = urlsplit(public_base_url)
    config = (
        json.dumps(
            {
                "documentId": artifacts.root.name,
                "initialPageIdx": artifacts.initial_page_idx,
                "parentOrigin": f"{base.scheme}://{base.netloc}",
                "pageLabels": MESSAGES["trace_source"],
                "layoutRegions": [asdict(region) for region in collect_layout_regions(middle_json.pages)],
            },
            ensure_ascii=True,
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    script = (Path(__file__).resolve().parents[2] / "resources" / "gradio_trace_html.js").read_text(encoding="utf-8")
    styles = """<style>
[data-docvortex-trace-id], .mineru-trace-pages {
    --trace-strong: color-mix(in srgb, var(--trace-color, #596673) 65%, black); }
[data-docvortex-trace-id] { cursor: pointer; scroll-margin: 24px; }
[data-docvortex-trace-id]:hover:not(:has([data-docvortex-trace-id]:hover)),
[data-docvortex-trace-id]:focus-visible {
    outline: 1px solid var(--trace-strong); outline-offset: 3px;
    background: color-mix(in srgb, var(--trace-color) 5%, transparent); }
[data-mineru-selected] { outline: 2px solid var(--trace-strong) !important; outline-offset: 3px;
    background: color-mix(in srgb, var(--trace-color) 8%, transparent) !important; }
.mineru-trace-pages { position: fixed; right: 8px; z-index: 100;
    display: flex; flex-wrap: wrap; gap: 4px;
    max-width: 75%; padding: 5px; background: Canvas;
    box-sizing: border-box; max-height: calc(100vh - var(--mineru-preview-offset, 8px) - 8px);
    overflow-y: auto; overscroll-behavior: contain;
    border: 1px solid var(--trace-strong); border-radius: 6px; }
.mineru-trace-pages[hidden] { display: none; }
.mineru-trace-pages button { cursor: pointer; color: CanvasText; background: Canvas;
    border: 1px solid #aaa; border-radius: 4px; padding: 3px 6px; }
.mineru-trace-pages button[aria-pressed=true] { border-color: var(--trace-strong);
    background: color-mix(in srgb, var(--trace-color) 8%, transparent); }
</style>"""
    document = document.replace("</head>", styles + "</head>", 1)
    return document.replace("</body>", f"<script>({script})({config});</script></body>", 1)


def _prepare_preview_links(document: str, *, include_outline: bool = False) -> str:
    """为预览中的网页外链指定新标签页，并按需复用当前 DOM 准备目录。"""
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
    if include_outline:
        _add_outline_preview(soup)
    return str(soup)


def _add_outline_preview(soup: BeautifulSoup) -> None:
    """复用预览 DOM 准备唯一标题锚点和纯文本标签，并注入当前 iframe 的目录资源。"""
    from .i18n import MESSAGES

    if soup.head is None or soup.body is None:
        return
    used_ids = {str(node["id"]) for node in soup.find_all(id=True)}
    count = 0
    for heading in soup.select("article.docvortex-document .docvortex-heading"):
        if heading.name not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        label_node = deepcopy(heading)
        for formula in label_node.select("[data-docvortex-latex]"):
            formula.replace_with(str(formula["data-docvortex-latex"]))
        label = " ".join(label_node.get_text().split())
        if not label:
            continue
        count += 1
        if not heading.get("id"):
            identifier = f"mineru-outline-{count}"
            while identifier in used_ids:
                identifier += "-toc"
            heading["id"] = identifier
            used_ids.add(identifier)
        heading["data-mineru-outline-label"] = label
    if not count:
        return
    resources = Path(__file__).resolve().parents[2] / "resources"
    style = soup.new_tag("style", attrs={"data-mineru-outline": ""})
    style.string = (resources / "gradio_outline.css").read_text(encoding="utf-8")
    soup.head.append(style)
    labels = {key: MESSAGES[f"outline_{key}"] for key in ("title", "expand", "collapse")}
    config = json.dumps(labels, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    script = soup.new_tag("script", attrs={"data-mineru-outline": ""})
    script.string = f"({(resources / 'gradio_outline.js').read_text(encoding='utf-8')})({config});"
    soup.body.append(script)


def _render_html(artifacts: RunArtifacts, *, public_base_url: str, include_source_map: bool = False) -> str:
    """把已物化语义树渲染为使用当前 Gradio 静态图片地址的独立 HTML。"""
    if urlsplit(public_base_url).scheme not in {"http", "https"} or not urlsplit(public_base_url).netloc:
        raise ValueError("HTML rendering requires the current Gradio HTTP base URL")
    result = ParseResult.from_json(artifacts.middle_json_path.read_text(encoding="utf-8"))
    asset_base_url = f"{public_base_url.rstrip('/')}/gradio_api/file={quote(artifacts.root.as_posix(), safe='/:')}"
    document = cast(
        str,
        render(
            result.middle_json,
            RenderFormat.HTML,
            options=HtmlRenderOptions(
                standalone=True,
                document_title=artifacts.stem,
                asset_base_url=asset_base_url,
                include_source_map=include_source_map,
            ),
        ),
    )
    if include_source_map:
        return _add_trace_preview(document, artifacts, middle_json=result.middle_json, public_base_url=public_base_url)
    return document


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


def _materialize_middle_json(middle_json: MiddleJson, output_dir: Path, *, asset_root: Path | None = None) -> MiddleJson:
    """由宿主提供素材字节，统一交给 DocVortex 命名、消解冲突并改写引用。"""
    asset_root = (asset_root or output_dir).resolve()
    images = output_dir / "images"
    _ensure_path_inside(output_dir, images)
    existing = AssetStore()
    for path in sorted(images.rglob("*")):
        _ensure_path_inside(images, path)
        if path.is_symlink():
            raise ValueError(f"Image assets must not contain symlinks: {path}")
        if path.is_file():
            existing.add(path.relative_to(output_dir).as_posix(), path.read_bytes())

    def resolve_image(block: ImagePayloadBlock, page_idx: int) -> tuple[bytes, str] | None:
        """仅消费已有素材；缺失或非法图片明确失败，bbox 不触发源文件补裁。"""
        data: bytes | None = None
        extension = "jpg"
        if block.image_base64 is not None:
            data, extension = parse_image_data_uri_strict(block.image_base64)
        elif block.image_path:
            safe_path = validate_image_sidecar_path(block.image_path)
            candidate = (asset_root / safe_path).resolve()
            _ensure_path_inside(asset_root, candidate)
            data = candidate.read_bytes()
            extension = candidate.suffix.lstrip(".") or extension
        return (data, extension) if data else None

    document, assets = materialize_middle(
        middle_json,
        existing,
        image_resolver=resolve_image,
        asset_resolver=_asset_resolver(asset_root),
    )
    validate_materialized_assets(document, assets)
    for relative_path, payload in assets.items():
        target = output_dir / relative_path
        _ensure_path_inside(output_dir, target)
        if target.is_symlink():
            raise ValueError(f"Image assets must not contain symlinks: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative_path not in existing:
            target.write_bytes(payload)
    return document


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


def _safe_stem(value: str) -> str:
    """生成文件系统安全且长度受限的任务名称。"""
    normalized = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")
    if not normalized:
        normalized = "document"
    # strip 必须放在截断之后：截断可能暴露出尾部 "_"，若先 strip 则二次调用结果不同，
    # from_state() 的 stem 严格相等校验会拒绝解析时生成的 stem。
    truncated = normalized.encode("utf-8")[:120].decode("utf-8", errors="ignore")
    return truncated.strip("._") or "document"


__all__ = [
    "DownloadFormat",
    "RunArtifacts",
    "create_run_artifacts",
    "render_html_preview",
    "persist_parse_result",
    "render_download",
]
