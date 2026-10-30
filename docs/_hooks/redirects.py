"""保留已迁移文档的旧网址，同时避免归档正文进入站点和搜索索引。"""

from __future__ import annotations

import html
import json
import posixpath
from pathlib import Path, PurePosixPath
from shutil import copyfile
from urllib.parse import quote

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files


def _document_path(value: str, docs_dir: Path) -> Path:
    """只接受文档目录内的相对路径，避免错误映射覆盖其他文件。"""
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid documentation path: {value}")
    path = docs_dir / value
    if not path.resolve().is_relative_to(docs_dir.resolve()):
        raise ValueError(f"Documentation path escapes docs_dir: {value}")
    return path


def _html_path(source: str, config: MkDocsConfig) -> str:
    """沿用 MkDocs 的 README 和目录网址规则计算旧页面输出位置。"""
    return File(source, config.docs_dir, config.site_dir, config.use_directory_urls).dest_uri


@event_priority(-200)
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """检查语言筛选后的页面，禁止两个源文件覆盖同一个公开网址。"""
    destinations: dict[str, str] = {}
    for file in files.documentation_pages():
        if previous := destinations.get(file.dest_uri):
            raise ValueError(f"Duplicate documentation URL: {previous} and {file.src_uri} -> {file.dest_uri}")
        destinations[file.dest_uri] = file.src_uri
    return files


@event_priority(-200)
def on_post_build(config: MkDocsConfig) -> None:
    """双语构建后生成无索引跳转页，历史附件则保留原下载字节。"""
    docs_dir = Path(config.docs_dir)
    site_dir = Path(config.site_dir)
    redirects: dict[str, str] = json.loads((docs_dir / "_redirects.json").read_text(encoding="utf-8"))
    languages = config.plugins["i18n"].config.languages
    prefixes = ["", *(f"{language.locale}/" for language in languages if language.build and not language.default)]
    for source, destination in redirects.items():
        _document_path(source, docs_dir)
        target = _document_path(destination, docs_dir)
        if not target.is_file():
            raise ValueError(f"Missing documentation redirect target: {destination}")
        for prefix in prefixes:
            if not source.endswith(".md"):
                # JSON/PDF 等历史下载仍返回原格式，不能替换为 HTML 重定向正文。
                output = site_dir / prefix / source
                output.parent.mkdir(parents=True, exist_ok=True)
                copyfile(target, output)
                continue
            output = site_dir / prefix / _html_path(source, config)
            if destination.startswith(("archive/", "proposals/")):
                url = f"{config.repo_url}/blob/master/docs/{quote(destination)}"
            else:
                public_source = destination.removeprefix("en/")
                public_output = _html_path(public_source, config)
                if not destination.startswith(("en/", "zh/")):
                    public_output = prefix + public_output
                url = posixpath.relpath(public_output, (PurePosixPath(prefix) / _html_path(source, config)).parent)
            if output.exists() and 'data-mineru-redirect="true"' not in output.read_text(encoding="utf-8"):
                raise ValueError(f"Documentation redirect would overwrite a page: {output}")
            escaped_url = html.escape(url, quote=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                '<!doctype html><html lang="en" data-mineru-redirect="true"><head>'
                '<meta charset="utf-8"><meta name="robots" content="noindex,follow">'
                f'<meta http-equiv="refresh" content="0; url={escaped_url}">'
                f'<link rel="canonical" href="{escaped_url}"><title>Documentation moved</title>'
                "</head><body><p>This document has moved. "
                f'<a href="{escaped_url}">Open its current location</a>.</p>'
                f"<script>location.replace({json.dumps(url)} + location.hash);</script></body></html>\n",
                encoding="utf-8",
            )
