"""Tasks 专用产物策略：每个输入只发布一个完整的可下载文件。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..task_types import TASK_OUTPUT_FORMATS, TaskOutputFormat
from .base import ParseResult


def build_task_output(
    result: ParseResult,
    formats: list[TaskOutputFormat],
    directory: Path,
) -> Path:
    """一次物化并渲染所选格式，全部成功后才返回主产物路径。"""
    from docvortex.export import embed_html_images, materialize_middle, validate_materialized_assets

    from ..render import render_docx, render_html, render_latex

    middle, assets = materialize_middle(result.middle_json)
    validate_materialized_assets(middle, assets)
    exported = ParseResult(middle_json=middle)
    entries: dict[str, bytes] = {}
    if "default" in formats:
        entries.update(
            {
                "markdown.md": exported.markdown().encode("utf-8", errors="replace"),
                "middle_json.json": exported.to_json().encode("utf-8", errors="replace"),
                "structured_content.json": json.dumps(exported.structured_content(), ensure_ascii=False, indent=2).encode(
                    "utf-8", errors="replace"
                ),
            }
        )
    if "docx" in formats:
        entries["document.docx"] = render_docx(middle, asset_resolver=assets.__getitem__)
    if "html" in formats:
        entries["document.html"] = embed_html_images(render_html(middle), assets).encode("utf-8", errors="replace")
    if "latex" in formats:
        entries["document.tex"] = render_latex(middle).encode("utf-8", errors="replace")
    directory.mkdir(parents=True, exist_ok=True)
    if len(formats) == 1 and formats[0] in {"docx", "html"}:
        name, content = next(iter(entries.items()))
        target = directory / name
        target.write_bytes(content)
        return target
    target = directory / "result.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
        if "default" in formats or "latex" in formats:
            archive.writestr("images/", b"")
            for name, content in sorted(assets.items()):
                archive.writestr(name, content)
    return target


__all__ = ["TASK_OUTPUT_FORMATS", "TaskOutputFormat", "build_task_output"]
