# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mineru.parser.base import ParseResult
from mineru.render import render_content_list, render_content_list_v2, render_markdown
from mineru.utils.title_level_postprocess import finalize_client_side_pages
from mineru.version import __version__

PDF_BACKENDS = {"pipeline", "vlm", "hybrid"}


def _write_json(path: Path, payload: Any) -> None:
    """按项目现有格式写入 JSON 文件，保持中文内容可读。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def _normalize_client_side_backend(backend: str) -> str:
    """将旧 CLI 传入的后端选项归一化为客户端重渲染使用的后端族。"""
    if backend == "office":
        return "office"
    if backend == "pipeline" or backend.startswith("pipeline"):
        return "pipeline"
    if backend.startswith("vlm"):
        return "vlm"
    if backend.startswith("hybrid"):
        return "hybrid"
    raise ValueError(f"Unsupported middle json backend for client-side output generation: {backend}")


def regenerate_client_side_outputs(
    parse_dir: str | Path,
    doc_stem: str,
    backend: str,
) -> tuple[Path, ...]:
    """读取服务端 staged/finalized middle json，并在客户端覆盖生成最终输出产物。"""
    parse_dir = Path(parse_dir)
    middle_json_path = parse_dir / f"{doc_stem}_middle.json"
    markdown_path = parse_dir / f"{doc_stem}.md"
    content_list_path = parse_dir / f"{doc_stem}_content_list.json"
    content_list_v2_path = parse_dir / f"{doc_stem}_content_list_v2.json"

    if not middle_json_path.exists():
        raise FileNotFoundError(f"Missing middle json file: {middle_json_path}")

    middle_json = json.loads(middle_json_path.read_text(encoding="utf-8"))
    if not isinstance(middle_json, dict):
        raise ValueError("middle_json must be a dict.")
    normalized_backend = _normalize_client_side_backend(backend)
    result = ParseResult.from_dict(middle_json)
    pages = result.pages

    if normalized_backend in PDF_BACKENDS:
        finalize_client_side_pages(pages, normalized_backend)

    image_dir = "images"

    markdown_path.write_text(
        render_markdown(pages, image_dir),
        encoding="utf-8",
    )
    _write_json(
        content_list_path,
        render_content_list(pages, image_dir),
    )
    _write_json(
        content_list_v2_path,
        render_content_list_v2(pages, image_dir),
    )
    output_middle_json = ParseResult(pages=pages).to_dict()
    output_middle_json["_backend"] = normalized_backend
    output_middle_json["_version_name"] = __version__
    _write_json(middle_json_path, output_middle_json)

    return (
        middle_json_path,
        markdown_path,
        content_list_path,
        content_list_v2_path,
    )
