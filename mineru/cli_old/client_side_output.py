# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mineru.render import render_content_list, render_content_list_v2, render_markdown
from mineru.utils.title_level_postprocess import finalize_client_side_middle_json

PDF_BACKENDS = {"pipeline", "vlm", "hybrid"}
SUPPORTED_BACKENDS = {*PDF_BACKENDS, "office"}


def _write_json(path: Path, payload: Any) -> None:
    """按项目现有格式写入 JSON 文件，保持中文内容可读。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def regenerate_client_side_outputs(
    parse_dir: str | Path,
    doc_stem: str,
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
    backend = middle_json.get("_backend")
    pdf_info = middle_json.get("pdf_info")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported middle json backend for client-side output generation: {backend}")
    if not isinstance(pdf_info, list):
        raise ValueError("middle_json must contain a list field named pdf_info.")

    if backend in PDF_BACKENDS:
        finalize_client_side_middle_json(middle_json)
        pdf_info = middle_json["pdf_info"]

    image_dir = "images"

    markdown_path.write_text(
        render_markdown(pdf_info, image_dir),
        encoding="utf-8",
    )
    _write_json(
        content_list_path,
        render_content_list(pdf_info, image_dir),
    )
    _write_json(
        content_list_v2_path,
        render_content_list_v2(pdf_info, image_dir),
    )
    if backend in PDF_BACKENDS:
        _write_json(middle_json_path, middle_json)

    return (
        middle_json_path,
        markdown_path,
        content_list_path,
        content_list_v2_path,
    )
