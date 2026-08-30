from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.pipeline import _analyze_native_document
from mineru.model.flash.pdf.text_styles import PDFTextScriptLine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output/pdf/flash_script_review"
PDF_PATHS = sorted((PROJECT_ROOT / "demo/pdfs").glob("*.pdf")) + [
    Path("/Users/myhloli/pdf/NPU_开发环境部署_参考指南.pdf"),
]
RENDER_DPI = 200
CONTEXT_CHARS = 40
CONTACT_COLUMNS = 3
CONTACT_ROWS = 5


def _sha256(path: Path) -> str:
    """计算源 PDF 摘要，供候选 ID 和审阅追踪使用。"""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(value: str) -> str:
    """生成适合 sidecar 与最终 TextSpan 对齐的稳定文本键。"""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = "".join(char for char in normalized if char.isprintable())
    return re.sub(r"\s+", "", normalized)


def _safe_stem(path: Path) -> str:
    """生成适合作为产物目录的稳定文件名。"""
    value = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("_.")
    return value[:64] or "document"


def _load_font(size: int) -> ImageFont.ImageFont:
    """优先加载支持中英文标签的系统字体。"""
    for font_path in (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


def _page_bbox_from_normalized(
    bbox: Any,
    page_size: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """把最终 Flash block bbox 转回页面 point。"""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return (
        values[0] * page_size[0],
        values[1] * page_size[1],
        values[2] * page_size[0],
        values[3] * page_size[1],
    )


def _script_candidates(
    diagnostic: dict[str, Any],
) -> tuple[dict[tuple[int | None, str, str], deque[dict[str, Any]]], list[dict[str, Any]]]:
    """把检测及最终投影 sidecar 转成按 block、角色和文本消费的候选队列。"""
    detected = []
    detected_by_key: dict[tuple[int, int, int, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    lines = sorted(
        diagnostic["lines"],
        key=lambda item: (item["source_index"], item["bbox"][1], item["bbox"][0]),
    )
    line_position = {item["source_index"]: index for index, item in enumerate(lines)}
    for script_line in diagnostic["script_lines"]:
        if not isinstance(script_line, PDFTextScriptLine):
            continue
        position = line_position.get(script_line.source_index)
        current = lines[position]["text"] if position is not None else script_line.text
        previous = lines[position - 1]["text"] if position is not None and position > 0 else ""
        following = lines[position + 1]["text"] if position is not None and position + 1 < len(lines) else ""
        for script_range in script_line.script_ranges:
            text = script_line.text[script_range.start : script_range.end]
            candidate = {
                "source_index": script_line.source_index,
                "range_start": script_range.start,
                "range_end": script_range.end,
                "role": script_range.style,
                "text": text,
                "bbox": list(script_range.bbox),
                "angle": script_line.angle,
                "formula_region": script_range.formula_region,
                "stable_body_count": script_range.stable_body_count,
                "previous_line": previous,
                "current_line": current,
                "next_line": following,
            }
            detected.append(candidate)
            detected_by_key[
                (
                    script_line.source_index,
                    script_range.start,
                    script_range.end,
                    script_range.style,
                    text,
                )
            ].append(candidate)

    queues: dict[tuple[int | None, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    materialized = diagnostic.get("materialized_ranges")
    if isinstance(materialized, list):
        candidates = []
        for mapped in sorted(
            materialized,
            key=lambda item: (
                int(item.get("block_index", -1)),
                int(item.get("raw_start", -1)),
                int(item.get("source_index", -1)),
                int(item.get("range_start", -1)),
            ),
        ):
            identity = (
                int(mapped["source_index"]),
                int(mapped["range_start"]),
                int(mapped["range_end"]),
                str(mapped["role"]),
                str(mapped["text"]),
            )
            candidate = dict(detected_by_key[identity].popleft()) if detected_by_key[identity] else dict(mapped)
            candidate.update(mapped)
            candidate["bbox"] = list(candidate["bbox"])
            candidates.append(candidate)
            queues[
                (
                    int(mapped["block_index"]),
                    str(mapped["role"]),
                    _normalized_text(str(mapped["text"])),
                )
            ].append(candidate)
        return queues, candidates

    for candidate in detected:
        queues[(None, str(candidate["role"]), _normalized_text(str(candidate["text"])))].append(candidate)
    return queues, detected


def _final_script_runs(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """收集最终真正物化的 Flash 上下标 TextSpan。"""
    runs = []
    for block_index, block in enumerate(blocks):
        block_type = str(block.get("type") or "")
        block_bbox = _page_bbox_from_normalized(block.get("bbox"), page_size)
        block_content = block.get("content")
        if block_type == "table" and isinstance(block_content, str) and block_content.lstrip().lower().startswith("<table"):
            soup = BeautifulSoup(block_content, "html.parser")
            for row_index, row in enumerate(soup.find_all("tr")):
                for cell_index, cell in enumerate(row.find_all(["td", "th"], recursive=False)):
                    for tag in cell.find_all(["sup", "sub"]):
                        text = tag.get_text()
                        if not _normalized_text(text):
                            continue
                        runs.append(
                            {
                                "block_index": block_index,
                                "block_type": block_type,
                                "block_bbox": list(block_bbox) if block_bbox is not None else None,
                                "role": "superscript" if tag.name == "sup" else "subscript",
                                "text": text,
                                "cell_row": row_index,
                                "cell_col": cell_index,
                                "cell_text": cell.get_text(),
                            }
                        )

        def walk(value: Any) -> None:
            """递归遍历 block 的 InlineSpan。"""
            if isinstance(value, dict):
                content = value.get("content")
                styles = value.get("styles")
                if value.get("type") == "text" and isinstance(content, str) and isinstance(styles, list):
                    for role in ("superscript", "subscript"):
                        if role in styles and _normalized_text(content):
                            runs.append(
                                {
                                    "block_index": block_index,
                                    "block_type": block_type,
                                    "block_bbox": list(block_bbox) if block_bbox is not None else None,
                                    "role": role,
                                    "text": content,
                                }
                            )
                if isinstance(content, (dict, list)):
                    walk(content)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(block_content)
    return runs


def _candidate_id(
    source_sha256: str,
    page_number: int,
    candidate: dict[str, Any],
) -> str:
    """用源文件、页、来源区间和角色构造稳定候选 ID。"""
    payload = "|".join(
        (
            source_sha256,
            str(page_number),
            str(candidate.get("source_index", -1)),
            str(candidate.get("range_start", -1)),
            str(candidate.get("range_end", -1)),
            str(candidate["role"]),
            str(candidate["text"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clip_context(value: str, styled_text: str) -> str:
    """围绕样式文本保留约 40 字符上下文。"""
    compact = re.sub(r"\s+", " ", value).strip()
    position = compact.find(styled_text)
    if position < 0:
        return compact[: 2 * CONTEXT_CHARS + len(styled_text)]
    start = max(0, position - CONTEXT_CHARS)
    end = min(len(compact), position + len(styled_text) + CONTEXT_CHARS)
    return compact[start:end]


def _render_crop(
    page_image: Image.Image,
    scale: float,
    bbox: tuple[float, float, float, float],
    record: dict[str, Any],
    output_path: Path,
) -> None:
    """裁剪上下标周围源页，并用角色颜色标出 tight bbox。"""
    x0, y0, x1, y1 = bbox
    line_height = max(1.0, y1 - y0)
    padding_x = max(96.0, 10.0 * line_height)
    padding_y = max(18.0, 1.75 * line_height)
    crop_box = (
        max(0, round((x0 - padding_x) * scale)),
        max(0, round((y0 - padding_y) * scale)),
        min(page_image.width, round((x1 + padding_x) * scale)),
        min(page_image.height, round((y1 + padding_y) * scale)),
    )
    crop = page_image.crop(crop_box).convert("RGB")
    label_height = 42
    canvas = Image.new("RGB", (crop.width, crop.height + label_height), "white")
    canvas.paste(crop, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    color = (220, 40, 40) if record["role"] == "superscript" else (30, 90, 220)
    draw.rectangle(
        (
            round(x0 * scale) - crop_box[0],
            round(y0 * scale) - crop_box[1] + label_height,
            round(x1 * scale) - crop_box[0],
            round(y1 * scale) - crop_box[1] + label_height,
        ),
        outline=color,
        width=4,
    )
    label = f"{record['id']}  p{record['page']}  {record['role']}  {record['text']}"
    draw.text((8, 8), label, fill=color, font=_load_font(20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _build_contact_sheets(
    records: list[dict[str, Any]],
    staging: Path,
    *,
    directory: str = "contact_sheets",
) -> list[str]:
    """按文档把候选 crop 组合成多张联系表。"""
    contact_paths = []
    page_size = (520, 250)
    per_sheet = CONTACT_COLUMNS * CONTACT_ROWS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["file"]].append(record)
    for document_name, document_records in sorted(grouped.items()):
        for sheet_index in range(math.ceil(len(document_records) / per_sheet)):
            subset = document_records[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
            canvas = Image.new(
                "RGB",
                (CONTACT_COLUMNS * page_size[0], CONTACT_ROWS * page_size[1]),
                "white",
            )
            for index, record in enumerate(subset):
                with Image.open(staging / record["crop"]) as source_crop:
                    crop = source_crop.convert("RGB")
                crop.thumbnail((page_size[0] - 12, page_size[1] - 12))
                x = (index % CONTACT_COLUMNS) * page_size[0] + 6
                y = (index // CONTACT_COLUMNS) * page_size[1] + 6
                canvas.paste(crop, (x, y))
            relative = Path(directory) / (f"{_safe_stem(Path(document_name))}_{sheet_index + 1:02d}.png")
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output)
            contact_paths.append(relative.as_posix())
    return contact_paths


def _write_review_markdown(records: list[dict[str, Any]], staging: Path) -> Path:
    """生成按文件和页码分组、带勾选框和 crop 的人工审阅 Markdown。"""
    lines = [
        "# Flash 上下标人工审阅",
        "",
        f"候选总数：{len(records)}",
        "",
        "- 红框：superscript",
        "- 蓝框：subscript",
        "",
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["file"], record["page"])].append(record)
    for (file_name, page_number), page_records in sorted(grouped.items()):
        lines.extend([f"## {file_name} - p{page_number}", ""])
        for record in page_records:
            context = _clip_context(record["current_line"], record["text"])
            lines.extend(
                [
                    f"- [ ] `{record['id']}` **{record['role']}** `{record['text']}`",
                    f"  - block: `{record['block_type']}`; angle: `{record['angle']}`; "
                    f"cell: `{record.get('cell_row')},{record.get('cell_col')}`; "
                    f"formula_region: `{record['formula_region']}`; stable_body_count: `{record['stable_body_count']}`",
                    f"  - previous: {record['previous_line']}",
                    f"  - context: {context}",
                    f"  - next: {record['next_line']}",
                    f"  - ![{record['id']}]({record['crop']})",
                    "",
                ]
            )
    output = staging / "review.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _promote(staging: Path) -> None:
    """把已验证 staging 目录原子提升为最终审阅目录。"""
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    backup = OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(backup)
    try:
        os.replace(staging, OUTPUT_DIR)
    except Exception:
        if backup.exists() and not OUTPUT_DIR.exists():
            backup.rename(OUTPUT_DIR)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    """扫描全部 Flash PDF，生成可追踪的上下标人工审阅产物。"""
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="flash_script_review_",
            dir=OUTPUT_DIR.parent,
        )
    )
    records = []
    documents = []
    try:
        for document_index, pdf_path in enumerate(PDF_PATHS, start=1):
            source_sha256 = _sha256(pdf_path)
            diagnostics: list[dict[str, Any]] = []
            print(f"解析 {document_index}/{len(PDF_PATHS)} {pdf_path.name}", flush=True)
            with PDFDocument(str(pdf_path)) as document:
                pages = _analyze_native_document(
                    document,
                    script_diagnostics=diagnostics,
                )
                page_images: dict[int, tuple[Image.Image, float]] = {}
                document_count = 0
                for page_index, blocks in enumerate(pages):
                    diagnostic = diagnostics[page_index]
                    queues, _candidates = _script_candidates(diagnostic)
                    final_runs = _final_script_runs(blocks, diagnostic["page_size"])
                    for final_index, final_run in enumerate(final_runs):
                        key = (
                            final_run["block_index"],
                            final_run["role"],
                            _normalized_text(final_run["text"]),
                        )
                        fallback_key = (
                            None,
                            final_run["role"],
                            _normalized_text(final_run["text"]),
                        )
                        candidate = (
                            queues[key].popleft()
                            if queues[key]
                            else queues[fallback_key].popleft()
                            if queues[fallback_key]
                            else {
                                "source_index": -1,
                                "range_start": final_index,
                                "range_end": final_index + 1,
                                "role": final_run["role"],
                                "text": final_run["text"],
                                "bbox": final_run["block_bbox"],
                                "angle": 0,
                                "formula_region": False,
                                "stable_body_count": 0,
                                "previous_line": "",
                                "current_line": final_run.get("cell_text", final_run["text"]),
                                "next_line": "",
                            }
                        )
                        bbox = candidate.get("bbox") or final_run["block_bbox"]
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            raise AssertionError((pdf_path, page_index, final_run))
                        record_id = _candidate_id(
                            source_sha256,
                            page_index + 1,
                            candidate,
                        )
                        relative_crop = Path("crops") / _safe_stem(pdf_path) / f"{record_id}.png"
                        record = {
                            "id": record_id,
                            "file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "source_sha256": source_sha256,
                            "page": page_index + 1,
                            "block_type": final_run["block_type"],
                            "role": final_run["role"],
                            "text": final_run["text"],
                            "bbox": [float(value) for value in bbox],
                            "angle": int(candidate.get("angle", 0)),
                            "source_index": int(candidate.get("source_index", -1)),
                            "range_start": int(candidate.get("range_start", -1)),
                            "range_end": int(candidate.get("range_end", -1)),
                            "formula_region": bool(candidate.get("formula_region", False)),
                            "stable_body_count": int(candidate.get("stable_body_count", 0)),
                            "previous_line": str(candidate.get("previous_line", "")),
                            "current_line": str(candidate.get("current_line", final_run["text"])),
                            "next_line": str(candidate.get("next_line", "")),
                            "cell_row": final_run.get("cell_row"),
                            "cell_col": final_run.get("cell_col"),
                            "cell_text": final_run.get("cell_text"),
                            "crop": relative_crop.as_posix(),
                        }
                        if page_index not in page_images:
                            rendered = document.render_page(
                                page_index,
                                scale=RENDER_DPI / 72,
                            )
                            page_images[page_index] = (
                                rendered.pil_image,
                                rendered.scale,
                            )
                        image, scale = page_images[page_index]
                        _render_crop(
                            image,
                            scale,
                            tuple(record["bbox"]),
                            record,
                            staging / relative_crop,
                        )
                        records.append(record)
                        document_count += 1
                documents.append(
                    {
                        "file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "source_sha256": source_sha256,
                        "page_count": document.page_count,
                        "candidate_count": document_count,
                    }
                )
        contact_sheets = _build_contact_sheets(records, staging)
        table_contact_sheets = _build_contact_sheets(
            [record for record in records if record["block_type"] == "table"],
            staging,
            directory="table_contact_sheets",
        )
        review_path = _write_review_markdown(records, staging)
        manifest = {
            "source_count": len(PDF_PATHS),
            "page_count": sum(document["page_count"] for document in documents),
            "candidate_count": len(records),
            "role_counts": dict(Counter(record["role"] for record in records)),
            "formula_region_count": sum(record["formula_region"] for record in records),
            "documents": documents,
            "contact_sheets": contact_sheets,
            "table_contact_sheets": table_contact_sheets,
            "records": records,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        crop_count = len(list((staging / "crops").rglob("*.png")))
        checkbox_count = review_path.read_text(encoding="utf-8").count("- [ ]")
        if not (
            len(records) == crop_count == checkbox_count
            and sum(document["candidate_count"] for document in documents) == len(records)
        ):
            raise AssertionError(
                {
                    "records": len(records),
                    "crops": crop_count,
                    "checkboxes": checkbox_count,
                }
            )
        _promote(staging)
        print(OUTPUT_DIR / "manifest.json")
        print(OUTPUT_DIR / "review.md")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
