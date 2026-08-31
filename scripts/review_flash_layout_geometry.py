"""生成 Flash 双几何全量回测、扰动不变性和逐页人工审阅产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(os.environ.get("MINERU_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pypdfium2 as pdfium  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pdftext.schema import Bbox  # noqa: E402

from mineru.model.flash.pdf.document import PDFDocument, PDFPageTextGeometry  # noqa: E402
from mineru.model.flash.pdf.pipeline import _analyze_native_document  # noqa: E402


DEFAULT_OUTPUT = SOURCE_ROOT / "output" / "pdf" / "flash_layout_geometry_review"
TRACKED_GOLD_MANIFEST = SOURCE_ROOT / "tests" / "fixtures" / "flash_layout_geometry_manifest.json"
PERTURBATION_NAMES = (
    "x_advance",
    "y_symmetric",
    "y_asymmetric",
    "latin_only",
    "punctuation",
)
_IGNORED_FINGERPRINT_KEYS = {
    "bbox",
    "lines",
    "image_path",
    "image_url",
    "img_path",
    "_layout_tree",
}
_TYPE_COLORS = {
    "doc_title": (214, 39, 40),
    "paragraph_title": (255, 127, 14),
    "text": (31, 119, 180),
    "header": (148, 103, 189),
    "footer": (140, 86, 75),
    "page_number": (227, 119, 194),
    "caption": (44, 160, 44),
    "footnote": (188, 189, 34),
    "page_footnote": (23, 190, 207),
    "table": (17, 127, 122),
    "image": (127, 127, 127),
    "equation": (188, 80, 144),
    "code": (57, 59, 121),
}


def _sha256_bytes(value: bytes) -> str:
    """返回字节内容的稳定 SHA256。"""

    return hashlib.sha256(value).hexdigest()


def _corpus_paths(manifest: dict[str, Any], source_root: Path) -> list[Path]:
    """按版本化 manifest 返回仓库相对路径语料。"""

    paths = []
    for document in manifest.get("documents", []):
        path = Path(str(document["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"manifest path must be repository-relative: {path}")
        resolved = source_root / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        expected_sha = str(document.get("sha256") or "")
        if expected_sha and _sha256_bytes(resolved.read_bytes()) != expected_sha:
            raise ValueError(f"manifest sha256 mismatch: {path}")
        paths.append(resolved)
    return paths


def _stable_factor(font_name: str) -> float:
    """由字体名确定性生成 loose bbox 扰动倍率。"""

    digest = hashlib.sha256(font_name.encode("utf-8", errors="replace")).digest()[0]
    return (1.75, 2.5, 4.0, 6.0)[digest % 4]


def _perturb_bbox(
    char: dict[str, Any],
    origin: tuple[float, float] | None,
    variant: str,
) -> None:
    """按指定变体改变 loose bbox，保留 tight 与 origin。"""

    raw_bbox = char.get("bbox")
    if raw_bbox is None or origin is None:
        return
    x0, y0, x1, y1 = [float(value) for value in raw_bbox]
    text = str(char.get("char") or "")
    font_name = str((char.get("font") or {}).get("name") or "")
    if variant == "latin_only" and not any(value.isascii() and value.isalnum() for value in text):
        return
    if variant == "punctuation" and not any(not value.isalnum() and not value.isspace() for value in text):
        return

    baseline = float(origin[1])
    ascent = max(0.1, baseline - y0)
    descent = max(0.1, y1 - baseline)
    factor = _stable_factor(font_name)
    if variant == "x_advance":
        new_x1 = x0 + max(0.1, x1 - x0) * max(1.75, factor)
        char["bbox"] = Bbox([x0, y0, new_x1, y1])
        return
    if variant == "y_asymmetric":
        new_y0 = baseline - ascent * factor
        new_y1 = baseline + descent * (0.5 if factor > 1 else 3.0)
    elif variant == "punctuation":
        new_y0 = baseline + 1.5 * max(ascent, descent)
        new_y1 = new_y0 + max(0.5, 0.25 * (ascent + descent))
    else:
        new_y0 = baseline - ascent * factor
        new_y1 = baseline + descent * factor
    char["bbox"] = Bbox([x0, min(new_y0, new_y1), x1, max(new_y0, new_y1)])


class _PerturbedPDFDocument:
    """代理 PDFDocument，并对读取出的 loose 字符框应用确定性扰动。"""

    def __init__(self, document: PDFDocument, variant: str) -> None:
        """保存真实文档和当前 loose 扰动名称。"""

        self._document = document
        self._variant = variant

    def __getattr__(self, name: str) -> Any:
        """把未覆盖的方法和属性转发给真实 PDFDocument。"""

        return getattr(self._document, name)

    def get_page_chars_with_geometry(self, page_idx: int) -> PDFPageTextGeometry:
        """返回 tight/origin 不变、loose bbox 已扰动的页面字符几何。"""

        geometry = self._document.get_page_chars_with_geometry(page_idx)
        chars = deepcopy(geometry.chars)
        loose_bboxes = dict(geometry.loose_bboxes)
        for char in chars:
            char_idx = char.get("char_idx")
            origin = geometry.origins.get(char_idx) if isinstance(char_idx, int) else None
            if isinstance(char_idx, int):
                raw_bbox = loose_bboxes.get(char_idx) or char.get("bbox")
                probe = {
                    "bbox": Bbox([float(value) for value in raw_bbox]),
                    "char": char.get("char"),
                    "font": char.get("font"),
                }
                _perturb_bbox(probe, origin, self._variant)
                loose_bboxes[char_idx] = tuple(float(value) for value in probe["bbox"])
        return PDFPageTextGeometry(
            chars=chars,
            tight_bboxes=geometry.tight_bboxes,
            origins=geometry.origins,
            loose_bboxes=loose_bboxes,
        )


def _canonical_value(value: Any) -> Any:
    """移除允许变化的几何与大载荷，保留语义标签、层级和可见内容。"""

    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key not in _IGNORED_FINGERPRINT_KEYS
        }
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str) and len(value) > 2048:
        return {"sha256": _sha256_bytes(value.encode("utf-8", errors="replace")), "length": len(value)}
    return value


def _page_fingerprint(page: list[dict[str, Any]]) -> str:
    """计算忽略输出 bbox 后的逐页语义指纹。"""

    payload = json.dumps(_canonical_value(page), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _page_bbox_fingerprint(page: list[dict[str, Any]]) -> str:
    """计算类型、文本顺序和公开 bbox 共同组成的逐页指纹。"""

    payload = [
        {
            "type": block.get("type"),
            "bbox": block.get("bbox"),
            "text_sha256": _sha256_bytes(_visible_text(block.get("content")).encode("utf-8", errors="replace")),
        }
        for block in page
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _visible_text(value: Any) -> str:
    """递归提取审阅标签使用的可见文本。"""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_visible_text(item) for item in value)
    if isinstance(value, dict):
        return _visible_text(value.get("content", ""))
    return ""


def _block_id(pdf_sha: str, page_index: int, block_index: int, block: dict[str, Any]) -> str:
    """生成与输出 bbox 无关的稳定人工金标 block ID。"""

    text_hash = _sha256_bytes(_visible_text(block.get("content")).encode("utf-8", errors="replace"))
    raw = f"{pdf_sha}:{page_index}:{block_index}:{block.get('type')}:{text_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _load_font(size: int) -> ImageFont.ImageFont:
    """加载可显示中英文 block 标签的系统字体。"""

    for path in (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _render_overlays(
    pdf_bytes: bytes,
    pages: list[list[dict[str, Any]]],
    geometry: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """渲染 block 与 loose/tight/origin/canonical 四层逐页 overlay。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(13)
    rendered: list[Path] = []
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        for page_index, blocks in enumerate(pages):
            page = document[page_index]
            image = page.render(scale=1.25).to_pil().convert("RGB")
            draw = ImageDraw.Draw(image)
            page_width, page_height = [float(value) for value in page.get_size()]

            def absolute_rectangle(bbox: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
                """把页面 point bbox 映射为当前渲染像素框。"""

                x0, y0, x1, y1 = [float(item) for item in bbox]
                return (
                    int(x0 / page_width * image.width),
                    int(y0 / page_height * image.height),
                    int(x1 / page_width * image.width),
                    int(y1 / page_height * image.height),
                )

            for char in geometry.get("char_repairs", []):
                if int(char["page_index"]) != page_index:
                    continue
                draw.rectangle(absolute_rectangle(char["source_bbox"]), outline=(220, 40, 40), width=1)
                draw.rectangle(absolute_rectangle(char["tight_bbox"]), outline=(40, 90, 220), width=1)
                draw.rectangle(absolute_rectangle(char["layout_bbox"]), outline=(30, 170, 70), width=1)
                ox = int(float(char["origin"][0]) / page_width * image.width)
                oy = int(float(char["origin"][1]) / page_height * image.height)
                draw.ellipse((ox - 2, oy - 2, ox + 2, oy + 2), fill=(255, 180, 0))
            for line in geometry.get("line_repairs", []):
                if int(line["page_index"]) != page_index:
                    continue
                draw.rectangle(absolute_rectangle(line["source_bbox"]), outline=(220, 40, 40), width=1)
                draw.rectangle(absolute_rectangle(line["layout_bbox"]), outline=(20, 180, 60), width=3)
            for block_index, block in enumerate(blocks):
                bbox = block.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                block_type = str(block.get("type") or "unknown")
                color = _TYPE_COLORS.get(block_type, (0, 0, 0))
                x0, y0, x1, y1 = [float(item) for item in bbox]
                rectangle = (
                    int(x0 * image.width),
                    int(y0 * image.height),
                    int(x1 * image.width),
                    int(y1 * image.height),
                )
                draw.rectangle(rectangle, outline=color, width=2)
                draw.text((rectangle[0] + 2, max(0, rectangle[1] - 15)), f"{block_index}:{block_type}", fill=color, font=font)
            output_path = output_dir / f"page_{page_index + 1:03d}.png"
            image.save(output_path)
            rendered.append(output_path)
            page.close()
    finally:
        document.close()
    return rendered


def _write_annotated_pdf(images: list[Path], output_path: Path) -> None:
    """把逐页几何 overlay 保存成稳定的多页审阅 PDF。"""

    members = [Image.open(path).convert("RGB") for path in images]
    if not members:
        return
    try:
        # 页面以 1.25x 渲染；90 DPI 可还原原始 72pt 页面尺寸。
        members[0].save(output_path, "PDF", save_all=True, append_images=members[1:], resolution=90.0)
    finally:
        for member in members:
            member.close()


def _build_contact_sheets(images: list[Path], output_dir: Path) -> list[Path]:
    """把逐页 overlay 按八页一张组合成可快速巡检的联系表。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(images), 8), start=1):
        members = [Image.open(path).convert("RGB") for path in images[start : start + 8]]
        try:
            thumb_width = 420
            thumbnails = []
            for member in members:
                height = max(1, round(member.height * thumb_width / member.width))
                thumbnails.append(member.resize((thumb_width, height)))
            row_heights = [max(image.height for image in thumbnails[row : row + 2]) for row in range(0, len(thumbnails), 2)]
            canvas = Image.new("RGB", (thumb_width * 2, sum(row_heights)), "white")
            y = 0
            for row_index, row_start in enumerate(range(0, len(thumbnails), 2)):
                for column, image in enumerate(thumbnails[row_start : row_start + 2]):
                    canvas.paste(image, (column * thumb_width, y))
                y += row_heights[row_index]
            path = output_dir / f"pages_{start + 1:03d}_{min(len(images), start + 8):03d}.png"
            canvas.save(path)
            output.append(path)
        finally:
            for member in members:
                member.close()
    return output


def _run_document(
    path: Path,
    *,
    perturb: bool,
    render: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """解析一份语料并返回正常、扰动、overlay 和金标页面信息。"""

    pdf_bytes = path.read_bytes()
    pdf_sha = _sha256_bytes(pdf_bytes)
    geometry_diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with PDFDocument(pdf_bytes) as document:
        mode = document.classify()
        pages = _analyze_native_document(document, geometry_diagnostics=geometry_diagnostics)
    elapsed = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    geometry = geometry_diagnostics[0] if geometry_diagnostics else {}
    fingerprints = [_page_fingerprint(page) for page in pages]
    bbox_fingerprints = [_page_bbox_fingerprint(page) for page in pages]
    perturbation_diffs: dict[str, list[int]] = {}
    perturbation_bbox_diffs: dict[str, list[int]] = {}
    if perturb and mode == "txt":
        for variant in PERTURBATION_NAMES:
            with PDFDocument(pdf_bytes) as document:
                variant_pages = _analyze_native_document(_PerturbedPDFDocument(document, variant))
            variant_fingerprints = [_page_fingerprint(page) for page in variant_pages]
            variant_bbox_fingerprints = [_page_bbox_fingerprint(page) for page in variant_pages]
            perturbation_diffs[variant] = [
                index
                for index, (normal, changed) in enumerate(zip(fingerprints, variant_fingerprints, strict=True))
                if normal != changed
            ]
            perturbation_bbox_diffs[variant] = [
                index
                for index, (normal, changed) in enumerate(zip(bbox_fingerprints, variant_bbox_fingerprints, strict=True))
                if normal != changed
            ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry.json").write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_list.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if render:
        page_images = _render_overlays(pdf_bytes, pages, geometry, output_dir / "pages")
        _build_contact_sheets(page_images, output_dir / "contact_sheets")
        _write_annotated_pdf(page_images, output_dir / "annotated.pdf")
        block_images = _render_overlays(pdf_bytes, pages, {}, output_dir / "block_pages")
        _build_contact_sheets(block_images, output_dir / "block_contact_sheets")
        _write_annotated_pdf(block_images, output_dir / "block_layout.pdf")

    return {
        "file": path.name,
        "path": str(path),
        "sha256": pdf_sha,
        "pages": len(pages),
        "parse_mode": mode,
        "page_fingerprints": fingerprints,
        "bbox_fingerprints": bbox_fingerprints,
        "perturbation_diffs": perturbation_diffs,
        "perturbation_bbox_diffs": perturbation_bbox_diffs,
        "elapsed_seconds": elapsed,
        "peak_rss_delta": max(0, rss_after - rss_before),
        "geometry_summary": {
            "repaired_chars": len(geometry.get("char_repairs", [])),
            "repaired_lines": sum(line.get("state") != "healthy" for line in geometry.get("line_repairs", [])),
            "split_y_candidates": sum(bool(line.get("split_y_candidate")) for line in geometry.get("line_repairs", [])),
            "strong_x_runs": sum(bool(run.get("strong_x_bad")) for run in geometry.get("run_diagnostics", [])),
            "sibling_x_runs": sum(bool(run.get("sibling_x_bad")) for run in geometry.get("run_diagnostics", [])),
        },
        "gold_pages": [
            {
                "page_index": page_index,
                "fingerprint": fingerprints[page_index],
                "bbox_fingerprint": bbox_fingerprints[page_index],
                "approved": False,
                "blocks": [
                    {
                        "id": _block_id(pdf_sha, page_index, block_index, block),
                        "type": str(block.get("type") or ""),
                        "text_sha256": _sha256_bytes(_visible_text(block.get("content")).encode("utf-8", errors="replace")),
                    }
                    for block_index, block in enumerate(page)
                ],
            }
            for page_index, page in enumerate(pages)
        ],
    }


def _parse_args() -> argparse.Namespace:
    """解析全量回测输出、扰动和渲染开关。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=TRACKED_GOLD_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--perturb-loose", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-performance-gate", action="store_true")
    parser.add_argument("--include", action="append", default=[], help="仅运行指定 PDF 文件名，可重复传入")
    parser.add_argument(
        "--role",
        action="append",
        choices=("normal", "abnormal", "ocr_control"),
        default=[],
        help="仅运行 manifest 中指定角色，可重复传入",
    )
    parser.add_argument("--approve", action="store_true", help="把已人工审阅的全部页面标成 approved")
    return parser.parse_args()


def main() -> int:
    """执行版本化 Flash 语料并写出几何、金标和视觉审阅产物。"""

    args = _parse_args()
    source_root = args.source_root.resolve()
    tracked_manifest_path = args.manifest.resolve()
    tracked_gold = json.loads(tracked_manifest_path.read_text(encoding="utf-8"))
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    role_by_name = {
        Path(str(document["path"])).name: str(document.get("role") or "") for document in tracked_gold.get("documents", [])
    }
    corpus_paths = [
        path
        for path in _corpus_paths(tracked_gold, source_root)
        if (not args.include or path.name in set(args.include))
        and (not args.role or role_by_name.get(path.name) in set(args.role))
    ]
    for index, path in enumerate(corpus_paths, start=1):
        print(f"[{index}/{len(corpus_paths)}] {path.name}", flush=True)
        document = _run_document(
            path,
            perturb=args.perturb_loose,
            render=not args.no_render,
            output_dir=output_dir / path.stem,
        )
        if args.approve:
            for page in document["gold_pages"]:
                page["approved"] = True
        documents.append(document)

    perturbation_failures = [
        {
            "file": document["file"],
            "variant": variant,
            "pages": pages,
        }
        for document in documents
        for variant, pages in document["perturbation_diffs"].items()
        if pages
    ]
    perturbation_bbox_failures = [
        {
            "file": document["file"],
            "variant": variant,
            "pages": pages,
        }
        for document in documents
        for variant, pages in document["perturbation_bbox_diffs"].items()
        if pages
    ]
    gold_mismatches: list[dict[str, Any]] = []
    tracked_by_file = {Path(str(document["path"])).name: document for document in tracked_gold.get("documents", [])}
    for document in documents:
        expected = tracked_by_file.get(document["file"])
        if expected is None or expected.get("sha256") != document["sha256"]:
            gold_mismatches.append({"file": document["file"], "reason": "source_missing_or_sha_mismatch"})
            continue
        expected_pages = expected.get("pages", [])
        if not expected_pages:
            continue
        if len(expected_pages) != len(document["page_fingerprints"]):
            gold_mismatches.append({"file": document["file"], "reason": "page_count_mismatch"})
            continue
        semantic_pages = [
            page_index
            for page_index, (expected_page, actual_fingerprint) in enumerate(
                zip(expected_pages, document["page_fingerprints"], strict=True)
            )
            if expected_page.get("fingerprint") != actual_fingerprint
        ]
        bbox_pages = [
            page_index
            for page_index, (expected_page, actual_fingerprint) in enumerate(
                zip(expected_pages, document["bbox_fingerprints"], strict=True)
            )
            if expected_page.get("bbox_fingerprint") != actual_fingerprint
        ]
        if semantic_pages:
            gold_mismatches.append(
                {"file": document["file"], "reason": "semantic_fingerprint_mismatch", "pages": semantic_pages}
            )
        if bbox_pages:
            gold_mismatches.append({"file": document["file"], "reason": "bbox_fingerprint_mismatch", "pages": bbox_pages})

    performance_failures = []
    if not args.skip_performance_gate:
        for document in documents:
            expected = tracked_by_file.get(document["file"], {})
            max_elapsed = expected.get("max_elapsed_seconds")
            max_rss = expected.get("max_peak_rss_delta")
            if isinstance(max_elapsed, (int, float)) and document["elapsed_seconds"] > float(max_elapsed):
                performance_failures.append({"file": document["file"], "metric": "elapsed_seconds"})
            if isinstance(max_rss, (int, float)) and document["peak_rss_delta"] > float(max_rss):
                performance_failures.append({"file": document["file"], "metric": "peak_rss_delta"})
    manifest = {
        "source_root": str(source_root),
        "source_count": len(documents),
        "page_count": sum(document["pages"] for document in documents),
        "txt_page_count": sum(document["pages"] for document in documents if document["parse_mode"] == "txt"),
        "perturbation_enabled": args.perturb_loose,
        "perturbations": list(PERTURBATION_NAMES),
        "perturbation_failures": perturbation_failures,
        "perturbation_bbox_failures": perturbation_bbox_failures,
        "performance_failures": performance_failures,
        "gold_mismatches": gold_mismatches,
        "documents": [{key: value for key, value in document.items() if key != "gold_pages"} for document in documents],
    }
    gold = {
        "contract": "Flash semantic labels and grouping ignore loose bbox coordinates",
        "baseline_git_sha": tracked_gold.get("baseline_git_sha"),
        "source_count": len(documents),
        "page_count": manifest["page_count"],
        "all_approved": all(page["approved"] for document in documents for page in document["gold_pages"]),
        "documents": [
            {
                "file": document["file"],
                "sha256": document["sha256"],
                "pages": document["gold_pages"],
            }
            for document in documents
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "flash_layout_gold_manifest.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    review_lines = [
        "# Flash layout geometry review",
        "",
        f"- Documents: {manifest['source_count']}",
        f"- Pages: {manifest['page_count']}",
        f"- TXT pages: {manifest['txt_page_count']}",
        f"- Perturbation enabled: {manifest['perturbation_enabled']}",
        f"- Perturbation failures: {len(perturbation_failures)}",
        f"- Perturbation bbox failures: {len(perturbation_bbox_failures)}",
        f"- Performance failures: {len(performance_failures)}",
        f"- Gold mismatches: {len(gold_mismatches)}",
        f"- All pages approved: {gold['all_approved']}",
        "",
    ]
    for document in documents:
        changed = sum(bool(pages) for pages in document["perturbation_diffs"].values())
        review_lines.append(
            f"- {document['file']}: {document['pages']} pages, {document['parse_mode']}, failed variants={changed}"
        )
    (output_dir / "review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return 1 if perturbation_failures or perturbation_bbox_failures or gold_mismatches or performance_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
