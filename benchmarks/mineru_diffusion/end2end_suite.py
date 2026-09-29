from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import pypdfium2 as pdfium
from PIL import Image

from benchmarks.mineru_diffusion.compare_results import (
    compare_layout_blocks,
    parse_layout_blocks,
)
from benchmarks.mineru_diffusion.end2end_openai import (
    End2EndConfig,
    End2EndClient,
    End2EndOpenAIClient,
    convert_otsl_to_html,
    run_end2end_batch,
    run_end2end,
    wrap_equation,
)


PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}
ROTATE_TOKENS = {
    None: "<|rotate_up|>",
    0: "<|rotate_up|>",
    90: "<|rotate_right|>",
    180: "<|rotate_down|>",
    270: "<|rotate_left|>",
}


@dataclass(frozen=True)
class PdfSource:
    doc_id: str
    pdf_path: Path
    pages: str = "all"


@dataclass(frozen=True)
class PageCase:
    case_id: str
    doc_id: str
    pdf_path: Path
    page_index: int
    image_path: Path
    width: int
    height: int


def parse_page_spec(spec: str, page_count: int) -> list[int]:
    spec = spec.strip().lower()
    if page_count < 1:
        return []
    if spec in {"", "all", "*"}:
        return list(range(page_count))

    pages: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token == "last":
            pages.append(page_count - 1)
            continue
        if token.startswith("first:"):
            count = int(token.split(":", 1)[1])
            pages.extend(range(min(count, page_count)))
            continue
        if token.startswith("last:"):
            count = int(token.split(":", 1)[1])
            start = max(0, page_count - count)
            pages.extend(range(start, page_count))
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = page_count - 1 if end_text == "last" else int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending page range: {token}")
            pages.extend(range(start, end + 1))
            continue
        pages.append(int(token))

    seen: set[int] = set()
    normalized: list[int] = []
    for page in pages:
        if page < 0 or page >= page_count:
            raise ValueError(
                f"Page index {page} out of range for document with {page_count} pages"
            )
        if page not in seen:
            seen.add(page)
            normalized.append(page)
    return normalized


def default_pdf_sources(repo_root: Path | None = None) -> list[PdfSource]:
    repo_root = (repo_root or Path.cwd()).resolve()
    candidates = [
        PdfSource("mineru_demo1", repo_root / "../MinerU/demo/pdfs/demo1.pdf", "0-2"),
        PdfSource("mineru_demo2", repo_root / "../MinerU/demo/pdfs/demo2.pdf", "0-2"),
        PdfSource("mineru_demo3", repo_root / "../MinerU/demo/pdfs/demo3.pdf", "0-2"),
        PdfSource("mineru_small_ocr", repo_root / "../MinerU/demo/pdfs/small_ocr.pdf", "0-2"),
        PdfSource("mineru_unit_test", repo_root / "../MinerU/tests/unittest/pdfs/test.pdf", "all"),
        PdfSource(
            "mineru_diffusion_paper",
            repo_root / "../MinerU-Diffusion/docs/MinerU-Diffusion-V1.pdf",
            "0-2,5,10,20,last",
        ),
        PdfSource(
            "ragflow_doc1",
            repo_root / "../ragflow-0.25.6/test/benchmark/test_docs/Doc1.pdf",
            "all",
        ),
        PdfSource(
            "ragflow_doc2",
            repo_root / "../ragflow-0.25.6/test/benchmark/test_docs/Doc2.pdf",
            "all",
        ),
        PdfSource(
            "ragflow_doc3",
            repo_root / "../ragflow-0.25.6/test/benchmark/test_docs/Doc3.pdf",
            "all",
        ),
    ]
    return [
        PdfSource(item.doc_id, item.pdf_path.resolve(), item.pages)
        for item in candidates
        if item.pdf_path.exists()
    ]


def _render_page(pdf_doc: pdfium.PdfDocument, page_index: int, scale: float) -> Image.Image:
    page = pdf_doc[page_index]
    try:
        return page.render(scale=scale).to_pil().convert("RGB")
    finally:
        page.close()


def render_pdf_sources(
    sources: Sequence[PdfSource],
    output_dir: Path,
    *,
    scale: float = 2.0,
) -> list[PageCase]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cases: list[PageCase] = []

    for source in sources:
        pdf_path = source.pdf_path.expanduser().resolve()
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        try:
            page_count = len(pdf_doc)
            page_indices = parse_page_spec(source.pages, page_count)
            for page_index in page_indices:
                image = _render_page(pdf_doc, page_index, scale)
                case_id = f"{source.doc_id}_p{page_index + 1:04d}"
                image_path = images_dir / f"{case_id}.png"
                image.save(image_path)
                cases.append(
                    PageCase(
                        case_id=case_id,
                        doc_id=source.doc_id,
                        pdf_path=pdf_path,
                        page_index=page_index,
                        image_path=image_path.resolve(),
                        width=image.width,
                        height=image.height,
                    )
                )
                image.close()
        finally:
            pdf_doc.close()

    write_manifest(output_dir / "manifest.json", cases)
    return cases


def write_manifest(path: Path, cases: Sequence[PageCase]) -> None:
    payload = {"cases": [_case_to_json(case) for case in cases]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> list[PageCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for row in payload["cases"]:
        cases.append(
            PageCase(
                case_id=row["case_id"],
                doc_id=row["doc_id"],
                pdf_path=Path(row["pdf_path"]),
                page_index=int(row["page_index"]),
                image_path=Path(row["image_path"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return cases


def _case_to_json(case: PageCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["pdf_path"] = str(case.pdf_path)
    payload["image_path"] = str(case.image_path)
    return payload


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def block_type_counts(blocks: Sequence[Any]) -> dict[str, int]:
    return dict(Counter(str(_block_get(block, "type", "")) for block in blocks))


def _block_content(block: Any) -> str:
    content = str(_block_get(block, "content", "") or "").strip()
    if not content:
        return ""
    block_type = str(_block_get(block, "type", ""))
    if block_type == "table":
        return convert_otsl_to_html(content)
    if block_type == "equation":
        return wrap_equation(content)
    return content


def blocks_to_markdown(blocks: Sequence[Any], *, keep_paratext: bool = False) -> str:
    parts: list[str] = []
    for block in blocks:
        block_type = str(_block_get(block, "type", ""))
        if not keep_paratext and block_type in PARATEXT_TYPES:
            continue
        content = _block_content(block)
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _bbox_to_1000(bbox: Sequence[float]) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    left, top, right, bottom = (float(value) for value in bbox)
    if right <= left or bottom <= top:
        return None
    return tuple(
        max(0, min(1000, round(value * 1000)))
        for value in (left, top, right, bottom)
    )


def blocks_to_layout_text(blocks: Sequence[Any]) -> str:
    lines: list[str] = []
    for block in blocks:
        bbox = _bbox_to_1000(_block_get(block, "bbox", []))
        if bbox is None:
            continue
        block_type = str(_block_get(block, "type", "unknown"))
        angle = _block_get(block, "angle", None)
        rotate = ROTATE_TOKENS.get(angle, "<|rotate_up|>")
        lines.append(
            "<|box_start|>"
            f"{bbox[0]:03d} {bbox[1]:03d} {bbox[2]:03d} {bbox[3]:03d}"
            "<|box_end|>"
            f"<|ref_start|>{block_type}<|ref_end|>{rotate}"
        )
    return "\n".join(lines)


def _jsonable_blocks(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for block in blocks:
        if isinstance(block, dict):
            rows.append(dict(block))
        else:
            rows.append(asdict(block))
    return rows


def make_result_row(
    case: PageCase,
    *,
    ok: bool,
    metrics: dict[str, Any] | None = None,
    blocks: Sequence[Any] = (),
    markdown: str = "",
    layout_output: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    blocks_json = _jsonable_blocks(blocks)
    layout_text = layout_output or blocks_to_layout_text(blocks_json)
    return {
        **_case_to_json(case),
        "ok": ok,
        "error": error,
        "metrics": metrics or {},
        "markdown": markdown,
        "layout_output": layout_text,
        "blocks": blocks_json,
        "block_type_counts": block_type_counts(blocks_json),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latency(row: dict[str, Any]) -> float | None:
    value = row.get("metrics", {}).get("total_elapsed")
    return float(value) if value is not None else None


def summarize_suite_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    latencies = [
        latency
        for latency in (_latency(row) for row in ok_rows)
        if latency is not None
    ]
    type_counts: Counter[str] = Counter()
    for row in ok_rows:
        type_counts.update(row.get("block_type_counts", {}))
    markdown_chars = sum(len(row.get("markdown", "")) for row in ok_rows)
    total_latency = sum(latencies)
    summary = {
        "num_cases": len(rows),
        "num_ok": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "p50_latency_s": statistics.median(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "total_latency_s": total_latency if latencies else None,
        "markdown_chars": markdown_chars,
        "markdown_chars_per_s": (
            markdown_chars / total_latency if total_latency > 0 else None
        ),
        "block_type_counts": dict(type_counts),
    }
    content_concurrency_values = {
        row.get("metrics", {}).get("content_concurrency")
        for row in ok_rows
        if row.get("metrics", {}).get("content_concurrency") is not None
    }
    if len(content_concurrency_values) == 1:
        summary["content_concurrency"] = content_concurrency_values.pop()
    layout_concurrency_values = {
        row.get("metrics", {}).get("layout_concurrency")
        for row in ok_rows
        if row.get("metrics", {}).get("layout_concurrency") is not None
    }
    if len(layout_concurrency_values) == 1:
        summary["layout_concurrency"] = layout_concurrency_values.pop()
    for source_key, summary_key in (
        ("throughput_wall_elapsed_s", "throughput_wall_elapsed_s"),
        ("throughput_layout_wall_elapsed_s", "throughput_layout_wall_elapsed_s"),
        ("throughput_extract_wall_elapsed_s", "throughput_extract_wall_elapsed_s"),
    ):
        values = {
            row.get("metrics", {}).get(source_key)
            for row in ok_rows
            if row.get("metrics", {}).get(source_key) is not None
        }
        if len(values) == 1:
            summary[summary_key] = values.pop()
    wall_elapsed = summary.get("throughput_wall_elapsed_s")
    if wall_elapsed:
        summary["throughput_pages_per_s"] = len(ok_rows) / wall_elapsed
        summary["throughput_markdown_chars_per_s"] = markdown_chars / wall_elapsed
    return summary


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _similarity(lhs: str, rhs: str) -> float | None:
    if not lhs or not rhs:
        return None
    return SequenceMatcher(
        None,
        _normalize_text(lhs),
        _normalize_text(rhs),
        autojunk=False,
    ).ratio()


def compare_suite_rows(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["case_id"]: row for row in baseline_rows}
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    case_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    cases = []
    for case_id in case_ids:
        baseline = baseline_by_id.get(case_id)
        candidate = candidate_by_id.get(case_id)
        baseline_latency = _latency(baseline) if baseline else None
        candidate_latency = _latency(candidate) if candidate else None
        baseline_blocks = parse_layout_blocks(baseline.get("layout_output", "")) if baseline else []
        candidate_blocks = parse_layout_blocks(candidate.get("layout_output", "")) if candidate else []
        matches, precision, recall, f1 = compare_layout_blocks(
            baseline_blocks,
            candidate_blocks,
        )
        cases.append(
            {
                "case_id": case_id,
                "doc_id": (baseline or candidate or {}).get("doc_id"),
                "page_index": (baseline or candidate or {}).get("page_index"),
                "baseline_ok": bool(baseline and baseline.get("ok")),
                "candidate_ok": bool(candidate and candidate.get("ok")),
                "baseline_latency_s": baseline_latency,
                "candidate_latency_s": candidate_latency,
                "speedup": (
                    baseline_latency / candidate_latency
                    if baseline_latency and candidate_latency
                    else None
                ),
                "markdown_similarity": _similarity(
                    baseline.get("markdown", "") if baseline else "",
                    candidate.get("markdown", "") if candidate else "",
                ),
                "baseline_blocks": len(baseline_blocks),
                "candidate_blocks": len(candidate_blocks),
                "layout_matched_blocks": matches,
                "layout_precision": precision,
                "layout_recall": recall,
                "layout_f1": f1,
                "baseline_type_counts": baseline.get("block_type_counts", {}) if baseline else {},
                "candidate_type_counts": candidate.get("block_type_counts", {}) if candidate else {},
            }
        )

    matched_ok = [
        item
        for item in cases
        if item["baseline_ok"] and item["candidate_ok"]
    ]
    similarities = [
        item["markdown_similarity"]
        for item in matched_ok
        if item["markdown_similarity"] is not None
    ]
    layout_f1s = [item["layout_f1"] for item in matched_ok]
    baseline_total = sum(
        item["baseline_latency_s"] or 0.0 for item in matched_ok
    )
    candidate_total = sum(
        item["candidate_latency_s"] or 0.0 for item in matched_ok
    )
    summary = {
        "num_cases": len(cases),
        "num_matched_ok": len(matched_ok),
        "baseline_total_latency_s": baseline_total,
        "candidate_total_latency_s": candidate_total,
        "total_speedup": (
            baseline_total / candidate_total if candidate_total > 0 else None
        ),
        "mean_speedup": statistics.mean(
            item["speedup"] for item in matched_ok if item["speedup"] is not None
        )
        if any(item["speedup"] is not None for item in matched_ok)
        else None,
        "mean_markdown_similarity": (
            statistics.mean(similarities) if similarities else None
        ),
        "min_markdown_similarity": min(similarities) if similarities else None,
        "mean_layout_f1": statistics.mean(layout_f1s) if layout_f1s else None,
        "min_layout_f1": min(layout_f1s) if layout_f1s else None,
    }
    return {"summary": summary, "cases": cases}


def write_run_outputs(output_dir: Path, rows: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    timestamp = int(time.time())
    results_path = output_dir / f"results_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"
    summary = summarize_suite_rows(rows)
    _write_jsonl(results_path, rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "latest_results.jsonl", rows)
    (output_dir / "latest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results_path, summary_path


def run_dllm_suite(
    cases: Sequence[PageCase],
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    timeout: float,
    layout_max_tokens: int,
    content_max_tokens: int,
    table_max_tokens: int,
    formula_max_tokens: int,
    block_size: int,
    dynamic_threshold: float,
    content_concurrency: int,
    max_denoising_steps: int | None = None,
    keep_paratext: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = End2EndOpenAIClient(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        block_size=block_size,
        dynamic_threshold=dynamic_threshold,
        max_denoising_steps=max_denoising_steps,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = run_end2end(
                End2EndConfig(
                    image_path=case.image_path,
                    output_dir=output_dir / "cases" / case.case_id,
                    endpoint=endpoint,
                    model=model,
                    timeout=timeout,
                    layout_max_tokens=layout_max_tokens,
                    content_max_tokens=content_max_tokens,
                    table_max_tokens=table_max_tokens,
                    formula_max_tokens=formula_max_tokens,
                    block_size=block_size,
                    max_denoising_steps=max_denoising_steps,
                    dynamic_threshold=dynamic_threshold,
                    content_concurrency=content_concurrency,
                    keep_paratext=keep_paratext,
                ),
                client=client,
            )
            rows.append(
                make_result_row(
                    case,
                    ok=True,
                    metrics=result.metrics,
                    blocks=result.blocks,
                    markdown=result.markdown,
                    layout_output=blocks_to_layout_text(result.blocks),
                )
            )
        except Exception as exc:
            rows.append(make_result_row(case, ok=False, error=repr(exc)))
    return rows


def run_dllm_throughput_suite(
    cases: Sequence[PageCase],
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    timeout: float,
    layout_max_tokens: int,
    content_max_tokens: int,
    table_max_tokens: int,
    formula_max_tokens: int,
    block_size: int,
    dynamic_threshold: float,
    layout_concurrency: int,
    content_concurrency: int,
    max_denoising_steps: int | None = None,
    keep_paratext: bool = False,
    client: End2EndClient | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = client or End2EndOpenAIClient(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        block_size=block_size,
        dynamic_threshold=dynamic_threshold,
        max_denoising_steps=max_denoising_steps,
    )
    configs = [
        End2EndConfig(
            image_path=case.image_path,
            output_dir=output_dir / "cases" / case.case_id,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            layout_max_tokens=layout_max_tokens,
            content_max_tokens=content_max_tokens,
            table_max_tokens=table_max_tokens,
            formula_max_tokens=formula_max_tokens,
            block_size=block_size,
            max_denoising_steps=max_denoising_steps,
            dynamic_threshold=dynamic_threshold,
            content_concurrency=content_concurrency,
            keep_paratext=keep_paratext,
        )
        for case in cases
    ]
    try:
        results = run_end2end_batch(
            configs,
            client=client,
            layout_concurrency=layout_concurrency,
            content_concurrency=content_concurrency,
        )
    except Exception as exc:
        return [make_result_row(case, ok=False, error=repr(exc)) for case in cases]

    rows: list[dict[str, Any]] = []
    for case, result in zip(cases, results):
        rows.append(
            make_result_row(
                case,
                ok=True,
                metrics=result.metrics,
                blocks=result.blocks,
                markdown=result.markdown,
                layout_output=blocks_to_layout_text(result.blocks),
            )
        )
    return rows


def _run_mineru_client_page(
    client: Any,
    image: Image.Image,
    *,
    image_analysis: bool,
) -> tuple[list[Any], dict[str, Any]]:
    started = time.perf_counter()
    layout_start = time.perf_counter()
    layout_result = client.layout_detect(image)
    layout_elapsed = time.perf_counter() - layout_start

    extract_start = time.perf_counter()
    block_images, prompts, params, indices = client.helper.prepare_for_extract(
        image,
        layout_result,
        None,
        image_analysis,
    )
    outputs = client._batch_predict(block_images, prompts, params, None, None)
    for index, output in zip(indices, outputs):
        layout_result[index].content = output.text
        layout_result[index].scored = output.scored
    blocks = client.helper.post_process(layout_result)
    extract_elapsed = time.perf_counter() - extract_start
    markdown = blocks_to_markdown(blocks)
    metrics = {
        "layout_elapsed": layout_elapsed,
        "extract_elapsed": extract_elapsed,
        "total_elapsed": time.perf_counter() - started,
        "num_blocks": len(blocks),
        "num_extracted_blocks": len(indices),
        "markdown_chars": len(markdown),
    }
    return blocks, metrics


def run_mineru_client_suite(
    cases: Sequence[PageCase],
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    timeout: int,
    max_concurrency: int,
    image_analysis: bool,
    keep_paratext: bool = False,
) -> list[dict[str, Any]]:
    from mineru_vl_utils import MinerUClient

    output_dir.mkdir(parents=True, exist_ok=True)
    client = MinerUClient(
        backend="http-client",
        server_url=endpoint,
        model_name=model,
        http_timeout=timeout,
        max_concurrency=max_concurrency,
        image_analysis=image_analysis,
        enable_table_formula_eq_wrap=True,
        enable_cross_page_table_merge=True,
        skip_model_name_checking=True,
        use_tqdm=False,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            with Image.open(case.image_path) as image:
                blocks, metrics = _run_mineru_client_page(
                    client,
                    image.convert("RGB"),
                    image_analysis=image_analysis,
                )
            rows.append(
                make_result_row(
                    case,
                    ok=True,
                    metrics=metrics,
                    blocks=blocks,
                    markdown=blocks_to_markdown(blocks, keep_paratext=keep_paratext),
                    layout_output=blocks_to_layout_text(blocks),
                )
            )
        except Exception as exc:
            rows.append(make_result_row(case, ok=False, error=repr(exc)))
    return rows


def run_mineru_client_throughput_suite(
    cases: Sequence[PageCase],
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    timeout: int,
    max_concurrency: int,
    image_analysis: bool,
    keep_paratext: bool = False,
) -> list[dict[str, Any]]:
    from mineru_vl_utils import MinerUClient

    output_dir.mkdir(parents=True, exist_ok=True)
    client = MinerUClient(
        backend="http-client",
        server_url=endpoint,
        model_name=model,
        http_timeout=timeout,
        max_concurrency=max_concurrency,
        image_analysis=image_analysis,
        enable_table_formula_eq_wrap=True,
        enable_cross_page_table_merge=True,
        skip_model_name_checking=True,
        use_tqdm=False,
    )
    images: list[Image.Image] = []
    try:
        for case in cases:
            images.append(Image.open(case.image_path).convert("RGB"))

        started = time.perf_counter()
        layout_start = time.perf_counter()
        layout_results = client.batch_layout_detect(images)
        layout_wall_elapsed = time.perf_counter() - layout_start

        extract_start = time.perf_counter()
        prepared_inputs = client.helper.batch_prepare_for_extract(
            client.executor,
            images,
            layout_results,
            None,
            image_analysis,
        )
        all_images, all_prompts, all_params, all_indices = client._flatten_prepared_inputs(
            prepared_inputs
        )
        outputs = client._batch_predict(
            all_images,
            all_prompts,
            all_params,
            None,
            None,
        )
        for (image_index, block_index), output in zip(all_indices, outputs):
            layout_results[image_index][block_index].content = output.text
            layout_results[image_index][block_index].scored = output.scored
        processed_list = client.helper.batch_post_process(
            client.executor,
            layout_results,
        )
        extract_wall_elapsed = time.perf_counter() - extract_start
        wall_elapsed = time.perf_counter() - started
    except Exception as exc:
        return [make_result_row(case, ok=False, error=repr(exc)) for case in cases]
    finally:
        for image in images:
            image.close()

    extracted_counts = Counter(image_index for image_index, _ in all_indices)
    rows: list[dict[str, Any]] = []
    for image_index, (case, blocks) in enumerate(zip(cases, processed_list)):
        markdown = blocks_to_markdown(blocks, keep_paratext=keep_paratext)
        metrics = {
            "num_blocks": len(blocks),
            "num_extracted_blocks": extracted_counts.get(image_index, 0),
            "markdown_chars": len(markdown),
            "layout_concurrency": max_concurrency,
            "content_concurrency": max_concurrency,
            "throughput_batch_size": len(cases),
            "throughput_wall_elapsed_s": wall_elapsed,
            "throughput_layout_wall_elapsed_s": layout_wall_elapsed,
            "throughput_extract_wall_elapsed_s": extract_wall_elapsed,
        }
        rows.append(
            make_result_row(
                case,
                ok=True,
                metrics=metrics,
                blocks=blocks,
                markdown=markdown,
                layout_output=blocks_to_layout_text(blocks),
            )
        )
    return rows


def _parse_source_args(values: Sequence[str]) -> list[PdfSource]:
    sources: list[PdfSource] = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) == 1:
            path = Path(parts[0])
            sources.append(PdfSource(path.stem, path, "all"))
        elif len(parts) == 2:
            doc_id, path = parts
            sources.append(PdfSource(doc_id, Path(path), "all"))
        else:
            doc_id, path, pages = parts
            sources.append(PdfSource(doc_id, Path(path), pages))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--scale", type=float, default=2.0)
    render.add_argument("--source", action="append", default=[])
    render.add_argument("--preset", choices=["coverage"], default="coverage")

    run_dllm = subparsers.add_parser("run-dllm")
    run_dllm.add_argument("--manifest", type=Path, required=True)
    run_dllm.add_argument("--output-dir", type=Path, required=True)
    run_dllm.add_argument("--endpoint", default="http://127.0.0.1:18084/v1/chat/completions")
    run_dllm.add_argument("--model", default="mineru-diffusion")
    run_dllm.add_argument("--timeout", type=float, default=600.0)
    run_dllm.add_argument("--layout-max-tokens", type=int, default=2048)
    run_dllm.add_argument("--content-max-tokens", type=int, default=1024)
    run_dllm.add_argument("--table-max-tokens", type=int, default=2048)
    run_dllm.add_argument("--formula-max-tokens", type=int, default=1024)
    run_dllm.add_argument("--block-size", type=int, default=32)
    run_dllm.add_argument("--dynamic-threshold", type=float, default=0.95)
    run_dllm.add_argument("--max-denoising-steps", type=int)
    run_dllm.add_argument("--content-concurrency", type=int, default=1)
    run_dllm.add_argument("--keep-paratext", action="store_true")

    run_dllm_throughput = subparsers.add_parser("run-dllm-throughput")
    run_dllm_throughput.add_argument("--manifest", type=Path, required=True)
    run_dllm_throughput.add_argument("--output-dir", type=Path, required=True)
    run_dllm_throughput.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18084/v1/chat/completions",
    )
    run_dllm_throughput.add_argument("--model", default="mineru-diffusion")
    run_dllm_throughput.add_argument("--timeout", type=float, default=600.0)
    run_dllm_throughput.add_argument("--layout-max-tokens", type=int, default=2048)
    run_dllm_throughput.add_argument("--content-max-tokens", type=int, default=1024)
    run_dllm_throughput.add_argument("--table-max-tokens", type=int, default=2048)
    run_dllm_throughput.add_argument("--formula-max-tokens", type=int, default=1024)
    run_dllm_throughput.add_argument("--block-size", type=int, default=32)
    run_dllm_throughput.add_argument("--dynamic-threshold", type=float, default=0.95)
    run_dllm_throughput.add_argument("--max-denoising-steps", type=int)
    run_dllm_throughput.add_argument("--layout-concurrency", type=int, default=4)
    run_dllm_throughput.add_argument("--content-concurrency", type=int, default=8)
    run_dllm_throughput.add_argument("--keep-paratext", action="store_true")

    run_mineru = subparsers.add_parser("run-mineru-client")
    run_mineru.add_argument("--manifest", type=Path, required=True)
    run_mineru.add_argument("--output-dir", type=Path, required=True)
    run_mineru.add_argument("--endpoint", default="http://127.0.0.1:30000")
    run_mineru.add_argument("--model", default="mineru2.5-pro")
    run_mineru.add_argument("--timeout", type=int, default=600)
    run_mineru.add_argument("--max-concurrency", type=int, default=8)
    run_mineru.add_argument("--disable-image-analysis", action="store_true")
    run_mineru.add_argument("--keep-paratext", action="store_true")

    run_mineru_throughput = subparsers.add_parser("run-mineru-client-throughput")
    run_mineru_throughput.add_argument("--manifest", type=Path, required=True)
    run_mineru_throughput.add_argument("--output-dir", type=Path, required=True)
    run_mineru_throughput.add_argument("--endpoint", default="http://127.0.0.1:30000")
    run_mineru_throughput.add_argument("--model", default="mineru2.5-pro")
    run_mineru_throughput.add_argument("--timeout", type=int, default=600)
    run_mineru_throughput.add_argument("--max-concurrency", type=int, default=8)
    run_mineru_throughput.add_argument("--disable-image-analysis", action="store_true")
    run_mineru_throughput.add_argument("--keep-paratext", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "render":
        sources = _parse_source_args(args.source) if args.source else default_pdf_sources()
        cases = render_pdf_sources(sources, args.output_dir, scale=args.scale)
        print(json.dumps({"num_cases": len(cases)}, indent=2, ensure_ascii=False))
    elif args.command == "run-dllm":
        rows = run_dllm_suite(
            load_manifest(args.manifest),
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            layout_max_tokens=args.layout_max_tokens,
            content_max_tokens=args.content_max_tokens,
            table_max_tokens=args.table_max_tokens,
            formula_max_tokens=args.formula_max_tokens,
            block_size=args.block_size,
            dynamic_threshold=args.dynamic_threshold,
            content_concurrency=args.content_concurrency,
            max_denoising_steps=args.max_denoising_steps,
            keep_paratext=args.keep_paratext,
        )
        results_path, summary_path = write_run_outputs(args.output_dir, rows)
        print(json.dumps({"results": str(results_path), "summary": str(summary_path)}, indent=2))
    elif args.command == "run-dllm-throughput":
        rows = run_dllm_throughput_suite(
            load_manifest(args.manifest),
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            layout_max_tokens=args.layout_max_tokens,
            content_max_tokens=args.content_max_tokens,
            table_max_tokens=args.table_max_tokens,
            formula_max_tokens=args.formula_max_tokens,
            block_size=args.block_size,
            dynamic_threshold=args.dynamic_threshold,
            layout_concurrency=args.layout_concurrency,
            content_concurrency=args.content_concurrency,
            max_denoising_steps=args.max_denoising_steps,
            keep_paratext=args.keep_paratext,
        )
        results_path, summary_path = write_run_outputs(args.output_dir, rows)
        print(json.dumps({"results": str(results_path), "summary": str(summary_path)}, indent=2))
    elif args.command == "run-mineru-client":
        rows = run_mineru_client_suite(
            load_manifest(args.manifest),
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
            image_analysis=not args.disable_image_analysis,
            keep_paratext=args.keep_paratext,
        )
        results_path, summary_path = write_run_outputs(args.output_dir, rows)
        print(json.dumps({"results": str(results_path), "summary": str(summary_path)}, indent=2))
    elif args.command == "run-mineru-client-throughput":
        rows = run_mineru_client_throughput_suite(
            load_manifest(args.manifest),
            output_dir=args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
            image_analysis=not args.disable_image_analysis,
            keep_paratext=args.keep_paratext,
        )
        results_path, summary_path = write_run_outputs(args.output_dir, rows)
        print(json.dumps({"results": str(results_path), "summary": str(summary_path)}, indent=2))
    elif args.command == "compare":
        payload = compare_suite_rows(
            _read_jsonl(args.baseline),
            _read_jsonl(args.candidate),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
