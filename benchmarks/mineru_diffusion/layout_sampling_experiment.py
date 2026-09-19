from __future__ import annotations

import argparse
import base64
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import requests
from PIL import Image

from benchmarks.mineru_diffusion.compare_results import (
    compare_layout_blocks,
    parse_layout_blocks,
)
from benchmarks.mineru_diffusion.end2end_openai import (
    SYSTEM_PROMPT,
    TASK_PROMPTS,
    parse_layout_output,
    prepare_layout_image,
)
from benchmarks.mineru_diffusion.end2end_suite import (
    PageCase,
    blocks_to_layout_text,
    load_manifest,
    make_result_row,
    write_run_outputs,
)
from benchmarks.mineru_diffusion.harness import STOP_STRINGS


@dataclass(frozen=True)
class LayoutSamplingVariant:
    name: str
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


DEFAULT_VARIANT = LayoutSamplingVariant(
    name="dllm_default_temp1",
    temperature=1.0,
)

MINERU_LAYOUT_VARIANT = LayoutSamplingVariant(
    name="dllm_mineru_layout_sampling",
    temperature=0.0,
    top_p=0.01,
    top_k=1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=100,
)


class LayoutClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout: float,
        block_size: int,
        dynamic_threshold: float,
        max_denoising_steps: int | None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.block_size = block_size
        self.dynamic_threshold = dynamic_threshold
        self.max_denoising_steps = max_denoising_steps
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._thread_local.session = session
        return session

    def infer_layout(self, image_url: str, variant: LayoutSamplingVariant) -> str:
        vllm_xargs: dict[str, Any] = {
            "block_size": self.block_size,
            "dynamic_threshold": self.dynamic_threshold,
        }
        if self.max_denoising_steps is not None:
            vllm_xargs["max_denoising_steps"] = self.max_denoising_steps
        if variant.no_repeat_ngram_size is not None:
            vllm_xargs["no_repeat_ngram_size"] = variant.no_repeat_ngram_size

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": TASK_PROMPTS["[layout]"]},
                    ],
                },
            ],
            "max_tokens": 2048,
            "temperature": variant.temperature,
            "stop": list(STOP_STRINGS),
            "block_size": self.block_size,
            "dynamic_threshold": self.dynamic_threshold,
            "vllm_xargs": vllm_xargs,
            "skip_special_tokens": False,
        }
        for key in (
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        ):
            value = getattr(variant, key)
            if value is not None:
                payload[key] = value
        if self.max_denoising_steps is not None:
            payload["max_denoising_steps"] = self.max_denoising_steps

        response = self._session().post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        for stop in STOP_STRINGS:
            text = text.split(stop, 1)[0]
        return text.strip()


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as image:
        layout_image = prepare_layout_image(image.convert("RGB"))
    with BytesIO() as buffer:
        layout_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_case_inputs(cases: Sequence[PageCase]) -> dict[str, str]:
    return {case.case_id: image_to_data_url(case.image_path) for case in cases}


def run_variant(
    cases: Sequence[PageCase],
    *,
    image_urls: dict[str, str],
    client: LayoutClient,
    variant: LayoutSamplingVariant,
    output_dir: Path,
    layout_concurrency: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def run_one(case: PageCase) -> dict[str, Any]:
        layout_start = time.perf_counter()
        layout_output = client.infer_layout(image_urls[case.case_id], variant)
        layout_elapsed = time.perf_counter() - layout_start
        blocks = parse_layout_output(layout_output)
        layout_text = blocks_to_layout_text(blocks)
        metrics = {
            "layout_elapsed": layout_elapsed,
            "total_elapsed": layout_elapsed,
            "num_blocks": len(blocks),
            "num_extracted_blocks": 0,
            "markdown_chars": 0,
            "layout_chars": len(layout_text),
            "layout_raw_chars": len(layout_output),
            "layout_concurrency": layout_concurrency,
            "content_concurrency": 0,
            "sampling_variant": variant.name,
            "temperature": variant.temperature,
            "top_p": variant.top_p,
            "top_k": variant.top_k,
            "presence_penalty": variant.presence_penalty,
            "frequency_penalty": variant.frequency_penalty,
            "repetition_penalty": variant.repetition_penalty,
            "no_repeat_ngram_size": variant.no_repeat_ngram_size,
        }
        return make_result_row(
            case,
            ok=True,
            metrics=metrics,
            blocks=blocks,
            markdown="",
            layout_output=layout_text,
        )

    rows_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=layout_concurrency) as executor:
        futures = {executor.submit(run_one, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows_by_id[case.case_id] = future.result()
            except Exception as exc:
                rows_by_id[case.case_id] = make_result_row(
                    case,
                    ok=False,
                    error=repr(exc),
                    metrics={
                        "sampling_variant": variant.name,
                        "layout_concurrency": layout_concurrency,
                    },
                )

    wall_elapsed = time.perf_counter() - started
    rows = [rows_by_id[case.case_id] for case in cases]
    for row in rows:
        if row.get("ok"):
            row["metrics"]["throughput_batch_size"] = len(cases)
            row["metrics"]["throughput_wall_elapsed_s"] = wall_elapsed
            row["metrics"]["throughput_layout_wall_elapsed_s"] = wall_elapsed
            row["metrics"]["throughput_extract_wall_elapsed_s"] = 0.0

    write_run_outputs(output_dir, rows)
    return rows


def summarize_layout_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    latencies = [float(row["metrics"]["layout_elapsed"]) for row in ok_rows]
    total_latency = sum(latencies)
    layout_chars = sum(int(row["metrics"].get("layout_chars", 0)) for row in ok_rows)
    raw_chars = sum(int(row["metrics"].get("layout_raw_chars", 0)) for row in ok_rows)
    blocks = sum(int(row["metrics"].get("num_blocks", 0)) for row in ok_rows)
    wall_values = {
        row["metrics"].get("throughput_wall_elapsed_s")
        for row in ok_rows
        if row["metrics"].get("throughput_wall_elapsed_s") is not None
    }
    wall_elapsed = wall_values.pop() if len(wall_values) == 1 else None
    return {
        "num_cases": len(rows),
        "num_ok": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "layout_total_latency_s": total_latency,
        "layout_mean_latency_s": statistics.mean(latencies) if latencies else None,
        "layout_p50_latency_s": statistics.median(latencies) if latencies else None,
        "layout_max_latency_s": max(latencies) if latencies else None,
        "throughput_wall_elapsed_s": wall_elapsed,
        "layout_chars": layout_chars,
        "layout_raw_chars": raw_chars,
        "num_blocks": blocks,
        "layout_chars_per_layout_s": (
            layout_chars / total_latency if total_latency > 0 else None
        ),
    }


def compare_layout_rows(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["case_id"]: row for row in baseline_rows if row.get("ok")}
    candidate_by_id = {row["case_id"]: row for row in candidate_rows if row.get("ok")}
    cases = []
    for case_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        baseline = baseline_by_id[case_id]
        candidate = candidate_by_id[case_id]
        baseline_blocks = parse_layout_blocks(baseline.get("layout_output", ""))
        candidate_blocks = parse_layout_blocks(candidate.get("layout_output", ""))
        matches, precision, recall, f1 = compare_layout_blocks(
            baseline_blocks,
            candidate_blocks,
        )
        baseline_latency = float(baseline["metrics"]["layout_elapsed"])
        candidate_latency = float(candidate["metrics"]["layout_elapsed"])
        cases.append(
            {
                "case_id": case_id,
                "baseline_latency_s": baseline_latency,
                "candidate_latency_s": candidate_latency,
                "speedup": (
                    baseline_latency / candidate_latency
                    if candidate_latency > 0
                    else None
                ),
                "baseline_blocks": len(baseline_blocks),
                "candidate_blocks": len(candidate_blocks),
                "layout_matched_blocks": matches,
                "layout_precision": precision,
                "layout_recall": recall,
                "layout_f1": f1,
            }
        )
    baseline_total = sum(item["baseline_latency_s"] for item in cases)
    candidate_total = sum(item["candidate_latency_s"] for item in cases)
    layout_f1s = [item["layout_f1"] for item in cases]
    return {
        "summary": {
            "num_matched_ok": len(cases),
            "baseline_total_layout_s": baseline_total,
            "candidate_total_layout_s": candidate_total,
            "total_speedup": (
                baseline_total / candidate_total if candidate_total > 0 else None
            ),
            "mean_layout_f1": statistics.mean(layout_f1s) if layout_f1s else None,
            "min_layout_f1": min(layout_f1s) if layout_f1s else None,
        },
        "cases": cases,
    }


def write_experiment_report(
    output_dir: Path,
    *,
    default_rows: Sequence[dict[str, Any]],
    mineru_rows: Sequence[dict[str, Any]],
    pro_rows: Sequence[dict[str, Any]] | None,
) -> None:
    default_summary = summarize_layout_rows(default_rows)
    mineru_summary = summarize_layout_rows(mineru_rows)
    default_vs_mineru = compare_layout_rows(default_rows, mineru_rows)

    payload: dict[str, Any] = {
        "default_variant": DEFAULT_VARIANT.__dict__,
        "mineru_layout_variant": MINERU_LAYOUT_VARIANT.__dict__,
        "default_summary": default_summary,
        "mineru_layout_summary": mineru_summary,
        "default_vs_mineru": default_vs_mineru,
    }
    if pro_rows is not None:
        pro_layout_rows = normalize_pro_layout_rows(pro_rows)
        payload["pro_layout_summary"] = summarize_layout_rows(pro_layout_rows)
        payload["pro_vs_default"] = compare_layout_rows(pro_layout_rows, default_rows)
        payload["pro_vs_mineru_layout"] = compare_layout_rows(
            pro_layout_rows,
            mineru_rows,
        )

    (output_dir / "layout_sampling_experiment_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# DLLM Layout Sampling Experiment",
        "",
        "Scope: layout-only requests on the 23-page PDF coverage suite.",
        "",
        "| Variant | OK | Layout total s | Wall s | Mean s | Layout chars | Blocks | Chars/layout-s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _summary_row("DLLM default temp=1", default_summary),
        _summary_row("DLLM MinerU layout sampling", mineru_summary),
    ]
    if pro_rows is not None:
        lines.append(_summary_row("MinerU2.5-Pro existing layout", payload["pro_layout_summary"]))
    comparison = default_vs_mineru["summary"]
    lines.extend(
        [
            "",
            "## Default vs MinerU Layout Sampling",
            "",
            f"- Total layout speedup: {comparison['total_speedup']:.4f}x "
            "(default / MinerU-sampling).",
            f"- Mean layout F1 between variants: {comparison['mean_layout_f1']:.4f}.",
        ]
    )
    if pro_rows is not None:
        pro_default = payload["pro_vs_default"]["summary"]
        pro_mineru = payload["pro_vs_mineru_layout"]["summary"]
        lines.extend(
            [
                "",
                "## Existing Pro Layout Reference",
                "",
                f"- Pro / DLLM default layout speedup: {pro_default['total_speedup']:.4f}x.",
                f"- Pro / DLLM MinerU-sampling layout speedup: {pro_mineru['total_speedup']:.4f}x.",
                f"- Pro vs DLLM MinerU-sampling mean layout F1: {pro_mineru['mean_layout_f1']:.4f}.",
            ]
        )
    lines.append("")
    (output_dir / "layout_sampling_experiment_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _summary_row(name: str, summary: dict[str, Any]) -> str:
    wall = summary.get("throughput_wall_elapsed_s")
    wall_text = f"{wall:.3f}" if wall is not None else "n/a"
    return (
        f"| {name} | {summary['num_ok']}/{summary['num_cases']} "
        f"| {summary['layout_total_latency_s']:.3f} "
        f"| {wall_text} "
        f"| {summary['layout_mean_latency_s']:.3f} "
        f"| {summary['layout_chars']} "
        f"| {summary['num_blocks']} "
        f"| {summary['layout_chars_per_layout_s']:.1f} |"
    )


def normalize_pro_layout_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        metrics = copied.setdefault("metrics", {})
        layout_elapsed = float(metrics.get("layout_elapsed", 0.0))
        metrics["total_elapsed"] = layout_elapsed
        metrics["layout_chars"] = len(copied.get("layout_output", ""))
        metrics["layout_raw_chars"] = metrics["layout_chars"]
        normalized.append(copied)
    return normalized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18084/v1/chat/completions",
    )
    parser.add_argument("--model", default="mineru-diffusion")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--dynamic-threshold", type=float, default=0.90)
    parser.add_argument("--max-denoising-steps", type=int)
    parser.add_argument("--layout-concurrency", type=int, default=4)
    parser.add_argument("--pro-results-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_manifest(args.manifest)
    image_urls = prepare_case_inputs(cases)
    client = LayoutClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        block_size=args.block_size,
        dynamic_threshold=args.dynamic_threshold,
        max_denoising_steps=args.max_denoising_steps,
    )
    default_rows = run_variant(
        cases,
        image_urls=image_urls,
        client=client,
        variant=DEFAULT_VARIANT,
        output_dir=args.output_dir / DEFAULT_VARIANT.name,
        layout_concurrency=args.layout_concurrency,
    )
    mineru_rows = run_variant(
        cases,
        image_urls=image_urls,
        client=client,
        variant=MINERU_LAYOUT_VARIANT,
        output_dir=args.output_dir / MINERU_LAYOUT_VARIANT.name,
        layout_concurrency=args.layout_concurrency,
    )
    pro_rows = read_jsonl(args.pro_results_jsonl) if args.pro_results_jsonl else None
    write_experiment_report(
        args.output_dir,
        default_rows=default_rows,
        mineru_rows=mineru_rows,
        pro_rows=pro_rows,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "default": summarize_layout_rows(default_rows),
                "mineru_layout": summarize_layout_rows(mineru_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
