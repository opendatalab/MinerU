from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from benchmarks.mineru_diffusion.harness import (
    BenchmarkResult,
    STOP_STRINGS,
    read_cases,
    summarize_results,
    write_default_cases,
)


def _extract_output(payload: dict[str, Any]) -> str:
    return payload["choices"][0]["message"]["content"]


def create_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def build_request_payload(case: dict[str, Any]) -> dict[str, Any]:
    block_size = case.get("block_size", 32)
    dynamic_threshold = case.get("dynamic_threshold", 0.95)
    vllm_xargs = {
        "block_size": block_size,
        "dynamic_threshold": dynamic_threshold,
    }
    payload = {
        "model": case.get("model", "mineru-diffusion"),
        "messages": case["messages"],
        "max_tokens": case.get("max_tokens", 1024),
        "temperature": case.get("temperature", 1.0),
        "stop": case.get("stop", list(STOP_STRINGS)),
        "block_size": block_size,
        "dynamic_threshold": dynamic_threshold,
        "vllm_xargs": vllm_xargs,
    }
    if "max_denoising_steps" in case:
        max_denoising_steps = int(case["max_denoising_steps"])
        payload["max_denoising_steps"] = max_denoising_steps
        vllm_xargs["max_denoising_steps"] = max_denoising_steps
    return payload


def parse_task_max_tokens(value: str) -> dict[str, int]:
    if not value:
        return {}
    parsed: dict[str, int] = {}
    for part in value.split(","):
        if not part:
            continue
        task, sep, raw_tokens = part.partition("=")
        if not sep or not task:
            raise ValueError(
                "--task-max-tokens entries must use task=tokens format"
            )
        parsed[task] = int(raw_tokens)
    return parsed


def run_case(endpoint: str, case: dict[str, Any], timeout: float) -> BenchmarkResult:
    case_id = str(case.get("case_id", "unknown"))
    request_payload = build_request_payload(case)
    started = time.perf_counter()
    try:
        response = create_session().post(endpoint, json=request_payload, timeout=timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        return BenchmarkResult(
            case_id=case_id,
            ok=True,
            latency_s=latency,
            output_text=_extract_output(response.json()),
            error=None,
        )
    except Exception as exc:
        return BenchmarkResult(
            case_id=case_id,
            ok=False,
            latency_s=time.perf_counter() - started,
            output_text="",
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18082/v1/chat/completions")
    parser.add_argument("--cases-jsonl", type=Path, default=None)
    parser.add_argument("--write-default-cases", type=Path, default=None)
    parser.add_argument(
        "--image",
        default=None,
        help="image URL used when generating default text/table/formula/layout cases",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/mineru_diffusion"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--task-max-tokens",
        default="",
        help="comma-separated task=tokens overrides for generated default cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.cases_jsonl is None
        or args.write_default_cases is not None
    ) and args.image is None:
        raise SystemExit("--image is required when generating default cases")

    if args.write_default_cases is not None:
        write_default_cases(
            args.write_default_cases,
            args.image,
            task_max_tokens=parse_task_max_tokens(args.task_max_tokens),
        )
        print(f"wrote {args.write_default_cases}")
        return

    cases_path = args.cases_jsonl
    if cases_path is None:
        cases_path = args.output_dir / "default_cases.jsonl"
        write_default_cases(
            cases_path,
            args.image,
            task_max_tokens=parse_task_max_tokens(args.task_max_tokens),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_cases(cases_path)
    results = [run_case(args.endpoint, case, args.timeout) for case in cases]
    summary = summarize_results(results)

    results_path = args.output_dir / f"results_{int(time.time())}.jsonl"
    summary_path = args.output_dir / "latest_summary.json"
    results_path.write_text(
        "".join(result.to_json() + "\n" for result in results),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
