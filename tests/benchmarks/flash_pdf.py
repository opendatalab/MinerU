# Copyright (c) Opendatalab. All rights reserved.
# ruff: noqa: E402
"""生成 Flash PDF 完整输出基线，并在独立进程中测量耗时和峰值内存。"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unittest"))

from _flash_pdf_test_utils import _page_bbox_fingerprint, _page_fingerprint

from mineru.backend.postprocess.document import model_json_to_middle_json
from mineru.config import LLMAidedConfig
from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.pipeline import _analyze_native_document
from mineru.types import ModelJson


def _read_pdf(path: Path) -> bytes:
    """读取原始语料，在内存中解码版本化 XOR 测试文件。"""
    payload = path.read_bytes()
    if path.suffix == ".xor":
        key = b"MinerU flash layout fixture"
        return bytes(value ^ key[index % len(key)] for index, value in enumerate(payload))
    return payload


def _digest(value: Any) -> str:
    """对完整 JSON 值计算稳定摘要，不忽略几何、空格或任何字段。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """写入可审查 JSON 产物，父目录由本次运行独立创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _predict(payload: bytes) -> list[list[dict[str, Any]]]:
    """在单次文档生命周期内执行原生分析，确保计时包含页面打开和关闭。"""
    with PDFDocument(payload) as document:
        return _analyze_native_document(document)


def _worker(path: Path, destination: Path, runs: int, profile: bool) -> None:
    """隔离一份文档的计时、完整输出和进程峰值内存，避免其它文档污染 RSS。"""
    from loguru import logger

    logger.disable("mineru")
    payload = _read_pdf(path)
    if runs:
        _predict(payload)
    durations = []
    expected_digest = None
    for _ in range(max(1, runs)):
        gc.collect()
        started = time.perf_counter()
        pages = _predict(payload)
        durations.append(time.perf_counter() - started)
        digest = _digest(pages)
        if expected_digest is not None and digest != expected_digest:
            raise AssertionError(f"Non-deterministic model-list: {path}")
        expected_digest = digest
        del pages
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pages = _predict(payload)
    middle = model_json_to_middle_json(
        ModelJson(
            pages=deepcopy(pages),
            page_index_map=[],
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="refactor-baseline",
        ),
        llm_aided_config=LLMAidedConfig(),
    ).model_dump(mode="json")
    output = {"model_list": pages, "middle_json": middle}
    _write_json(destination / "output.json", output)
    result = {
        "path": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pages": len(pages),
        "full_output_sha256": _digest(output),
        "page_fingerprints": [_page_fingerprint(page) for page in pages],
        "bbox_fingerprints": [_page_bbox_fingerprint(page) for page in pages],
        "seconds": durations,
        "median_seconds": statistics.median(durations),
        "peak_rss_bytes": peak_rss * (1024 if sys.platform != "darwin" else 1),
    }
    del pages, middle, output
    if profile:
        profiler = cProfile.Profile()
        profiler.runcall(_predict, payload)
        stats = pstats.Stats(profiler)
        result["profile"] = [
            {
                "file": str(Path(filename).relative_to(ROOT)),
                "line": line,
                "function": name,
                "calls": values[1],
                "self_seconds": values[2],
                "cumulative_seconds": values[3],
            }
            for (filename, line, name), values in stats.stats.items()
            if "/mineru/model/flash/pdf/" in filename
        ]
    _write_json(destination / "result.json", result)


def _compare(output: Path, baseline: Path, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """核对完整语料集合、源文件指纹及输出，并单独报告性能变化。"""
    previous = json.loads((baseline / "report.json").read_text())
    by_path = {record["path"]: record for record in previous["documents"]}
    if set(by_path) != {record["path"] for record in results}:
        raise AssertionError("Baseline and candidate corpus differ")
    comparisons = []
    for record in results:
        old = by_path[record["path"]]
        if old["source_sha256"] != record["source_sha256"]:
            raise AssertionError(f"Input changed: {record['path']}")
        comparisons.append(
            {
                "path": record["path"],
                "equal": old["full_output_sha256"] == record["full_output_sha256"],
                "time_ratio": record["median_seconds"] / old["median_seconds"],
                "rss_ratio": record["peak_rss_bytes"] / old["peak_rss_bytes"],
            }
        )
    _write_json(output / "comparison.json", comparisons)
    return comparisons


def main() -> None:
    """执行完整回归或指定样本基准；每份文档单独启动子进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--runs", type=int, default=5, help="预热一次后计时次数；0 只运行功能校验")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--profile", action="store_true", help="额外运行剖析；不计入耗时或 RSS 指标")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 0:
        parser.error("--runs must be non-negative")
    if args.worker:
        _worker(args.worker, args.output, args.runs, args.profile)
        return
    if (args.output / "report.json").exists():
        parser.error("output already contains a report; choose a new directory")
    manifest = json.loads((ROOT / "tests/fixtures/flash_layout_geometry_manifest.json").read_text())
    paths = args.path or [
        *(item["path"] for item in manifest["documents"]),
        *(str(path.relative_to(ROOT)) for path in sorted((ROOT / "tests/unittest/pdfs/native_pdf_tables").glob("*.pdf"))),
    ]
    results = []
    for index, relative in enumerate(dict.fromkeys(paths)):
        destination = args.output.resolve() / f"{index:02d}-{Path(relative).stem}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(ROOT / relative),
            "--output",
            str(destination),
            "--runs",
            str(args.runs),
        ]
        if args.profile:
            command.append("--profile")
        subprocess.run(command, cwd=ROOT, check=True)
        record = json.loads((destination / "result.json").read_text())
        record["artifact"] = str(destination.relative_to(args.output.resolve()))
        results.append(record)
        print(f"{index + 1}/{len(paths)} {relative}: {record['median_seconds']:.3f}s", flush=True)
    historical = {item["path"]: item for item in manifest["documents"]}
    history_differences = []
    for record in results:
        old = historical.get(record["path"])
        if old is not None:
            expected = [page["fingerprint"] for page in old["pages"]]
            expected_bbox = [page["bbox_fingerprint"] for page in old["pages"]]
            if expected != record["page_fingerprints"] or expected_bbox != record["bbox_fingerprints"]:
                history_differences.append(record["path"])
    report = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("pdftext", "pypdfium2", "numpy", "pydantic")},
        "runs": args.runs,
        "historical_baseline_sha": manifest["baseline_git_sha"],
        "historical_differences": history_differences,
        "documents": results,
    }
    _write_json(args.output / "report.json", report)
    if args.baseline and any(not item["equal"] for item in _compare(args.output, args.baseline, results)):
        raise SystemExit("Full output differs; inspect output.json artifacts and comparison.json")


if __name__ == "__main__":
    main()
