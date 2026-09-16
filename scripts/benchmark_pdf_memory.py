"""在独立解析进程外采样内存，比较不同 checkout 与堆回收开关的效果。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    """读取诊断参数，默认预热三轮并测量二十轮，输入文件按顺序循环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pdf", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("flash", "basic", "standard", "advanced"), default="flash")
    parser.add_argument("--mode", choices=("txt", "ocr", "auto"), default="txt")
    parser.add_argument("--trim", choices=("0", "1"), default="0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument(
        "--fingerprints", action="store_true", help="Compute full output hashes; use separately from memory runs"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.rounds < 1 or args.interval <= 0 or args.window_size < 1:
        parser.error("warmup >= 0; rounds, interval and window-size must be positive")
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    args.pdf = [path.resolve() for path in args.pdf]
    if not (args.repo / "mineru" / "parser").is_dir() or not all(path.is_file() for path in args.pdf):
        parser.error("repo must contain mineru/parser and every PDF must exist")
    return args


def _fingerprint(document: Any) -> str:
    """包含完整素材计算共享协议摘要，不使用会省略 PDF 图片的产品封装。"""
    value = document.to_dict(skip_defaults=False)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _worker(args: argparse.Namespace) -> None:
    """在指定 checkout 解析真实文件，每轮释放结果且不额外触发垃圾回收。"""
    sys.path.insert(0, str(args.repo))
    from mineru.parser.mineru_parser import MinerUParser
    from mineru.backend.analysis.pdf.images import shutdown_pdf_render_executor

    parser = MinerUParser(tier=args.tier, parse_mode=args.mode)
    try:
        with (args.output / "rounds.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(args.warmup + args.rounds):
                path = args.pdf[index % len(args.pdf)]
                started = time.monotonic()
                result = parser.parse(path)
                parsed = time.monotonic()
                record = {
                    "index": index,
                    "warmup": index < args.warmup,
                    "pdf": str(path),
                    "started": started,
                    "parse_seconds": parsed - started,
                    "pages": len(result.pages),
                }
                if args.fingerprints:
                    record["middle_sha256"] = _fingerprint(result.middle_json)
                    record["model_sha256"] = _fingerprint(result._model_output)
                del result
                record["released"] = time.monotonic()
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    finally:
        shutdown_pdf_render_executor()


def _sample_processes(root: Any) -> list[dict[str, Any]]:
    """分别采样解析进程与所有子进程，权限不足的 USS 保留为空而非零。"""
    import psutil

    records = []
    try:
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return records
    for process in processes:
        try:
            rss = process.memory_info().rss
            try:
                uss = getattr(process.memory_full_info(), "uss", None)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                uss = None
            records.append(
                {
                    "pid": process.pid,
                    "role": "parser" if process.pid == root.pid else "child",
                    "rss": rss,
                    "uss": uss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return records


def _summarize(args: argparse.Namespace, exit_code: int) -> dict[str, Any]:
    """汇总测量阶段耗时和各 PID 峰值，保留原始采样用于窗口及轮次对齐。"""
    rounds_path = args.output / "rounds.jsonl"
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()] if rounds_path.exists() else []
    measured = [record for record in rounds if not record["warmup"]]
    peaks: dict[int, dict[str, Any]] = {}
    total_rss_peak = 0
    total_uss_peak: int | None = None
    for line in (args.output / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if not measured or not (measured[0]["started"] <= sample["time"] <= measured[-1]["released"]):
            continue
        total_rss_peak = max(total_rss_peak, sum(item["rss"] for item in sample["processes"]))
        uss_values = [item["uss"] for item in sample["processes"]]
        if uss_values and all(value is not None for value in uss_values):
            total_uss_peak = max(total_uss_peak or 0, sum(uss_values))
        for item in sample["processes"]:
            peak = peaks.setdefault(item["pid"], {**item})
            peak["rss"] = max(peak["rss"], item["rss"])
            if item["uss"] is not None:
                peak["uss"] = max(peak["uss"] or 0, item["uss"])
    return {
        "exit_code": exit_code,
        "repo": str(args.repo),
        "python": sys.version,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "docvortex": importlib.metadata.version("docvortex"),
        "trim": args.trim,
        "tier": args.tier,
        "mode": args.mode,
        "window_size": args.window_size,
        "fingerprints": args.fingerprints,
        "measured_rounds": len(measured),
        "median_parse_seconds": statistics.median(record["parse_seconds"] for record in measured) if measured else None,
        "process_peaks": list(peaks.values()),
        "sampled_total_rss_peak": total_rss_peak if peaks else None,
        "sampled_total_uss_peak": total_uss_peak,
    }


def main() -> None:
    """启动独立解析进程，保存逐轮摘要、外部采样和解析日志。"""
    args = _arguments()
    if args.worker:
        _worker(args)
        return
    import psutil

    # 禁止覆盖上一组实验，避免把不同 checkout 或开关的证据混在一起。
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "MINERU_MALLOC_TRIM": args.trim,
        "MINERU_PROCESSING_WINDOW_SIZE": str(args.window_size),
    }
    # 子进程切换 cwd 后仍必须使用父进程已解析的绝对输入路径。
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo",
        str(args.repo),
        "--pdf",
        *(str(path) for path in args.pdf),
        "--output",
        str(args.output),
        "--tier",
        args.tier,
        "--mode",
        args.mode,
        "--trim",
        args.trim,
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--window-size",
        str(args.window_size),
    ]
    if args.fingerprints:
        command.append("--fingerprints")
    with (args.output / "parse.log").open("w") as log, (args.output / "samples.jsonl").open("w") as samples:
        process = subprocess.Popen(command, env=environment, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                samples.write(json.dumps({"time": time.monotonic(), "processes": _sample_processes(root)}) + "\n")
                samples.flush()
                time.sleep(args.interval)
            exit_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    summary = _summarize(args, exit_code)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
