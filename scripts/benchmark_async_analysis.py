# Copyright (c) Opendatalab. All rights reserved.
"""比较固定输入在同步线程桥接与原生异步解析下的吞吐和资源峰值。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import hashlib
import os
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


def percentile(values: list[float], fraction: float) -> float:
    """使用线性插值计算分位数，空样本返回零。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def gpu_memory_mib(pids: set[int]) -> float | None:
    """通过 NVIDIA 工具统计当前解析进程及其子进程显存，不把远端显存当成本地数据。"""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        output = subprocess.check_output(
            [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return sum(float(memory) for line in output.splitlines() for pid, memory in [line.split(",")] if int(pid) in pids)
    except (ValueError, subprocess.SubprocessError):
        return None


def save_result(result: Any, path: Path, directory: Path) -> None:
    """保存可复查的中间协议、Markdown 和内嵌图像 HTML，不计入性能时间。"""
    from mineru.parser.writer import FileBasedDataWriter
    from mineru.render import render_html

    destination = directory / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    result.save(FileBasedDataWriter(str(destination)))
    (destination / "document.html").write_text(render_html(result.middle_json), encoding="utf-8")


async def measure(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    """以固定并发解析同一组文件，同时采集延迟、内存、线程和事件循环响应。"""
    from mineru.config import config
    from mineru.parser import parse, parse_async

    settings = config.model.vlm.model_copy(deep=True)
    if args.server_url:
        settings.server_url = args.server_url
    if args.engine:
        settings.engine = args.engine
    settings.max_concurrency = args.request_limit
    settings = type(settings).model_validate(settings.model_dump())
    if not args.llm:
        config.llm_aided.features.title_leveling = False
        config.llm_aided.features.cross_page_table_cell_merge = False
    gate = asyncio.Semaphore(concurrency)
    process = psutil.Process()
    lag_ms: list[float] = []
    peak = {"rss_bytes": 0, "process_tree_rss_bytes": 0, "threads": 0, "gpu_memory_mib": None}
    done = asyncio.Event()

    async def invoke(path: Path) -> Any:
        """两种模式使用相同的解析参数，配置中的鉴权信息不写入报告。"""
        options = {"tier": args.tier, "ocr_mode": args.ocr_mode, "vlm_config": settings, "page_range": args.page_range}
        if args.mode == "sync-thread":
            return await asyncio.to_thread(parse, path, **options)
        return await parse_async(path, **options)

    async def heartbeat() -> None:
        """测量调用方事件循环调度延迟，并采样当前进程与子进程资源。"""
        while not done.is_set():
            expected = time.perf_counter() + 0.02
            await asyncio.sleep(0.02)
            lag_ms.append(max(0.0, time.perf_counter() - expected) * 1000)
            peak["rss_bytes"] = max(peak["rss_bytes"], process.memory_info().rss)
            peak["threads"] = max(peak["threads"], process.num_threads())
            children = process.children(recursive=True)
            tree_rss = process.memory_info().rss
            for child in children:
                try:
                    tree_rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak["process_tree_rss_bytes"] = max(peak["process_tree_rss_bytes"], tree_rss)

    async def sample_gpu() -> None:
        """独立采样 NVIDIA 显存，工具等待不阻塞解析事件循环。"""
        if shutil.which("nvidia-smi") is None:
            return
        while not done.is_set():
            pids = {process.pid, *(child.pid for child in process.children(recursive=True))}
            value = await asyncio.to_thread(gpu_memory_mib, pids)
            if value is not None:
                peak["gpu_memory_mib"] = max(peak["gpu_memory_mib"] or 0, value)
            try:
                await asyncio.wait_for(done.wait(), 1)
            except asyncio.TimeoutError:
                pass

    async def one(path: Path) -> dict[str, Any]:
        """分别记录排队延迟和实际解析耗时，失败也计入完整报告。"""
        submitted = time.perf_counter()
        async with gate:
            started = time.perf_counter()
            try:
                result = await invoke(path)
                return {
                    "file": str(path),
                    "pages": len(result.pages),
                    "parse_ms": (time.perf_counter() - started) * 1000,
                    "latency_ms": (time.perf_counter() - submitted) * 1000,
                    "error": None,
                }
            except Exception as exc:
                return {
                    "file": str(path),
                    "pages": 0,
                    "parse_ms": (time.perf_counter() - started) * 1000,
                    "latency_ms": (time.perf_counter() - submitted) * 1000,
                    "error": type(exc).__name__,
                }

    for _ in range(args.warmup):
        for path in args.files:
            result = await invoke(path)
            if args.save_results:
                await asyncio.to_thread(save_result, result, path, args.save_results)
    samplers = [asyncio.create_task(heartbeat()), asyncio.create_task(sample_gpu())]
    started = time.perf_counter()
    try:
        documents = await asyncio.gather(*(one(path) for _ in range(args.repeats) for path in args.files))
        elapsed = time.perf_counter() - started
    finally:
        done.set()
        await asyncio.gather(*samplers)
    successful = [item for item in documents if item["error"] is None]
    report = {
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "documents": documents,
        "successful": len(successful),
        "documents_per_second": len(successful) / elapsed,
        "pages_per_second": sum(item["pages"] for item in successful) / elapsed,
        "parse_p50_ms": percentile([item["parse_ms"] for item in successful], 0.5),
        "parse_p95_ms": percentile([item["parse_ms"] for item in successful], 0.95),
        "end_to_end_p95_ms": percentile([item["latency_ms"] for item in successful], 0.95),
        "loop_lag_p95_ms": percentile(lag_ms, 0.95),
        "loop_lag_max_ms": max(lag_ms, default=0),
        **peak,
    }
    if args.cancel_after is not None and args.mode == "async":
        operation = asyncio.create_task(invoke(args.files[0]))
        await asyncio.sleep(args.cancel_after)
        if operation.done():
            operation.result()
            report["cancel"] = {"status": "completed_before_cancel"}
        else:
            requested = time.perf_counter()
            operation.cancel()
            try:
                await operation
            except asyncio.CancelledError:
                report["cancel"] = {
                    "status": "canceled",
                    "return_ms": (time.perf_counter() - requested) * 1000,
                    "remote_abort_verified": False,
                }
    return report


def main() -> None:
    """执行可在当前源码与基线源码中复用的性能矩阵，并保存原始数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-results", type=Path)
    parser.add_argument("--mode", choices=["sync-thread", "async"], default="async")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tier", choices=["flash", "basic", "standard", "advanced"], default="standard")
    parser.add_argument("--ocr-mode", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--page-range", default="")
    parser.add_argument("--server-url", default="")
    parser.add_argument("--engine", choices=["auto", "vllm", "lmdeploy", "mlx", "llama-cpp"])
    parser.add_argument("--request-limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cancel-after", type=float)
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    if min(*args.concurrency, args.repeats, args.request_limit) < 1 or args.warmup < 0:
        parser.error("Concurrency, repeats and request limit must be positive; warmup cannot be negative")
    args.files = [path.resolve(strict=True) for path in args.files]
    import mineru
    import mineru_vl_utils
    from mineru.config import config

    report = {
        "source": str(Path(mineru.__file__).resolve()),
        "platform": platform.platform(),
        "mode": args.mode,
        "tier": args.tier,
        "ocr_mode": args.ocr_mode,
        "request_limit": args.request_limit,
        "remote_vlm": bool(args.server_url or config.model.vlm.server_url),
        "vl_utils_source_version": mineru_vl_utils.__version__,
        "vl_utils_source": str(Path(mineru_vl_utils.__file__).resolve()),
        "small_backend": config.model.small_backend,
        "device_mode": os.getenv("MINERU_DEVICE_MODE", "auto"),
        "onnx_intra_threads": os.getenv("MINERU_INTRA_OP_NUM_THREADS"),
        "onnx_inter_threads": os.getenv("MINERU_INTER_OP_NUM_THREADS"),
        "mineru_vl_utils": importlib.metadata.version("mineru-vl-utils"),
        "window_size": os.getenv("MINERU_PROCESSING_WINDOW_SIZE", "64"),
        "llm_enabled": args.llm,
        "files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.files],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for concurrency in args.concurrency:
            report["results"].append(asyncio.run(measure(args, concurrency)))
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        from mineru.model.vlm.runtime import shutdown_cached_models

        shutdown_cached_models()


if __name__ == "__main__":
    main()
