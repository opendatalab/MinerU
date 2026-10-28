"""在相同 Linux 主机和固定 PDF 上对照 Torch 线程、时间、RSS 与解析输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil


def parse_once(path: Path, tier: str) -> dict[str, str]:
    """通过公开解析接口执行一次 CPU Torch 推理，比较两类确定性输出的哈希。"""
    from mineru.parser import parse

    result = parse(str(path), tier=tier, image_analysis=False)
    structured = json.dumps(result.structured_content(), ensure_ascii=False, sort_keys=True).encode()
    return {
        "markdown_sha256": hashlib.sha256(result.markdown().encode()).hexdigest(),
        "structured_sha256": hashlib.sha256(structured).hexdigest(),
    }


def measure_round(path: Path, tier: str, concurrency: int) -> dict[str, Any]:
    """每轮新建工作线程，同时采样整个进程的线程数和 RSS，覆盖后续线程初始化。"""
    process = psutil.Process()
    stopped = threading.Event()
    samples: list[tuple[int, int]] = []

    def sample() -> None:
        """采集高水位，不改变模型或推理配置。"""
        while not stopped.is_set():
            samples.append((process.num_threads(), process.memory_info().rss))
            stopped.wait(0.02)

    before = process.num_threads()
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(parse_once, path, tier) for _ in range(concurrency)]
            outputs = [future.result() for future in futures]
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        monitor.join()
    return {
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "threads_before": before,
        "threads_after": process.num_threads(),
        "peak_threads": max(value[0] for value in samples),
        "peak_rss_bytes": max(value[1] for value in samples),
        "outputs": outputs,
    }


def main() -> None:
    """每个基线或候选进程预热一次，再交替测量至少六轮 1/3 并发。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tier", choices=("basic",), default="basic")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--require-linux", action="store_true")
    args = parser.parse_args()
    if args.rounds < 6:
        parser.error("At least six rounds are required")
    if args.require_linux and platform.system() != "Linux":
        parser.error("Linux acceptance requires a Linux host")
    if not args.pdf.is_file():
        parser.error("PDF does not exist")
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    from mineru.config import get_config
    import mineru
    import docvortex
    from mineru import version
    import torch

    get_config().model.small_backend = "torch"
    warmup = parse_once(args.pdf, args.tier)
    rounds = [measure_round(args.pdf, args.tier, concurrency) for _ in range(args.rounds) for concurrency in (1, 3)]
    report = {
        "label": args.label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mineru_version": version.__version__,
        "mineru_source": str(Path(mineru.__file__).resolve()),
        "docvortex_source": str(Path(docvortex.__file__).resolve()),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "torch_intra_op_threads": torch.get_num_threads(),
        "thread_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MINERU_CPU_NUM_THREADS", "MINERU_INTRA_OP_NUM_THREADS")
        },
        "warmup": warmup,
        "rounds": rounds,
        "outputs_consistent": all(output == warmup for value in rounds for output in value["outputs"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
