# Copyright (c) Opendatalab. All rights reserved.
"""按原始像素面积排序，串行测量两个公式 ONNX 的真实 batch 性能。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from mineru.model.mfr.pp_formulanet.processors import (
    LatexImageFormat,
    UniMERNetDecode,
    UniMERNetImgDecode,
    UniMERNetTestTransform,
)

__all__ = ["prepare", "worker", "benchmark"]


def save(path: Path, data: Any) -> None:
    """持久化每个独立配置，避免长测量中断丢失结果。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def prepare(images_dir: Path, output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """按来源分两组，并以实际图像面积稳定排序；预处理独立于推理计时。"""
    manifest = json.loads((images_dir / "manifest.json").read_text())
    groups: dict[str, list[dict[str, Any]]] = {"samples36": [], "demo1_demo2": []}
    decoder, transform, formatter = UniMERNetImgDecode(input_size=(384, 384)), UniMERNetTestTransform(), LatexImageFormat()
    for item in manifest:
        path = images_dir / item["file"]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        height, width = pixels.shape[:2]
        started = time.perf_counter()
        array = formatter.format(transform.transform(decoder.img_decode(pixels)))
        elapsed = time.perf_counter() - started
        np.save(output_dir / f"{path.stem}.npy", array)
        group = "samples36" if item["source"] == "samples36" else "demo1_demo2"
        groups[group].append(
            {
                "path": str(path),
                "file": path.name,
                "width": width,
                "height": height,
                "area": width * height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensor": str(output_dir / f"{path.stem}.npy"),
                "preprocess_seconds": elapsed,
            }
        )
    for group in groups.values():
        group.sort(key=lambda item: (item["area"], item["width"], item["height"], item["file"]))
    if [len(groups[key]) for key in ("samples36", "demo1_demo2")] != [36, 117]:
        raise ValueError("Expected 36 original samples and 117 demo1/demo2 formulas")
    save(output_dir / "datasets.json", groups)
    return groups


def trim(ids: np.ndarray) -> list[int]:
    """在首个 EOS 截断，排除同批其他样本结束较晚带来的 PAD。"""
    values = ids.tolist()
    return values[: values.index(2) + 1] if 2 in values else values


def worker(args: argparse.Namespace) -> None:
    """单独进程创建一个 ORT 会话，预热后按指定线程数和轮数测量完整数据集。"""
    samples = json.loads((args.output_dir / "datasets.json").read_text())[args.dataset]
    arrays = [np.load(item["tensor"]) for item in samples]
    batches = [
        np.concatenate(arrays[start : start + args.batch_size], axis=0) for start in range(0, len(arrays), args.batch_size)
    ]
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    model = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    initialization = time.perf_counter() - started
    name = model.get_inputs()[0].name
    started = time.perf_counter()
    model.run(None, {name: batches[0]})
    warmup = time.perf_counter() - started
    rounds, last_tokens = [], []
    for iteration in range(args.rounds):
        outputs, latencies = [], []
        for batch_index, batch in enumerate(batches):
            started = time.perf_counter()
            result = model.run(None, {name: batch})[0]
            latencies.append(time.perf_counter() - started)
            outputs.extend(trim(row) for row in result)
            # 长测试按四分之一轮输出进度；日志处理在 session.run 计时区间外。
            if (batch_index + 1) % max(1, len(batches) // 4) == 0:
                print(
                    f"progress {args.dataset} {args.label} b={args.batch_size} "
                    f"round={iteration + 1} batches={batch_index + 1}/{len(batches)}",
                    flush=True,
                )
        rounds.append({"seconds": sum(latencies), "batch_seconds": latencies})
        if last_tokens and outputs != last_tokens:
            raise AssertionError("Repeated run token outputs differ")
        last_tokens = outputs
        print(f"{args.dataset} {args.label} batch={args.batch_size} round={iteration + 1}: {sum(latencies):.3f}s", flush=True)
    decoder = UniMERNetDecode(character_list=yaml.safe_load(args.config.read_text())["PostProcess"]["character_dict"])
    latex = [decoder(np.array([ids], dtype=np.int64))[0] for ids in last_tokens]
    mean = statistics.mean(row["seconds"] for row in rounds)
    result = {
        "dataset": args.dataset,
        "model": args.label,
        "batch_size": args.batch_size,
        "intra_op_threads": args.threads,
        "inter_op_threads": 1,
        "count": len(samples),
        "batch_lengths": [len(batch) for batch in batches],
        "initialization_seconds": initialization,
        "warmup_seconds": warmup,
        "rounds": rounds,
        "mean_seconds": mean,
        "images_per_second": len(samples) / mean,
        "milliseconds_per_image": mean * 1000 / len(samples),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "tokens": last_tokens,
        "latex": latex,
        "files": [item["file"] for item in samples],
        "providers": model.get_providers(),
        "heavy_modules": [key for key in ("torch", "transformers") if key in sys.modules],
    }
    save(args.output_dir / f"{args.dataset}-{args.label}-b{args.batch_size}.json", result)


def benchmark(args: argparse.Namespace) -> None:
    """串行启动配置，交替模型先后次序，避免两个测量进程互相争抢 CPU。"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare(args.images_dir, args.output_dir)
    metadata = {
        "platform": platform.platform(),
        "python": sys.version,
        "onnxruntime": ort.__version__,
        "intra_op_threads": args.threads,
        "inter_op_threads": 1,
        "rounds": args.rounds,
        "sort": "original width*height ascending, tie-break width/height/filename",
        "timing": "session.run only; excludes session creation, preprocessing, concatenation and text decoding",
        "models": {
            label: {"path": str(path.resolve()), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
            for label, path in (("reference", args.reference), ("candidate", args.candidate))
        },
    }
    save(args.output_dir / "environment.json", metadata)
    results = []
    for dataset_index, dataset in enumerate(("samples36", "demo1_demo2")):
        for batch_index, batch in enumerate(args.batch_sizes):
            models = [("reference", args.reference), ("candidate", args.candidate)]
            if (dataset_index + batch_index) % 2:
                models.reverse()
            for label, path in models:
                command = [
                    sys.executable,
                    "-m",
                    "scripts.benchmark_formula_onnx",
                    "--worker",
                    "--threads",
                    str(args.threads),
                    "--rounds",
                    str(args.rounds),
                    "--dataset",
                    dataset,
                    "--label",
                    label,
                    "--model",
                    str(path),
                    "--batch-size",
                    str(batch),
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(args.output_dir),
                ]
                subprocess.run(
                    command, check=True, env={**os.environ, "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1"}
                )
                results.append(json.loads((args.output_dir / f"{dataset}-{label}-b{batch}.json").read_text()))
                save(args.output_dir / "results.json", results)


def main() -> None:
    """解析总控或独立工作进程参数，不修改生产模型及其注册。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    for key in ("images-dir", "reference", "candidate", "config", "output-dir", "model"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--label")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.threads < 1 or args.rounds < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("线程数、轮数和 batch size 必须为正整数")
    worker(args) if args.worker else benchmark(args)


if __name__ == "__main__":
    main()
