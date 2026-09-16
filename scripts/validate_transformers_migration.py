#!/usr/bin/env python3
"""在指定 checkout 和真实模型上保存 Transformers 迁移的输出、耗时及内存证据。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    """解析只读输入、模型目录和独立结果目录，不改变用户配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path.home() / ".mineru/models")
    parser.add_argument("--kind", choices=("layout", "mfr", "parse", "vlm-transformers", "vlm-mlx"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tier", choices=("basic", "standard"), default="basic")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0:
        parser.error("rounds must be positive and warmup must be nonnegative")
    return args


def _synchronize(device: str) -> None:
    """等待设备完成计算，避免把异步提交时间误认为推理耗时。"""
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _physical_footprint_reader() -> Callable[[], int] | None:
    """读取 macOS 统一内存的物理占用，避免 MPS 外部存储改变 RSS/框架计数口径。"""
    if sys.platform != "darwin":
        return None
    import ctypes

    class UsageInfo(ctypes.Structure):
        """对应系统 sys/resource.h 中稳定的 rusage_info_v0 结构。"""

        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64)
            for name in (
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
            )
        ]

    function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    pid = os.getpid()

    def read() -> int:
        """只读取当前进程统计，系统调用失败时使本次测量明确失败。"""
        info = UsageInfo()
        if function(pid, 0, ctypes.byref(info)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return info.phys_footprint

    return read


def _measure(operation: Callable[[], Any], device: str, rounds: int, warmup: int) -> tuple[Any, dict[str, Any]]:
    """预热后测量三轮，并每 20ms 采样进程 RSS 与 Torch 设备占用。"""
    import psutil
    import torch

    for _ in range(warmup):
        operation()
    _synchronize(device)
    process = psutil.Process()
    peaks = {"rss_bytes": 0, "torch_device_bytes": 0}
    footprint_reader = _physical_footprint_reader()
    if footprint_reader is not None:
        peaks["physical_footprint_bytes"] = 0
    sampling_errors = []
    stop = threading.Event()

    def sample() -> None:
        """持续采样同一进程，所有后端使用相同的采样间隔。"""
        while not stop.is_set():
            peaks["rss_bytes"] = max(peaks["rss_bytes"], process.memory_info().rss)
            if footprint_reader is not None:
                try:
                    peaks["physical_footprint_bytes"] = max(peaks["physical_footprint_bytes"], footprint_reader())
                except OSError as exc:
                    sampling_errors.append(exc)
                    return
            allocated = 0
            if device.startswith("mps"):
                allocated = torch.mps.current_allocated_memory()
            elif device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(device)
            peaks["torch_device_bytes"] = max(peaks["torch_device_bytes"], allocated)
            stop.wait(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    durations = []
    result = None
    try:
        for _ in range(rounds):
            started = time.perf_counter()
            result = operation()
            _synchronize(device)
            durations.append(time.perf_counter() - started)
    finally:
        stop.set()
        sampler.join()
    if sampling_errors:
        raise sampling_errors[0]
    return result, {"seconds": durations, "median_seconds": statistics.median(durations), **peaks}


def _operation(args: argparse.Namespace, manifest: dict[str, Any]) -> Callable[[], Any]:
    """通过实际产品模型和解析器构造测量操作，保留默认生成与分批策略。"""
    from PIL import Image
    import numpy as np

    model_root = args.model_root.expanduser()
    if args.kind in {"vlm-transformers", "vlm-mlx"}:
        from mineru.model.vlm.runtime import ModelSingleton
        from mineru_vl_utils.vlm_client.base_client import SamplingParams

        backend = "transformers" if args.kind == "vlm-transformers" else "mlx-engine"
        predictor = ModelSingleton().get_model(backend, str(model_root / "MinerU2.5-Pro-2605-1.2B"), None, batch_size=2)
        predictor.client.use_tqdm = False
        images = [Image.open(item["image"]).convert("RGB") for item in manifest["pages"][:2]]
        # 组件 smoke 只验证真实模型的协议链路；整页质量另由 parse 场景在原始分辨率验收。
        for image in images:
            image.thumbnail((448, 448))
        return lambda: predictor.client.batch_predict(
            images, "Read the text in this image.", SamplingParams(max_new_tokens=16, temperature=0)
        )
    if args.kind == "layout":
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

        model = PPDocLayoutV2LayoutModel(str(model_root / "MinerU-4_models_torch/Layout/PP-DocLayoutV2"), args.device)
        images = [Image.open(item["image"]).convert("RGB") for item in manifest["pages"]]
        return lambda: model.batch_predict(images, batch_size=2)
    if args.kind == "mfr":
        from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer

        model = FormulaRecognizer(str(model_root / "MinerU-4_models_torch/MFR/pp_formulanet_plus_m"), args.device)
        images = [np.asarray(Image.open(path).convert("RGB")) for path in manifest["formulas"]]
        if not images:
            raise ValueError("The manifest must contain at least one formula crop")
        detections = [[{"label": "display_formula", "bbox": [0, 0, image.shape[1], image.shape[0]]}] for image in images]
        return lambda: model.batch_predict(detections, images, batch_size=2)

    from mineru.config import VlmConfig, config
    from mineru.parser.mineru_parser import MinerUParser

    config.model.small_backend = "torch"
    config.llm_aided.features.title_leveling = False
    config.llm_aided.features.cross_page_table_cell_merge = False
    parser = MinerUParser(tier=args.tier, parse_mode="ocr", vlm_config=VlmConfig())

    def parse() -> list[dict[str, Any]]:
        """按清单页码解析真实 PDF，保存完整语义树与最终 Markdown。"""
        results = []
        for item in manifest["pages"]:
            result = parser.parse(item["pdf"], page_range=str(item["page"]))
            results.append({"name": item["name"], "markdown": result.markdown(), "middle": result.to_dict(skip_defaults=False)})
        return results

    return parse


def _json_value(value: Any) -> Any:
    """将模型返回的 NumPy/Torch 数据转换成可比较的 JSON。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def main() -> None:
    """保存版本、测量口径和推理结果，便于在两个版本之间复核。"""
    args = _arguments()
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["MINERU_DEVICE_MODE"] = args.device
    manifest = json.loads(args.manifest.read_text())
    operation = _operation(args, manifest)
    result, metrics = _measure(operation, args.device, args.rounds, args.warmup)
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "torch", "tokenizers", "docvortex")}
    report = {
        "versions": versions,
        "source_paths": {
            name: sys.modules[name].__file__ for name in ("mineru", "docvortex", "mineru_vl_utils") if name in sys.modules
        },
        "python": sys.version,
        "kind": args.kind,
        "device": args.device,
        "sample_interval_ms": 20,
        "warmup_rounds": args.warmup,
        "metrics": metrics,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value) + "\n")
    print(json.dumps({"output": str(args.output), **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
