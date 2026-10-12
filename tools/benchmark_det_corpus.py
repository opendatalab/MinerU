"""冻结生产 det 裁剪并独立重放 A/B/C 或融合 D/E，后续测速无需完整解析。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_mps_det_service import SAMPLES, write_json  # noqa: E402
from det_cpu_microbatch import CPU_MICROBATCH_STRATEGIES, project_cpu_microbatch  # noqa: E402
from det_shape_buckets import SHAPE_STRATEGIES, project_shape  # noqa: E402

DEFAULT_CORPUS = ROOT / "tmp/pdfs/issue5632-det-corpus-20261012"
DEFAULT_RESULTS = ROOT / "tmp/pdfs/issue5632-det-only-20261012"
CAPTURE = ROOT / "tmp/pdfs/issue5632-cross-document-run2-20261012/A"
SINGLE_POLICIES = {"J", "K"}
CPU_POLICIES = {"A", "L", "M", "N", "V", *CPU_MICROBATCH_STRATEGIES}
CPU_SPLIT_POLICIES = {"M", "N"}
FUSION_REPLAY_POLICIES = {"F": "E", "G": "E", "H": "E", "I": "E", "K": "E", "L": "E", "N": "E"}


def project_replay_shape(shape: list[int], policy: str) -> dict[str, Any]:
    """单图入口逐张运行 N=1，其他实验策略保留原来的形状投影与行为。"""
    if policy in CPU_MICROBATCH_STRATEGIES:
        return project_cpu_microbatch(shape, policy)
    projection = project_shape(shape, "F" if policy in SINGLE_POLICIES | CPU_SPLIT_POLICIES else policy)
    if policy in SINGLE_POLICIES:
        projection["entrypoint"] = "TextDetector.__call__"
    return projection


def det_source_fingerprint() -> str:
    """对本实验直接使用的普通 det 源码树取摘要，区分无关提交号变化与实际推理代码变化。"""
    paths = [ROOT / "mineru/model/registry.py"]
    for directory in (ROOT / "mineru/model/_internal/pytorchocr", ROOT / "mineru/model/runtime"):
        paths.extend(path for path in directory.rglob("*") if path.suffix in {".py", ".yaml", ".yml"})
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """读取已冻结的输入和原始结果。"""
    return json.loads(path.read_text(encoding="utf-8"))


def mac_memory_diagnostics() -> dict[str, Any]:
    """读取本进程 TASK_VM_INFO，补充 RSS 未计入的压缩页和物理 footprint。"""
    import ctypes
    import psutil

    result: dict[str, Any] = {
        "system_available": psutil.virtual_memory().available,
        "system_swap_used": psutil.swap_memory().used,
    }
    if sys.platform != "darwin":
        return result
    library = ctypes.CDLL(None)
    task = ctypes.c_uint.in_dll(library, "mach_task_self_").value
    values = (ctypes.c_uint64 * 128)()
    count = ctypes.c_uint(ctypes.sizeof(values) // ctypes.sizeof(ctypes.c_uint))
    library.task_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    library.task_info.restype = ctypes.c_int
    code = library.task_info(task, 22, values, ctypes.byref(count))
    # 本机 SDK mach/task_info.h 中 compressed、phys_footprint、ledger_phys_footprint_peak 的偏移。
    if code == 0 and count.value >= 44:
        result.update({"compressed": values[15], "phys_footprint": values[18], "phys_footprint_peak": values[21]})
    else:
        result["task_info_error"] = code
    return result


def fuse_det_eval(net: Any, policy: str) -> dict[str, Any]:
    """仅在实验模型的 CPU/eval 状态融合明确的 Conv-BN 对，保留生产模型实现和权重文件。"""
    import torch
    from torch.nn.utils import fuse_conv_bn_eval
    from mineru.model._internal.pytorchocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4ConvLayer
    from mineru.model._internal.pytorchocr.modeling.heads.det_db_head import PPOCRV6DBConvBatchnormLayer

    before = sum(isinstance(module, torch.nn.BatchNorm2d) for module in net.modules())
    report: dict[str, Any] = {"before_bn": before, "after_bn": before, "pairs": [], "seconds": 0.0}
    if policy not in {"D", "E"}:
        return report
    if before != 36 or net.training or any(parameter.device.type != "cpu" for parameter in net.parameters()):
        raise ValueError("融合要求 CPU/eval Tiny-det，且现有重参数化之后恰好剩余 36 个 BN")
    backbone = [(name, module) for name, module in net.named_modules() if isinstance(module, PPLCNetV4ConvLayer)]
    head = [(name, module) for name, module in net.named_modules() if isinstance(module, PPOCRV6DBConvBatchnormLayer)]
    if len(backbone) != 34 or len(head) != 2:
        raise ValueError(f"融合结构与约定不符：backbone={len(backbone)}, head={len(head)}")
    pairs = [(name, module, "normalization") for name, module in backbone]
    if policy == "E":
        pairs += [(name, module, "norm") for name, module in head]
    started = time.perf_counter()
    for name, module, bn_attribute in pairs:
        convolution = module.convolution
        normalization = getattr(module, bn_attribute)
        if not isinstance(convolution, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            raise TypeError(f"{name} 不是明确支持的卷积类型")
        if not isinstance(normalization, torch.nn.BatchNorm2d) or convolution.training or normalization.training:
            raise ValueError(f"{name} 的 Conv/BN 不满足 eval 融合条件")
        if normalization.running_mean is None or normalization.running_var is None:
            raise ValueError(f"{name} 缺少 BN running buffers")
        transpose = isinstance(convolution, torch.nn.ConvTranspose2d)
        module.convolution = fuse_conv_bn_eval(convolution, normalization, transpose=transpose)
        setattr(module, bn_attribute, torch.nn.Identity())
        report["pairs"].append(
            {"module": name, "bn_attribute": bn_attribute, "convolution": type(convolution).__name__, "transpose": transpose}
        )
    net.eval()
    report["seconds"] = time.perf_counter() - started
    report["after_bn"] = sum(isinstance(module, torch.nn.BatchNorm2d) for module in net.modules())
    expected = 2 if policy == "D" else 0
    if report["after_bn"] != expected:
        raise ValueError(f"融合之后应剩余 {expected} 个 BN，实际 {report['after_bn']}")
    return report


def selected_calls(document: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    """保留原调用编号，支持普通顺序重放和仅诊断指定调用集合。"""
    if args.call_indices:
        indices = [int(index) for index in args.call_indices.split(",")]
        if len(indices) != len(set(indices)) or any(index < 0 or index >= len(document["calls"]) for index in indices):
            raise ValueError("call-indices 包含重复或越界调用")
    else:
        indices = list(range(args.start_call, len(document["calls"])))
        if args.limit_calls:
            indices = indices[: args.limit_calls]
    if not indices:
        raise ValueError(f"{document['name']} 的调用范围为空")
    return [(index, document["calls"][index]) for index in indices]


def freeze(args: argparse.Namespace) -> None:
    """复制三份文档进入生产 det 的全部 PNG、参数、原 batch 和参考检测框。"""
    import numpy as np
    from PIL import Image
    from docvortex.document.pdf import PDFDocument

    if (args.corpus / "manifest.json").exists():
        raise FileExistsError("已有冻结语料，请使用新目录，避免覆盖旧输入")
    (args.corpus / "images").mkdir(parents=True, exist_ok=True)
    documents = []
    verified: set[str] = set()
    for index, (sample, source) in enumerate(SAMPLES.items()):
        label = f"{index + 1:02d}-{sample}"
        captured = args.capture / label
        job = read_json(captured / "job.json")
        assert job["status"] == "completed", f"源文档取样未完整完成：{label}"
        calls = [json.loads(line) for line in (captured / "det-trace.jsonl").read_text().splitlines()]
        for call in calls:
            for item in call["items"]:
                crop = item["crop"]
                if crop in verified:
                    continue
                source_png = args.capture / "crops" / f"{crop}.png"
                target_png = args.corpus / "images" / f"{crop}.png"
                shutil.copy2(source_png, target_png)
                with Image.open(target_png) as image:
                    array = np.array(image.convert("RGB"))[:, :, ::-1].copy()
                assert list(array.shape) == item["original_shape"]
                actual = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
                assert actual == crop, f"PNG 不能恢复原始 BGR 输入：{crop}"
                verified.add(crop)
        directory = args.corpus / sample
        directory.mkdir(exist_ok=True)
        (directory / "inputs").mkdir(exist_ok=True)
        input_index = 0
        with (directory / "inputs.jsonl").open("w", encoding="utf-8") as index_file:
            for call_index, call in enumerate(calls):
                for item_index, item in enumerate(call["items"]):
                    link = directory / "inputs" / f"{input_index:05d}.png"
                    link.symlink_to(Path("../../images") / f"{item['crop']}.png")
                    index_file.write(
                        json.dumps(
                            {
                                "document": sample,
                                "input_index": input_index,
                                "call_index": call_index,
                                "item_index": item_index,
                                "png": str(link.relative_to(args.corpus)),
                                "shape": call["shape"],
                                "box_thresh": call["box_thresh"],
                                "unclip_ratio": call["unclip_ratio"],
                                "reference_box_count": len(item["boxes"] or []),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    input_index += 1
        shutil.copy2(captured / "bundle/model_output.json", directory / "model_output.json")
        shutil.copy2(captured / "middle_json.json", directory / "middle_json.json")
        with PDFDocument(str(source)) as document:
            pages = document.page_count
        documents.append(
            {
                "name": sample,
                "source_pdf": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "pages": pages,
                "calls": calls,
                "crops": sum(len(call["items"]) for call in calls),
                "source_model_output": f"{sample}/model_output.json",
                "source_middle_json": f"{sample}/middle_json.json",
            }
        )
    write_json(
        args.corpus / "manifest.json",
        {
            "version": 1,
            "capture": str(args.capture),
            "capture_route": "V1 API / Basic / forced OCR / all pages",
            "documents": documents,
            "unique_images": len(verified),
            "parameters": {
                "det_algorithm": "DB",
                "det_limit_side_len": 960,
                "det_limit_type": "max",
                "det_max_side_limit": 4000,
                "det_db_thresh": 0.3,
                "det_box_type": "quad",
                "det_db_score_mode": "fast",
                "use_dilation": False,
                "lang": "ch",
            },
            "parameter_source": "box_thresh/unclip_ratio recorded per call; other fields are unchanged production defaults",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
    )
    (args.corpus / "README.md").write_text(
        "# 可复用 det 输入\n\n"
        "images/ 保存无损 PNG，manifest.json 保存原始 batch、NCHW、参数和参考框。\n"
        "每份文档 inputs/ 按 det 输入顺序编号，inputs.jsonl 映射到原调用；编号不是 PDF 页码或语义块编号。\n"
        "model_output.json 和 middle_json.json 保留源解析上下文。\n\n"
        "快速烟测：\n\n"
        f"`.venv4/bin/python tools/benchmark_det_corpus.py run --corpus {args.corpus} "
        "--output NEW_OUTPUT --limit-calls 12`\n\n"
        "定点调试追加 `--documents python --start-call 50 --limit-calls 1`；全量去掉 limit-calls。\n"
        "完成后用相同 corpus/output 执行 summarize；每轮请使用新 output 保留证据。\n",
        encoding="utf-8",
    )
    print(
        f"FROZEN documents={len(documents)} crops={sum(doc['crops'] for doc in documents)} unique_png={len(verified)}",
        flush=True,
    )


def worker(args: argparse.Namespace) -> None:
    """独立进程仅加载 det，重放冻结调用，并在文档之间执行约定清理。"""
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from mineru.model._internal.pytorchocr.infer.predict_det import TextDetector
    from mineru.model._internal.pytorchocr.infer.pytorchocr_utility import init_args
    from mineru.model.registry import MINERU_4_MODELS_TORCH
    from mineru.model.runtime.cpu_threads import initialize_torch_threads
    from mineru.model.runtime.memory import clean_memory

    initialize_torch_threads()
    nnpack_enabled_before = torch._C._get_nnpack_enabled()
    # 实验对照显式保留旧 NNPACK 开启基线，避免生产默认变化使后续重测失去对照。
    torch.backends.nnpack.set_flags(args.policy != "V")
    nnpack_enabled = torch._C._get_nnpack_enabled()
    source_before = det_source_fingerprint()
    device = "cpu" if args.policy in CPU_POLICIES else "mps"
    output = args.output / args.policy
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.corpus / "manifest.json")
    documents = {doc["name"]: doc for doc in manifest["documents"]}
    process = psutil.Process()
    totals: Counter[str] = Counter()
    shapes: Counter[tuple[int, ...]] = Counter()
    seen_shapes: set[tuple[int, ...]] = set()
    last_shape: tuple[int, ...] | None = None
    network_shapes: list[list[int]] = []

    def memory() -> dict[str, int]:
        """在同步边界记录 CPU RSS 和两类 MPS 内存，输入图像不跨文档保留。"""
        torch.mps.synchronize()
        return {
            "rss": process.memory_info().rss,
            "mps_allocated": torch.mps.current_allocated_memory(),
            "mps_driver": torch.mps.driver_allocated_memory(),
        }

    class MeasuredDetector(TextDetector):
        """计时生产前处理和网络执行，不保存或变换额外输入。"""

        def __init__(self, parameters: Any) -> None:
            """保留生产权重、reparam 和 FP32，只迁移本实验实例的设备。"""
            loading_started = time.perf_counter()
            super().__init__(parameters)
            loading_seconds = time.perf_counter() - loading_started
            self.fusion = fuse_det_eval(self.net, FUSION_REPLAY_POLICIES.get(args.policy, args.policy))
            transfer_started = time.perf_counter()
            self.device = device
            self._apply_inference_precision(device, precision_override="fp32")
            torch.mps.synchronize()
            self.loading_seconds = loading_seconds + time.perf_counter() - transfer_started
            self.started = 0.0
            self.last_maps = None
            self.capture_maps = args.save_maps
            self.bucket_padding_seconds = 0.0
            self.net.register_forward_pre_hook(self.before_forward)
            self.net.register_forward_hook(self.after_forward)

        def _preprocess_det_image(self, img: Any) -> Any:
            """记录真实 resize、归一化和 CHW 转换耗时。"""
            started = time.perf_counter()
            result = super()._preprocess_det_image(img)
            totals["preprocess"] += time.perf_counter() - started
            return result

        def _batch_process_preprocessed(self, batch_items: list[Any]) -> tuple[list[Any], float]:
            """仅在形状实验及 CPU N 拆分实验中替换网络批次，其他策略继续调用原实现。"""
            if args.policy in CPU_MICROBATCH_STRATEGIES and batch_items:
                shape = [len(batch_items), *batch_items[0][1].shape]
                if project_cpu_microbatch(shape, args.policy)["split_required"]:
                    from det_cpu_microbatch import process_cpu_microbatch

                    return process_cpu_microbatch(self, batch_items, args.policy)
                return super()._batch_process_preprocessed(batch_items)
            if args.policy in SHAPE_STRATEGIES or args.policy in CPU_SPLIT_POLICIES:
                from det_shape_buckets import process_bucketed_batch

                bucket_policy = "F" if args.policy in CPU_SPLIT_POLICIES else args.policy
                return process_bucketed_batch(self, batch_items, bucket_policy)
            return super()._batch_process_preprocessed(batch_items)

        def before_forward(self, module: Any, inputs: Any) -> None:
            """记录完整 NCHW 形状；这里只统计复用机会，不声称是底层命中率。"""
            nonlocal last_shape
            if device == "mps":
                torch.mps.synchronize()
            self.started = time.perf_counter()
            last_shape = tuple(inputs[0].shape)
            network_shapes.append(list(last_shape))
            totals["new_shapes" if last_shape not in seen_shapes else "reused_shapes"] += 1
            seen_shapes.add(last_shape)
            shapes[last_shape] += 1

        def after_forward(self, module: Any, inputs: Any, outputs: Any) -> None:
            """同步实际 GPU 执行完成，累计纯模型 forward 时间。"""
            if device == "mps":
                torch.mps.synchronize()
            totals["forward"] += time.perf_counter() - self.started

        def _build_det_preds(self, outputs: Any) -> Any:
            """记录结果转 FP32 并回传 CPU 的耗时。"""
            started = time.perf_counter()
            result = super()._build_det_preds(outputs)
            totals["output_transfer"] += time.perf_counter() - started
            if args.save_maps:
                self.last_maps = result["maps"]
            return result

        def _postprocess_det_batch(self, preds: Any, batch_shapes: Any, ori_shapes: Any) -> Any:
            """记录 DB 后处理、裁剪和检测框过滤的耗时。"""
            started = time.perf_counter()
            result = super()._postprocess_det_batch(preds, batch_shapes, ori_shapes)
            totals["postprocess"] += time.perf_counter() - started
            return result

    model_path = MINERU_4_MODELS_TORCH.pytorch_paddle.path("ch_PP-OCRv6_tiny_det_infer.safetensors").ensure()
    models: dict[tuple[float, float], MeasuredDetector] = {}
    model_loads = []
    records = []
    sequence = args.documents.split(",") * args.cycles
    for position, name in enumerate(sequence):
        document = documents[name]
        calls = selected_calls(document, args)
        totals.clear()
        shapes.clear()
        before = memory()
        diagnostics_before = mac_memory_diagnostics()
        peak = dict(before)
        peak_diagnostics = {key: diagnostics_before[key] for key in ("phys_footprint", "compressed")}
        results = []
        call_metrics = []
        differences = []
        seconds = 0.0
        input_seconds = 0.0
        for index, (call_index, call) in enumerate(calls):
            profile = (call["box_thresh"], call["unclip_ratio"])
            if profile not in models:
                parameters = init_args().parse_args([])
                for key, value in manifest["parameters"].items():
                    setattr(parameters, key, value)
                parameters.device = "cpu"
                parameters.det_model_path = str(model_path)
                parameters.det_db_box_thresh, parameters.det_db_unclip_ratio = profile
                started = time.perf_counter()
                models[profile] = MeasuredDetector(parameters)
                torch.mps.synchronize()
                model_loads.append(
                    {
                        "profile": list(profile),
                        "seconds": time.perf_counter() - started,
                        "loading_seconds": models[profile].loading_seconds,
                        "fusion": models[profile].fusion,
                    }
                )
            started = time.perf_counter()
            images = []
            for item in call["items"]:
                with Image.open(args.corpus / "images" / f"{item['crop']}.png") as image:
                    images.append(np.array(image.convert("RGB"))[:, :, ::-1].copy())
            input_seconds += time.perf_counter() - started
            detector = models[profile]
            stages_before = dict(totals)
            network_shapes.clear()
            padding_before = detector.bucket_padding_seconds
            started = time.perf_counter()
            if args.policy in SINGLE_POLICIES:
                detected = []
                single_maps = []
                for input_image in images:
                    detected.append(detector(input_image))
                    if args.save_maps:
                        single_maps.append(detector.last_maps)
                if args.save_maps:
                    detector.last_maps = np.concatenate(single_maps, axis=0)
            else:
                detected = detector.batch_predict(images, max_batch_size=call["shape"][0])
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            seconds += elapsed
            call_metrics.append(
                {
                    "call_index": call_index,
                    "shape": call["shape"],
                    "profile": list(profile),
                    "network_shapes": list(network_shapes),
                    "shape_projection": project_replay_shape(call["shape"], args.policy),
                    "bucket_padding_seconds": detector.bucket_padding_seconds - padding_before,
                    "seconds": elapsed,
                    "stages": {
                        key: totals[key] - stages_before.get(key, 0)
                        for key in ("preprocess", "forward", "output_transfer", "postprocess", "new_shapes", "reused_shapes")
                    },
                }
            )
            assert network_shapes == project_replay_shape(call["shape"], args.policy)["network_shapes"], (
                "实际网络 NCHW 与策略投影不同"
            )
            assert len(detected) == len(call["items"])
            for item_index, (item, (boxes, _)) in enumerate(zip(call["items"], detected)):
                points = boxes.tolist() if hasattr(boxes, "tolist") else boxes
                results.append(
                    {
                        "crop": item["crop"],
                        "profile": list(profile),
                        "call_index": call_index,
                        "item_index": item_index,
                        "boxes": points,
                        "reference_equal": points == item["boxes"],
                    }
                )
                if points != item["boxes"]:
                    difference = {
                        "policy": args.policy,
                        "document": name,
                        "call_index": call_index,
                        "item_index": item_index,
                        "crop": item["crop"],
                        "shape": call["shape"],
                        "profile": list(profile),
                        "reference": item["boxes"],
                        "actual": points,
                    }
                    differences.append(difference)
                    review = output / "review"
                    review.mkdir(exist_ok=True)
                    prefix = f"{position + 1:02d}-{name}-call{call_index:04d}-item{item_index:03d}"
                    shutil.copy2(args.corpus / "images" / f"{item['crop']}.png", review / f"{prefix}-source.png")
                    render_crop_comparison(args.corpus, review / f"{prefix}-boxes.png", item["crop"], item["boxes"], points)
            if args.save_maps:
                (output / "maps").mkdir(exist_ok=True)
                np.savez_compressed(
                    output / "maps" / f"{position + 1:02d}-{name}-call{call_index:04d}.npz", maps=detector.last_maps
                )
                detector.last_maps = None
            del images, detected
            snapshot = memory()
            diagnostics = mac_memory_diagnostics()
            call_metrics[-1]["memory"] = snapshot
            call_metrics[-1]["memory_diagnostics"] = diagnostics
            for key in peak:
                peak[key] = max(peak[key], snapshot[key])
            for key in peak_diagnostics:
                peak_diagnostics[key] = max(peak_diagnostics[key], diagnostics[key])
            if (index + 1) % 50 == 0:
                print(
                    f"PROGRESS {args.policy} {position + 1}-{name} calls={index + 1}/{len(calls)} det={seconds:.2f}s",
                    flush=True,
                )
        memory_before_cleanup = memory()
        diagnostics_before_cleanup = mac_memory_diagnostics()
        cleanup_started = time.perf_counter()
        if args.policy == "C":
            gc.collect()
        else:
            # A 也沿用宿主 MPS 的原回收边界，CPU 模型参数继续复用。
            clean_memory("mps")
            if device == "mps":
                seen_shapes.clear()
        cleanup_seconds = time.perf_counter() - cleanup_started
        diagnostics_after = mac_memory_diagnostics()
        for key in peak_diagnostics:
            peak_diagnostics[key] = max(peak_diagnostics[key], diagnostics_before_cleanup[key], diagnostics_after[key])
        label = f"{position + 1:02d}-{name}"
        record = {
            "label": label,
            "document": name,
            "cycle": position // len(args.documents.split(",")) + 1,
            "seconds": seconds,
            "crops": len(results),
            "calls": len(calls),
            "stages": dict(totals),
            "input_seconds_excluded": input_seconds,
            "cleanup_seconds": cleanup_seconds,
            "memory_before": before,
            "memory_before_cleanup": memory_before_cleanup,
            "memory_after": memory(),
            "memory_diagnostics_before": diagnostics_before,
            "memory_diagnostics_before_cleanup": diagnostics_before_cleanup,
            "memory_diagnostics_after": diagnostics_after,
            "peak_memory": peak,
            "peak_memory_diagnostics": peak_diagnostics,
            "call_metrics": call_metrics,
            "differences": differences,
            "models": [
                {
                    "profile": list(profile),
                    "id": id(model),
                    "net_id": id(model.net),
                    "device": model.device,
                    "dtype": str(model.ocr_inference_dtype),
                    "fusion": model.fusion,
                }
                for profile, model in models.items()
            ],
            "reference_exact_crops": sum(result["reference_equal"] for result in results),
            "shape_counts": [{"shape": list(shape), "calls": count} for shape, count in shapes.items()],
        }
        records.append(record)
        write_json(output / f"{label}-boxes.json", results)
        write_json(output / "results.json", records)
        print(
            f"DONE {args.policy} {label} det={seconds:.3f}s crops={len(results)} "
            f"reference_exact={record['reference_exact_crops']} "
            f"memory={record['memory_after']}",
            flush=True,
        )
        if args.policy == "C" and position + 1 == len(args.documents.split(",")) * args.cycles and args.cycles >= 2:
            first = records[len(args.documents.split(",")) - 1]["memory_after"]
            second = record["memory_after"]
            if any(second[key] - first[key] > max(256 * 1024**2, first[key] * 0.1) for key in first):
                sequence.extend(args.documents.split(","))
                print("EXTEND C: 同进程追加一遍输入，验证内存增长趋势", flush=True)
    shutdown_before = mac_memory_diagnostics()
    shutdown_started = time.perf_counter()
    del detector
    models.clear()
    gc.collect()
    clean_memory("mps")
    shutdown_seconds = time.perf_counter() - shutdown_started
    write_json(
        output / "environment.json",
        {
            "pid": os.getpid(),
            "policy": args.policy,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_commit": torch.version.git_version,
            "platform": platform.platform(),
            "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "nnpack_enabled_before": nnpack_enabled_before,
            "nnpack_enabled": nnpack_enabled,
            "nnpack_enabled_at_exit": torch._C._get_nnpack_enabled(),
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "model_loads": model_loads,
            "shutdown_memory": memory(),
            "shutdown_memory_diagnostics_before": shutdown_before,
            "shutdown_memory_diagnostics_after": mac_memory_diagnostics(),
            "shutdown_seconds": shutdown_seconds,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shape_adapter_sha256": hashlib.sha256((ROOT / "tools/det_shape_buckets.py").read_bytes()).hexdigest(),
            "cpu_microbatch_adapter_sha256": hashlib.sha256((ROOT / "tools/det_cpu_microbatch.py").read_bytes()).hexdigest(),
            "det_source_sha256_before": source_before,
            "det_source_sha256_after": det_source_fingerprint(),
            "limit_calls": args.limit_calls,
            "start_call": args.start_call,
        },
    )


def run(args: argparse.Namespace) -> None:
    """串行启动独立策略进程，复测仅依赖冻结 PNG 和 manifest。"""
    policies = args.policies.split(",")
    if len(set(policies)) != len(policies) or any(
        policy not in "ABCDEFGHIJKLMNOPQRSTUV" or len(policy) != 1 for policy in policies
    ):
        raise ValueError("policies 必须是无重复的 A 至 V 逗号列表")
    names = args.documents.split(",")
    available = {document["name"] for document in read_json(args.corpus / "manifest.json")["documents"]}
    if any(name not in available for name in names):
        raise ValueError(f"documents 必须取自 {sorted(available)}")
    if (args.output / "protocol.json").exists():
        raise FileExistsError("请用新的 --output 保留此前结果")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "protocol.json",
        {
            "corpus": str(args.corpus),
            "documents": args.documents,
            "cycles": args.cycles,
            "limit_calls": args.limit_calls,
            "start_call": args.start_call,
            "policies": args.policies,
            "call_indices": args.call_indices,
            "save_maps": args.save_maps,
        },
    )
    for policy in policies:
        directory = args.output / policy
        if (directory / "results.json").exists():
            raise FileExistsError("请用新的 --output 保留此前结果")
        directory.mkdir(exist_ok=True)
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--policy",
                    policy,
                    "--corpus",
                    str(args.corpus),
                    "--output",
                    str(args.output),
                    "--documents",
                    args.documents,
                    "--cycles",
                    str(args.cycles),
                    "--limit-calls",
                    str(args.limit_calls),
                    "--start-call",
                    str(args.start_call),
                    "--call-indices",
                    args.call_indices,
                    *(["--save-maps"] if args.save_maps else []),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"FINISHED policy={policy} report={directory / 'results.json'}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    """汇总冷热 det 性能和缓存成本，保存差异图以及可直接复测的命令。"""
    import numpy as np

    manifest = read_json(args.corpus / "manifest.json")
    protocol = read_json(args.output / "protocol.json")
    document_count = len(protocol["documents"].split(","))
    policies = protocol["policies"].split(",")
    if not {"A", "C"}.issubset(policies):
        raise ValueError("速度验收汇总至少需要 A 和 C；其他定点运行可直接读取 results.json")
    results = {policy: read_json(args.output / policy / "results.json") for policy in policies}
    environments = {policy: read_json(args.output / policy / "environment.json") for policy in policies}
    aggregate = {}
    for policy, records in results.items():
        aggregate[policy] = {
            "first_seconds": sum(record["seconds"] for record in records[:document_count]),
            "second_seconds": sum(record["seconds"] for record in records[document_count : 2 * document_count]),
            "reference_exact": all(record["reference_exact_crops"] == record["crops"] for record in records),
            "crops_per_cycle": sum(record["crops"] for record in records[:document_count]),
        }
    cpu = aggregate["A"]["second_seconds"]
    candidate = aggregate["C"]["second_seconds"]
    first_memory = results["C"][document_count - 1]["memory_after"]
    second_memory = results["C"][2 * document_count - 1]["memory_after"]
    memory_growth = {
        key: {
            "first": first_memory[key],
            "second": second_memory[key],
            "delta": second_memory[key] - first_memory[key],
            "limit": max(first_memory[key] * 0.1, 256 * 1024**2),
        }
        for key in first_memory
    }
    gate = {
        "det_reduction": 1 - candidate / cpu,
        "det_speedup": cpu / candidate,
        "det_performance_pass": candidate <= cpu * 0.75,
        "outputs_exact": all(value["reference_exact"] for value in aggregate.values()),
        "weights_equal": len({env["model_sha256"] for env in environments.values()}) == 1,
        "det_fp32": all(
            model["dtype"] == "torch.float32" for records in results.values() for row in records for model in row["models"]
        ),
        "models_reused": all(
            len({model["net_id"] for row in records for model in row["models"] if tuple(model["profile"]) == profile}) == 1
            for records in results.values()
            for profile in {tuple(model["profile"]) for row in records for model in row["models"]}
        ),
        "memory_screen_pass": all(value["delta"] <= value["limit"] for value in memory_growth.values()),
        "reverse_order_needed": abs(1 - candidate / cpu) < 0.1,
        "additional_cycle_executed": len(results["C"]) > 2 * document_count,
    }
    if gate["additional_cycle_executed"]:
        third_memory = results["C"][3 * document_count - 1]["memory_after"]
        gate["third_cycle_memory_growth"] = {
            key: {"delta": third_memory[key] - second_memory[key], "limit": max(second_memory[key] * 0.1, 256 * 1024**2)}
            for key in third_memory
        }
    checks = []
    review = args.output / "review"
    review.mkdir(exist_ok=True)
    for policy, records in results.items():
        for record in records:
            captured = read_json(args.output / policy / f"{record['label']}-boxes.json")
            document = next(doc for doc in manifest["documents"] if doc["name"] == record["document"])
            source_calls = document["calls"][protocol.get("start_call", 0) :]
            if protocol["limit_calls"]:
                source_calls = source_calls[: protocol["limit_calls"]]
            references = [item for call in source_calls for item in call["items"]]
            for index, (item, reference) in enumerate(zip(captured, references)):
                assert item["crop"] == reference["crop"], "重放顺序不同"
                if item["reference_equal"]:
                    continue
                delta = {
                    "policy": policy,
                    "document": record["document"],
                    "label": record["label"],
                    "input_index": index,
                    "crop": item["crop"],
                    "reference": reference["boxes"],
                    "actual": item["boxes"],
                }
                checks.append(delta)
                render_crop_comparison(
                    args.corpus,
                    review / f"{policy}-{record['label']}-{index:05d}.png",
                    item["crop"],
                    reference["boxes"],
                    item["boxes"],
                )
    # 每份文档另留一个真实长文本裁剪，方便人工确认输入和检测框的含义。
    for document in manifest["documents"]:
        if document["name"] not in protocol["documents"].split(","):
            continue
        case = next(record for record in results["C"] if record["document"] == document["name"])
        replay = read_json(args.output / "C" / f"{case['label']}-boxes.json")
        if not replay:
            continue
        item = max(replay, key=lambda value: len(value["boxes"] or []))
        reference = next(
            source
            for call in document["calls"]
            for source in call["items"]
            if source["crop"] == item["crop"] and [call["box_thresh"], call["unclip_ratio"]] == item["profile"]
        )
        render_crop_comparison(
            args.corpus, review / f"sample-{document['name']}.png", item["crop"], reference["boxes"], item["boxes"]
        )
    summary = {
        "aggregates": aggregate,
        "gates": gate,
        "memory_growth": memory_growth,
        "differences": checks,
        "requests": results,
        "environments": environments,
    }
    write_json(args.output / "summary.json", summary)
    rows = []
    diagnostic_rows = []
    for policy, records in results.items():
        for record in records:
            rows.append(
                f"| {policy} | {record['label']} | {record['seconds']:.3f} | {record['crops'] / record['seconds']:.1f} | "
                f"{record['stages']['preprocess']:.3f} | {record['stages']['forward']:.3f} | "
                f"{record['stages']['postprocess']:.3f} | {record['memory_after']['rss'] / 1024**3:.2f} | "
                f"{record['memory_after']['mps_driver'] / 1024**3:.2f} |"
            )
            diagnostics = record.get("memory_diagnostics_after", {})
            if "phys_footprint" in diagnostics:
                diagnostic_rows.append(
                    f"| {policy} | {record['label']} | {diagnostics['phys_footprint'] / 1024**3:.2f} | "
                    f"{diagnostics['compressed'] / 1024**3:.2f} | {diagnostics['system_available'] / 1024**3:.2f} | "
                    f"{diagnostics['system_swap_used'] / 1024**3:.2f} |"
                )
    diagnostics_report = ""
    if diagnostic_rows:
        diagnostics_report = (
            "\n补充 macOS TASK_VM_INFO 记账值：\n\n"
            "| 策略 | 输入 | footprint GiB | compressed GiB | 系统 available GiB | 系统 swap used GiB |\n"
            "| --- | --- | ---: | ---: | ---: | ---: |\n"
            + "\n".join(diagnostic_rows)
            + "\n\nfootprint 包含压缩/换出页记账，不能理解为当前驻留物理 RAM，不能与 RSS/driver 相加。"
            "原内存门槛只检查 RSS/allocated/driver 的增长，不能排除高绝对 footprint 的成本。\n"
        )
    command = f".venv4/bin/python tools/benchmark_det_corpus.py run --corpus {args.corpus} --output NEW_OUTPUT"
    report = f"""**仅 det 的冻结语料 A/B/C 对照**

完整语料保存 3 份真实文档的全部普通文字 det 输入：95、88、114 页，共 2280 次裁剪、2262 张唯一 PNG。
语料来自实际 Basic/OCR 生产调用，包含正文和表格 det，保留原始 batch 及每次调用的 box_thresh/unclip_ratio。
PNG 逐个恢复 BGR 并验证像素 SHA256；manifest 保存每次调用的输入形状、图像键和参考框，model_output 保留源块上下文。
只有输入准备阶段经过完整解析；以下测试不调用 layout、rec、表格识别、VLM 或文档导出。

A=CPU det+每本文档原 MPS 清理；B=MPS det+每本文档原 MPS 清理；C=MPS det+每本文档仅 GC 保留缓存。
本次顺序 {protocol["policies"]}，文档顺序 {protocol["documents"]}，每轮起始调用 {protocol.get("start_call", 0)}，
每本文档调用上限 {protocol["limit_calls"]}（0 为全部）。策略使用独立进程串行运行、相同权重、FP32、生产线程配置。
相同 det 参数的模型在同一进程复用。
每份文档严格重放捕获的 batch，断言实际 NCHW 一致。不存在手工补齐形状、融合算子或其他算法优化。

| 策略 | 输入 | det 秒 | 裁剪/秒 | 预处理秒 | forward 秒 | 后处理秒 | 结束 RSS GiB | MPS driver GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

第二遍所选输入的 det 合计：A {cpu:.3f} 秒，C {candidate:.3f} 秒。
C/A 耗时比 {candidate / cpu:.3f}，耗时降低 {1 - candidate / cpu:.2%}（负值表示回退）。
预定速度门槛（降低至少 25%）：{gate["det_performance_pass"]}。
与真实捕获参考框逐项一致：{gate["outputs_exact"]}；三种策略权重相同：{gate["weights_equal"]}。
det FP32：{gate["det_fp32"]}；模型实例持续复用：{gate["models_reused"]}。
内存初筛通过：{gate["memory_screen_pass"]}；追加第三遍：{gate["additional_cycle_executed"]}。
内存分项与第三遍趋势保存在 summary.json，门槛为增长不超过 max(10%,256 MiB)。
{diagnostics_report}
不同 NCHW 的首次执行仍可能慢；有限输入集合的稳定性不能证明无限种类文档的缓存有界。

det 时间包含真实预处理、张量创建、同步推理、输出回传和 DB 后处理。
模型加载、PNG 读取、结果 JSON 写出、文档间 GC/empty_cache 单独记录，不计入 det 秒。
内存高水位在每次调用结束同步采样，未宣称捕获每个算子内部的瞬时峰值。
此结果只代表普通文字 det 阶段，不代表整本 API 或 Standard/VLM 的速度。

变化框全部保存在 review/ 和 summary.json；每份文档另保存一个真实长文本裁剪供人工核查。
语料目录：{args.corpus}
全量复测：`{command}`。
快速烟测：在上述命令追加 `--limit-calls 12`；只测某份文档可用 `--documents python`。
定点调试：用 `--start-call 50 --limit-calls 1 --documents python` 重放指定的原始调用（从 0 开始编号）。
所有重复调用使用 PNG 和 manifest，无需重跑 PDF 解析。运行结束会释放模型和 MPS 缓存。
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for policy, records in results.items():
        indices = np.arange(1, len(records) + 1)
        axes[0].plot(indices, [record["seconds"] for record in records], "o-", label=policy)
        axes[1].plot(indices, [record["memory_after"]["mps_driver"] / 1024**3 for record in records], "o-", label=policy)
    axes[0].set(xlabel="Document replay index", ylabel="det seconds", title="A CPU / B MPS clear / C MPS retain")
    axes[1].set(xlabel="Document replay index", ylabel="MPS driver GiB", title="Cache residency after cleanup")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(args.output / "performance-memory.png", dpi=160)
    plt.close(figure)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def render_crop_comparison(corpus: Path, target: Path, crop: str, reference: Any, actual: Any) -> None:
    """绘制真实 PNG 的参考/实际检测框，颜色不同但原像素相同。"""
    from PIL import Image, ImageDraw

    with Image.open(corpus / "images" / f"{crop}.png") as opened:
        image = opened.convert("RGB")
    width, height = image.size
    output = Image.new("RGB", (2 * width, height + 32), "white")
    for column, (name, boxes, color) in enumerate((("Captured reference", reference, "red"), ("Replay", actual, "blue"))):
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        for box in boxes or []:
            draw.line([tuple(point) for point in box] + [tuple(box[0])], fill=color, width=2)
        output.paste(panel, (column * width, 32))
        ImageDraw.Draw(output).text((column * width + 5, 8), name, fill="black")
    output.save(target)


def main() -> None:
    """提供一次性冻结、完整 det 重放和少量调用烟测的可复用入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "run",
            "worker",
            "summarize",
            "fusion-experiment",
            "fusion-summarize",
            "shape-experiment",
            "shape-summarize",
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--policy", choices=tuple("ABCDEFGHIJKLMNOPQRSTUV"), default="A")
    parser.add_argument("--policies", default="A,B,C")
    parser.add_argument("--documents", default="tender,python,114")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--limit-calls", type=int, default=0)
    parser.add_argument("--start-call", type=int, default=0)
    parser.add_argument("--call-indices", default="", help="只诊断指定原调用编号，逗号分隔；不用于正式性能对照")
    parser.add_argument("--save-maps", action="store_true", help="诊断时另存概率图，不纳入正式性能对照")
    args = parser.parse_args()
    args.corpus, args.output, args.capture = args.corpus.resolve(), args.output.resolve(), args.capture.resolve()
    if args.cycles < 1 or args.limit_calls < 0 or args.start_call < 0:
        parser.error("cycles 必须为正，调用范围参数不能为负")
    if args.command == "summarize" and read_json(args.output / "protocol.json")["cycles"] < 2:
        parser.error("冷热对照汇总需要至少两遍输入；单遍调试可直接查看 results.json")
    if args.command.startswith("fusion-"):
        from analyze_det_fusion import fusion_experiment, fusion_summarize

        {"fusion-experiment": fusion_experiment, "fusion-summarize": fusion_summarize}[args.command](args)
    elif args.command.startswith("shape-"):
        from analyze_det_shapes import shape_experiment, shape_summarize

        {"shape-experiment": shape_experiment, "shape-summarize": shape_summarize}[args.command](args)
    else:
        {"freeze": freeze, "run": run, "worker": worker, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
