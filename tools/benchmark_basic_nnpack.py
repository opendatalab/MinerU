"""在 CPU 上用 demo1/demo2 强制 OCR 跑 Basic，简单比较全局 NNPACK 开关。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_mps_det_service import differences  # noqa: E402
from benchmark_det_corpus import mac_memory_diagnostics  # noqa: E402
from benchmark_det_corpus import det_source_fingerprint  # noqa: E402
from benchmark_mps_det_service import write_json  # noqa: E402

DOCUMENTS = (ROOT / "demo/pdfs/demo1.pdf", ROOT / "demo/pdfs/demo2.pdf")
STAGES = {
    "pdf.layout": "Layout",
    "pdf.formulas": "MFR（含首次加载）",
    "pdf.ocr_detection": "正文 OCR det",
    "pdf.ocr_recognition": "正文 OCR rec",
    "pdf.table_recognition": "表格识别（含表内 OCR）",
    "pdf.formula_numbers": "公式编号（含 OCR rec）",
    "model.initialize": "模型初始化（嵌套，不能重复相加）",
}


def sha256(path: Path) -> str:
    """分块校验权重与样本，避免一次性读取大权重增加内存占用。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_models() -> list[dict[str, Any]]:
    """核对已加载 Torch 模型的全部参数确实为 CPU FP32，并记录权重位置及实例复用。"""
    import torch

    from mineru.model.runtime.hybrid import AtomModelSingleton

    records = []
    for key, wrapper in AtomModelSingleton._models.items():
        components = [wrapper]
        if hasattr(wrapper, "text_sys"):
            components = [wrapper.text_sys.text_detector, wrapper.text_sys.text_recognizer]
        elif hasattr(wrapper, "text_detector"):
            components = [wrapper.text_detector, wrapper.text_recognizer]
        for model in components:
            network = getattr(model, "net", None)
            if network is None:
                network = getattr(model, "model", None)
            if not isinstance(network, torch.nn.Module):
                continue
            parameters = list(network.parameters())
            devices = sorted({str(parameter.device) for parameter in parameters})
            dtypes = sorted({str(parameter.dtype) for parameter in parameters if parameter.is_floating_point()})
            assert devices == ["cpu"] and dtypes == ["torch.float32"], (type(model).__name__, devices, dtypes)
            weight = getattr(model, "weights_path", None) or getattr(model, "model_dir", None)
            records.append(
                {
                    "key": str(key),
                    "class": type(model).__name__,
                    "id": id(model),
                    "network_id": id(network),
                    "devices": devices,
                    "dtypes": dtypes,
                    "weight": str(weight) if weight else None,
                }
            )
    assert any(row["class"] == "PPDocLayoutV2LayoutModel" for row in records), "未核对到 Torch Layout"
    assert any(row["class"] == "FormulaRecognizer" for row in records), "未核对到 Torch MFR"
    assert any(row["class"] == "TextDetector" for row in records), "未核对到 Torch det"
    assert any(row["class"] == "TextRecognizer" for row in records), "未核对到 Torch rec"
    return records


def worker(args: argparse.Namespace) -> None:
    """在独立进程只改变 NNPACK 开关，通过公开 SDK 完成两份 PDF 的真实 CPU Basic/OCR 解析。"""
    os.environ["MINERU_DEVICE_MODE"] = "cpu"
    os.environ["MINERU_TABLE_DEVICE"] = "cpu"
    os.environ["MINERU_PROFILE_STAGES"] = "1"
    import psutil
    import torch
    from loguru import logger

    from mineru.config import get_config
    from mineru.model.runtime.cpu_threads import initialize_torch_threads
    from mineru.parser import parse
    from mineru.parser.writer import FileBasedDataWriter

    get_config().model.small_backend = "torch"
    get_config().llm_aided.features.title_leveling = False
    get_config().llm_aided.features.cross_page_table_cell_merge = False
    initialize_torch_threads()
    source_before = det_source_fingerprint()
    torch.backends.nnpack.set_flags(args.policy == "on")
    target = args.output / args.policy
    target.mkdir(parents=True, exist_ok=False)
    current: dict[str, Any] | None = None
    process = psutil.Process()
    stopped = threading.Event()

    def record_stage(message: Any) -> None:
        """收集生产 stage_timer 日志，保留阶段嵌套关系对应的原始完成时间。"""
        match = re.search(r"MinerU stage: name=([^,]+), elapsed_s=([0-9.]+)", message.record["message"])
        if match is not None and current is not None:
            current["stage_events"].append({"name": match[1], "seconds": float(match[2]), "finished_at": time.perf_counter()})

    def sample_memory() -> None:
        """半秒采集本进程 RSS/footprint，采样峰不代表算子内部瞬时峰值。"""
        with (target / "memory-samples.jsonl").open("w", encoding="utf-8") as stream:
            while not stopped.wait(0.5):
                value = {"time": time.time(), "rss": process.memory_info().rss, **mac_memory_diagnostics()}
                if current is not None:
                    value["document"] = current["document"]
                stream.write(json.dumps(value) + "\n")
                stream.flush()

    sink = logger.add(record_stage, level="INFO", format="{message}")
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    records = []
    weights: dict[str, str] = {}
    try:
        for path in DOCUMENTS:
            current = {"document": path.stem, "stage_events": []}
            started = time.perf_counter()
            result = parse(path, tier="basic", ocr_mode="ocr", image_analysis=False, page_range="all")
            current["parse_seconds"] = time.perf_counter() - started
            current["pages"] = len(result.pages)
            current["metadata"] = result.middle_json.extensions["mineru"]
            assert current["pages"] == (13 if path.stem == "demo1" else 6)
            assert current["metadata"] == {"tier": "basic", "parse_mode": "ocr"}
            stages: Counter[str] = Counter()
            for event in current["stage_events"]:
                stages[event["name"]] += event["seconds"]
            current["stages"] = dict(stages)
            for stage in ("pdf.layout", "pdf.formulas", "pdf.ocr_detection", "pdf.ocr_recognition"):
                assert stages[stage] > 0, f"{path.stem} 未实际执行 {stage}"
            current["models"] = inspect_models()
            current["memory_after_parse"] = {"rss": process.memory_info().rss, **mac_memory_diagnostics()}
            export_started = time.perf_counter()
            result.save(FileBasedDataWriter(str(target / path.stem)))
            current["export_seconds_excluded"] = time.perf_counter() - export_started
            for model in current["models"]:
                if not model["weight"]:
                    continue
                weight_path = Path(model["weight"])
                files = list(weight_path.glob("*.safetensors")) if weight_path.is_dir() else [weight_path]
                for weight_file in files:
                    if str(weight_file) not in weights:
                        weights[str(weight_file)] = sha256(weight_file)
            records.append(current)
            write_json(target / "results.json", records)
            print(f"DONE {args.policy} {path.stem} parse={current['parse_seconds']:.3f}s stages={dict(stages)}", flush=True)
            current = None
            del result
    finally:
        stopped.set()
        monitor.join()
        logger.remove(sink)
    write_json(
        target / "environment.json",
        {
            "pid": os.getpid(),
            "policy": args.policy,
            "nnpack_enabled": torch._C._get_nnpack_enabled(),
            "device": os.environ["MINERU_DEVICE_MODE"],
            "table_device": os.environ["MINERU_TABLE_DEVICE"],
            "small_backend": get_config().model.small_backend,
            "torch": torch.__version__,
            "torch_commit": torch.version.git_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "documents_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in DOCUMENTS},
            "weights_sha256": weights,
            "script_sha256": sha256(Path(__file__)),
            "det_source_before": source_before,
            "det_source_after": det_source_fingerprint(),
            "crop_fix_sha256": {
                name: sha256(ROOT / name)
                for name in (
                    "mineru/model/ocr/results.py",
                    "mineru/model/ocr/image.py",
                    "mineru/model/ocr/pytorch_paddle.py",
                )
            },
        },
    )


def summarize(args: argparse.Namespace) -> None:
    """逐项比较完整结果，按两份 PDF 合计中位数汇总各阶段，不把嵌套初始化重复相加。"""
    groups: dict[str, dict[str, Any]] = {}
    fingerprints = set()
    for round_name in ("round-1", "round-2"):
        groups[round_name] = {}
        for policy in ("on", "off"):
            directory = args.output / round_name / policy
            rows = json.loads((directory / "results.json").read_text())
            env = json.loads((directory / "environment.json").read_text())
            assert env["nnpack_enabled"] == (policy == "on")
            assert env["device"] == env["table_device"] == "cpu" and env["small_backend"] == "torch"
            assert env["det_source_before"] == env["det_source_after"]
            fingerprints.add(
                json.dumps(
                    {key: value for key, value in env.items() if key not in {"pid", "policy", "nnpack_enabled"}}, sort_keys=True
                )
            )
            stage_totals: Counter[str] = Counter()
            for row in rows:
                stage_totals.update(row["stages"])
            groups[round_name][policy] = {
                "rows": rows,
                "parse_seconds": sum(row["parse_seconds"] for row in rows),
                "stages": dict(stage_totals),
                "environment": env,
            }
    assert len(fingerprints) == 1, "CPU 配置、线程、权重或代码不一致"
    comparisons = []
    comparison_root = args.output / "comparisons"
    comparison_root.mkdir(exist_ok=True)
    baseline = args.output / "round-1" / "on"
    for round_name in groups:
        for policy in groups[round_name]:
            for path in DOCUMENTS:
                left, right = baseline / path.stem, args.output / round_name / policy / path.stem
                target = comparison_root / f"{round_name}-{policy}-{path.stem}"
                target.mkdir(exist_ok=True)
                result = {"round": round_name, "policy": policy, "document": path.stem}
                for name in ("middle_json.json", "model_output.json", "structured_content.json"):
                    delta = differences(json.loads((left / name).read_text()), json.loads((right / name).read_text()))
                    write_json(target / (name + ".differences.json"), delta)
                    result[name] = {"exact": not delta, "changed_leaves": len(delta)}
                result["markdown_exact"] = (left / "markdown.md").read_bytes() == (right / "markdown.md").read_bytes()
                if not result["markdown_exact"]:
                    import difflib

                    (target / "markdown.diff").write_text(
                        "".join(
                            difflib.unified_diff(
                                (left / "markdown.md").read_text().splitlines(True),
                                (right / "markdown.md").read_text().splitlines(True),
                            )
                        )
                    )
                comparisons.append(result)
    medians = {
        policy: {
            "parse_seconds": statistics.median(groups[name][policy]["parse_seconds"] for name in groups),
            "stages": {
                stage: statistics.median(groups[name][policy]["stages"].get(stage, 0) for name in groups) for stage in STAGES
            },
        }
        for policy in ("on", "off")
    }
    tables = [
        f"| SDK parse 合计 | {medians['on']['parse_seconds']:.3f} | {medians['off']['parse_seconds']:.3f} | "
        f"{1 - medians['off']['parse_seconds'] / medians['on']['parse_seconds']:.1%} |"
    ]
    for stage, label in STAGES.items():
        on, off = medians["on"]["stages"][stage], medians["off"]["stages"][stage]
        tables.append(
            f"| {label} | {on:.3f} | {off:.3f} | {1 - off / on:.1%} |" if on else f"| {label} | {on:.3f} | {off:.3f} | — |"
        )
    per_document = []
    for name, group in groups.items():
        for policy, value in group.items():
            for row in value["rows"]:
                s = row["stages"]
                per_document.append(
                    f"| {name} | {policy} | {row['document']} | {row['parse_seconds']:.3f} | {s['pdf.layout']:.3f} | "
                    f"{s['pdf.formulas']:.3f} | {s['pdf.ocr_detection']:.3f} | {s['pdf.ocr_recognition']:.3f} |"
                )
    write_json(args.output / "summary.json", {"groups": groups, "medians": medians, "comparisons": comparisons})
    env = groups["round-1"]["on"]["environment"]
    report = f"""# CPU Basic/OCR：demo1＋demo2，NNPACK 开关

demo1 13 页，demo2 6 页，每个独立进程按此顺序各解析一次。两轮策略顺序 on→off、off→on，共 76 页。
通过公开 `mineru.parser.parse()`：tier=basic、ocr_mode=ocr、image_analysis=False、整本。
强制 MINERU_DEVICE_MODE=cpu、MINERU_TABLE_DEVICE=cpu、small_backend=torch；核对 Layout/MFR/det/rec 全部参数为 CPU FP32。
使用 .venv4，Python {env["python"]}、PyTorch {env["torch"]}，生产线程 {env["threads"]}/{env["interop_threads"]}。
没有前向预热、实验融合或 N/H/W 拆分补齐；保留生产内部已有的优化。每本文档执行真实生产清理。
两种策略都使用相同的零尺寸 OCR 裁图修复：先执行既有窄框过滤，并丢弃无法生成至少一像素裁图的退化框。
初次未修复基线在 demo2 崩溃，失败现场单独保留，不混入本轮正式统计。
首份 demo1 含模型冷加载，demo2 复用模型；不混为全热态或单模型纯 forward 速度。
使用生产 MINERU_PROFILE_STAGES 日志；MFR 阶段含首次 MFR 初始化，model.initialize 嵌套于准备/MFR，不能重复相加。
表格与公式编号还会执行 OCR，正文 det/rec 列不包括这部分；对应时间保留单独阶段。
SDK parse 包括读取、模型初始化、推理、PDF 清理和确定性后处理；结果导出、权重校验排除在计时之外。

| 阶段（两份 PDF 合计，两轮中位数） | NNPACK 开 秒 | NNPACK 关 秒 | 耗时降低 |
| --- | ---: | ---: | ---: |
{chr(10).join(tables)}

| 轮次 | NNPACK | 文档 | parse 秒 | Layout 秒 | MFR 秒 | 正文 det 秒 | 正文 rec 秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(per_document)}

输出逐项比较完整 Middle/Model/Structured JSON 与 Markdown，未删除源元数据或对数值施加容差；完整差异见 comparisons/。
一致性结果：`{json.dumps(comparisons, ensure_ascii=False)}`。
环境、权重与输入摘要在各进程 environment.json；生产阶段事件与内存采样单独保留。
NNPACK N≥16 属于 PyTorch 启发式，并非保证本机快：此前固定 N=16 的原 batch 在关闭后转到普通卷积路径，耗时明显降低。
FFT/Winograd 变换、打包、线程调度与硬件适配可能影响实际收益；本实验不能单独裁定其中哪一项占主导。
不修改生产配置、公开接口或权重；仅适用于这两份文档的本机 CPU Torch Basic/OCR。

```bash
.venv4/bin/python tools/benchmark_basic_nnpack.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/benchmark_basic_nnpack.py summarize --output {args.output}
```
"""
    (args.output / "report.md").write_text(report)
    print(json.dumps({"medians": medians, "comparisons": comparisons}, ensure_ascii=False), flush=True)


def experiment(args: argparse.Namespace) -> None:
    """串行执行两轮反序的独立进程对照，并保存实际使用脚本与完整结果。"""
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / "executed-script.py")
    for name, policies in (("round-1", ("on", "off")), ("round-2", ("off", "on"))):
        directory = args.output / name
        directory.mkdir()
        for policy in policies:
            with (directory / f"{policy}.log").open("w") as stream:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "worker", "--policy", policy, "--output", str(directory)],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"FINISHED {name} {policy}", flush=True)
    summarize(args)


def main() -> None:
    """提供简单的完整 Basic/OCR 对照及复测入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("experiment", "worker", "summarize"))
    parser.add_argument("--policy", choices=("on", "off"), default="on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"experiment": experiment, "worker": worker, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
