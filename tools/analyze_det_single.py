"""对照 3.4 单图入口、当前 batch 和 N=1 拆分，仅重放冻结 det 输入。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from analyze_det_fusion import execute_run, load_runs
from benchmark_det_corpus import DEFAULT_CORPUS, ROOT, read_json
from benchmark_mps_det_service import write_json

NAMES = {
    "A": "CPU 原 batch，无融合",
    "E": "MPS 原 batch，融合 36 组",
    "F": "MPS batch 内拆 N=1，融合 36 组",
    "J": "MPS 逐图 3.4 __call__，无融合",
    "K": "MPS 逐图 3.4 __call__，融合 36 组",
}


def method_body(source: str, method: str) -> str:
    """抽取 TextDetector 方法的 AST 主体，忽略中文注释和文档字符串比较真实执行逻辑。"""
    tree = ast.parse(source)
    detector = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TextDetector")
    function = next(node for node in detector.body if isinstance(node, ast.FunctionDef) and node.name == method)
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def inspect_history(output: Path) -> dict[str, Any]:
    """归档 3.4 发布源码并核实单图及共享方法等价，避免将不同模型版本混为入口收益。"""
    target = output / "history"
    target.mkdir(exist_ok=True)
    current = (ROOT / "mineru/model/_internal/pytorchocr/infer/predict_det.py").read_text()
    methods = ("__call__", "_preprocess_det_image", "_build_det_preds", "_postprocess_det_batch", "_batch_process_preprocessed")
    records = []
    for tag in ("mineru-3.4.0-released", "mineru-3.4.5-released"):
        path = "mineru/model/utils/tools/infer/predict_det.py"
        source = subprocess.check_output(["git", "show", f"{tag}:{path}"], cwd=ROOT, text=True)
        (target / f"{tag}-predict_det.py").write_text(source)
        config = subprocess.check_output(
            ["git", "show", f"{tag}:mineru/model/utils/pytorchocr/utils/resources/models_config.yml"], cwd=ROOT, text=True
        )
        (target / f"{tag}-models_config.yml").write_text(config)
        records.append(
            {
                "tag": tag,
                "commit": subprocess.check_output(["git", "rev-parse", tag], cwd=ROOT, text=True).strip(),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "equal_method_bodies": {
                    method: method_body(current, method) == method_body(source, method) for method in methods
                },
                "default_ch_model": "ch_PP-OCRv6_small_det_infer.safetensors",
                "experiment_model": "ch_PP-OCRv6_tiny_det_infer.safetensors",
            }
        )
    if not all(all(record["equal_method_bodies"].values()) for record in records):
        raise ValueError("当前调用链与 3.4 单图方法不等价，需先调查差异，不能直接称为 3.4 入口")
    write_json(output / "history-proof.json", records)
    return {"tags": records, "single_n": 1, "shared_batch_helper": True, "weights_and_fp32_controlled": True}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """配对比较单图融合、batch 融合与 N=1 拆分，验证实际形状、参考框和推理源码指纹。"""
    runs = load_runs(args.output)
    fingerprints = set()
    smoke = {}
    rounds = {}
    fp32, reused, source_stable, shapes_valid = True, True, True, True
    for entry in runs:
        env, rows = entry["environment"], entry["records"]
        fingerprints.add(
            tuple(
                env[key]
                for key in (
                    "python",
                    "torch_commit",
                    "threads",
                    "interop_threads",
                    "model_sha256",
                    "script_sha256",
                    "shape_adapter_sha256",
                    "det_source_sha256_before",
                    "det_source_sha256_after",
                )
            )
        )
        source_stable &= env["det_source_sha256_before"] == env["det_source_sha256_after"]
        fp32 &= all(model["dtype"] == "torch.float32" for row in rows for model in row["models"])
        profiles = {tuple(model["profile"]) for row in rows for model in row["models"]}
        reused &= all(
            len({model["net_id"] for row in rows for model in row["models"] if tuple(model["profile"]) == profile}) == 1
            for profile in profiles
        )
        for row in rows:
            for metric in row["call_metrics"]:
                if entry["policy"] in {"F", "J", "K"}:
                    shapes_valid &= len(metric["network_shapes"]) == metric["shape"][0] and all(
                        shape[0] == 1 and shape[1:] == metric["shape"][1:] for shape in metric["network_shapes"]
                    )
        required = {
            (call["box_thresh"], call["unclip_ratio"])
            for doc in read_json(args.corpus / "manifest.json")["documents"]
            for call in doc["calls"]
        }
        value = {
            "det_seconds": sum(row["seconds"] for row in rows),
            "inputs": sum(row["crops"] for row in rows),
            "exact": all(row["reference_exact_crops"] == row["crops"] for row in rows),
            "differences": sum(len(row["differences"]) for row in rows),
            "profiles_complete": profiles == required,
            "peak_footprint": max(
                *(row["peak_memory_diagnostics"]["phys_footprint"] for row in rows),
                env["shutdown_memory_diagnostics_before"]["phys_footprint"],
                env["shutdown_memory_diagnostics_after"]["phys_footprint"],
            ),
            "forward_calls": sum(len(metric["network_shapes"]) for row in rows for metric in row["call_metrics"]),
            "documents": rows,
            "environment": env,
        }
        loading = sum(model["loading_seconds"] for model in env["model_loads"])
        fusion = sum(model["fusion"]["seconds"] for model in env["model_loads"])
        cleanup = sum(row["cleanup_seconds"] for row in rows) + env["shutdown_seconds"]
        value.update({"loading_seconds": loading, "fusion_seconds": fusion, "cleanup_seconds": cleanup})
        value["operational_seconds"] = value["det_seconds"] + loading + fusion + cleanup
        if entry["directory"] == "smoke":
            smoke[entry["policy"]] = value
        elif entry["directory"].startswith("round-"):
            protocol = entry["protocol"]
            if (
                protocol["cycles"] != 1
                or protocol["limit_calls"]
                or protocol["start_call"]
                or protocol["call_indices"]
                or protocol["save_maps"]
            ):
                raise ValueError("正式对照必须三份文档全量单遍，不能混入诊断")
            if value["inputs"] != 2280:
                raise ValueError("正式对照输入数量不完整")
            rounds.setdefault(entry["directory"], {})[entry["policy"]] = value
    medians = {}
    for policy in NAMES:
        values = [group[policy] for group in rounds.values() if policy in group]
        if values:
            medians[policy] = {
                "det_seconds": statistics.median(value["det_seconds"] for value in values),
                "operational_seconds": statistics.median(value["operational_seconds"] for value in values),
                "peak_footprint": max(value["peak_footprint"] for value in values),
                "outputs_exact": all(value["exact"] for value in values),
            }
    return {
        "smoke": smoke,
        "rounds": rounds,
        "medians": medians,
        "fp32": fp32,
        "models_reused": reused,
        "source_stable": source_stable,
        "matched_execution_environment": len(fingerprints) == 1,
        "n1_shapes_verified": shapes_valid,
    }


def summarize(args: argparse.Namespace) -> None:
    """保存单图入口收益、融合收益和 N 拆分重叠关系，结论只针对当前 tiny/FP32 det。"""
    summary = collect(args)
    medians = summary["medians"]
    rows, docs = [], []
    for name, group in summary["rounds"].items():
        for policy, value in group.items():
            rows.append(
                f"| {name} | {policy} | {value['det_seconds']:.3f} | {value['operational_seconds']:.3f} | "
                f"{value['peak_footprint'] / 1024**3:.3f} | {value['exact']} | {value['forward_calls']} |"
            )
            for row in value["documents"]:
                stages = row["stages"]
                docs.append(
                    f"| {name} | {policy} | {row['document']} | {row['seconds']:.3f} | {stages['preprocess']:.3f} | "
                    f"{stages['forward']:.3f} | {stages['output_transfer']:.3f} | {stages['postprocess']:.3f} | "
                    f"{len(row['shape_counts'])} |"
                )
    comparisons = {
        "single_fusion_reduction_j_to_k": 1 - medians["K"]["det_seconds"] / medians["J"]["det_seconds"],
        "fused_single_reduction_vs_fused_batch": 1 - medians["K"]["det_seconds"] / medians["E"]["det_seconds"],
        "single_seconds_ratio_to_n1_split": medians["K"]["det_seconds"] / medians["F"]["det_seconds"],
        "single_reduction_vs_cpu": 1 - medians["K"]["det_seconds"] / medians["A"]["det_seconds"],
    }
    summary["comparisons"] = comparisons
    write_json(args.output / "summary.json", summary)
    headline = [
        f"| {policy} | {NAMES[policy]} | {value['det_seconds']:.3f} | "
        f"{value['peak_footprint'] / 1024**3:.3f} | {value['outputs_exact']} |"
        for policy, value in medians.items()
    ]
    report = f"""# 3.4 单图入口与 batch/N=1 det 对照

单图入口融合后，相对未融合单图耗时降低 {comparisons["single_fusion_reduction_j_to_k"]:.1%}；
相对融合 batch E 降低 {comparisons["fused_single_reduction_vs_fused_batch"]:.1%}。
K/F 耗时比 {comparisons["single_seconds_ratio_to_n1_split"]:.3f}；相对 CPU A 降低
{comparisons["single_reduction_vs_cpu"]:.1%}（负数表示慢于 CPU）。

3.4.0/3.4.5 的 TextDetector.__call__ 及四个共享方法，与当前方法执行主体 AST 完全相同，证据见 history-proof.json。
单图入口使用共享底层方法，实际每次网络前向 N=1。3.4 默认中文 det 是 small；本实验继续使用同一 tiny 权重及 FP32，
只比较调用路径，不能推导完整 3.4 默认模型/精度/解析流程速度。

融合与调用路径无关：在 CPU/eval/现有 reparam 后使用原融合器，K 剩余 BN=0；J 不融合。
单图入口已经 N=1，无法再通过 N 拆分降低 N；F 与 K 均为 N=1，区别在批量前后处理与逐图前后处理。
F/J/K 不改变 H/W，不补齐空间或虚假样本。实际网络 NCHW 全部保存并验证。

使用 2280 个冻结输入、两组原后处理参数、FP32、生产线程，两轮 A/E/F/J/K 与反序，独立新进程串行。
三文档原顺序，每本文档完整清理，无额外前向预热；模型在进程内复用。
执行环境一致：{summary["matched_execution_environment"]}；det 源码运行前后稳定：{summary["source_stable"]}；
FP32：{summary["fp32"]}；实例复用：{summary["models_reused"]}；N=1 形状核实：{summary["n1_shapes_verified"]}。
实际执行源码在 executed-scripts/，历史源码在 history/。

| 策略 | 说明 | 三文档 det 中位数秒 | 峰 footprint GiB | 与冻结参考框完全一致 |
| --- | --- | ---: | ---: | --- |
{chr(10).join(headline)}

| 轮次 | 策略 | det 秒 | 加载＋融合＋det＋清理秒 | 峰 footprint GiB | 框一致 | 网络前向次数 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
{chr(10).join(rows)}

| 轮次 | 策略 | 文档 | det 秒 | 前处理秒 | 同步 forward 秒 | 回传秒 | 后处理秒 | 唯一 NCHW |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(docs)}

总 det 包括前处理、张量创建、同步 forward、回传、DB 后处理和循环调度。
PNG 读取、采样与结果写盘排除在 det 外。模型加载、融合及清理单独记录。
所有策略沿用冻结原调用结束的同步采样边界；J/K 在该边界内逐张推理，没有额外插入单张内存采样。
采样峰值不代表算子内部瞬时峰值；footprint 包含压缩页记账，与 RSS/driver 重叠，不能相加。全机 swap 单独记录。
本轮不更改生产设备默认、公开接口和原权重，不再测试 H/W 补齐。短测不证明持久服务内存稳定。

```bash
.venv4/bin/python tools/analyze_det_single.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/analyze_det_single.py summarize --output {args.output}
.venv4/bin/python tools/benchmark_det_corpus.py run --policies J,K --cycles 1 --limit-calls 12 --output NEW_SMOKE
```
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"medians": medians, "comparisons": comparisons}, ensure_ascii=False), flush=True)


def experiment(args: argparse.Namespace) -> None:
    """先确认历史单图方法与烟测输出，再按正反两轮独立进程比较五种策略。"""
    if args.output.exists() and any(path.name != "smoke" for path in args.output.iterdir()):
        raise FileExistsError("实验要求全新目录或仅包含本次已完成的 smoke")
    args.output.mkdir(parents=True, exist_ok=True)
    inspect_history(args.output)
    archive = args.output / "executed-scripts"
    archive.mkdir()
    for name in (
        "benchmark_det_corpus.py",
        "benchmark_mps_det_service.py",
        "det_shape_buckets.py",
        "analyze_det_fusion.py",
        Path(__file__).name,
    ):
        shutil.copy2(Path(__file__).parent / name, archive / name)
    if not (args.output / "smoke/protocol.json").exists():
        execute_run(args, "smoke", "A,E,F,J,K", limit_calls=12)
    summary = collect(args)
    if set(summary["smoke"]) != set(NAMES):
        raise ValueError("烟测策略尚未全部完成")
    if not all(row["exact"] and row["profiles_complete"] for row in summary["smoke"].values()):
        write_json(args.output / "smoke-failed.json", summary)
        raise ValueError("烟测框差异或参数覆盖缺失，停止正式计时")
    state = {"phase": "round-1", "det_only": True, "original_weights": True}
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-1", "A,E,F,J,K")
    state["phase"] = "round-2"
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-2", "K,J,F,E,A")
    summarize(args)
    state["phase"] = "complete"
    write_json(args.output / "experiment.json", state)


def main() -> None:
    """提供单图入口实验和独立汇总命令，既有 ABC/融合/形状汇总接口保持不变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("experiment", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    args.output, args.corpus = args.output.resolve(), args.corpus.resolve()
    args.documents = "tender,python,114"
    {"experiment": experiment, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
