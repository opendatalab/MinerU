"""比较原 CPU batch、仅关闭 NNPACK 和 N≥16 拆分，不修改生产默认。"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from analyze_det_cpu import collect as collect_cpu
from analyze_det_fusion import execute_run
from benchmark_det_corpus import DEFAULT_CORPUS, read_json, worker
from benchmark_mps_det_service import write_json

NAMES = {"A": "原 batch / NNPACK 开", "V": "原 batch / NNPACK 关", "R": "N≥16 拆成 N≤8 / NNPACK 开"}
ROUNDS = {"round-1": "A,V,R", "round-2": "R,A,V", "repeat": "V,R,A"}
NNPACK_DOC = "https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.nnpack"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """独立逐框复核，并断言设备、线程、融合及 NNPACK 状态符合唯一变量对照。"""
    summary = collect_cpu(args)
    for group in summary["groups"].values():
        for policy, value in group.items():
            assert policy in NAMES
            env = value["environment"]
            assert env["nnpack_enabled"] == env["nnpack_enabled_at_exit"] == (policy != "V")
            for load in env["model_loads"]:
                assert load["fusion"]["before_bn"] == load["fusion"]["after_bn"] == 36
                assert not load["fusion"]["pairs"] and load["fusion"]["seconds"] == 0
            if policy in {"A", "V"}:
                for row in value["documents"]:
                    assert all(call["network_shapes"] == [call["shape"]] for call in row["call_metrics"])
    return summary


def background_snapshot(output: Path, label: str) -> None:
    """保存轮次边界的后台进程占用快照，仅记录负载，不改变其他应用。"""
    process = subprocess.run(["ps", "-axo", "pid,pcpu,pmem,etime,comm"], capture_output=True, text=True, check=True)
    lines = process.stdout.splitlines()
    ordered = sorted(lines[1:], key=lambda line: float(line.split()[1]), reverse=True)
    record = {"time": datetime.now().astimezone().isoformat(), "label": label, "processes": ordered[:20]}
    with (output / "background-load.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def profile_route(args: argparse.Namespace) -> None:
    """独立定点追踪真实 N=16 调用的卷积选路，诊断数据不参与正式测速。"""
    import torch

    from mineru.model.runtime.cpu_threads import initialize_torch_threads

    initialize_torch_threads()
    manifest = read_json(args.corpus / "manifest.json")
    document = next(row for row in manifest["documents"] if row["name"] == "tender")
    index = next(index for index, call in enumerate(document["calls"]) if call["shape"][0] == 16)
    measured = argparse.Namespace(
        corpus=args.corpus,
        output=args.output,
        policy=args.policy,
        documents="tender",
        cycles=1,
        start_call=0,
        limit_calls=0,
        call_indices=str(index),
        save_maps=False,
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        worker(measured)
    operators = [
        {"operator": event.key, "count": event.count, "self_cpu_us": event.self_cpu_time_total}
        for event in profile.key_averages()
        if any(name in event.key.lower() for name in ("nnpack", "slow_conv", "thnn", "winograd", "mkldnn"))
    ]
    target = args.output / args.policy
    write_json(target / "operators.json", {"call_index": index, "operators": operators})
    profile.export_chrome_trace(str(target / "cpu-trace.json"))
    rows = read_json(target / "results.json")
    assert all(row["reference_exact_crops"] == row["crops"] for row in rows)
    uses_nnpack = any(row["operator"] == "aten::_nnpack_spatial_convolution" for row in operators)
    assert uses_nnpack == (args.policy == "A"), "实际 NNPACK 卷积选路与预期不同"
    print(json.dumps({"policy": args.policy, "call_index": index, "operators": operators}), flush=True)


def summarize(args: argparse.Namespace) -> None:
    """汇总三轮平衡顺序的独立进程中位数、N 分组收益、内存及严格框一致性。"""
    summary = collect(args)
    groups = summary["groups"]
    medians: dict[str, Any] = {}
    details, documents, strata = [], [], []
    for policy in NAMES:
        values = [groups[name][policy] for name in ROUNDS]
        medians[policy] = {
            "det_seconds": statistics.median(value["det_seconds"] for value in values),
            "operational_seconds": statistics.median(value["operational_seconds"] for value in values),
            "peak_footprint": max(value["peak_footprint"] for value in values),
            "peak_rss": max(row["peak_memory"]["rss"] for value in values for row in value["documents"]),
            "outputs_exact": all(value["exact"] for value in values),
            "inputs_checked": sum(value["inputs"] for value in values),
            "per_round_reductions_vs_a": [
                1 - groups[name][policy]["det_seconds"] / groups[name]["A"]["det_seconds"] for name in ROUNDS
            ],
        }
        for n_range in ("N<16", "N≥16"):
            totals = [
                sum(
                    call["seconds"]
                    for row in value["documents"]
                    for call in row["call_metrics"]
                    if (call["shape"][0] >= 16) == (n_range == "N≥16")
                )
                for value in values
            ]
            medians[policy][n_range] = {"round_seconds": totals, "median_seconds": statistics.median(totals)}
            strata.append(
                f"| {policy} | {n_range} | {' / '.join(f'{value:.3f}' for value in totals)} | {statistics.median(totals):.3f} |"
            )
    for value in medians.values():
        value["reduction_vs_a"] = 1 - value["det_seconds"] / medians["A"]["det_seconds"]
        value["reduction_vs_r"] = 1 - value["det_seconds"] / medians["R"]["det_seconds"]
    for name in ROUNDS:
        assert set(groups[name]) == set(NAMES)
        for policy, value in groups[name].items():
            details.append(
                f"| {name} | {policy} | {value['det_seconds']:.3f} | {value['loading_seconds']:.3f} | "
                f"{value['cleanup_seconds']:.3f} | {value['operational_seconds']:.3f} | {value['forward_calls']} |"
            )
            for row in value["documents"]:
                stages = row["stages"]
                documents.append(
                    f"| {name} | {policy} | {row['document']} | {row['seconds']:.3f} | {stages['preprocess']:.3f} | "
                    f"{stages['forward']:.3f} | {stages['output_transfer']:.3f} | {stages['postprocess']:.3f} |"
                )
    summary.update({"medians": medians, "primary_rounds": list(ROUNDS), "orders": ROUNDS})
    write_json(args.output / "summary.json", summary)
    per_round_display = {
        policy: " / ".join(format(groups[name][policy]["det_seconds"], ".3f") for name in ROUNDS) for policy in NAMES
    }
    headlines = [
        f"| {policy} {NAMES[policy]} | {per_round_display[policy]} | "
        f"{value['det_seconds']:.3f} | {value['reduction_vs_a']:.1%} | {value['reduction_vs_r']:.1%} | "
        f"{value['peak_footprint'] / 1024**3:.3f} | {value['peak_rss'] / 1024**3:.3f} | {value['outputs_exact']} |"
        for policy, value in medians.items()
    ]
    env = groups["round-1"]["A"]["environment"]
    report = f"""# 本机 CPU det：仅关闭 NNPACK

复用三份冻结文档（95/88/114 页），每遍 2280 个输入、638 次原调用；不重新解析 PDF。
同一 Tiny 权重、CPU FP32、生产线程 4/10、两组后处理模型复用，每本文档原清理。
环境：{env["cpu"]}；Python {env["python"]}；PyTorch {env["torch"]}（{env["torch_commit"]}）。
权重 SHA256：`{env["model_sha256"]}`；冻结 manifest SHA256：`{summary["manifest_sha256"]}`。
A/V 使用完全相同的原 batch 方法，唯一干预为 V 在实验子进程调用 `torch.backends.nnpack.set_flags(False)`。
R 为上一轮 N≥16 才拆成 N≤8 的对照；三种策略均无融合、BN=36，无 H/W 补齐或额外前向预热。
三轮顺序 A→V→R、R→A→V、V→R→A，各策略在每个运行位置出现一次，均为独立新进程串行。
repeat/ 是第三轮完整数据，三轮均纳入中位数。后台 CPU 有占用，边界快照见 background-load.jsonl；并非空闲机器校准。

| 策略 | 三轮 det 秒 | 中位数秒 | 相对 A 耗时降低 | 相对 R 耗时降低 | 峰 footprint GiB | 峰 RSS GiB | 框完全一致 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(headlines)}

| 轮次 | 策略 | det 秒 | 加载秒 | 清理秒 | 加载＋det＋清理秒 | 前向次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(details)}

| 轮次 | 策略 | 文档 | det 秒 | 前处理秒 | 同步 forward 秒 | 回传秒 | 后处理秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(documents)}

| 策略 | 原 N 分组 | 三轮 det 秒 | 中位数秒 |
| --- | --- | --- | ---: |
{chr(10).join(strata)}

全量逐项复核检测框数量、顺序、四点坐标，模型实例复用、实际 NCHW、FP32、线程和源码/权重指纹。
每策略三轮共 6840 个输入；差异输入、参数、原始/候选框及叠框由原重放器保存在 review/。
定点 profiler 对 tender 首个 N=16 调用另存 diagnostics/，确认 A 使用 NNPACK，V/R 不使用；诊断开销不混入正式测速。
PNG 读取、模型加载、清理、同步内存采样和结果保存排除在 det 外；det 包括前处理、张量准备、网络执行和 DB 后处理。
内存为调用结束及清理边界采样；采样峰不是算子内部瞬时峰值。footprint/RSS/compressed 不相加，swap 是全机值。
NNPACK 开关是进程全局设置，其他 CPU 模型会共享该状态：[PyTorch 文档]({NNPACK_DOC})。
本轮只验证本机 det；未修改生产默认，不推导其他模型、CPU 平台、MPS/CUDA 或完整解析收益。

```bash
.venv4/bin/python tools/analyze_det_nnpack.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/analyze_det_nnpack.py summarize --output {args.output}
.venv4/bin/python tools/benchmark_det_corpus.py run --policies A,V --cycles 1 --output NEW_OUTPUT
```
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(medians, ensure_ascii=False), flush=True)


def experiment(args: argparse.Namespace) -> None:
    """先烟测和独立选路诊断，再运行三轮平衡顺序的全量对照，全部证据另存新目录。"""
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "executed-scripts"
    archive.mkdir()
    for name in (
        "benchmark_det_corpus.py",
        "benchmark_mps_det_service.py",
        "det_shape_buckets.py",
        "det_cpu_microbatch.py",
        "analyze_det_fusion.py",
        "analyze_det_cpu.py",
        Path(__file__).name,
    ):
        shutil.copy2(Path(__file__).parent / name, archive / name)
    background_snapshot(args.output, "before-smoke")
    execute_run(args, "smoke", ",".join(NAMES), limit_calls=12)
    smoke = collect(args)["groups"]["smoke"]
    assert all(value["exact"] for value in smoke.values()), "烟测检测框不一致，暂停正式测速"
    for policy in NAMES:
        with (args.output / f"profile-{policy}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "profile",
                    "--policy",
                    policy,
                    "--corpus",
                    str(args.corpus),
                    "--output",
                    str(args.output / "diagnostics"),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    for name, policies in ROUNDS.items():
        write_json(args.output / "experiment.json", {"phase": name, "policies": policies})
        background_snapshot(args.output, f"before-{name}")
        execute_run(args, name, policies)
        background_snapshot(args.output, f"after-{name}")
        collect(args)
    summarize(args)
    write_json(args.output / "experiment.json", {"phase": "complete", "orders": ROUNDS})


def main() -> None:
    """提供仅关闭 NNPACK 的实验、汇总及独立 profiler 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("experiment", "summarize", "profile"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--policy", choices=tuple(NAMES), default="V")
    args = parser.parse_args()
    args.output, args.corpus = args.output.resolve(), args.corpus.resolve()
    args.documents = "tender,python,114"
    {"experiment": experiment, "summarize": summarize, "profile": profile_route}[args.command](args)


if __name__ == "__main__":
    main()
