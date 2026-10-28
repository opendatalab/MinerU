"""在融合 E 基线上调度 N/H/W 归并实验，保留严格输出与 8 GiB 内存门槛。"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from analyze_det_fusion import diagnose_differences, execute_run, load_runs
from benchmark_det_corpus import read_json
from benchmark_mps_det_service import write_json
from det_shape_buckets import SHAPE_STRATEGIES, project_shape

NAMES = {
    "A": "CPU 原模型",
    "E": "MPS 全融合，原 N/H/W",
    "F": "MPS 全融合，拆为 N=1",
    "G": "MPS 全融合，拆分并补齐 N=4",
    "H": "MPS 全融合，H/W 补齐到 64 倍数",
    "I": "MPS 全融合，H/W 补齐到 128 倍数",
}


def shape_distribution(args: argparse.Namespace) -> dict[str, Any]:
    """在执行前统计准确网络形状和计算量，避免只凭桶数量估计速度。"""
    documents = read_json(args.corpus / "manifest.json")["documents"]
    result = {}
    for policy in ("E", *SHAPE_STRATEGIES):
        rows = []
        for document in documents:
            projected = [project_shape(call["shape"], policy) for call in document["calls"]]
            shapes = {tuple(shape) for item in projected for shape in item["network_shapes"]}
            rows.append(
                {
                    "document": document["name"],
                    "unique_shapes": len(shapes),
                    "forward_calls": sum(len(item["network_shapes"]) for item in projected),
                    "processed_pixels": sum(item["processed_pixels"] for item in projected),
                    "original_pixels": sum(item["original_pixels"] for item in projected),
                }
            )
        result[policy] = rows
    return result


def collect_shape_summary(args: argparse.Namespace) -> dict[str, Any]:
    """使用同轮 A/E 配对验收新策略，烟测失败或内存超预算都不算有效收益。"""
    runs = load_runs(args.output)
    smoke = [entry for entry in runs if entry["directory"].startswith("smoke")]
    formal = [entry for entry in runs if entry["directory"].startswith("round-")]
    checks = {}
    required = {
        (call["box_thresh"], call["unclip_ratio"])
        for document in read_json(args.corpus / "manifest.json")["documents"]
        for call in document["calls"]
    }
    for policy in NAMES:
        rows = [row for entry in smoke if entry["policy"] == policy for row in entry["records"]]
        profiles = {tuple(model["profile"]) for row in rows for model in row["models"]}
        checks[policy] = {
            "inputs": sum(row["crops"] for row in rows),
            "differences": sum(len(row["differences"]) for row in rows),
            "exact": bool(rows) and all(row["reference_exact_crops"] == row["crops"] for row in rows),
            "profiles_complete": profiles == required,
            "det_seconds": sum(row["seconds"] for row in rows),
        }
    fingerprints = {
        (
            entry["environment"]["model_sha256"],
            entry["environment"]["torch_commit"],
            entry["environment"]["python"],
            entry["environment"]["threads"],
            entry["environment"]["interop_threads"],
            entry["environment"]["git_head"],
            entry["environment"]["script_sha256"],
            entry["environment"]["shape_adapter_sha256"],
        )
        for entry in runs
    }
    matched = len(fingerprints) == 1
    fp32 = all(model["dtype"] == "torch.float32" for entry in runs for row in entry["records"] for model in row["models"])
    reuse = all(
        len({model["net_id"] for row in entry["records"] for model in row["models"] if tuple(model["profile"]) == profile}) == 1
        for entry in runs
        for profile in {tuple(model["profile"]) for row in entry["records"] for model in row["models"]}
    )
    rounds: dict[str, dict[str, Any]] = {}
    for entry in formal:
        protocol = entry["protocol"]
        if (
            protocol["documents"] != "tender,python,114"
            or protocol["cycles"] != 1
            or protocol["limit_calls"]
            or protocol["start_call"]
            or protocol["call_indices"]
            or protocol["save_maps"]
        ):
            raise ValueError("正式形状对照必须完整单遍重放，不能混入诊断或生命周期数据")
        env = entry["environment"]
        rows = entry["records"]
        loading = sum(item["loading_seconds"] for item in env["model_loads"])
        fusion = sum(item["fusion"]["seconds"] for item in env["model_loads"])
        cleanup = sum(row["cleanup_seconds"] for row in rows) + env["shutdown_seconds"]
        det = sum(row["seconds"] for row in rows)
        metrics = [metric for row in rows for metric in row["call_metrics"]]
        rounds.setdefault(entry["directory"], {})[entry["policy"]] = {
            "det_seconds": det,
            "loading_seconds": loading,
            "fusion_seconds": fusion,
            "cleanup_seconds": cleanup,
            "operational_seconds": det + loading + fusion + cleanup,
            "peak_footprint": max(
                *(row["peak_memory_diagnostics"]["phys_footprint"] for row in rows),
                env["shutdown_memory_diagnostics_before"]["phys_footprint"],
                env["shutdown_memory_diagnostics_after"]["phys_footprint"],
            ),
            "exact": all(row["reference_exact_crops"] == row["crops"] for row in rows),
            "forward_calls": sum(len(metric["network_shapes"]) for metric in metrics),
            "pixel_ratio": sum(metric["shape_projection"]["processed_pixels"] for metric in metrics)
            / sum(metric["shape_projection"]["original_pixels"] for metric in metrics),
            "bucket_padding_seconds": sum(metric["bucket_padding_seconds"] for metric in metrics),
        }
    baseline = all(checks[policy]["exact"] and checks[policy]["profiles_complete"] for policy in ("A", "E"))
    baseline &= all(value["exact"] for group in rounds.values() for policy, value in group.items() if policy in ("A", "E"))
    candidates = {}
    for policy in SHAPE_STRATEGIES:
        paired = [group for group in rounds.values() if {"A", "E", policy}.issubset(group)]
        if not paired:
            continue
        seconds = statistics.median(group[policy]["det_seconds"] for group in paired)
        cpu = statistics.median(group["A"]["det_seconds"] for group in paired)
        original = statistics.median(group["E"]["det_seconds"] for group in paired)
        peak = max(group[policy]["peak_footprint"] for group in paired)
        exact = checks[policy]["exact"] and all(group[policy]["exact"] for group in paired)
        gains = [1 - group[policy]["det_seconds"] / group["E"]["det_seconds"] for group in paired]
        reduction = 1 - seconds / original
        cpu_reduction = 1 - seconds / cpu
        eligible = matched and fp32 and reuse and baseline and exact and peak <= 8 * 1024**3
        candidates[policy] = {
            "rounds": len(paired),
            "median_det_seconds": seconds,
            "matched_e_seconds": original,
            "matched_cpu_seconds": cpu,
            "reduction_vs_e": reduction,
            "reduction_vs_a": cpu_reduction,
            "round_gains": gains,
            "peak_footprint": peak,
            "outputs_exact": exact,
            "eligible": eligible,
            "research_pass": eligible
            and len(paired) >= 2
            and reduction >= 0.25
            and sum(gain > 0 for gain in gains) > len(gains) / 2,
            "repeat_needed": eligible
            and (
                reduction < 0.1
                or (any(gain <= 0 for gain in gains) and any(gain > 0 for gain in gains))
                or abs(reduction - 0.25) < 0.05
                or abs(cpu_reduction - 0.25) < 0.05
            ),
            "lifecycle_pass": None,
            "default_candidate": False,
        }
    for entry in runs:
        if entry["directory"] != "lifecycle":
            continue
        policy = entry["policy"]
        rows = entry["records"]
        if len(rows) != 6 or policy not in candidates:
            raise ValueError("生命周期要求候选连续两遍三文档")
        before = rows[2]["memory_after"] | {"footprint": rows[2]["memory_diagnostics_after"]["phys_footprint"]}
        after = rows[5]["memory_after"] | {"footprint": rows[5]["memory_diagnostics_after"]["phys_footprint"]}
        growth = {
            key: {
                "delta": after[key] - before[key],
                "limit": max(before[key] * 0.1, 256 * 1024**2),
                "first": before[key],
                "second": after[key],
            }
            for key in before
        }
        peak = max(row["peak_memory_diagnostics"]["phys_footprint"] for row in rows)
        passed = (
            all(value["delta"] <= value["limit"] for value in growth.values())
            and all(row["crops"] == row["reference_exact_crops"] for row in rows)
            and peak <= 8 * 1024**3
        )
        candidates[policy].update(
            {
                "lifecycle_pass": passed,
                "lifecycle_peak_footprint": peak,
                "lifecycle_growth": growth,
                "default_candidate": passed
                and candidates[policy]["research_pass"]
                and candidates[policy]["reduction_vs_a"] >= 0.25,
            }
        )
    return {
        "smoke": checks,
        "rounds": rounds,
        "candidates": candidates,
        "baseline_exact": baseline,
        "matched_environment": matched,
        "fp32": fp32,
        "models_reused": reuse,
        "runs": runs,
    }


def shape_experiment(args: argparse.Namespace) -> None:
    """按烟测、正反两轮、必要复测和生命周期顺序验收形状归并，输出失败者仅保留诊断证据。"""
    if args.documents != "tender,python,114" or args.start_call or args.limit_calls or args.call_indices or args.save_maps:
        raise ValueError("自动形状实验使用固定完整语料，定点调用请使用 run")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("请用全新输出目录保留上一轮证据")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "executed-scripts"
    source.mkdir()
    for name in (
        "benchmark_det_corpus.py",
        "benchmark_mps_det_service.py",
        "analyze_det_fusion.py",
        "analyze_det_shapes.py",
        "det_shape_buckets.py",
    ):
        shutil.copy2(Path(__file__).parent / name, source / name)
    write_json(args.output / "shape-distribution.json", shape_distribution(args))
    state: dict[str, Any] = {"phase": "smoke", "footprint_budget": 8 * 1024**3, "baseline": "E", "corpus": str(args.corpus)}
    write_json(args.output / "experiment.json", state)
    execute_run(args, "smoke", "A,E,F,G,H,I", limit_calls=12)
    summary = collect_shape_summary(args)
    if not all(value["profiles_complete"] for value in summary["smoke"].values()):
        manifest = read_json(args.corpus / "manifest.json")
        seen = {tuple(model["profile"]) for entry in summary["runs"] for row in entry["records"] for model in row["models"]}
        missing = {(call["box_thresh"], call["unclip_ratio"]) for doc in manifest["documents"] for call in doc["calls"]} - seen
        for number, profile in enumerate(sorted(missing)):
            doc, index = next(
                (doc["name"], index)
                for doc in manifest["documents"]
                for index, call in enumerate(doc["calls"])
                if (call["box_thresh"], call["unclip_ratio"]) == profile
            )
            execute_run(args, f"smoke-profile-{number + 1}", "A,E,F,G,H,I", documents=doc, call_indices=str(index))
    summary = collect_shape_summary(args)
    candidates = [
        policy
        for policy in SHAPE_STRATEGIES
        if summary["smoke"][policy]["exact"] and summary["smoke"][policy]["profiles_complete"]
    ]
    if not summary["baseline_exact"]:
        state["phase"] = "baseline_rejected"
    elif not candidates:
        state["phase"] = "candidates_rejected_at_smoke"
    else:
        state.update({"phase": "formal", "formal_candidates": candidates})
        write_json(args.output / "experiment.json", state)
        policies = ["A", "E", *candidates]
        execute_run(args, "round-1", ",".join(policies))
        execute_run(args, "round-2", ",".join(reversed(policies)))
        summary = collect_shape_summary(args)
        valid = [policy for policy, value in summary["candidates"].items() if value["eligible"]]
        best = min(valid, key=lambda policy: summary["candidates"][policy]["median_det_seconds"]) if valid else None
        if best and summary["candidates"][best]["repeat_needed"]:
            execute_run(args, "round-3", f"A,E,{best}")
        summary = collect_shape_summary(args)
        passed = [policy for policy, value in summary["candidates"].items() if value["research_pass"]]
        if passed:
            best = min(passed, key=lambda policy: summary["candidates"][policy]["median_det_seconds"])
            state["lifecycle_policy"] = best
            execute_run(args, "lifecycle", best, cycles=2)
        state["phase"] = "complete"
    write_json(args.output / "experiment.json", state)
    diagnose_differences(args, baseline_policies=("A", "E"))
    shape_summarize(args)


def shape_summarize(args: argparse.Namespace) -> None:
    """保存形状收益、真实算量、框差异和内存门槛，区分烟测淘汰与正式性能结论。"""
    summary = collect_shape_summary(args)
    summary["experiment"] = read_json(args.output / "experiment.json")
    summary["distribution"] = read_json(args.output / "shape-distribution.json")
    summary["probability_diagnostics"] = (
        read_json(args.output / "probability-diagnostics.json")
        if (args.output / "probability-diagnostics.json").exists()
        else []
    )
    write_json(args.output / "summary.json", summary)
    smoke_rows = []
    for policy, value in summary["smoke"].items():
        smoke_rows.append(
            f"| {policy} | {NAMES[policy]} | {value['inputs']} | {value['differences']} | "
            f"{value['exact']} | {value['det_seconds']:.3f} |"
        )
    distribution_rows = []
    for policy, rows in summary["distribution"].items():
        for row in rows:
            distribution_rows.append(
                f"| {policy} | {row['document']} | {row['unique_shapes']} | {row['forward_calls']} | "
                f"{row['processed_pixels'] / row['original_pixels']:.3f} |"
            )
    performance = []
    for round_name, group in summary["rounds"].items():
        for policy, value in group.items():
            performance.append(
                f"| {round_name} | {policy} | {value['det_seconds']:.3f} | {value['operational_seconds']:.3f} | "
                f"{value['peak_footprint'] / 1024**3:.2f} | {value['forward_calls']} | "
                f"{value['pixel_ratio']:.3f} | {value['exact']} |"
            )
    gates = []
    for policy, value in summary["candidates"].items():
        gates.append(
            f"| {policy} | {value['median_det_seconds']:.3f} | {value['reduction_vs_e']:.1%} | {value['reduction_vs_a']:.1%} | "
            f"{value['peak_footprint'] / 1024**3:.2f} | {value['outputs_exact']} | {value['research_pass']} | "
            f"{value['lifecycle_pass']} | {value['default_candidate']} |"
        )
    document_rows = []
    for entry in summary["runs"]:
        for row in entry["records"]:
            document_rows.append(
                f"| {entry['directory']} | {entry['policy']} | {row['label']} | {row['seconds']:.3f} | "
                f"{len(row['shape_counts'])} | {row['peak_memory_diagnostics']['phys_footprint'] / 1024**3:.2f} | "
                f"{row['reference_exact_crops']}/{row['crops']} |"
            )
    env = summary["runs"][0]["environment"]
    failed = [policy for policy in SHAPE_STRATEGIES if not summary["smoke"][policy]["exact"]]
    conclusions = []
    for policy, value in summary["candidates"].items():
        conclusions.append(
            f"{policy}：det 中位数 {value['median_det_seconds']:.2f} 秒，相对 E 降低 {value['reduction_vs_e']:.1%}，"
            f"采样峰 footprint {value['peak_footprint'] / 1024**3:.2f} GiB，框一致 {value['outputs_exact']}。"
        )
        if value["lifecycle_pass"] is False:
            growth = value["lifecycle_growth"]
            conclusions.append(
                f"{policy} 生命周期未通过：结束 footprint 从 {growth['footprint']['first'] / 1024**3:.2f} "
                f"增至 {growth['footprint']['second'] / 1024**3:.2f} GiB，"
                f"RSS 增长 {growth['rss']['delta'] / 1024**3:.2f} GiB；allocated/driver 的增长见 summary.json。"
            )
    lifetime_rows = []
    for entry in summary["runs"]:
        if entry["directory"] == "lifecycle":
            for row in entry["records"]:
                diagnostic = row["memory_diagnostics_after"]
                lifetime_rows.append(
                    f"| {entry['policy']} | {row['label']} | {row['seconds']:.3f} | "
                    f"{row['peak_memory_diagnostics']['phys_footprint'] / 1024**3:.2f} | "
                    f"{diagnostic['phys_footprint'] / 1024**3:.2f} | {row['memory_after']['rss'] / 1024**3:.2f} | "
                    f"{row['memory_after']['mps_driver'] / 1024**3:.3f} |"
                )
    difference_counts = {}
    unchanged_spatial_differences = 0
    for entry in summary["runs"]:
        for row in entry["records"]:
            by_call = {metric["call_index"]: metric for metric in row["call_metrics"]}
            for difference in row["differences"]:
                counter = difference_counts.setdefault(entry["policy"], {"inputs": 0, "box_count_changes": 0})
                counter["inputs"] += 1
                counter["box_count_changes"] += len(difference["reference"] or []) != len(difference["actual"] or [])
                if entry["policy"] in ("H", "I"):
                    metric = by_call[difference["call_index"]]
                    unchanged_spatial_differences += metric["shape"][2:] == metric["network_shapes"][0][2:]
    write_json(args.output / "difference-counts.json", difference_counts)
    count_description = (
        "；".join(
            f"{policy} 有 {value['inputs']} 个差异输入，其中 {value['box_count_changes']} 个改变框数量"
            for policy, value in difference_counts.items()
        )
        or "没有检测框差异"
    )
    defaults = [policy for policy, value in summary["candidates"].items() if value["default_candidate"]]
    default_description = (
        f"下一轮默认策略验证候选：{defaults}。" if defaults else "本轮没有满足默认策略要求的候选，保持 CPU 默认。"
    )
    report = f"""# 融合 E 基线上的 det 形状归并

{chr(10).join(conclusions)}

状态：{summary["experiment"]["phase"]}。烟测框一致性失败的候选：{failed}，这些策略不进入正式性能验收。
仅改实验脚本；使用冻结的 2280 个普通文字 det 输入，不再解析 PDF。生产默认和原权重不变。
沿用完全一致的检测框门槛与 8 GiB 采样峰 footprint 预算；研究收益要求相对 E 至少降低 25%。
全部策略独立进程串行，正式两轮顺序互逆，每本文档执行原 MPS 清理；不保留跨文档图缓存、不预热。
环境一致：{summary["matched_environment"]}，FP32：{summary["fp32"]}，模型复用：{summary["models_reused"]}。
{env["cpu"]} / {env["platform"]} / Python {env["python"]} / Torch {env["torch"]}。
intra/inter-op {env["threads"]}/{env["interop_threads"]}。
原权重 SHA256：`{env["model_sha256"]}`。实际执行源码保存在 executed-scripts/。

## 实现和烟测

E/F/G/H/I 都融合同样的 36 组 Conv/BN，A 为 CPU 原模型。形状阶段的对照是 E，不将上一轮融合收益计为归并收益。
保留原 resize、归一化与原始坐标信息；F/G 只改变 N，G 用最后一个真实样本补齐并丢弃额外输出。
H/I 在归一化后的 CPU 输入右侧/底部补 0，不改变原图像素；先将输出回传 CPU，再裁切到原 H/W，最后执行原 DB 后处理。
Tiny 模型包含全局池化和 SE，裁切输出不能保证空间补齐等价；通过逐项检测框比较裁决。
记录原调用与实际网络形状、拆批次数、实际像素算量和补齐耗时（已计入 det）。原调用输出顺序不变。

| 策略 | 方案 | 烟测输入 | 差异输入 | 框一致 | 烟测 det 秒 |
| --- | --- | ---: | ---: | --- | ---: |
{chr(10).join(smoke_rows)}

## 全量形状预估

| 策略 | 文档 | 唯一 NCHW | forward 次数 | 像素算量相对 E |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(distribution_rows)}

## 正式性能与验收

| 轮次 | 策略 | det 秒 | 加载＋融合＋det＋清理秒 | 峰 footprint GiB | forward 次数 | 像素算量比 | 框一致 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(performance) if performance else "| 未执行 | — | — | — | — | — | — | — |"}

| 候选 | det 中位数秒 | 相对 E 降低 | 相对 A 降低 | 峰 footprint GiB | 框一致 | 研究门槛 | 生命周期 | 默认策略候选 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
{chr(10).join(gates) if gates else "| 无合格候选 | — | — | — | — | False | False | 未执行 | False |"}

速度取独立轮次三文档合计中位数；研究门槛为输出严格一致、≤8 GiB、相对 E 降低 ≥25% 且多数轮次同向。
通过者另作同进程两遍三文档生命周期验证；清理后 RSS/footprint/allocated/driver 增长均须 ≤max(10%,256 MiB)。
相对 CPU A 降低 ≥25% 且通过生命周期才可成为下一轮默认策略候选。
{default_description} 生命周期的内存增长不能直接等同于已确认泄漏。

## 生命周期

| 策略 | 文档 | det 秒 | 峰 footprint GiB | 清理后 footprint GiB | 清理后 RSS GiB | 清理后 driver GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lifetime_rows) if lifetime_rows else "| 未执行 | — | — | — | — | — | — |"}

## 逐文档记录和差异证据

| 运行 | 策略 | 文档 | det 秒 | 实际唯一 NCHW | 峰 footprint GiB | 框一致输入 |
| --- | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(document_rows)}

所有差异的原 PNG、坐标和叠框位于各策略 review/ 与 results.json；CPU、E 及涉及候选的差异调用概率图保存在 diagnostics/。
probability-diagnostics.json 给出概率图最大/平均误差和跨 0.3 阈值像素数，仅用于诊断，不能放宽框一致性门槛。
difference-counts.json 另区分坐标变化与框数量变化：{count_description}。
H/I 未改变 H/W 的输入发生框差异 {unchanged_spatial_differences} 次；裁切最终概率图不能保证消除 padding 的影响。
det 包含真实前处理、补齐/拆批、张量传输、同步网络执行、输出恢复及后处理。读 PNG、写结果、采样与清理在外。
采样峰值不是内部瞬时峰值；footprint 包含压缩页记账，与 RSS/driver 重叠。全机 swap 不能全部归属 det。
仅代表 det 阶段，不推导完整 Basic/OCR 或 Standard/VLM。

```bash
.venv4/bin/python tools/benchmark_det_corpus.py shape-experiment --output NEW_OUTPUT
.venv4/bin/python tools/benchmark_det_corpus.py shape-summarize --output {args.output}
```

![det 性能与 footprint]({args.output}/performance-memory.png)
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    plotted_phase = "round-1" if summary["rounds"] else "smoke"
    for policy in NAMES:
        entries = [entry for entry in summary["runs"] if entry["policy"] == policy and entry["directory"] == plotted_phase]
        if not entries:
            continue
        chosen = entries[0]
        rows = chosen["records"]
        axes[0].plot(range(1, len(rows) + 1), [row["seconds"] for row in rows], "o-", label=f"{policy} ({chosen['directory']})")
        samples = [metric["memory_diagnostics"]["phys_footprint"] / 1024**3 for row in rows for metric in row["call_metrics"]]
        axes[1].plot(range(1, len(samples) + 1), samples, label=policy)
    axes[0].set(xlabel="Document replay index", ylabel="det seconds", title=f"det performance: {plotted_phase}")
    axes[1].set(xlabel="Original call index", ylabel="Sampled footprint GiB", title="Memory accounting")
    axes[1].axhline(8, color="red", linestyle="--", label="8 GiB budget")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.savefig(args.output / "performance-memory.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {"phase": summary["experiment"]["phase"], "smoke": summary["smoke"], "candidates": summary["candidates"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
