"""调度并验收冻结 det 语料的融合实验，不改变生产模型或设备策略。"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from benchmark_det_corpus import read_json, run
from benchmark_mps_det_service import write_json


def execute_run(args: argparse.Namespace, directory: str, policies: str, **changes: Any) -> None:
    """使用同一个 det 重放入口启动独立进程，所有策略始终串行执行。"""
    values = vars(args) | {
        "output": args.output / directory,
        "policies": policies,
        "cycles": 1,
        "limit_calls": 0,
        "start_call": 0,
        "call_indices": "",
        "save_maps": False,
    }
    run(argparse.Namespace(**(values | changes)))


def load_runs(root: Path) -> list[dict[str, Any]]:
    """读取已完成的实验运行，不把概率图诊断混入正式计时和内存验收。"""
    runs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == "diagnostics" or not (directory / "protocol.json").exists():
            continue
        protocol = read_json(directory / "protocol.json")
        for policy in protocol["policies"].split(","):
            target = directory / policy
            if not (target / "environment.json").exists():
                continue
            runs.append(
                {
                    "directory": directory.name,
                    "policy": policy,
                    "protocol": protocol,
                    "records": read_json(target / "results.json"),
                    "environment": read_json(target / "environment.json"),
                }
            )
    return runs


def select_candidate(summary: dict[str, Any], passed_only: bool) -> str | None:
    """从输出和内存合格的候选选最快者，差距不足 5% 时优先融合范围较小的 D。"""
    candidates = summary["candidates"]
    available = [
        policy
        for policy in ("D", "E")
        if policy in candidates and candidates[policy]["research_pass" if passed_only else "eligible"]
    ]
    if not available:
        return None
    winner = min(available, key=lambda policy: candidates[policy]["median_det_seconds"])
    if set(available) == {"D", "E"}:
        slower = max(candidates[policy]["median_det_seconds"] for policy in available)
        faster = min(candidates[policy]["median_det_seconds"] for policy in available)
        if 1 - faster / slower < 0.05:
            return "D"
    return winner


def collect_summary(args: argparse.Namespace) -> dict[str, Any]:
    """按独立进程轮次计算配对中位数、严格框一致性、8 GiB 预算与生命周期门槛。"""
    runs = load_runs(args.output)
    smoke = [entry for entry in runs if entry["directory"].startswith("smoke")]
    formal = [entry for entry in runs if entry["directory"].startswith("round-")]
    lifecycle = [entry for entry in runs if entry["directory"] == "lifecycle"]
    required_profiles = {
        (call["box_thresh"], call["unclip_ratio"])
        for document in read_json(args.corpus / "manifest.json")["documents"]
        for call in document["calls"]
    }
    smoke_checks = {}
    for policy in "ABDE":
        entries = [entry for entry in smoke if entry["policy"] == policy]
        records = [row for entry in entries for row in entry["records"]]
        profiles = {tuple(model["profile"]) for row in records for model in row["models"]}
        smoke_checks[policy] = {
            "exact": bool(records) and all(row["crops"] == row["reference_exact_crops"] for row in records),
            "profiles_complete": profiles == required_profiles,
            "inputs": sum(row["crops"] for row in records),
            "differences": sum(len(row["differences"]) for row in records),
        }
    by_round: dict[str, dict[str, Any]] = {}
    for entry in formal:
        protocol = entry["protocol"]
        if (
            protocol["cycles"] != 1
            or protocol["limit_calls"]
            or protocol["start_call"]
            or protocol["call_indices"]
            or protocol["save_maps"]
        ):
            raise ValueError("正式轮次必须整份语料单遍重放，不能筛选调用、另存 maps 或混入生命周期运行")
        if protocol["documents"] != "tender,python,114":
            raise ValueError("正式文档顺序与约定不同")
        environment = entry["environment"]
        records = entry["records"]
        loading = sum(load["loading_seconds"] for load in environment["model_loads"])
        fusion = sum(load["fusion"]["seconds"] for load in environment["model_loads"])
        det = sum(row["seconds"] for row in records)
        cleanup = sum(row["cleanup_seconds"] for row in records) + environment["shutdown_seconds"]
        peak = max(row["peak_memory_diagnostics"]["phys_footprint"] for row in records)
        peak = max(
            peak,
            environment["shutdown_memory_diagnostics_before"]["phys_footprint"],
            environment["shutdown_memory_diagnostics_after"]["phys_footprint"],
        )
        by_round.setdefault(entry["directory"], {})[entry["policy"]] = {
            "det_seconds": det,
            "loading_seconds": loading,
            "fusion_seconds": fusion,
            "cleanup_seconds": cleanup,
            "operational_seconds": loading + fusion + det + cleanup,
            "peak_footprint": peak,
            "exact": all(row["crops"] == row["reference_exact_crops"] for row in records),
        }
    environments = [entry["environment"] for entry in runs]
    matched_environment = (
        len(
            {
                (
                    environment["model_sha256"],
                    environment["torch_commit"],
                    environment["python"],
                    environment["threads"],
                    environment["interop_threads"],
                    environment["git_head"],
                    environment["script_sha256"],
                )
                for environment in environments
            }
        )
        == 1
        if environments
        else False
    )
    fp32 = all(model["dtype"] == "torch.float32" for entry in runs for row in entry["records"] for model in row["models"])
    models_reused = all(
        len({model["net_id"] for row in entry["records"] for model in row["models"] if tuple(model["profile"]) == profile}) == 1
        for entry in runs
        for profile in {tuple(model["profile"]) for row in entry["records"] for model in row["models"]}
    )
    baseline_exact = all(smoke_checks[policy]["exact"] and smoke_checks[policy]["profiles_complete"] for policy in "AB")
    baseline_exact &= all(value["exact"] for group in by_round.values() for policy, value in group.items() if policy in "AB")
    candidates = {}
    for policy in "DE":
        paired = [group for group in by_round.values() if {"A", "B", policy}.issubset(group)]
        if not paired:
            continue
        seconds = statistics.median(group[policy]["det_seconds"] for group in paired)
        cpu = statistics.median(group["A"]["det_seconds"] for group in paired)
        mps = statistics.median(group["B"]["det_seconds"] for group in paired)
        exact = smoke_checks[policy]["exact"] and all(group[policy]["exact"] for group in paired)
        peak = max(group[policy]["peak_footprint"] for group in paired)
        eligible = exact and peak <= 8 * 1024**3 and matched_environment and fp32 and models_reused and baseline_exact
        improvements = [1 - group[policy]["det_seconds"] / group["B"]["det_seconds"] for group in paired]
        majority = sum(value > 0 for value in improvements) > len(paired) / 2
        reduction = 1 - seconds / mps
        cpu_reduction = 1 - seconds / cpu
        candidates[policy] = {
            "rounds": len(paired),
            "median_det_seconds": seconds,
            "matched_cpu_seconds": cpu,
            "matched_mps_seconds": mps,
            "reduction_vs_b": reduction,
            "reduction_vs_a": cpu_reduction,
            "round_improvements_vs_b": improvements,
            "peak_footprint": peak,
            "outputs_exact": exact,
            "eligible": eligible,
            "research_pass": eligible and len(paired) >= 2 and reduction >= 0.25 and majority,
            "repeat_needed": eligible
            and (
                reduction < 0.1
                or (any(value <= 0 for value in improvements) and any(value > 0 for value in improvements))
                or abs(reduction - 0.25) < 0.05
                or abs(cpu_reduction - 0.25) < 0.05
            ),
            "lifecycle_pass": None,
            "default_candidate": False,
        }
    for entry in lifecycle:
        policy = entry["policy"]
        records = entry["records"]
        if len(records) != 6 or policy not in candidates:
            raise ValueError("生命周期验证必须为候选连续两遍三文档")
        first = records[2]["memory_after"] | {"footprint": records[2]["memory_diagnostics_after"]["phys_footprint"]}
        second = records[5]["memory_after"] | {"footprint": records[5]["memory_diagnostics_after"]["phys_footprint"]}
        growth = {
            key: {
                "first": first[key],
                "second": second[key],
                "delta": second[key] - first[key],
                "limit": max(first[key] * 0.1, 256 * 1024**2),
            }
            for key in first
        }
        peak = max(row["peak_memory_diagnostics"]["phys_footprint"] for row in records)
        exact = all(row["crops"] == row["reference_exact_crops"] for row in records)
        passed = exact and peak <= 8 * 1024**3 and all(value["delta"] <= value["limit"] for value in growth.values())
        candidates[policy].update({"lifecycle_pass": passed, "lifecycle_growth": growth, "lifecycle_peak_footprint": peak})
        candidates[policy]["default_candidate"] = (
            candidates[policy]["research_pass"] and passed and candidates[policy]["reduction_vs_a"] >= 0.25
        )
    return {
        "smoke": smoke_checks,
        "rounds": by_round,
        "candidates": candidates,
        "matched_environment": matched_environment,
        "fp32": fp32,
        "models_reused": models_reused,
        "baseline_exact": baseline_exact,
        "runs": runs,
    }


def diagnose_differences(args: argparse.Namespace, baseline_policies: tuple[str, ...] = ("A", "B")) -> None:
    """按文档集中重放所有差异调用，独立保存 CPU、MPS 基线和候选的概率图。"""
    import numpy as np

    groups: dict[str, dict[str, set[Any]]] = {}
    for entry in load_runs(args.output):
        for row in entry["records"]:
            for difference in row["differences"]:
                group = groups.setdefault(row["document"], {"indices": set(), "policies": set(baseline_policies)})
                group["indices"].add(difference["call_index"])
                group["policies"].add(entry["policy"])
    comparisons = []
    for document, group in groups.items():
        directory = f"diagnostics/{document}"
        indices = sorted(group["indices"])
        policies = sorted(group["policies"])
        execute_run(
            args, directory, ",".join(policies), documents=document, call_indices=",".join(map(str, indices)), save_maps=True
        )
        for index in indices:
            filename = f"01-{document}-call{index:04d}.npz"
            with np.load(args.output / directory / "A/maps" / filename) as saved:
                reference = saved["maps"]
            for policy in policies:
                if policy == "A":
                    continue
                with np.load(args.output / directory / policy / "maps" / filename) as saved:
                    actual = saved["maps"]
                delta = np.abs(actual - reference)
                comparisons.append(
                    {
                        "document": document,
                        "call_index": index,
                        "policy": policy,
                        "shape": list(actual.shape),
                        "finite": bool(np.isfinite(actual).all()),
                        "max_abs": float(delta.max()),
                        "mean_abs": float(delta.mean()),
                        "threshold_changed_pixels": int(np.count_nonzero((actual > 0.3) != (reference > 0.3))),
                    }
                )
    write_json(args.output / "probability-diagnostics.json", comparisons)


def fusion_experiment(args: argparse.Namespace) -> None:
    """先烟测再按门槛执行两轮、必要复测和生命周期验证，失败候选不进入正式测速。"""
    if args.documents != "tender,python,114" or args.start_call or args.limit_calls or args.call_indices or args.save_maps:
        raise ValueError("自动融合实验使用固定完整语料；定点调试请使用 run")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("融合实验必须使用全新的输出目录")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "executed-scripts"
    source.mkdir()
    for name in ("benchmark_det_corpus.py", "analyze_det_fusion.py", "benchmark_mps_det_service.py", "det_shape_buckets.py"):
        shutil.copy2(Path(__file__).parent / name, source / name)
    state: dict[str, Any] = {"phase": "smoke", "corpus": str(args.corpus), "footprint_budget": 8 * 1024**3}
    write_json(args.output / "experiment.json", state)
    execute_run(args, "smoke", "A,B,D,E", limit_calls=12)
    summary = collect_summary(args)
    manifest = read_json(args.corpus / "manifest.json")
    seen = {tuple(model["profile"]) for entry in summary["runs"] for row in entry["records"] for model in row["models"]}
    missing = {(call["box_thresh"], call["unclip_ratio"]) for doc in manifest["documents"] for call in doc["calls"]} - seen
    for number, profile in enumerate(sorted(missing)):
        document, index = next(
            (doc["name"], index)
            for doc in manifest["documents"]
            for index, call in enumerate(doc["calls"])
            if (call["box_thresh"], call["unclip_ratio"]) == profile
        )
        execute_run(args, f"smoke-profile-{number + 1}", "A,B,D,E", documents=document, call_indices=str(index))
    summary = collect_summary(args)
    candidates = [
        policy for policy in "DE" if summary["smoke"][policy]["exact"] and summary["smoke"][policy]["profiles_complete"]
    ]
    if not summary["baseline_exact"]:
        state["phase"] = "baseline_rejected"
    elif not candidates:
        state["phase"] = "candidates_rejected_at_smoke"
    else:
        state.update({"phase": "formal", "formal_candidates": candidates})
        write_json(args.output / "experiment.json", state)
        policies = ["A", "B", *candidates]
        execute_run(args, "round-1", ",".join(policies))
        execute_run(args, "round-2", ",".join(reversed(policies)))
        summary = collect_summary(args)
        best = select_candidate(summary, passed_only=False)
        if best and summary["candidates"][best]["repeat_needed"]:
            execute_run(args, "round-3", f"A,B,{best}")
        summary = collect_summary(args)
        best = select_candidate(summary, passed_only=True)
        if best:
            state["lifecycle_policy"] = best
            execute_run(args, "lifecycle", best, cycles=2)
        state["phase"] = "complete"
    write_json(args.output / "experiment.json", state)
    diagnose_differences(args)
    fusion_summarize(args)


def fusion_summarize(args: argparse.Namespace) -> None:
    """输出独立融合验收报告、融合清单和内存曲线，不覆盖第一轮 ABC 证据。"""
    summary = collect_summary(args)
    state = read_json(args.output / "experiment.json")
    summary["experiment"] = state
    summary["probability_diagnostics"] = (
        read_json(args.output / "probability-diagnostics.json")
        if (args.output / "probability-diagnostics.json").exists()
        else []
    )
    summary["validation"] = read_json(args.output / "validation.json") if (args.output / "validation.json").exists() else None
    write_json(args.output / "summary.json", summary)
    smoke_rows = [
        f"| {policy} | {value['inputs']} | {value['differences']} | {value['exact']} | {value['profiles_complete']} |"
        for policy, value in summary["smoke"].items()
    ]
    rows = []
    for entry in summary["runs"]:
        for row in entry["records"]:
            diagnostic = row["memory_diagnostics_after"]
            rows.append(
                f"| {entry['directory']} | {entry['policy']} | {row['label']} | {row['seconds']:.3f} | "
                f"{row['stages']['forward']:.3f} | {row['peak_memory_diagnostics']['phys_footprint'] / 1024**3:.2f} | "
                f"{diagnostic['phys_footprint'] / 1024**3:.2f} | {row['memory_after']['rss'] / 1024**3:.2f} | "
                f"{row['memory_after']['mps_driver'] / 1024**3:.2f} | {row['reference_exact_crops']}/{row['crops']} |"
            )
    totals = [
        f"| {round_name} | {policy} | {value['det_seconds']:.3f} | {value['loading_seconds']:.3f} | "
        f"{value['fusion_seconds']:.3f} | {value['cleanup_seconds']:.3f} | {value['operational_seconds']:.3f} |"
        for round_name, group in summary["rounds"].items()
        for policy, value in group.items()
    ]
    gates = [
        f"| {policy} | {value['median_det_seconds']:.3f} | {value['reduction_vs_b']:.1%} | {value['reduction_vs_a']:.1%} | "
        f"{value['peak_footprint'] / 1024**3:.2f} | {value['outputs_exact']} | {value['research_pass']} | "
        f"{value['lifecycle_pass']} | {value['default_candidate']} |"
        for policy, value in summary["candidates"].items()
    ]
    fusion = {
        entry["policy"]: entry["environment"]["model_loads"] for entry in summary["runs"] if entry["directory"] == "smoke"
    }
    write_json(args.output / "fusion-inventory.json", fusion)
    if state["phase"] == "candidates_rejected_at_smoke":
        conclusion = "D/E 在烟测阶段未通过严格检测框一致性，因此没有进入正式两轮性能验收。局部烟测耗时不能代表全量收益。"
    elif state["phase"] == "baseline_rejected":
        conclusion = "CPU/MPS 基线或参数覆盖未通过烟测，正式验收暂停；先调查基线，不能归因于融合。"
    else:
        conclusion = "正式对照已完成。研究收益和默认策略候选分别按下表判断；生命周期数据不混入速度中位数。"
        if summary["candidates"]:
            conclusions = []
            for policy, value in summary["candidates"].items():
                conclusions.append(
                    f"{policy} det 中位数 {value['median_det_seconds']:.2f} 秒，比原始 MPS 降低 "
                    f"{value['reduction_vs_b']:.1%}；采样峰 footprint {value['peak_footprint'] / 1024**3:.2f} GiB。"
                )
            conclusion += "\n\n" + "\n".join(conclusions)
            if all(not value["eligible"] for value in summary["candidates"].values()):
                conclusion += "\n\n没有满足严格输出与 8 GiB 预算的候选，未触发追加速度复测或生命周期验证。"
    environment = summary["runs"][0]["environment"]
    differences = sum(len(row["differences"]) for entry in summary["runs"] for row in entry["records"])
    total_inputs = sum(row["crops"] for entry in summary["runs"] for row in entry["records"])
    validation_report = ""
    if summary["validation"]:
        validation_report = (
            "\n## 工具验证\n\n"
            f"旧 ABC 原始记录的聚合、门槛和差异结果逐项不变：{summary['validation']['abc_legacy_summary_unchanged']}。\n"
            "另以真实 Python 第 50 次调用验证 A/B/D/E 的概率图另存路径；各 4 个输入，"
            "float32、形状 (4,1,160,352)、数值有限且框完全一致。\n"
            "该调用的概率值并非逐像素相同，最大偏差约 1.7e-5 至 2.7e-5，"
            "没有像素跨越 0.3 阈值；结果不混入正式速度与内存门槛。\n"
            "详细证据链接保存在 validation.json。\n"
        )
    report = f"""# 每本文档清理条件下的 MPS det 融合实验

{conclusion}

状态：{state["phase"]}。仅改动实验脚本；生产设备默认、公开接口和权重文件保持原样。
语料：{args.corpus}，95/88/114 页 PDF 的 2280 个普通文字 det 输入、2262 张唯一 PNG。
本实验只加载冻结 PNG 和 det；不再解析 PDF，不执行 layout/rec/VLM/导出。
A=CPU 原模型，B=MPS 原模型，D=MPS backbone 34 组融合，E=MPS 全部 36 组融合（包含 head 反卷积）。
全部 FP32、相同权重与生产线程；融合在 CPU/eval/既有 reparam 之后、MPS 迁移之前执行，不预热。
每本文档执行原 clean_memory('mps')，模型按两组后处理参数在同进程复用。
环境及原权重一致：{summary["matched_environment"]}；实际 FP32：{summary["fp32"]}；模型实例复用：{summary["models_reused"]}。
环境：{environment["cpu"]}，{environment["platform"]}，Python {environment["python"]}，Torch {environment["torch"]}。
Torch commit：`{environment["torch_commit"]}`；intra/inter-op 线程：{environment["threads"]}/{environment["interop_threads"]}。
原权重 SHA256：`{environment["model_sha256"]}`；仓库版本：`{environment["git_head"]}`。
详细版本、权重 SHA256、脚本 SHA256 和融合路径在 environment.json / fusion-inventory.json。
executed-scripts/ 保留实际执行时的实验源码；基准脚本与各子进程记录的指纹一致。

## 烟测与一致性

| 策略 | 输入次数 | 差异输入数 | 框完全一致 | 参数覆盖完整 |
| --- | ---: | ---: | --- | --- |
{chr(10).join(smoke_rows)}

检测框按数量、顺序、四点坐标逐项比较，任何变化都判定未通过。
烟测和正式运行共比较 {total_inputs} 次输入，差异输入记录 {differences} 条。
每个差异保留原输入 PNG、参考/候选叠框、参数与原始坐标；位于各策略 review/ 和 results.json。
差异调用另在 diagnostics/ 中以独立进程重放 CPU、原始 MPS、涉及候选，保存原始概率图 npz。
probability-diagnostics.json 保存最大/平均概率偏差及跨 0.3 阈值像素数；这些不是放宽检测框门槛的依据。
无框差异时不产生差异概率图诊断，probability-diagnostics.json 为空列表。

## 正式性能与验收

| 轮次 | 策略 | 三文档 det 秒 | 加载/迁移秒 | 融合秒 | 清理秒（含退出） | 操作合计秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(totals) if totals else "| 未执行 | — | — | — | — | — | — |"}

| 候选 | det 中位数秒 | 比 B 降低 | 比 A 降低 | 采样峰 footprint GiB | 框一致 | 研究门槛 | 生命周期 | 默认策略候选 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
{chr(10).join(gates) if gates else "| 无合格候选 | — | — | — | — | False | False | 未执行 | False |"}

正式第一轮 A→B→D→E，第二轮反向；未通过烟测的候选被剔除。三份文档顺序始终不变，每本均清理图缓存。
速度取各独立轮次三文档合计的中位数；追加轮次使用候选与当轮 A/B 配对计算。
研究门槛：比 B 降低至少 25%、多数轮次同向、严格框一致、采样峰 footprint ≤8 GiB。
默认策略候选还须比 CPU A 降低至少 25%，并通过独立两遍生命周期验证。
生命周期增长门槛：RSS、footprint、MPS allocated/driver 均不超过 max(10%,256 MiB)。
无合格候选时保持 CPU 默认；仅改善 B 而仍慢于 A 时，只认定研究收益。

## 逐文档性能与内存

| 运行 | 策略 | 文档 | det 秒 | forward 秒 | 峰 footprint GiB | 清理后 footprint GiB | RSS GiB | driver GiB | 框一致输入 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

det 包含预处理、设备传输、同步推理、输出回传与 DB 后处理；模型加载/融合、PNG 读取、写出、比较、清理和采样在外。
“操作合计”只包含加载＋融合＋det＋清理，不冒充完整请求或整个 worker 墙钟。
每次调用结束同步采样 footprint/compressed/RSS/allocated/driver，另记录文档清理前后和退出清理前后。
采样峰值不是算子内部瞬时峰值。footprint 包含压缩/换出页记账，与 RSS/driver 重叠，不能相加或当作当前驻留 RAM。
全机 available/swap 为系统指标，不全部归属于 det；详见每次调用 memory_diagnostics。

## 复测

```bash
.venv4/bin/python tools/benchmark_det_corpus.py fusion-experiment --output NEW_OUTPUT
.venv4/bin/python tools/benchmark_det_corpus.py fusion-summarize --output {args.output}
```

新目录保留所有原始证据；原 ABC 默认行为与入口继续可用。新增函数均有中文注释。
本结果不推导完整 Basic/OCR 或 Standard/VLM 性能，没有开展形状归并、预热或 FP16 实验。
{validation_report}

![逐文档 det 与调用期间内存曲线]({args.output}/performance-memory.png)
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for policy in "ABDE":
        entries = [
            entry for entry in summary["runs"] if entry["policy"] == policy and entry["directory"] in ("round-1", "smoke")
        ]
        if not entries:
            continue
        entry = next((entry for entry in entries if entry["directory"] == "round-1"), entries[0])
        rows = entry["records"]
        x = range(1, len(rows) + 1)
        axes[0].plot(x, [row["seconds"] for row in rows], "o-", label=policy)
        footprint = []
        for row in rows:
            footprint.extend(metric["memory_diagnostics"]["phys_footprint"] / 1024**3 for metric in row["call_metrics"])
            footprint.append(row["memory_diagnostics_after"]["phys_footprint"] / 1024**3)
        axes[1].plot(range(1, len(footprint) + 1), footprint, label=policy)
    axes[0].set(xlabel="Document replay index", ylabel="det seconds", title="Formal first round or rejected smoke")
    axes[1].set(
        xlabel="Call samples plus document cleanup boundaries",
        ylabel="Sampled footprint GiB",
        title="Footprint during the first formal round or smoke",
    )
    axes[1].axhline(8, color="red", linestyle="--", label="8 GiB budget")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.savefig(args.output / "performance-memory.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {"phase": state["phase"], "smoke": summary["smoke"], "candidates": summary["candidates"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
