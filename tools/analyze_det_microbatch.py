"""在本机冻结 det 语料上验收无融合 CPU 阈值拆分及 microbatch。"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from analyze_det_cpu import collect as collect_cpu
from analyze_det_fusion import execute_run
from benchmark_det_corpus import DEFAULT_CORPUS, read_json
from benchmark_mps_det_service import write_json
from det_cpu_microbatch import CPU_MICROBATCH_STRATEGIES

NAMES = {
    "A": "原 batch",
    "M": "全部拆 N=1（上一轮实现）",
    "O": "N≥16 才拆，每批 1",
    "P": "N≥16 才拆，每批 2",
    "Q": "N≥16 才拆，每批 4",
    "R": "N≥16 才拆，每批 8",
    "S": "全局 microbatch 上限 2",
    "T": "全局 microbatch 上限 4",
    "U": "全局 microbatch 上限 8",
}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """复用逐框独立复核，并额外保证本轮所有模型保持 36 个 BN、融合成本为零。"""
    summary = collect_cpu(args)
    for group in summary["groups"].values():
        for policy, value in group.items():
            if policy not in NAMES:
                raise ValueError(f"混入本轮范围之外的策略 {policy}")
            for load in value["environment"]["model_loads"]:
                assert load["fusion"]["before_bn"] == load["fusion"]["after_bn"] == 36
                assert not load["fusion"]["pairs"] and load["fusion"]["seconds"] == 0
    return summary


def summarize(args: argparse.Namespace) -> None:
    """以完整正反两轮的中位数排名，必要补测单列，不混入其他策略的比较样本。"""
    summary = collect(args)
    groups = summary["groups"]
    formal = {name: groups[name] for name in ("round-1", "round-2")}
    policies = list(formal["round-1"])
    assert set(formal["round-2"]) == set(policies)
    medians = {}
    for policy in policies:
        values = [group[policy] for group in formal.values()]
        seconds = statistics.median(value["det_seconds"] for value in values)
        peak = max(value["peak_footprint"] for value in values)
        exact = all(value["exact"] for value in values)
        medians[policy] = {
            "det_seconds": seconds,
            "operational_seconds": statistics.median(value["operational_seconds"] for value in values),
            "peak_footprint": peak,
            "outputs_exact": exact,
            "eligible": exact and peak <= 8 * 1024**3,
            "per_round_reductions": [1 - group[policy]["det_seconds"] / group["A"]["det_seconds"] for group in formal.values()],
        }
    for value in medians.values():
        value["reduction_vs_a"] = 1 - value["det_seconds"] / medians["A"]["det_seconds"]
        value["reduction_vs_m"] = 1 - value["det_seconds"] / medians["M"]["det_seconds"]
    candidates = [policy for policy in policies if policy in CPU_MICROBATCH_STRATEGIES and medians[policy]["eligible"]]
    ranking = sorted(candidates, key=lambda policy: medians[policy]["det_seconds"])
    summary.update({"medians": medians, "ranking": ranking, "primary_rounds": list(formal)})
    write_json(args.output / "summary.json", summary)
    headlines = [
        f"| {policy} {NAMES[policy]} | {formal['round-1'][policy]['det_seconds']:.3f} | "
        f"{formal['round-2'][policy]['det_seconds']:.3f} | {value['det_seconds']:.3f} | "
        f"{value['reduction_vs_a']:.1%} | {value['reduction_vs_m']:.1%} | "
        f"{value['peak_footprint'] / 1024**3:.3f} | {value['outputs_exact']} |"
        for policy, value in medians.items()
    ]
    documents, details, repeats = [], [], []
    for name, group in groups.items():
        if name == "smoke":
            continue
        for policy, value in group.items():
            if name == "repeat":
                repeats.append(
                    f"| {policy} | {value['det_seconds']:.3f} | {value['peak_footprint'] / 1024**3:.3f} | {value['exact']} |"
                )
            details.append(
                f"| {name} | {policy} | {value['det_seconds']:.3f} | {value['loading_seconds']:.3f} | "
                f"{value['cleanup_seconds']:.3f} | {value['operational_seconds']:.3f} | {value['forward_calls']} |"
            )
            for row in value["documents"]:
                stages = row["stages"]
                documents.append(
                    f"| {name} | {policy} | {row['document']} | {row['seconds']:.3f} | "
                    f"{stages['preprocess']:.3f} | {stages['forward']:.3f} | "
                    f"{stages['output_transfer']:.3f} | {stages['postprocess']:.3f} |"
                )
    report = f"""# 本机 CPU det：大 batch 才拆分及 microbatch

相同三份冻结文档（95/88/114 页），原调用顺序，每遍 2280 个输入、638 次原调用。
同一 tiny 权重、FP32、生产线程 4/10、两组后处理模型复用；独立新进程串行，每本文档原清理。
本轮无融合，所有模型 BN=36；不修改 H/W，不补齐空间或尾批，不做额外前向预热。
未拆分的调用直接执行原方法，拆分调用使用连续数组视图并按原顺序恢复概率图，后处理参数不变。
O/P/Q/R 只在原 N≥16 且超出上限时拆分；S/T/U 对所有超出上限的调用拆分。
本语料最大的原 N 是 16，“≥16”组实际只有 33 次 N=16，不能泛化为其他大小的最优阈值。

| 策略 | 第一轮 det 秒 | 第二轮 det 秒 | 中位数秒 | 相对 A 耗时降低 | 相对 M 耗时降低 | 峰 footprint GiB | 框完全一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(headlines)}

满足严格框一致及 ≤8 GiB 的候选排序：{ranking}。
完整两轮才参与主比较；必要补测数据单列。主指标为三文档 det 合计中位数，包含前处理、张量准备、forward、回传和 DB 后处理。
PNG 读取、同步内存采样、结果保存、模型加载和清理排除在 det 之外；下表保留准备及清理成本。

| 轮次 | 策略 | det 秒 | 加载秒 | 清理秒 | 加载＋det＋清理秒 | 实际前向次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(details)}

| 轮次 | 策略 | 文档 | det 秒 | 前处理秒 | forward 秒 | 回传秒 | 后处理秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(documents)}

必要补测：

| 策略 | det 秒 | 峰 footprint GiB | 框完全一致 |
| --- | ---: | ---: | --- |
{chr(10).join(repeats)}

保存的框独立逐项核对数量、顺序及四点坐标，实际 NCHW 与投影断言一致，差异图保留在各策略 review/。
每个模型为 CPU/FP32；权重、线程、推理源码以及拆分脚本指纹已核对；实际运行脚本另存 executed-scripts/。
内存沿原调用结束采样，采样峰不代表算子内部瞬时峰值；footprint 与 RSS/driver/compressed 不能相加，swap 为全机指标。
仅测本机 det，不推导完整 Basic/OCR、Standard/VLM、CUDA 或其他 CPU 平台的收益；本轮未修改生产默认、公开接口或权重。
未做持久服务生命周期验收。

```bash
.venv4/bin/python tools/analyze_det_microbatch.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/analyze_det_microbatch.py summarize --output {args.output}
.venv4/bin/python tools/benchmark_det_corpus.py run --policies A,M,O,P,Q,R,S,T,U --cycles 1 --limit-calls 12 --output NEW_SMOKE
```
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"medians": medians, "ranking": ranking}, ensure_ascii=False), flush=True)


def experiment(args: argparse.Namespace) -> None:
    """先验收烟测，再完整正反两轮；接近的优选只追加一次配对补测。"""
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
    execute_run(args, "smoke", ",".join(NAMES), limit_calls=12)
    smoke = collect(args)["groups"]["smoke"]
    assert smoke["A"]["exact"] and smoke["M"]["exact"], "原基线或上一轮 N=1 烟测不一致，先调查"
    eligible = [policy for policy in NAMES if smoke[policy]["exact"]]
    state = {"phase": "round-1", "policies": eligible, "excluded": [policy for policy in NAMES if policy not in eligible]}
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-1", ",".join(eligible))
    state["phase"] = "round-2"
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-2", ",".join(reversed(eligible)))
    summarize(args)
    summary = read_json(args.output / "summary.json")
    ranking, medians = summary["ranking"], summary["medians"]
    if ranking:
        best = ranking[0]
        reductions = medians[best]["per_round_reductions"]
        close = len(ranking) > 1 and 1 - medians[best]["det_seconds"] / medians[ranking[1]]["det_seconds"] < 0.05
        noisy = medians[best]["reduction_vs_a"] < 0.1 or min(reductions) <= 0 < max(reductions)
        near_gate = abs(medians[best]["reduction_vs_a"] - 0.25) < 0.05
        if close or noisy or near_gate:
            chosen = ["A", "M", *ranking[:2]]
            state.update(
                {
                    "phase": "repeat",
                    "repeat_policies": chosen,
                    "repeat_reason": {"close": close, "noisy": noisy, "near_25_percent": near_gate},
                }
            )
            write_json(args.output / "experiment.json", state)
            execute_run(args, "repeat", ",".join(chosen))
            summarize(args)
    state["phase"] = "complete"
    write_json(args.output / "experiment.json", state)


def main() -> None:
    """提供可复用的 CPU microbatch 实验入口，原 ABC 和融合实验入口不变。"""
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
