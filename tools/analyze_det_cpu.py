"""仅重放冻结 det 输入，拆开比较 CPU Conv-BN 融合及 N=1 拆分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from analyze_det_fusion import execute_run, load_runs
from benchmark_det_corpus import DEFAULT_CORPUS, project_replay_shape, read_json
from benchmark_mps_det_service import write_json

NAMES = {"A": "CPU 原 batch", "L": "CPU 原 batch＋融合 36 组", "M": "CPU N=1 拆分，无融合", "N": "CPU N=1 拆分＋融合 36 组"}
FUSION_DOC = "https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.fusion.fuse_conv_bn_eval.html"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """独立复核已保存框、实际 NCHW、CPU/FP32、融合结构和源码指纹后汇总各轮数据。"""
    manifest = read_json(args.corpus / "manifest.json")
    documents = {row["name"]: row for row in manifest["documents"]}
    required = {(call["box_thresh"], call["unclip_ratio"]) for doc in documents.values() for call in doc["calls"]}
    fingerprints = set()
    groups: dict[str, dict[str, Any]] = {}
    for entry in load_runs(args.output):
        if entry["directory"] not in {"smoke", "round-1", "round-2", "repeat"}:
            continue
        policy, protocol, env, rows = entry["policy"], entry["protocol"], entry["environment"], entry["records"]
        assert protocol["cycles"] == 1 and protocol["documents"] == "tender,python,114"
        assert not protocol["start_call"] and not protocol["call_indices"] and not protocol["save_maps"]
        if entry["directory"] != "smoke":
            assert not protocol["limit_calls"]
        assert env["threads"] == 4 and env["interop_threads"] == 10
        assert env["det_source_sha256_before"] == env["det_source_sha256_after"]
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
            + (env.get("cpu_microbatch_adapter_sha256", ""),)
        )
        profiles = {tuple(model["profile"]) for row in rows for model in row["models"]}
        assert profiles == required
        for profile in profiles:
            assert len({model["net_id"] for row in rows for model in row["models"] if tuple(model["profile"]) == profile}) == 1
        for row in rows:
            for model in row["models"]:
                assert model["device"] == "cpu" and model["dtype"] == "torch.float32"
                assert model["fusion"]["before_bn"] == 36
                assert model["fusion"]["after_bn"] == (0 if policy in {"L", "N"} else 36)
                assert len(model["fusion"]["pairs"]) == (36 if policy in {"L", "N"} else 0)
            calls = documents[row["document"]]["calls"]
            if protocol["limit_calls"]:
                calls = calls[: protocol["limit_calls"]]
            actual = read_json(args.output / entry["directory"] / policy / f"{row['label']}-boxes.json")
            expected = [
                (call_index, item_index, call, item)
                for call_index, call in enumerate(calls)
                for item_index, item in enumerate(call["items"])
            ]
            assert len(actual) == len(expected) == row["crops"]
            for result, (call_index, item_index, call, item) in zip(actual, expected):
                assert (
                    result["crop"] == item["crop"] and result["call_index"] == call_index and result["item_index"] == item_index
                )
                assert result["profile"] == [call["box_thresh"], call["unclip_ratio"]]
                assert result["reference_equal"] == (result["boxes"] == item["boxes"])
            assert row["reference_exact_crops"] == sum(result["reference_equal"] for result in actual)
            for metric in row["call_metrics"]:
                assert metric["network_shapes"] == project_replay_shape(metric["shape"], policy)["network_shapes"]
        count = sum(row["crops"] for row in rows)
        if entry["directory"] != "smoke":
            assert count == 2280 and sum(row["calls"] for row in rows) == 638
        loading = sum(model["loading_seconds"] for model in env["model_loads"])
        fusion = sum(model["fusion"]["seconds"] for model in env["model_loads"])
        cleanup = sum(row["cleanup_seconds"] for row in rows) + env["shutdown_seconds"]
        seconds = sum(row["seconds"] for row in rows)
        value = {
            "det_seconds": seconds,
            "loading_seconds": loading,
            "fusion_seconds": fusion,
            "cleanup_seconds": cleanup,
            "operational_seconds": loading + fusion + seconds + cleanup,
            "inputs": count,
            "exact": all(row["reference_exact_crops"] == row["crops"] for row in rows),
            "differences": sum(len(row["differences"]) for row in rows),
            "forward_calls": sum(len(metric["network_shapes"]) for row in rows for metric in row["call_metrics"]),
            "peak_footprint": max(
                [row["peak_memory_diagnostics"]["phys_footprint"] for row in rows]
                + [
                    env["shutdown_memory_diagnostics_before"]["phys_footprint"],
                    env["shutdown_memory_diagnostics_after"]["phys_footprint"],
                ]
            ),
            "documents": rows,
            "environment": env,
        }
        groups.setdefault(entry["directory"], {})[policy] = value
    if len(fingerprints) > 1:
        raise ValueError("运行环境、权重或推理源码不同，不能合并比较")
    return {
        "groups": groups,
        "matched_environment": len(fingerprints) == 1,
        "manifest_sha256": hashlib.sha256((args.corpus / "manifest.json").read_bytes()).hexdigest(),
    }


def summarize(args: argparse.Namespace) -> None:
    """区分融合和拆批的独立收益，展示逐文档阶段耗时、内存和加载清理成本。"""
    summary = collect(args)
    groups = summary["groups"]
    formal = {name: group for name, group in groups.items() if name != "smoke"}
    assert {"round-1", "round-2"}.issubset(formal)
    policies = list(groups["round-1"])
    assert all(set(group) == set(policies) for group in formal.values())
    medians = {}
    rows, docs, memories = [], [], []
    for policy in policies:
        values = [group[policy] for group in formal.values()]
        medians[policy] = {
            "det_seconds": statistics.median(value["det_seconds"] for value in values),
            "operational_seconds": statistics.median(value["operational_seconds"] for value in values),
            "outputs_exact": all(value["exact"] for value in values),
            "peak_footprint": max(value["peak_footprint"] for value in values),
        }
    for name, group in formal.items():
        for policy, value in group.items():
            rows.append(
                f"| {name} | {policy} | {value['det_seconds']:.3f} | {value['loading_seconds']:.3f} | "
                f"{value['fusion_seconds']:.4f} | {value['cleanup_seconds']:.3f} | {value['operational_seconds']:.3f} | "
                f"{value['peak_footprint'] / 1024**3:.3f} | {value['exact']} | {value['forward_calls']} |"
            )
            for row in value["documents"]:
                stages = row["stages"]
                docs.append(
                    f"| {name} | {policy} | {row['document']} | {row['seconds']:.3f} | {stages['preprocess']:.3f} | "
                    f"{stages['forward']:.3f} | {stages['output_transfer']:.3f} | {stages['postprocess']:.3f} |"
                )
                diagnostics = row["memory_diagnostics_after"]
                memories.append(
                    f"| {name} | {policy} | {row['document']} | {row['memory_after']['rss'] / 1024**3:.3f} | "
                    f"{diagnostics['phys_footprint'] / 1024**3:.3f} | {diagnostics['compressed'] / 1024**3:.3f} | "
                    f"{diagnostics['system_swap_used'] / 1024**3:.3f} |"
                )
    baseline = medians["A"]["det_seconds"]
    for policy, value in medians.items():
        value["reduction_vs_a"] = 1 - value["det_seconds"] / baseline
        value["per_round_reductions"] = [
            1 - group[policy]["det_seconds"] / group["A"]["det_seconds"] for group in formal.values()
        ]
    comparisons = {
        f"{before}_to_{after}": 1 - medians[after]["det_seconds"] / medians[before]["det_seconds"]
        for before, after in (("A", "L"), ("A", "M"), ("M", "N"), ("L", "N"))
        if before in medians and after in medians
    }
    summary.update({"medians": medians, "comparisons": comparisons})
    write_json(args.output / "summary.json", summary)
    headline = [
        f"| {policy} {NAMES[policy]} | {value['det_seconds']:.3f} | {value['reduction_vs_a']:.1%} | "
        f"{value['operational_seconds']:.3f} | {value['peak_footprint'] / 1024**3:.3f} | {value['outputs_exact']} |"
        for policy, value in medians.items()
    ]
    report = f"""# CPU det：Conv＋BN 融合与 N=1 拆分

相同三份冻结文档（95/88/114 页），2280 个输入，原顺序、原 H/W、两组后处理参数；不重跑 PDF。
所有策略为 CPU/FP32，生产线程 4/10，两组模型在各自进程内复用，每本文档沿用原清理。
烟测每份文档前 12 次调用，参数覆盖及框检查通过的策略才进入正式轮次。
全量正序 A/L/M/N、反序 N/M/L/A，独立新进程串行，无额外前向预热。
若小幅候选收益或两轮方向相反，再追加一轮对照；重复数据始终单列，不冒充原定轮次。
实际源码归档在 executed-scripts/，权重及推理源码校验在每个 environment.json。

| 策略 | 三文档 det 中位数秒 | 相对 A 耗时降低 | 加载＋融合＋det＋清理中位数秒 | 采样峰 footprint GiB | 框完全一致 |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(headline)}

融合仅限 eval 下已有的 36 组 Conv＋BN，在 CPU 加载及原重参数化后使用
[fuse_conv_bn_eval]({FUSION_DOC})，head 反卷积使用 transpose=True；融合后的 BN 全部替换为 Identity。
M/N 保留 batch 前后处理，仅逐个 N=1 网络前向；没有 H/W 或虚假样本补齐。
各次调用的实际 NCHW 已断言。A/L 为 638 次前向，M/N 为 2280 次。
比较 A→L 看原 batch 融合收益，A→M 看未融合 N 拆分收益，M→N 看拆批后的融合收益，L→N 看融合后的拆批收益。
上述配对的中位数耗时降低率：{json.dumps(comparisons, ensure_ascii=False)}。

| 轮次 | 策略 | det 秒 | 加载秒 | 融合秒 | 清理秒 | 合计秒 | 峰 footprint GiB | 框一致 | 前向次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
{chr(10).join(rows)}

| 轮次 | 策略 | 文档 | det 秒 | 前处理秒 | forward 秒 | 回传秒 | 后处理秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(docs)}

| 轮次 | 策略 | 文档 | 清理后 RSS GiB | footprint GiB | compressed GiB | 全机 swap GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(memories)}

det 包含真实前处理、张量准备、forward、回传、DB 后处理及循环调度；PNG 读取、采样和保存排除在外。
原始框被独立重读并核对数量、顺序和四点坐标，差异输入及叠框保存在对应 review/。
CPU 执行同步完成。内存沿用原调用结束采样，采样峰值不代表算子内部瞬时峰值；各内存指标不能相加，swap 属于全机。
仅代表本机 M4、当前 tiny/FP32 det，不能泛化其他 CPU 平台、完整 Basic/OCR 或 Standard/VLM。
本轮不更改生产默认、公开接口或原权重文件。

```bash
.venv4/bin/python tools/analyze_det_cpu.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/analyze_det_cpu.py summarize --output {args.output}
.venv4/bin/python tools/benchmark_det_corpus.py run --policies A,L,M,N --cycles 1 --limit-calls 12 --output NEW_SMOKE
```
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"medians": medians, "comparisons": comparisons}, ensure_ascii=False), flush=True)


def experiment(args: argparse.Namespace) -> None:
    """先烟测四种 CPU 策略，再正反两轮测速；小收益或方向变化时补一轮确认。"""
    args.output.mkdir(parents=True, exist_ok=False)
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
    execute_run(args, "smoke", "A,L,M,N", limit_calls=12)
    smoke = collect(args)["groups"]["smoke"]
    assert smoke["A"]["exact"], "CPU 基线烟测框不一致，停止正式对照"
    eligible = [policy for policy in NAMES if smoke[policy]["exact"]]
    state = {"phase": "round-1", "policies": eligible, "excluded": [policy for policy in NAMES if policy not in eligible]}
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-1", ",".join(eligible))
    state["phase"] = "round-2"
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-2", ",".join(reversed(eligible)))
    groups = collect(args)["groups"]
    noisy = []
    for policy in eligible:
        if policy == "A":
            continue
        reductions = [
            1 - groups[name][policy]["det_seconds"] / groups[name]["A"]["det_seconds"] for name in ("round-1", "round-2")
        ]
        if any(0 < value < 0.1 for value in reductions) or min(reductions) <= 0 < max(reductions):
            noisy.append(policy)
    if noisy:
        state.update({"phase": "repeat", "repeat_reason": noisy})
        write_json(args.output / "experiment.json", state)
        execute_run(args, "repeat", ",".join(eligible))
    summarize(args)
    state["phase"] = "complete"
    write_json(args.output / "experiment.json", state)


def main() -> None:
    """提供可复测的 CPU 对照入口，默认设备和既有 ABC 实验行为不变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("experiment", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    args.output, args.corpus = args.output.resolve(), args.corpus.resolve()
    args.documents = "tender,python,114"
    {"experiment": experiment, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
