"""按逐输入数量一致、忽略顺序和人工几何裁决重新评估 H/W 补齐，仅用于实验。"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from analyze_det_fusion import execute_run, load_runs
from benchmark_det_corpus import DEFAULT_CORPUS, ROOT, read_json
from benchmark_mps_det_service import write_json


def match_boxes(reference: list[Any], actual: list[Any]) -> dict[str, Any]:
    """使用四边形 IoU 的最优一对一匹配忽略输出顺序，并记录原图像素中的角点误差。"""
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    count_equal = len(reference) == len(actual)
    if reference == actual:
        return {"count_equal": True, "ordered_exact": True, "unordered_exact": True, "pairs": [], "triage": "exact"}
    overlaps = np.zeros((len(reference), len(actual)), dtype=np.float64)
    for i, left in enumerate(reference):
        left_array = np.asarray(left, dtype=np.float32)
        left_area = abs(cv2.contourArea(left_array))
        for j, right in enumerate(actual):
            right_array = np.asarray(right, dtype=np.float32)
            intersection, _ = cv2.intersectConvexConvex(left_array, right_array)
            union = left_area + abs(cv2.contourArea(right_array)) - intersection
            overlaps[i, j] = intersection / union if union > 0 else 0.0
    indices, candidates = linear_sum_assignment(-overlaps)
    pairs = []
    for i, j in zip(indices.tolist(), candidates.tolist()):
        left = np.asarray(reference[i], dtype=np.float64)
        right = np.asarray(actual[j], dtype=np.float64)
        variants = [np.roll(points, offset, axis=0) for points in (right, right[::-1]) for offset in range(4)]
        aligned = min(variants, key=lambda points: float(np.square(left - points).sum()))
        max_shift = float(np.linalg.norm(left - aligned, axis=1).max())
        # 相对短边可区分大表格的几像素变化与小字框的同样位移。
        short_side = max(1.0, float(np.linalg.norm(left - np.roll(left, -1, axis=0), axis=1).min()))
        pairs.append(
            {
                "reference_index": i,
                "actual_index": j,
                "iou": float(overlaps[i, j]),
                "max_corner_shift_px": max_shift,
                "shift_relative_short_side": max_shift / short_side,
                "corners_exact": bool(np.array_equal(left, aligned)),
            }
        )
    unordered_exact = count_equal and all(pair["corners_exact"] for pair in pairs)
    small = count_equal and all(
        pair["max_corner_shift_px"] <= 3 and pair["iou"] >= 0.8 and pair["shift_relative_short_side"] <= 0.15 for pair in pairs
    )
    return {
        "count_equal": count_equal,
        "ordered_exact": False,
        "unordered_exact": unordered_exact,
        "reference_count": len(reference),
        "actual_count": len(actual),
        "pairs": pairs,
        "unmatched_reference": sorted(set(range(len(reference))) - set(indices.tolist())),
        "unmatched_actual": sorted(set(range(len(actual))) - set(candidates.tolist())),
        "max_corner_shift_px": max((pair["max_corner_shift_px"] for pair in pairs), default=0.0),
        "min_iou": min((pair["iou"] for pair in pairs), default=0.0),
        "triage": "count_failure" if not count_equal else "order_only" if unordered_exact else "small" if small else "review",
    }


def audit_run(entry: dict[str, Any], root: Path, documents: dict[str, Any]) -> dict[str, Any]:
    """逐个输入对照冻结参考，保留全部匹配明细，不用文档总框数抵消漏检或新增。"""
    changes = []
    inputs = 0
    for row in entry["records"]:
        calls = documents[row["document"]]["calls"]
        output = root / entry["directory"] / entry["policy"]
        boxes = read_json(output / f"{row['label']}-boxes.json")
        if len(boxes) != row["crops"]:
            raise ValueError("结果数量与原运行记录不一致")
        for result in boxes:
            reference = calls[result["call_index"]]["items"][result["item_index"]]
            if reference["crop"] != result["crop"]:
                raise ValueError("输入图像键与冻结调用不一致")
            comparison = match_boxes(reference["boxes"] or [], result["boxes"] or [])
            inputs += 1
            if comparison["ordered_exact"]:
                continue
            changes.append(
                {
                    "directory": entry["directory"],
                    "policy": entry["policy"],
                    "document": row["document"],
                    "label": row["label"],
                    "call_index": result["call_index"],
                    "item_index": result["item_index"],
                    "crop": result["crop"],
                    "profile": result["profile"],
                    "reference": reference["boxes"],
                    "actual": result["boxes"],
                    "comparison": comparison,
                }
            )
    return {
        "inputs": inputs,
        "ordered_exact": inputs - len(changes),
        "count_failures": sum(not item["comparison"]["count_equal"] for item in changes),
        "order_only": sum(item["comparison"]["triage"] == "order_only" for item in changes),
        "small_triage": sum(item["comparison"]["triage"] == "small" for item in changes),
        "review_triage": sum(item["comparison"]["triage"] == "review" for item in changes),
        "changes": changes,
    }


def make_inspection(corpus: Path, changes: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """按数量变化和最大几何差异生成局部原图/参考/候选三栏图，人工裁决与自动分流分开保存。"""
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    unique = {}
    for change in changes:
        key = (change["policy"], change["document"], change["call_index"], change["item_index"])
        unique.setdefault(key, change)
    selected = sorted(
        unique.values(),
        key=lambda item: (
            item["comparison"]["count_equal"],
            item["comparison"].get("min_iou", 1.0),
            -item["comparison"].get("max_corner_shift_px", 0.0),
        ),
    )
    inventory = []
    for index, item in enumerate(selected):
        comparison = item["comparison"]
        focus = [item["reference"][i] for i in comparison["unmatched_reference"]]
        focus += [item["actual"][i] for i in comparison["unmatched_actual"]]
        if not focus and comparison["pairs"]:
            worst = min(comparison["pairs"], key=lambda pair: pair["iou"])
            focus = [item["reference"][worst["reference_index"]], item["actual"][worst["actual_index"]]]
        with Image.open(corpus / "images" / f"{item['crop']}.png") as opened:
            source = opened.convert("RGB")
        points = [point for box in focus for point in box]
        region = (
            (
                max(0, int(min(point[0] for point in points)) - 20),
                max(0, int(min(point[1] for point in points)) - 20),
                min(source.width, int(max(point[0] for point in points)) + 21),
                min(source.height, int(max(point[1] for point in points)) + 21),
            )
            if points
            else (0, 0, source.width, source.height)
        )
        width, height = region[2] - region[0], region[3] - region[1]
        scale = min(2.0, 460 / width, 360 / height)
        panel_width, panel_height = max(1, int(width * scale)), max(1, int(height * scale))
        canvas = Image.new("RGB", (panel_width * 3, panel_height + 65), "white")
        title = (
            f"{index:04d} {item['policy']} {item['document']} call{item['call_index']} item{item['item_index']} "
            f"count {len(item['reference'])}->{len(item['actual'])} "
            f"IoU {comparison.get('min_iou', 1):.3f} shift {comparison.get('max_corner_shift_px', 0):.1f}px"
        )
        ImageDraw.Draw(canvas).text((4, 3), title, fill="black")
        for column, (label, boxes, color) in enumerate(
            (("source", [], "black"), ("reference", item["reference"], "red"), ("padded", item["actual"], "blue"))
        ):
            panel = source.copy()
            draw = ImageDraw.Draw(panel)
            for box in boxes:
                draw.line([tuple(point) for point in box] + [tuple(box[0])], fill=color, width=1)
            panel = panel.crop(region).resize((panel_width, panel_height))
            canvas.paste(panel, (column * panel_width, 65))
            ImageDraw.Draw(canvas).text((column * panel_width + 4, 38), label, fill=color)
        name = f"{index:04d}-{item['policy']}-{item['document']}-call{item['call_index']:04d}-item{item['item_index']:03d}.png"
        canvas.save(output / name)
        inventory.append({"image": name, "region": list(region), "change": item, "verdict": None})
    write_json(output / "inventory.json", inventory)
    return inventory


def review_git_revisions(heads: set[str]) -> dict[str, Any]:
    """仅容许已核实未被 det 重放入口调用的印章修复及测试提交，其他源码漂移仍阻止环境通过。"""
    ignored_runtime = {
        "mineru/model/ocr/pp_ocr_v6_onnx.py",
        "mineru/model/ocr/pytorch_paddle.py",
        "mineru/model/ocr/seal_crop.py",
    }
    revisions = sorted(heads)
    changes = []
    for revision in revisions[1:]:
        paths = subprocess.check_output(
            ["git", "diff", "--name-only", revisions[0], revision], cwd=ROOT, text=True
        ).splitlines()
        relevant = [path for path in paths if not path.startswith("tests/") and path not in ignored_runtime]
        changes.append({"from": revisions[0], "to": revision, "changed_paths": paths, "unreviewed_paths": relevant})
    return {
        "heads": revisions,
        "same_head": len(revisions) == 1,
        "det_source_equivalent": bool(revisions) and all(not change["unreviewed_paths"] for change in changes),
        "changes": changes,
        "reason": "worker 直接调用 _internal.pytorchocr.infer.predict_det.TextDetector，不执行三处印章裁剪/引擎代码。",
    }


def collect(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """汇总配对耗时与逐输入数量检查，细微 bbox 分流阈值只用于选图，不自动判合格。"""
    state_file = root / "experiment.json"
    if state_file.exists() and read_json(state_file).get("timing_valid") is False:
        raise ValueError("该目录已标记并行 VLM 干扰，耗时禁止用于性能汇总；请在安静时段另开目录重测")
    documents = {document["name"]: document for document in read_json(args.corpus / "manifest.json")["documents"]}
    runs = load_runs(root)
    groups = {}
    environments = set()
    execution_environments = set()
    heads = set()
    fp32 = True
    reused = True
    required_profiles = {(call["box_thresh"], call["unclip_ratio"]) for doc in documents.values() for call in doc["calls"]}
    profiles_complete = True
    for entry in runs:
        rows, env = entry["records"], entry["environment"]
        environments.add(
            tuple(
                env[key]
                for key in (
                    "model_sha256",
                    "torch_commit",
                    "python",
                    "threads",
                    "interop_threads",
                    "git_head",
                    "script_sha256",
                    "shape_adapter_sha256",
                )
            )
        )
        execution_environments.add(
            tuple(
                env[key]
                for key in (
                    "model_sha256",
                    "torch_commit",
                    "python",
                    "threads",
                    "interop_threads",
                    "script_sha256",
                    "shape_adapter_sha256",
                )
            )
        )
        heads.add(env["git_head"])
        fp32 &= all(model["dtype"] == "torch.float32" for row in rows for model in row["models"])
        profiles = {tuple(model["profile"]) for row in rows for model in row["models"]}
        profiles_complete &= profiles == required_profiles
        reused &= all(
            len({model["net_id"] for row in rows for model in row["models"] if tuple(model["profile"]) == profile}) == 1
            for profile in profiles
        )
        quality = audit_run(entry, root, documents)
        det = sum(row["seconds"] for row in rows)
        loading = sum(model["loading_seconds"] for model in env["model_loads"])
        fusion = sum(model["fusion"]["seconds"] for model in env["model_loads"])
        cleanup = sum(row["cleanup_seconds"] for row in rows) + env["shutdown_seconds"]
        groups.setdefault(entry["directory"], {})[entry["policy"]] = {
            "det_seconds": det,
            "operational_seconds": det + loading + fusion + cleanup,
            "loading_seconds": loading,
            "fusion_seconds": fusion,
            "cleanup_seconds": cleanup,
            "peak_footprint": max(
                *(row["peak_memory_diagnostics"]["phys_footprint"] for row in rows),
                env["shutdown_memory_diagnostics_before"]["phys_footprint"],
                env["shutdown_memory_diagnostics_after"]["phys_footprint"],
            ),
            "quality": quality,
            "documents": [
                {
                    key: row[key]
                    for key in (
                        "document",
                        "seconds",
                        "crops",
                        "stages",
                        "cleanup_seconds",
                        "memory_after",
                        "memory_diagnostics_after",
                    )
                }
                for row in rows
            ],
        }
    revision_review = review_git_revisions(heads)
    return {
        "groups": groups,
        "matched_environment": len(environments) == 1,
        "matched_execution_environment": len(execution_environments) == 1 and revision_review["det_source_equivalent"],
        "revision_review": revision_review,
        "fp32": fp32,
        "models_reused": reused,
        "profiles_complete": profiles_complete,
        "environments": [{key: value for key, value in entry["environment"].items() if key != "model_loads"} for entry in runs],
    }


def render_metrics(summary: dict[str, Any], source: Path, output: Path) -> None:
    """绘制独立轮次的耗时及逐调用 footprint 曲线，避免将数量失败的速度误画成验收通过。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {name: group for name, group in summary["groups"].items() if name.startswith("round-")}
    if not groups:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    policies = [policy for policy in ("A", "E", "H", "I") if policy in summary["policies"]]
    medians = [summary["policies"][policy]["median_det_seconds"] for policy in policies]
    axes[0].bar(policies, medians, alpha=0.6)
    for index, policy in enumerate(policies):
        axes[0].scatter([index] * len(groups), [group[policy]["det_seconds"] for group in groups.values()], c="black", s=18)
    axes[0].set(ylabel="det seconds", title="Three-document median; dots = independent runs")
    axes[1].bar(policies, [summary["policies"][policy]["peak_footprint"] / 1024**3 for policy in policies], alpha=0.6)
    axes[1].axhline(8, color="red", linestyle="--", label="8 GiB budget")
    axes[1].set(ylabel="Sampled footprint GiB", title="Quality is evaluated separately")
    axes[1].legend()
    figure.savefig(output / "performance-memory.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False, constrained_layout=True)
    for axis, (name, group) in zip(axes[:, 0], groups.items()):
        for policy in group:
            rows = read_json(source / name / policy / "results.json")
            samples = [
                metric["memory_diagnostics"]["phys_footprint"] / 1024**3 for row in rows for metric in row["call_metrics"]
            ]
            axis.plot(range(1, len(samples) + 1), samples, label=policy)
        axis.axhline(8, color="red", linestyle="--")
        axis.set(xlabel="Original det call index; cleanup after each document", ylabel="footprint GiB", title=name)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "memory-curves.png", dpi=160)
    plt.close(figure)


def summarize(args: argparse.Namespace, root: Path | None = None) -> None:
    """保存放宽坐标门槛后的独立报告，数量失败的性能数据始终标为探索结果。"""
    source = root or args.output
    summary = collect(args, source)
    args.output.mkdir(parents=True, exist_ok=True)
    changes = [
        change for group in summary["groups"].values() for value in group.values() for change in value["quality"]["changes"]
    ]
    make_inspection(args.corpus, changes, args.output / "inspection")
    summaries = {}
    for policy in ("A", "E", "H", "I"):
        groups = [group for name, group in summary["groups"].items() if name.startswith("round-") and policy in group]
        if not groups:
            continue
        values = [group[policy] for group in groups]
        paired = [group for group in groups if "E" in group]
        total = statistics.median(value["det_seconds"] for value in values)
        quality = [value["quality"] for value in values]
        summaries[policy] = {
            "rounds": len(groups),
            "median_det_seconds": total,
            "reduction_vs_e": 1 - total / statistics.median(group["E"]["det_seconds"] for group in paired) if paired else None,
            "round_gains_vs_e": [1 - group[policy]["det_seconds"] / group["E"]["det_seconds"] for group in paired],
            "peak_footprint": max(value["peak_footprint"] for value in values),
            "inputs": sum(value["inputs"] for value in quality),
            "count_failures": sum(value["count_failures"] for value in quality),
            "count_gate_pass": all(value["count_failures"] == 0 for value in quality),
            "bbox_visual_acceptance": (
                "not_eligible_count_failed"
                if any(value["count_failures"] for value in quality)
                else "not_required"
                if all(not value["changes"] for value in quality)
                else "pending"
            ),
            "research_pass": False,
        }
    summary["policies"] = summaries
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "source-revision-review.json", summary["revision_review"])
    render_metrics(summary, source, args.output)
    rows, detail = [], []
    for name, group in summary["groups"].items():
        for policy, value in group.items():
            quality = value["quality"]
            rows.append(
                f"| {name} | {policy} | {value['det_seconds']:.3f} | {value['operational_seconds']:.3f} | "
                f"{value['peak_footprint'] / 1024**3:.3f} | {quality['count_failures']}/{quality['inputs']} | "
                f"{quality['order_only']} | {quality['small_triage']} | {quality['review_triage']} |"
            )
            for doc in value["documents"]:
                stage = doc["stages"]
                detail.append(
                    f"| {name} | {policy} | {doc['document']} | {doc['seconds']:.3f} | {stage['preprocess']:.3f} | "
                    f"{stage['forward']:.3f} | {stage['output_transfer']:.3f} | {stage['postprocess']:.3f} |"
                )
    medians = [
        f"| {policy} | {value['median_det_seconds']:.3f} | {value['reduction_vs_e']:.1%} | "
        f"{value['peak_footprint'] / 1024**3:.3f} | {value['count_gate_pass']} |"
        for policy, value in summaries.items()
        if value["reduction_vs_e"] is not None
    ]
    report = f"""# H/W 补齐：放宽顺序及细微 bbox 后的 det 实验

数量门槛按每个 det 输入检查；允许框顺序不同，四边形以最大 IoU 最优匹配。
细微 bbox 必须结合原图判断。角点 ≤3 px、IoU ≥0.8、位移 ≤短边 15% 仅用于选图分流，**不自动代表验收**。
数量变化无论正负都失败，不能用同本文档的总框数相同抵消。

A 为 CPU 原模型；E 为 36 组 Conv/BN 全融合 MPS、原形状；H/I 与 E 相同，但归一化输入右/下补 0 到 64/128 倍数。
使用冻结 2280 个输入、原顺序/batch、FP32、两种原后处理参数、每本文档原清理、同进程模型复用。
基线与候选独立进程串行，第二轮策略顺序反转；没有前向预热、跨文档保留图缓存或重跑 PDF。
即使先前烟测已发现数量问题，本轮也执行速度探索，**不将速度收益等同于通过输出验收**。
环境指纹一致：{summary["matched_environment"]}；FP32：{summary["fp32"]}；模型复用：{summary["models_reused"]}；
两种后处理参数覆盖：{summary["profiles_complete"]}。详细环境、权重及源码 SHA256 保存在各策略 environment.json。
提交号一致：{summary["revision_review"]["same_head"]}；普通 det 源码等价核验：
{summary["revision_review"]["det_source_equivalent"]}；排除已核实的印章提交元信息后，执行环境一致：
{summary["matched_execution_environment"]}。提交差异的原始文件清单保存在 source-revision-review.json，未隐藏原提交号。

| 轮次 | 策略 | det 秒 | 加载+融合+det+清理 秒 | 峰 footprint GiB | 数量变化/输入 | 仅顺序变化 | 细微差异分流 | 需重点审阅 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

| 策略 | 三文档 det 中位数秒 | 相对 E 降低 | 峰 footprint GiB | 逐输入数量门槛 |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(medians)}

| 轮次 | 策略 | 文档 | det 秒 | 前处理秒 | 同步 forward 秒 | 回传秒 | 后处理秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(detail)}

det 包含前处理、补齐、同步 forward、回传、原形状恢复和后处理。
PNG 读取、输出写盘、差异图绘制、逐调用采样、模型加载/融合及清理不计入 det；这些记录仍保存在原始数据。
采样 footprint 包含压缩页记账，不能与 RSS/driver 相加，采样峰值不代表算子内部瞬时峰值。
内存预算仍为 8 GiB。全机 swap 不归属于单个进程。仅代表 det，不推导整本解析速度。

全部原框/候选框/最优匹配保存在 summary.json；原图与全图叠框在各策略 review/，
局部三栏审阅图在 inspection/，其中 inventory.json 的 verdict 初始为 null，不冒充已完成视觉验收。
旧烟测及概率图证据：{args.previous}。本轮保留旧报告不覆盖。

```bash
.venv4/bin/python tools/analyze_det_spatial.py experiment --output NEW_OUTPUT
.venv4/bin/python tools/analyze_det_spatial.py summarize --output {args.output}
.venv4/bin/python tools/benchmark_det_corpus.py run --policies E,H,I --cycles 1 \\
  --documents 114 --call-indices 0,1 --save-maps --output NEW_DIAGNOSTIC
```
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps({"matched_environment": summary["matched_environment"], "policies": summaries}, ensure_ascii=False),
        flush=True,
    )


def experiment(args: argparse.Namespace) -> None:
    """在全新目录执行正反两轮全量速度探索，数量失败者不进入生命周期或生产候选验收。"""
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("实验要求全新目录，不能覆盖旧证据")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "executed-scripts"
    source.mkdir()
    for path in Path(__file__).parent.glob("*det*.py"):
        if path.name in {
            "benchmark_det_corpus.py",
            "benchmark_mps_det_service.py",
            "analyze_det_fusion.py",
            "det_shape_buckets.py",
            Path(__file__).name,
        }:
            shutil.copy2(path, source / path.name)
    state = {
        "phase": "round-1",
        "previous": str(args.previous),
        "mode": "speed_exploration_count_gate_required",
        "timing_valid": True,
        "context_note": args.context_note,
    }
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-1", "A,E,H,I")
    state["phase"] = "round-2"
    write_json(args.output / "experiment.json", state)
    execute_run(args, "round-2", "I,H,E,A")
    summarize(args)
    state.update({"phase": "complete", "lifecycle": "not_executed_for_count_failed_candidates"})
    write_json(
        args.output / "experiment.json",
        state,
    )


def main() -> None:
    """提供独立速度探索、汇总和旧烟测重新几何评估入口，不修改既有严格实验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("experiment", "summarize", "audit"))
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("tmp/pdfs/issue5632-det-shapes-20261012-1826"))
    parser.add_argument("--context-note", default="", help="记录开始计时前的外部任务完成状态，不替代环境数据")
    args = parser.parse_args()
    args.corpus, args.output, args.previous = args.corpus.resolve(), args.output.resolve(), args.previous.resolve()
    args.documents = "tender,python,114"
    if args.command == "experiment":
        experiment(args)
    elif args.command == "audit":
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("旧烟测重新审阅要求全新目录")
        summarize(args, args.previous)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
