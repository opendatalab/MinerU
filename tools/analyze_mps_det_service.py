"""汇总独立服务实验的时间、内存、完整输出和逐裁剪检测框差异。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_mps_det_service import DEFAULT_OUTPUT, SAMPLES, write_json  # noqa: E402


def read_json(path: Path) -> Any:
    """读取完整原始结果，不删除源文档的创建时间等语义元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """逐项保存所有不同叶子的路径和值，完整输入另保存在原始文件中。"""
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right}]
    if isinstance(left, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result.append({"path": f"{path}/{key}", "left": left.get(key), "right": right.get(key)})
            else:
                result.extend(differences(left[key], right[key], f"{path}/{key}"))
        return result
    if isinstance(left, list):
        result = []
        if len(left) != len(right):
            result.append({"path": f"{path}/length", "left": len(left), "right": len(right)})
        for index, (first, second) in enumerate(zip(left, right)):
            result.extend(differences(first, second, f"{path}/{index}"))
        for index in range(min(len(left), len(right)), max(len(left), len(right))):
            result.append(
                {
                    "path": f"{path}/{index}",
                    "left": left[index] if index < len(left) else None,
                    "right": right[index] if index < len(right) else None,
                }
            )
        return result
    return [] if left == right else [{"path": path, "left": left, "right": right}]


def det_items(path: Path) -> dict[str, list[Any]]:
    """按裁剪内容和后处理参数对齐检测结果，允许模型实例地址不同。"""
    result: dict[str, list[Any]] = defaultdict(list)
    for line in path.read_text().splitlines():
        batch = json.loads(line)
        for item in batch["items"]:
            key = f"{batch['box_thresh']}:{batch['unclip_ratio']}:{item['crop']}"
            result[key].append(item["boxes"])
    return dict(result)


def det_overlay(
    root: Path, target: Path, left_policy: str, right_policy: str, key: str, left: list[Any] | None, right: list[Any] | None
) -> None:
    """逐个保存变化裁剪的前后检测框，缺少对应输入时明确标注。"""
    from PIL import Image, ImageDraw

    crop_hash = key.split(":")[-1]
    source = root / left_policy / "crops" / f"{crop_hash}.png"
    if not source.exists():
        source = root / right_policy / "crops" / f"{crop_hash}.png"
    with Image.open(source) as opened:
        original = opened.convert("RGB")
    width, height = original.size
    canvas = Image.new("RGB", (2 * width, height + 36), "white")
    for column, (policy, groups, color) in enumerate(((left_policy, left, "red"), (right_policy, right, "blue"))):
        panel = original.copy()
        draw = ImageDraw.Draw(panel)
        if groups:
            for boxes in groups:
                for box in boxes or []:
                    draw.line([tuple(point) for point in box] + [tuple(box[0])], fill=color, width=2)
        canvas.paste(panel, (column * width, 36))
        ImageDraw.Draw(canvas).text(
            (column * width + 5, 8),
            f"{policy}: {sum(len(boxes or []) for boxes in groups) if groups else 'no matching input'}",
            fill="black",
        )
    canvas.save(target)


def draw_blocks(image: Any, blocks: list[dict[str, Any]], color: str) -> Any:
    """递归绘制共享 MiddleJson 的归一化 block 几何，保留原始页面像素。"""
    from PIL import ImageDraw

    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    width, height = image.size

    def visit(items: list[dict[str, Any]]) -> None:
        """遍历有几何信息的块及子块，不把内容文本当作几何数据。"""
        for block in items:
            bbox = block.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                draw.rectangle([bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height], outline=color, width=2)
            for field in ("children", "blocks"):
                if isinstance(block.get(field), list):
                    visit(block[field])

    visit(blocks)
    return panel


def page_overlays(directory: Path, sample: str, left: dict[str, Any], right: dict[str, Any], pages: list[int]) -> None:
    """为所有语义变化页保存原文、CPU 和候选整页叠框，不声称未查看页面已人工审阅。"""
    from docvortex.document.pdf import PDFDocument

    left_pages = {page["page_idx"]: page for page in left["pages"]}
    right_pages = {page["page_idx"]: page for page in right["pages"]}
    with PDFDocument(str(SAMPLES[sample])) as document:
        for index in pages:
            image = document.render_page(index).pil_image.convert("RGB")
            image.save(directory / f"page-{index + 1:03d}-source.png")
            draw_blocks(image, left_pages.get(index, {}).get("blocks", []), "red").save(
                directory / f"page-{index + 1:03d}-baseline.png"
            )
            draw_blocks(image, right_pages.get(index, {}).get("blocks", []), "blue").save(
                directory / f"page-{index + 1:03d}-candidate.png"
            )


def compare_request(root: Path, label: str, policy: str, sample: str, baseline_label: str | None = None) -> dict[str, Any]:
    """比较 CPU 基线与一份候选请求，完整保留数值变化与视觉证据。"""
    baseline_label = baseline_label or label
    first = root / "A" / baseline_label
    second = root / policy / label
    target = root / "comparisons" / f"A-{baseline_label}-vs-{policy}-{label}"
    target.mkdir(parents=True, exist_ok=True)
    middle_a = read_json(first / "middle_json.json")
    middle_b = read_json(second / "middle_json.json")
    # 当前共享协议没有运行计时字段；源文档元数据和所有语义字段均原样比较。
    middle_diff = differences(middle_a, middle_b)
    model_diff = differences(read_json(first / "bundle/model_output.json"), read_json(second / "bundle/model_output.json"))
    a, b = det_items(first / "det-trace.jsonl"), det_items(second / "det-trace.jsonl")
    det_diff = [
        {"key": key, "left": a.get(key), "right": b.get(key)} for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)
    ]
    write_json(target / "middle-differences.json", middle_diff)
    write_json(target / "model-differences.json", model_diff)
    write_json(target / "det-differences.json", det_diff)
    markdown_equal = (first / "markdown.md").read_bytes() == (second / "markdown.md").read_bytes()
    if not markdown_equal:
        import difflib

        delta = difflib.unified_diff(
            (first / "markdown.md").read_text().splitlines(True),
            (second / "markdown.md").read_text().splitlines(True),
            fromfile="baseline",
            tofile="candidate",
        )
        (target / "markdown.diff").write_text("".join(delta), encoding="utf-8")
    changed_pages = sorted(
        {
            int(item["path"].split("/")[2])
            for item in middle_diff
            if item["path"].startswith("/pages/") and item["path"].split("/")[2].isdigit()
        }
    )
    page_overlays(target, sample, middle_a, middle_b, changed_pages)
    for index, delta in enumerate(det_diff):
        det_overlay(root, target / f"det-{index:05d}.png", "A", policy, delta["key"], delta["left"], delta["right"])
    return {
        "policy": policy,
        "label": label,
        "baseline_label": baseline_label,
        "markdown_equal": markdown_equal,
        "middle_equal": not middle_diff,
        "model_equal": not model_diff,
        "det_equal": not det_diff,
        "middle_changed_leaves": len(middle_diff),
        "model_changed_leaves": len(model_diff),
        "det_changed_keys": len(det_diff),
        "changed_pages": [index + 1 for index in changed_pages],
        "directory": str(target),
    }


def analyze(root: Path) -> None:
    """计算预先约定的速度、内存和输出门槛，给出下一步决策所需证据。"""
    requests = {policy: read_json(root / policy / "requests.json") for policy in ("A", "B", "C")}
    comparisons = []
    for policy, records in requests.items():
        for record in records:
            if policy != "A":
                comparisons.append(compare_request(root, record["label"], policy, record["sample"]))
            if record["label"].startswith(("04-", "05-", "06-")):
                baseline_label = f"{int(record['label'][:2]) - 3:02d}-{record['sample']}"
                comparisons.append(compare_request(root, record["label"], policy, record["sample"], baseline_label))
    aggregate: dict[str, Any] = {}
    for policy, records in requests.items():
        aggregate[policy] = {}
        for cycle, subset in (("first", records[:3]), ("second", records[3:6])):
            aggregate[policy][cycle] = {
                key: sum(record[key] for record in subset)
                for key in ("request_seconds", "server_seconds", "det_seconds", "capture_seconds")
            }
        aggregate[policy]["memory_growth"] = {
            key: {
                "first": records[2]["memory_after"][key],
                "second": records[5]["memory_after"][key],
                "growth": records[5]["memory_after"][key] - records[2]["memory_after"][key],
                "limit": max(256 * 1024**2, records[2]["memory_after"][key] * 0.1),
            }
            for key in ("rss", "mps_allocated", "mps_driver")
        }
    cpu, candidate = aggregate["A"]["second"], aggregate["C"]["second"]
    det_reduction = 1 - candidate["det_seconds"] / cpu["det_seconds"]
    wall_ratio = candidate["request_seconds"] / cpu["request_seconds"]
    gate = {
        "det_reduction": det_reduction,
        "det_pass": det_reduction >= 0.25,
        "request_time_ratio": wall_ratio,
        "request_pass": wall_ratio <= 1.05,
        "memory_pass": all(value["growth"] <= value["limit"] for value in aggregate["C"]["memory_growth"].values()),
        "output_exact": all(
            check["middle_equal"] and check["markdown_equal"] and check["det_equal"] and check["model_equal"]
            for check in comparisons
        ),
        "additional_memory_cycle_needed": any(
            value["growth"] > value["limit"] for value in aggregate["C"]["memory_growth"].values()
        ),
        "reverse_order_needed": abs(1 - wall_ratio) < 0.1 or abs(det_reduction) < 0.1,
    }
    summary = {"aggregates": aggregate, "gates": gate, "comparisons": comparisons}
    write_json(root / "summary.json", summary)
    rows = []
    for policy, records in requests.items():
        for record in records:
            rows.append(
                f"| {policy} | {record['label']} | {record['request_seconds']:.2f} | {record['det_seconds']:.2f} | "
                f"{record['memory_after']['rss'] / 1024**3:.2f} | {record['memory_after']['mps_driver'] / 1024**3:.2f} | "
                f"{record['crops']} | {record['new_shapes']}/{record['reused_shapes']} |"
            )
    report = f"""**MPS det 跨文档 API 实验：第一轮验收**

三种策略在独立进程中串行运行。A=CPU det+原清理，B=MPS det+原清理，C=MPS det+仅 GC 保留缓存。
每个进程使用本地真实 V1 API、Basic、强制 OCR、全部页面、关闭可选图像描述、并发 1。
顺序为 95 页招标文件、88 页 Python.pdf、114 页 _114.pdf，再重复一次；没有 VLM、预热或算法优化。

| 策略 | 文档 | API 完整秒 | 普通 det 秒 | 结束 RSS GiB | 结束 MPS driver GiB | 裁剪数 | 新形状/复用形状调用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

第二遍三本文档合计：C 相对 A 的 det 耗时降低 {det_reduction:.2%}，API 完整请求耗时比值 {wall_ratio:.3f}。
预定门槛：det 降低至少 25%，完整请求不回退超过 5%；分别为 {gate["det_pass"]}、{gate["request_pass"]}。
内存筛查通过：{gate["memory_pass"]}；输出逐项完全一致：{gate["output_exact"]}。
需要追加内存轮次：{gate["additional_memory_cycle_needed"]}；需要逆序复测：{gate["reverse_order_needed"]}。

内存门槛为第二遍结束对第一遍结束的增长不超过 max(10%,256 MiB)，同时检查 RSS、MPS allocated、driver。
这只验证有限形状集合的两遍输入，不证明无限种类文档下缓存有界。
完整服务结束后会清空本地模型并释放 MPS 缓存，证据在各策略 shutdown.json。

普通 det 时间包含所有非印章 det 实例的真实预处理与同步推理/后处理，涵盖正文和表格使用的 det。
预处理截图、哈希和取证 JSON 写出耗时单独记录，不计入普通 det 时间。
原始 API 完整墙钟包含这些取证开销、生产导出和调度，但不包括结果下载。
第二遍已有裁剪不重复保存 PNG。NCHW 复用统计表示输入形状再次出现，不是底层 MPSGraph 命中率；原清理后会重置观察集合。
初始模型加载计入第一份文档完整请求，不计入 det 内部执行时间。其他模型的设备、精度和推理流程均不改。

所有请求保存 job.json、middle_json.json、markdown.md、自包含 zip/model_output.json、逐裁剪 det-trace.jsonl。
comparisons/ 保存所有变化叶子、Markdown diff、变化页的原文/基线/候选叠框，以及变化裁剪的前后 det 框。
源文档元数据全部保留，没有以删除 metadata 或模糊数值容差制造输出一致；job ID/运行计时属于单独的 API 请求记录。
自动差异和人工视觉裁决需分别报告，本报告生成时尚未自动把所有图片宣称为人工审阅完成。

本轮结论限定为本机 Basic/OCR 的整本流程及共享 det，不泛化为 Standard/VLM 全流程，也不直接修改默认设备或缓存策略。
"""
    (root / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def main() -> None:
    """允许显式选择实验目录，分析阶段不与正式推理同时运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(args.output.resolve())


if __name__ == "__main__":
    main()
