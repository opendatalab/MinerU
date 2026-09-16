# Copyright (c) Opendatalab. All rights reserved.
"""比较 PTH 导出的 Plus-M 与现用 ONNX：权重、图结构、完整 token、LaTeX 和中间数值。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer
from mineru.model.mfr.pp_formulanet_plus_m_onnx import PPFormulaNetPlusMONNX
from .export_formula_onnx import DecoderStep, Encoder

__all__ = ["compare", "diagnostic_model", "difference"]


def write_json(path: Path, data: Any) -> None:
    """逐步保存结果，长验证中断后仍可查看已经完成的证据。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def session(path: Path) -> ort.InferenceSession:
    """为所有对照统一 CPU provider、线程数及优化级别。"""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """以 FP64 累计误差，记录固定容差下的统计量及形状。"""
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "shape_equal": True,
        "shape": list(left.shape),
        "exact_equal": bool(np.array_equal(left, right)),
        "max_abs": float(delta.max(initial=0)),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "rtol": 1e-4,
        "atol": 1e-5,
        "allclose": bool(np.allclose(left, right, rtol=1e-4, atol=1e-5)),
        "outside_tolerance": int(np.count_nonzero(delta > 1e-5 + 1e-4 * np.abs(right))),
    }


def graph_summary(path: Path) -> dict[str, Any]:
    """递归统计控制流子图，避免只统计最外层节点误判模型复杂度。"""
    model = onnx.load(path)
    counts: Counter[str] = Counter()

    def visit(graph: onnx.GraphProto) -> None:
        """递归遍历 Loop/If 内部图节点。"""
        counts.update(node.op_type for node in graph.node)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)

    visit(model.graph)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "ir": model.ir_version,
        "opset": {op.domain: op.version for op in model.opset_import},
        "top_level_nodes": len(model.graph.node),
        "all_nodes": sum(counts.values()),
        "operators": dict(counts),
        "inputs": [
            {"name": item.name, "shape": [d.dim_param or d.dim_value for d in item.type.tensor_type.shape.dim]}
            for item in model.graph.input
        ],
    }


def diagnostic_model(source: Path, target: Path, *, candidate: bool) -> None:
    """只给诊断副本增加视觉特征与前 16 步 logits 输出，保留正式模型原文件。"""
    model = onnx.load(source)
    loop = next(node for node in model.graph.node if node.op_type == "Loop")
    loop.input[0] = "diagnostic_limit"
    model.graph.initializer.append(numpy_helper.from_array(np.array(16, np.int64), "diagnostic_limit"))
    body = next(attribute.g for attribute in loop.attribute if attribute.name == "body")
    argmax = next(node for node in body.node if node.op_type == "ArgMax")
    body.output.append(helper.make_tensor_value_info(argmax.input[0], TensorProto.FLOAT, ["batch", 50000]))
    loop.output.append("diagnostic_logits")
    feature_name = (
        "encoder/features" if candidate else next(node.output[0] for node in model.graph.node if node.op_type == "Transpose")
    )
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("diagnostic_logits", TensorProto.FLOAT, ["steps", "batch", 50000]),
            helper.make_tensor_value_info(feature_name, TensorProto.FLOAT, ["batch", 144, 2048]),
        ]
    )
    # 旧模型的原始 Loop iter 类型写成 [1]，ORT 仍能运行；不在此改写来源图的其他语义。
    onnx.save(model, target)


def weights_comparison(weights: Path, reference: Path) -> dict[str, Any]:
    """按值匹配原始权重，兼容 Paddle Linear 与 Torch Linear 的二维转置。"""
    model = onnx.load(reference)
    arrays = []

    def collect(graph: onnx.GraphProto) -> None:
        """权重也可能位于 Loop/If 子图，必须递归读取。"""
        arrays.extend(numpy_helper.to_array(value) for value in graph.initializer)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect(attribute.g)
                elif attribute.type == onnx.AttributeProto.TENSOR:
                    arrays.append(numpy_helper.to_array(attribute.t))

    collect(model.graph)
    signatures: set[tuple[tuple[int, ...], str]] = set()
    for value in arrays:
        signatures.add((value.shape, hashlib.sha256(value.tobytes()).hexdigest()))
        if value.ndim == 2:
            signatures.add((value.T.shape, hashlib.sha256(value.T.tobytes()).hexdigest()))
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    matched, unmatched, ignored = [], [], []
    for name, tensor in state.items():
        if name.endswith("num_batches_tracked"):
            ignored.append(name)
            continue
        array = tensor.numpy()
        item = {"name": name, "shape": list(array.shape), "elements": array.size}
        (matched if (array.shape, hashlib.sha256(array.tobytes()).hexdigest()) in signatures else unmatched).append(item)
    return {
        "matched_tensors": len(matched),
        "unmatched_tensors": len(unmatched),
        "ignored_tracking_tensors": len(ignored),
        "matched_elements": sum(item["elements"] for item in matched),
        "unmatched": unmatched,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """用同一预处理输入顺序执行两个完整模型，并对采样公式做三方数值对照。"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(item) for item in json.loads(args.inputs_json.read_text())]
    processor = PPFormulaNetPlusMONNX(str(args.reference), str(args.config), intra_op_num_threads=2)
    sessions = {"reference": processor.session, "candidate": session(args.candidate)}
    report: dict[str, Any] = {
        "environment": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "ort": ort.__version__,
            "providers": ["CPUExecutionProvider"],
            "threads": 2,
        },
        "graphs": {"reference": graph_summary(args.reference), "candidate": graph_summary(args.candidate)},
        "weights": weights_comparison(args.weights_dir / "PP-FormulaNet_plus-M.pth", args.reference),
        "samples": [],
    }
    inputs = []
    for index, path in enumerate(paths):
        image = np.asarray(Image.open(path).convert("RGB"))
        value = processor._preprocess(image)
        inputs.append(value)
        outputs, timings = {}, {}
        for label, model in sessions.items():
            started = time.perf_counter()
            outputs[label] = model.run(None, {model.get_inputs()[0].name: value})[0]
            timings[label] = time.perf_counter() - started
        left, right = outputs["reference"], outputs["candidate"]
        sample = {
            "path": str(path),
            "token_match": bool(np.array_equal(left, right)),
            "reference_tokens": left.tolist(),
            "candidate_tokens": right.tolist(),
            "reference_latex": processor._decode_tokens(left),
            "candidate_latex": processor._decode_tokens(right),
            "seconds": timings,
        }
        sample["latex_match"] = sample["reference_latex"] == sample["candidate_latex"]
        report["samples"].append(sample)
        if (index + 1) % 10 == 0 or index == len(paths) - 1:
            print(
                f"complete {index + 1}/{len(paths)} token_matches={sum(s['token_match'] for s in report['samples'])}",
                flush=True,
            )
            write_json(args.output_dir / "comparison.json", report)
    report["count"] = len(paths)
    report["token_matches"] = sum(sample["token_match"] for sample in report["samples"])
    report["latex_matches"] = sum(sample["latex_match"] for sample in report["samples"])
    report["seconds"] = {key: sum(sample["seconds"][key] for sample in report["samples"]) for key in sessions}
    del sessions, processor.session
    for label, source in (("reference", args.reference), ("candidate", args.candidate)):
        diagnostic_model(source, args.output_dir / f"{label}-diagnostic.onnx", candidate=label == "candidate")
    diagnostic_sessions = {label: session(args.output_dir / f"{label}-diagnostic.onnx") for label in ("reference", "candidate")}
    torch.set_num_threads(2)
    recognizer = FormulaRecognizer(str(args.weights_dir), "cpu")
    encoder, step = Encoder(recognizer.net).eval(), DecoderStep(recognizer.net).eval()
    # 均匀选取不同 PDF 来源和长短公式，比较相同 token 前缀下的中间 logits。
    selected = sorted(set(np.linspace(0, len(inputs) - 1, min(8, len(inputs)), dtype=int).tolist()))
    report["numerics"] = []
    for index in selected:
        value = inputs[index]
        outputs = {key: model.run(None, {model.get_inputs()[0].name: value}) for key, model in diagnostic_sessions.items()}
        with torch.inference_mode():
            features = encoder(torch.from_numpy(value))
            caches = tuple(
                tensor
                for i in range(6)
                for tensor in (torch.zeros(1, 16, 0, 32), torch.zeros(1, 16, 0, 32), features[1 + 2 * i], features[2 + 2 * i])
            )
            logits = []
            tokens = outputs["candidate"][0]
            for iteration in range(outputs["candidate"][1].shape[0]):
                result = step(torch.from_numpy(tokens[:, iteration : iteration + 1]), caches)
                logits.append(result[0].numpy())
                caches = tuple(
                    tensor
                    for i in range(6)
                    for tensor in (result[1 + 2 * i], result[2 + 2 * i], features[1 + 2 * i], features[2 + 2 * i])
                )
        old, new = outputs["reference"], outputs["candidate"]
        item = {
            "index": index,
            "path": str(paths[index]),
            "prefix_tokens_match": bool(np.array_equal(old[0], new[0])),
            "features_new_vs_old": difference(new[2], old[2]),
            "features_new_vs_torch": difference(new[2], features[0].numpy()),
            "features_old_vs_torch": difference(old[2], features[0].numpy()),
            "logits_new_vs_old": difference(new[1], old[1]),
            "logits_new_vs_torch": difference(new[1], np.stack(logits)),
            "logits_old_vs_torch": difference(old[1], np.stack(logits)),
        }
        report["numerics"].append(item)
        print(
            f"numerics {index}: features max={item['features_new_vs_old'].get('max_abs')}, "
            f"logits max={item['logits_new_vs_old'].get('max_abs')}",
            flush=True,
        )
        write_json(args.output_dir / "comparison.json", report)
    # 动态 batch=2 与单项调用对照，按每个样本首个 EOS 截断，排除不同结束长度的 PAD 差异。
    report["batch_two"] = []
    for label, source in (("reference", args.reference), ("candidate", args.candidate)):
        model = session(source)
        value = np.concatenate([inputs[0], inputs[-1]], axis=0)
        result = model.run(None, {model.get_inputs()[0].name: value})[0]
        decoded = [processor._decode_tokens(row) for row in result]
        expected = [report["samples"][index][f"{label}_latex"] for index in (0, -1)]
        report["batch_two"].append({"model": label, "latex_matches_single": decoded == expected, "latex": decoded})
        del model
    write_json(args.output_dir / "comparison.json", report)
    return report


def main() -> None:
    """解析实验模型路径并输出可审查的 JSON 报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "config", "weights-dir", "inputs-json", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
