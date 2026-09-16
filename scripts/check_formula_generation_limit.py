# Copyright (c) Opendatalab. All rights reserved.
"""在诊断副本中固定预测 token，验证两个 ONNX 原有循环的超长序列停止条件。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .compare_formula_onnx import session, write_json

__all__ = ["check_limit"]


def free_inputs(graph: onnx.GraphProto) -> set[str]:
    """收集子图捕获的外部值，保证裁剪时不误删 If/Loop 的依赖。"""
    defined = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    defined.update(name for node in graph.node for name in node.output)
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                used.update(free_inputs(attr.g))
    return used - defined


def prune(graph: onnx.GraphProto) -> None:
    """只裁剪不再影响输出的神经网络计算，保留原停止条件的全部依赖。"""
    needed = {value.name for value in graph.output}
    retained = []
    for node in reversed(graph.node):
        if not needed.intersection(node.output):
            continue
        retained.append(node)
        needed.update(name for name in node.input if name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                prune(attr.g)
                needed.update(free_inputs(attr.g))
    del graph.node[:]
    graph.node.extend(reversed(retained))
    weights = [value for value in graph.initializer if value.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(weights)
    # 来源图中缓存的旧 shape 已不适用于诊断替身，让 ORT 从实际初值重新推导。
    del graph.value_info[:]


def check_limit(source: Path, target: Path, *, candidate: bool) -> dict:
    """固定 argmax=7 且让无关缓存原样传递，执行原图剩余的停止逻辑。"""
    model = onnx.load(source)
    loop = next(node for node in model.graph.node if node.op_type == "Loop")
    body = next(attr.g for attr in loop.attribute if attr.name == "body")
    argmax = next(node for node in body.node if node.op_type == "ArgMax")
    output = argmax.output[0]
    argmax.CopyFrom(helper.make_node("Identity", ["fixed_prediction"], [output]))
    body.initializer.append(numpy_helper.from_array(np.array([7], np.int64), "fixed_prediction"))
    keep_states = {0, 1, 2} if candidate else {0, 1, 2, 3, 4, 29}
    for index, value in enumerate(body.output[1:]):
        if index not in keep_states:
            name = f"unchanged_state_{index}"
            body.node.append(helper.make_node("Identity", [body.input[index + 2].name], [name]))
            value.CopyFrom(body.input[index + 2])
            value.name = name
            value.type.tensor_type.ClearField("shape")
            body.input[index + 2].type.tensor_type.ClearField("shape")
    prune(model.graph)
    onnx.save(model, target)
    runtime = session(target)
    result = runtime.run(None, {runtime.get_inputs()[0].name: np.zeros((1, 1, 384, 384), np.float32)})[0]
    return {
        "source": str(source.resolve()),
        "diagnostic_model": str(target.resolve()),
        "method": "Only replace argmax with token 7 and pass through unused cache state; retain original stop logic",
        "tokens_including_bos": int(result.shape[1]),
        "generated_tokens": int(result.shape[1] - 1),
        "last_token": int(result[0, -1]),
        "eos_present": bool(np.any(result == 2)),
        "diagnostic_bytes": target.stat().st_size,
    }


def main() -> None:
    """对两个本地模型生成小型停止逻辑诊断图并保存结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        label: check_limit(source, args.output_dir / f"{label}-generation-limit.onnx", candidate=label == "candidate")
        for label, source in (("reference", args.reference), ("candidate", args.candidate))
    }
    write_json(args.output_dir / "generation-limits.json", result)
    print(result)


if __name__ == "__main__":
    main()
