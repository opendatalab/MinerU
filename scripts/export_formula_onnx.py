# Copyright (c) Opendatalab. All rights reserved.
"""从现有 Plus-M PTH 导出编码器、增量解码器和独立的完整 ONNX 模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer

__all__ = ["export_model", "Encoder", "DecoderStep"]


class Encoder(torch.nn.Module):
    """将视觉编码、维度投影及各层交叉注意力缓存放在循环外计算。"""

    def __init__(self, net: torch.nn.Module) -> None:
        """共享已加载 PTH 的原始参数，不再初始化或修改权重。"""
        super().__init__()
        self.backbone = net.backbone
        self.projection = net.head.enc_to_dec_proj
        self.cross_attention = torch.nn.ModuleList([layer.encoder_attn for layer in net.head.decoder.model.decoder.layers])

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """输出视觉特征和每层固定不变的交叉注意力 K/V。"""
        features = self.backbone(image).last_hidden_state
        projected = self.projection(features)
        caches = []
        for attention in self.cross_attention:
            for linear in (attention.k_proj, attention.v_proj):
                caches.append(linear(projected).reshape(image.shape[0], -1, 16, 32).transpose(1, 2))
        return (features, *caches)


class DecoderStep(torch.nn.Module):
    """保持现有 eager attention 的单 token 增量解码步骤。"""

    def __init__(self, net: torch.nn.Module) -> None:
        """只启用单步导出分支；不使用 growing cache 或 SDPA。"""
        super().__init__()
        self.decoder = net.head.decoder
        self.decoder.model.decoder.is_export = True
        self.layers = len(self.decoder.model.decoder.layers)

    def forward(self, token: torch.Tensor, caches: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """输入各层 self K/V 与 cross K/V，返回 logits 和增长后的 self K/V。"""
        past = tuple(tuple(caches[4 * i : 4 * i + 4]) for i in range(self.layers))
        # 仅形状参与原始 attention 的缓存选择；实际交叉注意力使用传入的 K/V。
        memory = caches[2].transpose(1, 2).reshape(token.shape[0], -1, 512)
        result = self.decoder(
            input_ids=token,
            encoder_hidden_states=memory,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return (result.logits[:, -1, :], *(value for layer in result.past_key_values for value in layer[:2]))


def compose_loop(encoder_path: Path, decoder_path: Path, target: Path) -> None:
    """用公开 ONNX API 组装动态 Loop，保留 BOS、EOS、PAD 和强制 EOS 行为。"""
    encoder = onnx.compose.add_prefix(onnx.load(encoder_path), "encoder/")
    decoder = onnx.compose.add_prefix(onnx.load(decoder_path), "decoder/")
    node = helper.make_node
    info = helper.make_tensor_value_info
    f32, i64, boolean = TensorProto.FLOAT, TensorProto.INT64, TensorProto.BOOL
    initializers = list(encoder.graph.initializer) + list(decoder.graph.initializer)
    constants = {
        "trip_count": np.array(2560, np.int64),
        "initial_condition": np.array(True),
        "zero_index": np.array([0], np.int64),
        "one_dim": np.array([1], np.int64),
        "cache_dims": np.array([16, 0, 32], np.int64),
        "eos": np.array(2, np.int64),
        "pad": np.array(1, np.int64),
        "forced_iteration": np.array(1535, np.int64),
        "axes_one": np.array([1], np.int64),
    }
    initializers += [numpy_helper.from_array(value, key) for key, value in constants.items()]
    nodes = list(encoder.graph.node) + [
        node("Shape", ["encoder/image"], ["image_shape"]),
        node("Gather", ["image_shape", "zero_index"], ["batch_dim"], axis=0),
        node("Concat", ["batch_dim", "one_dim"], ["token_shape"], axis=0),
        node("Concat", ["batch_dim", "cache_dims"], ["cache_shape"], axis=0),
        node("ConstantOfShape", ["token_shape"], ["bos"], value=numpy_helper.from_array(np.array([0], np.int64))),
        node("ConstantOfShape", ["batch_dim"], ["unfinished"], value=numpy_helper.from_array(np.array([True]))),
        node("ConstantOfShape", ["cache_shape"], ["empty_cache"]),
    ]
    # cross K/V 来自外层作用域；只有 self K/V 和生成状态进入循环状态。
    names = {"decoder/token": "current_token"}
    for i in range(6):
        for j, key in enumerate(("self_k", "self_v", "cross_k", "cross_v")):
            names[f"decoder/cache_{i}_{key}"] = f"past_{i}_{j}" if j < 2 else f"encoder/cross_{i}_{j - 2}"
    body_nodes = []
    for original in decoder.graph.node:
        copied = onnx.NodeProto()
        copied.CopyFrom(original)
        for index, name in enumerate(copied.input):
            copied.input[index] = names.get(name, name)
        body_nodes.append(copied)
    body_nodes += [
        node("ArgMax", ["decoder/logits"], ["argmax"], axis=-1, keepdims=0),
        node("Equal", ["iteration", "forced_iteration"], ["force_eos"]),
        node("Where", ["force_eos", "eos", "argmax"], ["predicted"]),
        node("Where", ["active", "predicted", "pad"], ["next_ids"]),
        node("Unsqueeze", ["next_ids", "axes_one"], ["next_token"]),
        node("Concat", ["history", "next_token"], ["next_history"], axis=1),
        node("Equal", ["next_ids", "eos"], ["is_eos"]),
        node("Not", ["is_eos"], ["not_eos"]),
        node("And", ["active", "not_eos"], ["next_active"]),
        node("Cast", ["next_active"], ["active_int"], to=i64),
        node("ReduceMax", ["active_int"], ["any_active"], keepdims=0),
        node("Cast", ["any_active"], ["continue"], to=boolean),
    ]
    cache_inputs = [info(f"past_{i}_{j}", f32, ["batch", 16, "past", 32]) for i in range(6) for j in range(2)]
    cache_outputs = [info(f"decoder/present_{i}_{j}", f32, ["batch", 16, "next_past", 32]) for i in range(6) for j in range(2)]
    body = helper.make_graph(
        body_nodes,
        "autoregressive_decoder",
        [
            info("iteration", i64, []),
            info("condition", boolean, []),
            info("current_token", i64, ["batch", 1]),
            info("history", i64, ["batch", "length"]),
            info("active", boolean, ["batch"]),
            *cache_inputs,
        ],
        [
            info("continue", boolean, []),
            info("next_token", i64, ["batch", 1]),
            info("next_history", i64, ["batch", "next_length"]),
            info("next_active", boolean, ["batch"]),
            *cache_outputs,
        ],
    )
    nodes.append(
        node(
            "Loop",
            ["trip_count", "initial_condition", "bos", "bos", "unfinished", *["empty_cache"] * 12],
            ["last_token", "token_ids", "final_active", *[f"final_cache_{i}" for i in range(12)]],
            body=body,
        )
    )
    graph = helper.make_graph(
        nodes,
        "PP-FormulaNet_plus-M_from_torch",
        list(encoder.graph.input),
        [info("token_ids", i64, ["batch", "length"])],
        initializers,
    )
    model = helper.make_model(graph, producer_name="MinerU torch export", opset_imports=[helper.make_opsetid("", 17)])
    # 当前图只需 IR 8；明确固定格式版本，避免 helper 随 ONNX 包升级抬高运行时门槛。
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, target)


def export_model(weights_dir: Path, output_dir: Path) -> dict[str, Any]:
    """导出完整图并记录来源哈希、版本和图接口，便于独立复现。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    recognizer = FormulaRecognizer(str(weights_dir), "cpu")
    encoder, decoder = Encoder(recognizer.net).eval(), DecoderStep(recognizer.net).eval()
    image = torch.zeros(1, 1, 384, 384)
    with torch.inference_mode():
        encoded = encoder(image)
        caches = tuple(
            tensor
            for i in range(6)
            for tensor in (torch.zeros(1, 16, 1, 32), torch.zeros(1, 16, 1, 32), encoded[1 + 2 * i], encoded[2 + 2 * i])
        )
    cross_names = [f"cross_{i}_{j}" for i in range(6) for j in range(2)]
    cache_names = [f"cache_{i}_{key}" for i in range(6) for key in ("self_k", "self_v", "cross_k", "cross_v")]
    present_names = [f"present_{i}_{j}" for i in range(6) for j in range(2)]
    encoder_path, decoder_path = output_dir / "encoder.onnx", output_dir / "decoder_step.onnx"
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (image,),
            encoder_path,
            dynamo=False,
            opset_version=17,
            input_names=["image"],
            output_names=["features", *cross_names],
            dynamic_axes={name: {0: "batch"} for name in ["image", "features", *cross_names]},
        )
        axes = {name: {0: "batch"} for name in ["token", "logits", *cache_names, *present_names]}
        for name in cache_names:
            if "self_" in name:
                axes[name][2] = "past"
        for name in present_names:
            axes[name][2] = "next_past"
        torch.onnx.export(
            decoder,
            (torch.zeros(1, 1, dtype=torch.int64), caches),
            decoder_path,
            dynamo=False,
            opset_version=17,
            input_names=["token", *cache_names],
            output_names=["logits", *present_names],
            dynamic_axes=axes,
        )
    target = output_dir / "PP-FormulaNet_plus-M.from_torch.onnx"
    compose_loop(encoder_path, decoder_path, target)
    files = [Path(recognizer.weights_path), Path(recognizer.infer_yaml_path), encoder_path, decoder_path, target]
    report = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "opset": 17,
        "ir": 8,
        "precision": "CPU FP32",
        "dynamic_axes": ["batch", "sequence", "self_attention_cache"],
        "input": ["batch", 1, 384, 384],
        "forced_eos_generated_token": 1536,
        "max_new_tokens": 2560,
        "files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            }
            for path in files
        ],
    }
    (output_dir / "export.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    """命令行接收本地 PTH 目录与实验输出目录，不修改正式注册模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_model(args.weights_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
