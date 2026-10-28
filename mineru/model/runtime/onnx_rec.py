# Copyright (c) Opendatalab. All rights reserved.
"""Content-addressed ONNX CTC output reduction shared by CPU, DML and WebGPU."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    import numpy as np

REC_OUTPUT_FORMAT = "ctc-ids-scores-v1"
REC_IDS_OUTPUT = "mineru_rec_ids"
REC_SCORES_OUTPUT = "mineru_rec_scores"


@dataclass(frozen=True)
class CompactRecModel:
    """Derived graph identity; contains no native session or device handles."""

    path: str
    num_classes: int
    source_sha256: str
    model_sha256: str


def _atomic_write(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reduce_graph(source: bytes) -> tuple[bytes, int]:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_model_from_string(source)
    # The distributed rec graph embeds its weights. Reject external tensors rather than
    # reusing a cache key which ignores changes to an external weights file.
    try:
        onnx.external_data_helper.convert_model_from_external_data(model)
    except ValueError as error:
        raise ValueError("Compact OCR rec requires an ONNX model with embedded weights") from error
    if len(model.graph.output) != 1:
        raise ValueError("OCR rec requires one [batch, time, classes] probability output")
    original = model.graph.output[0]
    tensor = original.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 3 or tensor.shape.dim[2].dim_value < 2:
        raise ValueError("OCR rec requires float32 probabilities with a fixed class dimension")
    classes = tensor.shape.dim[2].dim_value
    opset = next((item.version for item in model.opset_import if item.domain == ""), 0)
    if opset < 11:
        raise ValueError("Compact OCR rec requires ONNX opset >= 11")
    names = {name for node in model.graph.node for name in (*node.input, *node.output, node.name)}
    names.update(item.name for item in model.graph.initializer)
    names.update(item.name for item in model.graph.input)
    axes_name = "mineru_rec_class_axis"
    if names.intersection((REC_IDS_OUTPUT, REC_SCORES_OUTPUT, axes_name)):
        raise ValueError("OCR rec model contains reserved compact output names")
    model.graph.node.append(
        helper.make_node("ArgMax", [original.name], [REC_IDS_OUTPUT], name=REC_IDS_OUTPUT, axis=2, keepdims=0)
    )
    if opset >= 18:
        model.graph.initializer.append(helper.make_tensor(axes_name, TensorProto.INT64, [1], [2]))
        reduction = helper.make_node(
            "ReduceMax", [original.name, axes_name], [REC_SCORES_OUTPUT], name=REC_SCORES_OUTPUT, keepdims=0
        )
    else:
        reduction = helper.make_node(
            "ReduceMax", [original.name], [REC_SCORES_OUTPUT], name=REC_SCORES_OUTPUT, axes=[2], keepdims=0
        )
    model.graph.node.append(reduction)
    outputs = []
    for name, dtype in ((REC_IDS_OUTPUT, TensorProto.INT64), (REC_SCORES_OUTPUT, TensorProto.FLOAT)):
        output = helper.make_tensor_value_info(name, dtype, None)
        output.type.tensor_type.shape.dim.extend(tensor.shape.dim[:2])
        outputs.append(output)
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    onnx.checker.check_model(model)
    return model.SerializeToString(), classes


def prepare_rec_model(model_path: str, *, cache_dir: Path | None = None) -> CompactRecModel:
    """Keep the source immutable and atomically cache a checked, compact-output graph."""
    if cache_dir is None:
        from ...config import get_config

        cache_dir = Path(get_config().model.base_dir).expanduser() / ".derived" / "ocr-rec"
    source = Path(model_path).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    target = cache_dir / f"{REC_OUTPUT_FORMAT}-{source_sha}.onnx"
    manifest = target.with_suffix(".json")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target.with_suffix(".lock")), timeout=30):
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                info["format"] == REC_OUTPUT_FORMAT
                and info["source_sha256"] == source_sha
                and isinstance(info["num_classes"], int)
                and info["num_classes"] >= 2
                and hashlib.sha256(target.read_bytes()).hexdigest() == info["model_sha256"]
            ):
                return CompactRecModel(str(target), info["num_classes"], source_sha, info["model_sha256"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        derived, classes = _reduce_graph(source)
        derived_sha = hashlib.sha256(derived).hexdigest()
        _atomic_write(target, derived)
        _atomic_write(
            manifest,
            json.dumps(
                {"format": REC_OUTPUT_FORMAT, "source_sha256": source_sha, "model_sha256": derived_sha, "num_classes": classes}
            ).encode("utf-8"),
        )
    return CompactRecModel(str(target), classes, source_sha, derived_sha)


def validate_rec_outputs(outputs: list[np.ndarray], *, batch_size: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate the compact contract before CTC decoding or an automatic device decision."""
    import numpy as np

    if len(outputs) != 2:
        raise ValueError("OCR rec requires compact ID and score outputs")
    ids, scores = outputs
    if (
        ids.ndim != 2
        or ids.shape[0] != batch_size
        or ids.shape[1] == 0
        or scores.shape != ids.shape
        or ids.dtype != np.int64
        or scores.dtype != np.float32
        or np.any(ids < 0)
        or np.any(ids >= num_classes)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1 + 1e-5)
    ):
        raise ValueError("Invalid compact OCR rec IDs or confidence scores")
    return ids, scores


__all__ = [
    "CompactRecModel",
    "REC_IDS_OUTPUT",
    "REC_OUTPUT_FORMAT",
    "REC_SCORES_OUTPUT",
    "prepare_rec_model",
    "validate_rec_outputs",
]
