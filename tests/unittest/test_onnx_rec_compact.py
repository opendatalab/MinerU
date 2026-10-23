"""Real graph conversion, cache publication and shared ONNX rec output semantics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest

from mineru.model.runtime.onnx_rec import REC_IDS_OUTPUT, REC_SCORES_OUTPUT, prepare_rec_model, validate_rec_outputs


def make_model(path: Path, opset: int = 11, classes: int = 5) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["x"], ["probs"], axis=2)],
        "rec",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", "T", classes])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["B", "T", classes])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    onnx.save(model, path)
    return path


def cpu_session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("opset", [11, 13, 18, 20])
def test_compact_graph_matches_dense_probabilities(tmp_path: Path, opset: int) -> None:
    source = make_model(tmp_path / "original.onnx", opset)
    original_bytes = source.read_bytes()
    compact = prepare_rec_model(str(source), cache_dir=tmp_path / "cache")
    dense, reduced = cpu_session(str(source)), cpu_session(compact.path)
    assert [out.name for out in reduced.get_outputs()] == [REC_IDS_OUTPUT, REC_SCORES_OUTPUT]
    for batch, steps in [(1, 1), (6, 40), (2, 127), (7, 5)]:
        values = np.random.default_rng(42).normal(size=(batch, steps, 5)).astype(np.float32)
        values[0, 0] = 0  # Ties must select the first class, matching NumPy CTC decoding.
        expected = dense.run(None, {"x": values})[0]
        ids, scores = validate_rec_outputs(reduced.run(None, {"x": values}), batch_size=batch, num_classes=5)
        np.testing.assert_array_equal(ids, expected.argmax(-1))
        np.testing.assert_array_equal(scores, expected.max(-1))
    assert source.read_bytes() == original_bytes
    assert compact.source_sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_cache_reuse_invalidation_and_corruption_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model.runtime import onnx_rec

    source = make_model(tmp_path / "original.onnx")
    cache = tmp_path / "cache"
    first = prepare_rec_model(str(source), cache_dir=cache)
    with monkeypatch.context() as context:
        context.setattr(onnx_rec, "_reduce_graph", lambda _: pytest.fail("warm cache must not convert again"))
        assert prepare_rec_model(str(source), cache_dir=cache) == first
    target = Path(first.path)
    for damaged in [target, target.with_suffix(".json")]:
        damaged.write_bytes(b"broken")
        assert prepare_rec_model(str(source), cache_dir=cache) == first
    make_model(source, classes=6)
    second = prepare_rec_model(str(source), cache_dir=cache)
    assert second.path != first.path and second.num_classes == 6
    assert target.is_file()


def test_concurrent_preparation_publishes_one_complete_model(tmp_path: Path) -> None:
    source = make_model(tmp_path / "original.onnx")
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: prepare_rec_model(str(source), cache_dir=cache), range(8)))
    assert len(set(values)) == 1
    onnx.checker.check_model(values[0].path)
    assert len(list(cache.glob("*.onnx"))) == len(list(cache.glob("*.json"))) == 1
    assert not list(cache.glob(".*"))


def test_external_weights_are_rejected_without_publishing_cache(tmp_path: Path) -> None:
    source = make_model(tmp_path / "original.onnx")
    model = onnx.load(source)
    weight = helper.make_tensor("external", TensorProto.FLOAT, [1], [1.0])
    weight.data_location = TensorProto.EXTERNAL
    weight.external_data.add(key="location", value="weights.bin")
    model.graph.initializer.append(weight)
    source.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="embedded weights"):
        prepare_rec_model(str(source), cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.onnx"))


@pytest.mark.parametrize("bad", ["nan", "negative", "too_large", "id_bounds", "id_dtype", "shape", "empty", "count"])
def test_invalid_compact_outputs_are_rejected(bad: str) -> None:
    ids = np.zeros((2, 3), dtype=np.int64)
    scores = np.ones((2, 3), dtype=np.float32)
    if bad == "nan":
        scores[0, 0] = np.nan
    elif bad == "negative":
        scores[0, 0] = -0.1
    elif bad == "too_large":
        scores[0, 0] = 2
    elif bad == "id_bounds":
        ids[0, 0] = 5
    elif bad == "id_dtype":
        ids = ids.astype(np.float32)
    elif bad == "shape":
        scores = scores[:, :1]
    elif bad == "empty":
        ids, scores = ids[:, :0], scores[:, :0]
    outputs = [ids] if bad == "count" else [ids, scores]
    with pytest.raises(ValueError):
        validate_rec_outputs(outputs, batch_size=2, num_classes=5)


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "DmlExecutionProvider", "WebGpuExecutionProvider"])
def test_all_backends_use_compact_graph_and_restore_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from mineru.model.ocr import pp_ocr_v6_onnx as ocr

    source = make_model(tmp_path / "original.onnx")
    compact = prepare_rec_model(str(source), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(ocr, "prepare_rec_model", lambda path: compact)
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("A\nB\nC\n", encoding="utf-8")
    batches = []

    def run(names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == [REC_IDS_OUTPUT, REC_SCORES_OUTPUT]
        labels = feeds["x"][:, 0, 0, 0].astype(np.int64)
        batches.append(len(labels))
        ids = np.stack([labels, labels, np.zeros_like(labels), labels], axis=1)
        return [ids, np.full(ids.shape, 0.75, dtype=np.float32)]

    def session(path: str, *args: object, **kwargs: object) -> SimpleNamespace:
        assert path == compact.path
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="x")], run=run, get_providers=lambda: [provider])

    monkeypatch.setattr(ocr, "ort_session", session)
    rec = ocr.TextRecognizerONNX(str(source), str(dictionary), rec_batch_num=2)
    monkeypatch.setattr(rec, "_resize_norm_img", lambda image, ratio: np.full((3, 48, 320), image[0, 0, 0], dtype=np.float32))
    inputs = [np.full((10, width, 3), label, dtype=np.uint8) for width, label in [(30, 1), (10, 2), (20, 3), (40, 0), (50, 1)]]
    result, _ = rec(inputs)
    assert result == [("AA", 0.75), ("BB", 0.75), ("CC", 0.75), ("", 1.0), ("AA", 0.75)]
    assert batches == [2, 2, 1]
    assert rec([]) == ([], 0.0)
    manifest = json.loads(Path(compact.path).with_suffix(".json").read_text())
    assert manifest["num_classes"] == len(rec.character)
