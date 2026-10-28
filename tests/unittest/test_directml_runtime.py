"""Production DML presets, explicit calibration sessions, serialization and MFR tails."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.runtime.gpu import classify_d3d12_gpu, with_gpu_classification
from mineru.model.runtime import onnx
from mineru.model.runtime.onnx_device import OnnxSelection
from mineru.model.runtime.windows_gpu import GpuAdapter


@pytest.mark.parametrize(
    ("vendor", "enabled"),
    [
        (0x8086, [False, True, False, False]),
        (0x1002, [True, True, True, True]),
        (0x10DE, [True, True, False, False]),
        (0xFFFF, [None, None, None, None]),
    ],
)
def test_vendor_presets_and_actual_index(monkeypatch: pytest.MonkeyPatch, vendor: int, enabled: list[bool | None]) -> None:
    adapter = with_gpu_classification(GpuAdapter(7, "luid", "gpu", vendor, 1, 0), classify_d3d12_gpu(True))
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    native = SimpleNamespace(get_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"], disable_fallback=Mock())
    factory = Mock(return_value=native)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    for kind, preset in zip(("layout", "mfr", "ocr_rec", "unet"), enabled):
        session = onnx.create_ort_session("test.onnx", adapter=adapter, model_kind=kind)
        assert isinstance(session, onnx.DirectMLSession)
        kwargs = factory.call_args.kwargs
        options = kwargs["providers"][0][1]
        assert options["device_id"] == 7
        assert options.get("disable_metacommands") == (None if preset is None else "false" if preset else "true")
        assert not kwargs["sess_options"].enable_mem_pattern
        assert kwargs["sess_options"].execution_mode == onnx.ort.ExecutionMode.ORT_SEQUENTIAL
    assert native.disable_fallback.call_count == 4


def test_cpu_models_do_not_select_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx, "sys", SimpleNamespace(platform="win32"))
    select = Mock(side_effect=AssertionError("selection"))
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(onnx, "resolve_onnx_device", select)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    for kind in ("", "ocr_det", "seal_det", "table", "mfr_tail_cpu"):
        onnx.ort_session("test.onnx", model_kind=kind)
    select.assert_not_called()
    assert all(call.kwargs["providers"] == onnx.ort_providers() for call in factory.call_args_list)


@pytest.mark.parametrize("failure", ["unavailable", "exception", "silent"])
@pytest.mark.parametrize("kind", ["layout", "unet"])
def test_production_fallback_does_not_save_gpu(monkeypatch: pytest.MonkeyPatch, failure: str, kind: str) -> None:
    adapter = with_gpu_classification(GpuAdapter(1, "id", "gpu", 0x8086, 1, 0), classify_d3d12_gpu(False))
    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(
        onnx.ort, "get_available_providers", lambda: [] if failure == "unavailable" else ["DmlExecutionProvider"]
    )
    factory = Mock(side_effect=[RuntimeError("gpu init"), cpu] if failure == "exception" else None, return_value=cpu)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    save = Mock()
    monkeypatch.setattr(onnx, "confirm_gpu_session", save)
    opts = onnx.ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_profiling = True
    opts.profile_file_prefix = "unet-profile"
    assert (
        onnx.ort_session(
            "test.onnx", model_kind=kind, selection=OnnxSelection("gpu", adapter, "discrete_gpu", "dml"), sess_options=opts
        )
        is cpu
    )
    save.assert_not_called()
    assert factory.call_args.kwargs["providers"] == onnx.ort_providers()
    for call in factory.call_args_list:
        assert call.kwargs["sess_options"] is opts
        assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (2, 1)
        assert not opts.enable_cpu_mem_arena
        assert opts.enable_profiling and opts.profile_file_prefix == "unet-profile"


def test_calibration_factory_fails_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["DmlExecutionProvider"])
    monkeypatch.setattr(
        onnx.ort, "InferenceSession", Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    )
    with pytest.raises(RuntimeError, match="fell back"):
        onnx.create_ort_session(
            "test.onnx", adapter=with_gpu_classification(GpuAdapter(0, "id", "gpu", 1, 1, 0), classify_d3d12_gpu(True))
        )


def test_run_is_serialized_per_session() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def run(names: object, feed: object, options: object) -> list[int]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        time.sleep(0.01)
        with lock:
            active -= 1
        return [1]

    session = onnx.DirectMLSession(SimpleNamespace(run=run))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: session.run(None, {}), range(8))) == [[1]] * 8
    assert maximum == 1


@pytest.mark.parametrize(("count", "batch"), [(1, 8), (7, 8), (8, 8), (9, 8), (16, 8), (17, 8), (8, 4)])
def test_mfr_only_actual_batch_eight_uses_gpu(monkeypatch: pytest.MonkeyPatch, count: int, batch: int) -> None:
    from mineru.model.mfr import pp_formulanet_plus_m_onnx as mfr

    model = object.__new__(mfr.PPFormulaNetPlusMONNX)
    model.cpu_tail_enabled = True
    model._cpu_tail_session = None
    model._cpu_tail_lock = threading.Lock()
    model._model_path = "mfr.onnx"
    model._intra_op_num_threads = 4
    model.input_name = "image"
    gpu_calls, cpu_calls = [], []

    def run(calls: list[int], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = feeds["image"]
        calls.append(len(tensor))
        return [tensor[:, 0, 0, 0].astype(np.int64)[:, None]]

    model.session = SimpleNamespace(run=lambda names, feeds: run(gpu_calls, feeds))
    factory = Mock(return_value=SimpleNamespace(run=lambda names, feeds: run(cpu_calls, feeds)))
    monkeypatch.setattr(mfr, "create_ort_session", factory)
    monkeypatch.setattr(model, "_preprocess", lambda crop: np.full((1, 1, 2, 2), crop[0, 0, 0]) if crop.size else None)
    monkeypatch.setattr(model, "_decode_tokens", lambda tokens: str(tokens[0]))
    crops = [np.full((value, value, 3), value, dtype=np.uint8) for value in range(count, 0, -1)]
    crops.insert(1, np.empty((0, 0, 3), dtype=np.uint8))
    expected = [str(value) for value in range(count, 0, -1)]
    expected.insert(1, "")
    for _ in range(2):
        assert model._infer_batch(crops, batch_size=batch) == expected
    assert all(size == 8 for size in gpu_calls)
    assert all(0 < size < 8 for size in cpu_calls)
    assert sum(gpu_calls + cpu_calls) == count * 2
    assert factory.call_count == (1 if cpu_calls else 0)
