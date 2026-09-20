"""表格 CUDA provider 的显式选择、回退与 CPU 模型边界。"""

from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest

from mineru.model.runtime import onnx


def test_table_default_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch 使用 CUDA 不应隐式改变表格模型设备。"""
    monkeypatch.delenv("MINERU_TABLE_DEVICE", raising=False)
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda")
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    assert onnx.table_ort_session("model.onnx") is session
    assert factory.call_args.kwargs["providers"] == onnx.ort_providers()


def test_cuda_is_explicit_and_other_onnx_models_stay_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """表格 CUDA 使用低冷启动开销选项，OCR/Layout ONNX 仍保留 CPU 策略。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cuda")
    monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "4")
    monkeypatch.setenv("MINERU_INTER_OP_NUM_THREADS", "1")
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    onnx.table_ort_session("model.onnx")
    call = factory.call_args.kwargs
    assert call["providers"][0][0] == "CUDAExecutionProvider"
    assert call["providers"][0][1]["cudnn_conv_algo_search"] == "HEURISTIC"
    assert call["sess_options"].intra_op_num_threads == 4
    assert call["sess_options"].inter_op_num_threads == 1
    onnx.ort_session("ocr.onnx", device="cuda")
    assert factory.call_args.kwargs["providers"] == onnx.ort_providers()


@pytest.mark.parametrize("mode", ["unavailable", "exception", "silent"])
def test_cuda_failure_is_reported_and_cpu_remains_available(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """覆盖未安装 CUDA、动态库失败以及 ORT 自动降级，不能把回退误报为 GPU。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cuda")
    monkeypatch.setattr(
        onnx.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"] if mode == "unavailable" else ["CUDAExecutionProvider"],
    )
    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(side_effect=[RuntimeError("missing CUDA library"), cpu] if mode == "exception" else None, return_value=cpu)
    warning = Mock()
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    monkeypatch.setattr(onnx.logger, "warning", warning)
    assert onnx.table_ort_session("table.onnx") is cpu
    assert warning.called
    if mode in {"unavailable", "exception"}:
        assert factory.call_args.kwargs["providers"] == onnx.ort_providers()


def test_invalid_device_fails_before_session_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """设备拼写错误不能静默退化成 CPU。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cdua")
    factory = Mock()
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    with pytest.raises(ValueError, match="MINERU_TABLE_DEVICE"):
        onnx.table_ort_session("model.onnx")
    factory.assert_not_called()


def test_all_table_wrappers_use_shared_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """分类、有线和无线表格三个实际入口都必须经过同一个 provider 决策。"""
    from mineru.model.table.cls import paddle_table_cls
    from mineru.model.table.rec.slanet_plus import table_structure_utils
    from mineru.model.table.rec.unet_table import utils

    path = tmp_path / "table.onnx"
    path.touch()
    factory = Mock()
    for module in (paddle_table_cls, table_structure_utils, utils):
        monkeypatch.setattr(module, "table_ort_session", factory)
    paddle_table_cls.PaddleTableClsModel(model_path=str(path))
    table_structure_utils.OrtInferSession({"model_path": str(path)})
    utils.OrtInferSession({"model_path": str(path)})
    assert factory.call_count == 3
    assert all(call.args == (str(path),) for call in factory.call_args_list)
