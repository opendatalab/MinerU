"""表格 CUDA provider 的自动选择、显式覆盖、回退与 CPU 模型边界。"""

from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest

from mineru.model.runtime import onnx


@pytest.fixture(autouse=True)
def clear_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离开发机线程配置，确保默认值与覆盖行为可以独立验证。"""
    monkeypatch.delenv("MINERU_INTRA_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MINERU_INTER_OP_NUM_THREADS", raising=False)


@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize(
    ("intra", "inter", "expected"),
    [
        (None, None, (4, 1)),
        ("", "invalid", (4, 1)),
        ("0", "-1", (4, 1)),
        ("-2", "1.5", (4, 1)),
        ("8", "2", (8, 2)),
        ("2", None, (2, 1)),
        (None, "3", (4, 3)),
    ],
)
def test_session_thread_defaults_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    table: bool,
    intra: str | None,
    inter: str | None,
    expected: tuple[int, int],
) -> None:
    """两个会话入口都必须显式设置线程，非法环境值不能退回 ORT 自动扩张。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu")
    if intra is not None:
        monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", intra)
    if inter is not None:
        monkeypatch.setenv("MINERU_INTER_OP_NUM_THREADS", inter)
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    if table:
        onnx.table_ort_session("table.onnx")
    else:
        onnx.ort_session("model.onnx")
    opts = factory.call_args.kwargs["sess_options"]
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == expected
    assert opts.execution_mode == onnx.ort.ExecutionMode.ORT_SEQUENTIAL


@pytest.mark.parametrize(("explicit", "expected"), [(2, 2), (0, 8), (-1, 8)])
def test_explicit_intra_threads_take_precedence(monkeypatch: pytest.MonkeyPatch, explicit: int, expected: int) -> None:
    """显式正数优先于环境变量，非正数参数沿用统一回退规则。"""
    monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "8")
    factory = Mock()
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    onnx.ort_session("model.onnx", intra_op_num_threads=explicit)
    opts = factory.call_args.kwargs["sess_options"]
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (expected, 1)


def test_explicit_table_session_options_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """表格工厂保留显式线程和内存配置，仅补齐未指定的线程设置。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu")
    monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "8")
    monkeypatch.setenv("MINERU_INTER_OP_NUM_THREADS", "3")
    opts = onnx.ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.enable_cpu_mem_arena = False
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    onnx.table_ort_session("table.onnx", sess_options=opts)
    assert factory.call_args.kwargs["sess_options"] is opts
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (2, 3)
    assert opts.enable_cpu_mem_arena is False


def test_table_default_uses_cpu_without_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU ORT 包保持 CPU 执行，Torch 设备不参与表格 provider 决策。"""
    monkeypatch.delenv("MINERU_TABLE_DEVICE", raising=False)
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda")
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    assert onnx.table_ort_session("model.onnx") is session
    assert factory.call_args.kwargs["providers"] == onnx.ort_providers()


@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_cuda_selection_and_other_onnx_models_stay_cpu(monkeypatch: pytest.MonkeyPatch, device: str | None) -> None:
    """默认自动和显式 CUDA 都选择 GPU，但 OCR/Layout ONNX 仍保留 CPU 策略。"""
    if device is None:
        monkeypatch.delenv("MINERU_TABLE_DEVICE", raising=False)
    else:
        monkeypatch.setenv("MINERU_TABLE_DEVICE", device)
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


def test_explicit_cpu_overrides_available_cuda_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Space 的显式 CPU 配置不能因安装 GPU ORT 包而申请 CUDA 会话。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu")
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    onnx.table_ort_session("table.onnx")
    assert factory.call_args.kwargs["providers"] == onnx.ort_providers()


@pytest.mark.parametrize("device", ["auto", "cuda"])
@pytest.mark.parametrize("mode", ["unavailable", "exception", "silent"])
def test_cuda_failure_is_reported_and_cpu_remains_available(monkeypatch: pytest.MonkeyPatch, mode: str, device: str) -> None:
    """覆盖未安装 CUDA、动态库失败以及 ORT 自动降级，不能把回退误报为 GPU。"""
    monkeypatch.setenv("MINERU_TABLE_DEVICE", device)
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
    assert warning.called == (mode != "unavailable" or device == "cuda")
    for call in factory.call_args_list:
        opts = call.kwargs["sess_options"]
        assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (4, 1)
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


@pytest.mark.parametrize("explicit", [False, True])
def test_table_structure_entrypoints_use_shared_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit: bool
) -> None:
    """真实结构模型入口保留配置字典和显式线程，两个包装器共用环境与默认策略。"""
    from mineru.model.table.rec.slanet_plus.table_structure import TableStructurer
    from mineru.model.table.rec.unet_table.table_structure_unet import TSRUnet

    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu")
    monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "8")
    monkeypatch.setenv("MINERU_INTER_OP_NUM_THREADS", "3")
    model = tmp_path / "table.onnx"
    model.touch()
    config = {"model_path": str(model)}
    if explicit:
        config.update(intra_op_num_threads=2, inter_op_num_threads=1)
    original = config.copy()
    session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"character": "<td>\n</td>"}),
    )
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    TableStructurer(config)
    TSRUnet(config)
    assert config == original
    assert factory.call_count == 2
    for call in factory.call_args_list:
        opts = call.kwargs["sess_options"]
        assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == ((2, 1) if explicit else (8, 3))
        assert opts.enable_cpu_mem_arena is False
