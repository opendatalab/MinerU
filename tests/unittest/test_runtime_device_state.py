"""验证进程设备快照、并发初始化与局部模型覆盖。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.config import get_config
from mineru.model.runtime import device, device_state, onnx_device
from mineru.model.runtime.device_types import OnnxSelection, RuntimeDevice, TorchDeviceSelection


@pytest.mark.parametrize(
    "selection,backend,mode",
    [
        (TorchDeviceSelection("cpu"), "torch", "cpu"),
        (TorchDeviceSelection("cuda:1"), "torch", "gpu"),
        (TorchDeviceSelection("mps"), "torch", "gpu"),
        (TorchDeviceSelection("npu:0"), "torch", "gpu"),
        (OnnxSelection(), "onnx", "cpu"),
        (OnnxSelection("gpu"), "onnx", "gpu"),
    ],
)
def test_device_mode_is_derived(selection: OnnxSelection | TorchDeviceSelection, backend: str, mode: str) -> None:
    """模式与后端从同一选型派生，不能构造相互矛盾的字段。"""
    result = RuntimeDevice(selection)
    assert result.backend == backend and result.device == mode
    assert result.is_cpu == (mode == "cpu")


def test_import_and_cached_read_do_not_load_inference_backends() -> None:
    """独立解释器导入及查询不加载重依赖，也不创建初始化状态。"""
    script = """
import sys
from mineru.model.runtime import device_state
assert device_state.cached_runtime_device() is None
assert not device_state._states
assert not ({'torch', 'onnxruntime', 'transformers', 'mineru_llama_cpp'} & sys.modules.keys())
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)


def test_concurrent_initialization_is_once_and_peek_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """初始化期间只读查询立即返回 None，所有等待者最终共享同一快照。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    entered, release = threading.Event(), threading.Event()

    def select(**kwargs: object) -> OnnxSelection:
        """阻塞底层选型以验证实际等待者及无锁查询。"""
        entered.set()
        assert release.wait(5)
        return OnnxSelection("gpu", reason="test")

    probe = Mock(side_effect=select)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", probe)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(device_state.resolve_runtime_device) for _ in range(8)]
        try:
            assert entered.wait(5)
            assert device_state.cached_runtime_device() is None
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]
    assert all(value is results[0] for value in results)
    probe.assert_called_once()


def test_backend_config_environment_and_options_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """首轮后的配置、环境和参数变化都不能改变正式快照。"""
    monkeypatch.setattr(get_config().model, "small_backend", "torch")
    torch = Mock(return_value=TorchDeviceSelection("cuda:1"))
    monkeypatch.setattr(device, "resolve_torch_device", torch)
    onnx = Mock(side_effect=AssertionError("unrelated ONNX"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", onnx)
    first = device_state.resolve_runtime_device()
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    monkeypatch.setattr(get_config().model, "onnx_device", "gpu")
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cpu")
    assert device_state.resolve_runtime_device(allow_download=False, intra_op_num_threads=8) is first
    assert device_state.resolve_model_device() is first
    torch.assert_called_once()
    onnx.assert_not_called()


def test_failed_initialization_does_not_publish_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """未完成选型的异常不留下 CPU 占位，也不阻止后续合法初始化。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    probe = Mock(side_effect=[ValueError("invalid startup"), OnnxSelection("gpu")])
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", probe)
    with pytest.raises(ValueError, match="invalid startup"):
        device_state.resolve_runtime_device()
    assert device_state.cached_runtime_device() is None
    assert device_state.resolve_runtime_device().device == "gpu"


def test_formal_offline_cpu_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """正式离线运行确认 CPU 回退后，在线调用不能悄悄重新选型。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    probe = Mock(return_value=OnnxSelection(reason="selection_failed"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", probe)
    first = device_state.resolve_runtime_device(allow_download=False)
    assert first.is_cpu
    assert device_state.resolve_runtime_device() is first
    probe.assert_called_once_with(allow_download=False, intra_op_num_threads=0, inter_op_num_threads=0)


def test_new_pid_does_not_reuse_parent_decision_or_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """子进程按新 PID 创建状态，不读取父进程快照或等待父进程锁。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    probe = Mock(side_effect=[OnnxSelection(), OnnxSelection("gpu")])
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", probe)
    parent_pid = os.getpid()
    parent = device_state.resolve_runtime_device()
    parent_state = device_state._states[parent_pid]
    monkeypatch.setattr(device_state, "os", SimpleNamespace(getpid=lambda: parent_pid + 1))
    assert device_state.cached_runtime_device() is None
    child = device_state.resolve_runtime_device()
    assert child is not parent and child.device == "gpu"
    assert device_state._states[parent_pid + 1].lock is not parent_state.lock


def test_explicit_model_scope_preserves_default_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """局部模型覆盖只传递给嵌套构造，其他线程和默认策略继续使用原快照。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    default = device_state.resolve_runtime_device()
    override = device_state.resolve_model_device("torch", "cuda:2")
    with device_state.model_device_scope(override):
        assert device_state.resolve_model_device() is override
        assert device_state.resolve_runtime_device() is default
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(device_state.resolve_model_device).result(timeout=5) is default
    assert device_state.resolve_model_device() is default


def test_explicit_torch_cpu_does_not_initialize_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """明确的局部 Torch CPU 构造不触发无关默认后端的下载或校准。"""
    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    forbidden = Mock(side_effect=AssertionError("default selection"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", forbidden)
    assert device_state.resolve_model_device("torch", "cpu").is_cpu
    assert device_state.cached_runtime_device() is None
    forbidden.assert_not_called()


def test_asset_and_dependency_selection_follow_frozen_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """运行前的资产和依赖查询无副作用，决策发布后也不能随配置切换到另一后端。"""
    from mineru.model.registry import MINERU_4_MODELS_ONNX, MINERU_4_MODELS_TORCH, small_model_repo
    from mineru.parser.tier import required_modules_for_tier

    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    assert small_model_repo() is MINERU_4_MODELS_ONNX
    assert required_modules_for_tier("basic") == ["onnxruntime"]
    assert device_state.cached_runtime_device() is None
    device_state.resolve_runtime_device()
    monkeypatch.setattr(get_config().model, "small_backend", "torch")
    assert small_model_repo() is MINERU_4_MODELS_ONNX
    assert required_modules_for_tier("basic") == ["onnxruntime"]
    assert small_model_repo("torch") is MINERU_4_MODELS_TORCH


@pytest.mark.parametrize("order", list(permutations(("tier", "llama", "model"))))
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_consumers_share_one_decision_in_any_order(monkeypatch: pytest.MonkeyPatch, order: tuple[str, ...], mode: str) -> None:
    """档位、llama 和实际 ONNX 会话的不同首次调用顺序都只执行一次选型。"""
    from mineru.model.runtime import llama_device, onnx, tier_recommendation
    from mineru.model.runtime.windows_gpu import GpuAdapter

    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    monkeypatch.setattr(get_config().model.vlm, "engine", "llama-cpp")
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(llama_device, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: 8)
    monkeypatch.setattr(tier_recommendation, "inspect_calibration_record", lambda _: None)
    adapter = GpuAdapter(1, "id", "test", 0x10DE, 1, 0) if mode == "gpu" else None
    choice = OnnxSelection(mode, adapter, "rec_calibration", "dml")
    probe = Mock(return_value=choice)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", probe)
    session = SimpleNamespace(get_providers=lambda: ["DmlExecutionProvider" if adapter else "CPUExecutionProvider"])
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnx, "create_ort_session", factory)
    for consumer in order:
        if consumer == "tier":
            report = tier_recommendation.recommend_tiers()
            assert report.ceiling == ("advanced" if mode == "gpu" else "basic")
        elif consumer == "llama":
            assert llama_device.resolve_llama_device().device == mode
        else:
            assert onnx.ort_session("test.onnx", model_kind="layout") is session
            assert factory.call_args.kwargs["adapter"] is adapter
        # 配置写回或用户修改不应改变后续消费者的运行设备。
        monkeypatch.setattr(get_config().model, "onnx_device", "cpu" if mode == "gpu" else "gpu")
    probe.assert_called_once()
    assert device_state.cached_runtime_device().selection is choice


def test_session_cpu_fallback_does_not_change_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """单个模型 GPU 初始化失败后实际使用 CPU，进程默认策略仍保持最初决策。"""
    from mineru.model.runtime import onnx
    from mineru.model.runtime.windows_gpu import GpuAdapter

    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    choice = OnnxSelection("gpu", GpuAdapter(1, "id", "test", 0x10DE, 1, 0), "discrete_gpu", "dml")
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", Mock(return_value=choice))
    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(side_effect=[RuntimeError("GPU init"), cpu])
    monkeypatch.setattr(onnx, "create_ort_session", factory)
    confirm = Mock()
    monkeypatch.setattr(onnx, "confirm_gpu_session", confirm)
    assert onnx.ort_session("test.onnx", model_kind="layout") is cpu
    assert device_state.cached_runtime_device().device == "gpu"
    assert factory.call_count == 2
    confirm.assert_not_called()


def test_hardware_summary_reads_only_current_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """诊断将配置与本进程快照分开展示，查询本身不触发正式选型。"""
    from mineru.model.runtime.hardware import config_summary

    monkeypatch.setattr(get_config().model, "small_backend", "onnx")
    summary = config_summary()
    assert summary["process_runtime"] == {"pid": os.getpid(), "device": None, "small_backend": None}
    assert device_state.cached_runtime_device() is None
    device_state.resolve_runtime_device()
    monkeypatch.setattr(get_config().model, "onnx_device", "gpu")
    summary = config_summary()
    assert summary["settings"]["model.onnx_device"]["value"] == "gpu"
    assert summary["process_runtime"]["device"] == "cpu"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="需要 POSIX fork 验证继承锁边界")
def test_fork_child_ignores_lock_held_by_parent_thread() -> None:
    """独立解释器实际 fork 时，子进程绕过被父线程占用的初始化锁。"""
    script = """
import os
import threading
from mineru.config import get_config
from mineru.model.runtime import device_state
get_config().model.small_backend = 'torch'
os.environ['MINERU_DEVICE_MODE'] = 'cpu'
parent = device_state.resolve_runtime_device()
state = device_state._states[os.getpid()]
held, release = threading.Event(), threading.Event()
def hold():
    # 在父线程占用初始化锁，复现 fork 时继承不可释放锁的边界。
    with state.lock:
        held.set()
        release.wait(10)
thread = threading.Thread(target=hold)
thread.start()
assert held.wait(5)
pid = os.fork()
if pid == 0:
    try:
        assert device_state.cached_runtime_device() is None
        child = device_state.resolve_runtime_device()
        assert child is not parent and child.is_cpu
    except BaseException:
        os._exit(1)
    os._exit(0)
try:
    _, status = os.waitpid(pid, 0)
    assert status == 0
finally:
    release.set()
    thread.join(5)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=20)
