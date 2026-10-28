"""Device policy, one-time persistence, and calibration failures without GPU hardware."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from mineru.model.runtime.gpu import classify_d3d12_gpu, classify_vulkan_gpu, with_gpu_classification
from mineru.config import Config
from mineru.model.runtime import onnx_device as policy
from mineru.model.runtime.windows_gpu import GpuAdapter, preferred_gpu
from mineru.model.runtime.webgpu import WebGPUDevice

settings = importlib.import_module("mineru.config")


def adapter(*, index: int = 0, uma: bool | None = True, vendor: int = 0x8086, rank: int = 0) -> GpuAdapter:
    return with_gpu_classification(GpuAdapter(index, f"luid-{index}", "test GPU", vendor, 1, rank), classify_d3d12_gpu(uma))


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "config.yaml"
    target.write_text("model:\n  onnx_device: auto\n  source: local\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_file_path", lambda: str(target))
    monkeypatch.setattr(settings, "get_config_source", lambda key: "file")
    monkeypatch.setattr(settings.get_config().model, "onnx_device", "auto")
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(policy, "_selections", {})
    monkeypatch.setattr(policy, "ONNX_GPU_TIME_RATIO_THRESHOLDS", {"dml": 0.75, "webgpu": 0.8})
    monkeypatch.setattr(policy, "discover_gpus", lambda **_: (adapter(),))
    import onnxruntime
    from mineru.model import registry
    from mineru.model.runtime import onnx_calibration

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=lambda: "rec.onnx")))
    monkeypatch.setattr(
        onnx_calibration, "run_calibration", Mock(return_value={"decision_valid": True, "cpu_elapsed": 2.0, "gpu_elapsed": 1.0})
    )
    return target


def test_config_values_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert Config().model.onnx_device == "auto"
    for value in ("auto", "cpu", "gpu"):
        assert Config(model={"onnx_device": value}).model.onnx_device == value
    with pytest.raises(ValidationError):
        Config(model={"onnx_device": "cuda"})
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  onnx_device: cpu\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_CONFIG", str(path))
    monkeypatch.setenv("MINERU_MODEL_ONNX_DEVICE", "gpu")
    loaded = settings._load_effective_config()
    assert loaded.config.model.onnx_device == "gpu"
    assert loaded.sources[("model", "onnx_device")] == "env"


def test_discrete_priority_and_rank_mapping() -> None:
    igpu = adapter(index=0, rank=0)
    discrete = adapter(index=4, uma=False, rank=1)
    external = adapter(index=7, uma=False, vendor=0x10DE, rank=0)
    assert preferred_gpu((igpu, discrete, external)) == external
    assert preferred_gpu((igpu, discrete)) == discrete  # Intel discrete is still discrete.
    assert preferred_gpu(()) is None
    assert preferred_gpu((adapter(index=2, uma=None, rank=1), igpu)) == igpu


def test_gpu_mode_uses_performance_order_without_creating_devices() -> None:
    import ctypes
    from mineru.model.runtime import windows_gpu

    create = Mock(return_value=1)  # D3D12CreateDevice returns S_FALSE for a successful capability check.
    assert windows_gpu._architecture(ctypes.c_void_p(1), create, classify=False) == (True, None)
    assert create.call_args.args[-1] is None
    igpu = adapter(index=0, uma=None, rank=1)
    discrete = adapter(index=1, uma=None, vendor=0x10DE, rank=0)
    assert preferred_gpu((igpu, discrete)) == discrete


def test_gpu_mode_requests_fast_probe(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.get_config().model, "onnx_device", "gpu")
    discover = Mock(return_value=(adapter(uma=None),))
    monkeypatch.setattr(policy, "discover_gpus", discover)
    assert policy.resolve_onnx_device().device == "gpu"
    discover.assert_called_once_with(classify=False)


@pytest.mark.parametrize(("gpu_time", "expected"), [(0.74, True), (0.75, True), (0.751, False)])
def test_threshold_boundary(gpu_time: float, expected: bool) -> None:
    assert policy.choose_gpu(1.0, gpu_time, 0.75) is expected


@pytest.mark.parametrize("values", [(0, 1, 0.75), (1, -1, 0.75), (1, float("nan"), 0.75), (1, 1, 1.0)])
def test_invalid_timings(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        policy.choose_gpu(*values)


def test_cpu_never_discovers_or_benchmarks(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.get_config().model, "onnx_device", "cpu")
    discover = Mock(side_effect=AssertionError("discovery"))
    monkeypatch.setattr(policy, "discover_gpus", discover)
    assert policy.resolve_onnx_device().device == "cpu"
    discover.assert_not_called()


@pytest.mark.parametrize(("gpu_time", "expected"), [(1.0, "gpu"), (1.6, "cpu")])
def test_calibration_saved_once(config_file: Path, monkeypatch: pytest.MonkeyPatch, gpu_time: float, expected: str) -> None:
    from mineru.model.runtime import onnx_calibration

    measure = Mock(return_value={"decision_valid": True, "cpu_elapsed": 2.0, "gpu_elapsed": gpu_time})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    assert policy.resolve_onnx_device().device == expected
    assert policy.resolve_onnx_device().device == expected
    assert measure.call_count == 1
    assert yaml.safe_load(config_file.read_text())["model"] == {"onnx_device": expected, "source": "local"}
    policy._selections.clear()  # A new process reads the persisted choice instead of benchmarking again.
    assert policy.resolve_onnx_device().device == expected
    assert measure.call_count == 1


def test_offline_calibration_never_downloads_or_poison_online_cache(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model import registry

    def ensure(*, source: str | None = None) -> str:
        if source == "local":
            raise FileNotFoundError("not cached")
        return "rec.onnx"

    loader = Mock(side_effect=ensure)
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=loader)))
    assert policy.resolve_onnx_device(allow_download=False).device == "cpu"
    loader.assert_called_once_with(source="local")
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"
    assert policy.resolve_onnx_device().device == "gpu"


def test_discrete_skips_calibration_and_persists_only_after_session(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model.runtime import onnx_calibration

    monkeypatch.setattr(policy, "discover_gpus", lambda **_: (adapter(uma=False),))
    result = policy.resolve_onnx_device()
    assert result.device == "gpu"
    onnx_calibration.run_calibration.assert_not_called()
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"
    policy.confirm_gpu_session(result)
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "gpu"


@pytest.mark.parametrize("failure", [RuntimeError("GPU failed"), TimeoutError("worker timeout")])
def test_failures_do_not_save_or_retry(config_file: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    from mineru.model.runtime import onnx_calibration

    measure = Mock(side_effect=failure)
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    assert policy.resolve_onnx_device().device == "cpu"
    assert policy.resolve_onnx_device().device == "cpu"
    assert measure.call_count == 1
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"


def test_environment_choice_is_not_saved(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "get_config_source", lambda key: "env")
    assert policy.resolve_onnx_device().device == "gpu"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"


def test_user_edit_wins_compare_and_swap(config_file: Path) -> None:
    config_file.write_text("model:\n  onnx_device: cpu\n  source: local\n", encoding="utf-8")
    assert not settings.update_config_file({"model": {"onnx_device": "gpu"}}, only_if_onnx_auto=True)
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "cpu"
    assert policy.resolve_onnx_device().device == "cpu"


def test_unwritable_config_keeps_process_choice(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "update_config_file", Mock(side_effect=PermissionError("read only")))
    assert policy.resolve_onnx_device().device == "gpu"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"


@pytest.mark.parametrize("mode", ["no_gpu", "no_provider", "discovery_error", "pending", "non_windows"])
def test_discovery_and_unavailable_states(config_file: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import onnxruntime

    if mode == "no_gpu":
        monkeypatch.setattr(policy, "discover_gpus", lambda **_: ())
    elif mode == "no_provider":
        monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    elif mode == "discovery_error":
        monkeypatch.setattr(policy, "discover_gpus", Mock(side_effect=OSError("DXGI error")))
    elif mode == "pending":
        monkeypatch.setitem(policy.ONNX_GPU_TIME_RATIO_THRESHOLDS, "dml", None)
    else:
        monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="darwin"))
    assert policy.resolve_onnx_device().device == "cpu"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == ("cpu" if mode == "no_gpu" else "auto")


def test_manual_gpu_retained_without_gpu(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.get_config().model, "onnx_device", "gpu")
    config_file.write_text("model:\n  onnx_device: gpu\n", encoding="utf-8")
    monkeypatch.setattr(policy, "discover_gpus", lambda **_: ())
    assert policy.resolve_onnx_device().device == "cpu"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "gpu"


def test_waiting_process_rechecks_choice_after_lock(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model.runtime import onnx_calibration

    class OtherProcessSaved:
        def acquire(self) -> None:
            config_file.write_text("model:\n  onnx_device: cpu\n", encoding="utf-8")

        def release(self) -> None:
            pass

    monkeypatch.setattr(policy, "FileLock", lambda *args, **kwargs: OtherProcessSaved())
    assert policy.resolve_onnx_device().device == "cpu"
    onnx_calibration.run_calibration.assert_not_called()


def test_concurrent_model_initializers_share_one_decision(config_file: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from mineru.model.runtime import onnx_calibration

    with ThreadPoolExecutor(max_workers=4) as pool:
        choices = list(pool.map(lambda _: policy.resolve_onnx_device(), range(8)))
    assert all(choice is choices[0] for choice in choices)
    assert onnx_calibration.run_calibration.call_count == 1


def test_manual_reset_to_auto_recalibrates(config_file: Path) -> None:
    from mineru.model.runtime import onnx_calibration

    policy.resolve_onnx_device()
    settings.update_config_file({"model": {"onnx_device": "auto"}})
    policy._selections.clear()  # New process after the explicit reset.
    policy.resolve_onnx_device()
    assert onnx_calibration.run_calibration.call_count == 2


@pytest.mark.parametrize(
    ("requested", "uma", "gpu_time", "expected"),
    [
        ("cpu", True, 1.0, "cpu"),
        ("gpu", True, 1.0, "gpu"),
        ("auto", True, 1.0, "gpu"),
        ("auto", True, 1.6, "cpu"),
        ("auto", False, 1.0, "gpu"),
    ],
)
def test_unet_first_model_shares_selection_and_persistence(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    uma: bool,
    gpu_time: float,
    expected: str,
) -> None:
    from mineru.model.runtime import onnx, onnx_calibration
    from mineru.model.table.rec.unet_table.table_structure_unet import TSRUnet

    monkeypatch.setattr(settings.get_config().model, "onnx_device", requested)
    config_file.write_text(f"model:\n  onnx_device: {requested}\n  source: local\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu" if expected == "gpu" else "cuda")
    discover = Mock(return_value=(adapter(index=7, uma=uma),))
    monkeypatch.setattr(policy, "discover_gpus", discover)
    measure = Mock(return_value={"decision_valid": True, "cpu_elapsed": 2.0, "gpu_elapsed": gpu_time})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    factory = Mock(
        side_effect=lambda *args, **kwargs: SimpleNamespace(
            get_providers=lambda: [name for name, options in kwargs["providers"]], disable_fallback=Mock()
        )
    )
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    model = TSRUnet({"model_path": "unet.onnx", "small_backend": "onnx", "intra_op_num_threads": 2})
    provider = "DmlExecutionProvider" if expected == "gpu" else "CPUExecutionProvider"
    assert model.session.session.get_providers()[0] == provider
    onnx.ort_session("layout.onnx", model_kind="layout")
    onnx.ort_session("rec.onnx", model_kind="ocr_rec")
    assert all(call.kwargs["providers"][0][0] == provider for call in factory.call_args_list)
    assert discover.call_count == (0 if requested == "cpu" else 1)
    assert measure.call_count == (1 if requested == "auto" and uma else 0)
    if measure.called:
        assert measure.call_args.kwargs["intra_op_num_threads"] == 2
    if expected == "gpu":
        assert factory.call_args_list[0].kwargs["providers"][0][1] == {"device_id": 7, "disable_metacommands": "true"}
    assert yaml.safe_load(config_file.read_text())["model"] == {"onnx_device": expected, "source": "local"}


@pytest.fixture
def linux_config(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """为 Linux 选型提供可见集显，避免读取测试宿主的 Vulkan 设备。"""
    from mineru.model.runtime import vulkan, webgpu

    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(webgpu, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(webgpu, "_runtime_pid", None)
    monkeypatch.setattr(webgpu, "_runtime_device", None)
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(0x1002, 1, "integrated"), classify_vulkan_gpu(1)),),
    )
    monkeypatch.setattr(policy, "discover_webgpu_devices", Mock(return_value=(WebGPUDevice(0x1002, 1),)))
    return config_file


@pytest.mark.parametrize(("gpu_time", "expected"), [(1.5, "gpu"), (1.6, "gpu"), (1.601, "cpu")])
@pytest.mark.parametrize("hardware", ["integrated", "discrete"])
def test_linux_calibration_uses_vulkan_type_instead_of_ep_metadata(
    linux_config: Path, monkeypatch: pytest.MonkeyPatch, gpu_time: float, expected: str, hardware: str
) -> None:
    """EP 元数据即使声称独显，Vulkan 确认的集显仍按 0.8 阈值校准。"""
    from mineru.model.runtime import onnx_calibration

    descriptor = WebGPUDevice(0x1002, 1, (("type", hardware),))
    monkeypatch.setattr(policy, "discover_webgpu_devices", Mock(return_value=(descriptor,)))
    measure = Mock(return_value={"decision_valid": True, "cpu_elapsed": 2.0, "gpu_elapsed": gpu_time})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    selection = policy.resolve_onnx_device()
    assert selection.device == expected and selection.backend == "webgpu"
    assert policy.resolve_onnx_device() is selection
    measure.assert_called_once_with(
        "rec.onnx",
        with_gpu_classification(descriptor, classify_vulkan_gpu(1)),
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        threshold=0.8,
    )
    assert yaml.safe_load(linux_config.read_text())["model"] == {"onnx_device": expected, "source": "local"}


@pytest.mark.parametrize("identity", [(0x10DE, 0x2489), (0x1002, 0x73DF), (0x8086, 0x56A0)])
@pytest.mark.parametrize("source", ["file", "env"])
def test_linux_discrete_skips_calibration_and_saves_after_actual_session(
    linux_config: Path, monkeypatch: pytest.MonkeyPatch, identity: tuple[int, int], source: str
) -> None:
    """三家独显均免下载和校准，只在 WebGPU 会话确认成功后按配置来源保存。"""
    from mineru.model import registry
    from mineru.model.runtime import onnx, onnx_calibration, vulkan, webgpu

    descriptor = WebGPUDevice(*identity)
    monkeypatch.setattr(policy, "discover_webgpu_devices", lambda: (descriptor,))
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(*identity, "GPU"), classify_vulkan_gpu(2)),),
    )
    monkeypatch.setattr(settings, "get_config_source", lambda key: source)
    monkeypatch.setitem(policy.ONNX_GPU_TIME_RATIO_THRESHOLDS, "webgpu", None)
    download = Mock(side_effect=AssertionError("discrete GPU must not prepare calibration weights"))
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=download)))
    selected = policy.resolve_onnx_device(allow_download=False)
    assert selected == policy.OnnxSelection(
        "gpu",
        with_gpu_classification(replace(descriptor, vulkan_device=identity), classify_vulkan_gpu(2)),
        "discrete_gpu",
        "webgpu",
    )
    assert yaml.safe_load(linux_config.read_text(encoding="utf-8"))["model"]["onnx_device"] == "auto"

    native = SimpleNamespace(get_providers=lambda: [webgpu.WEBGPU_PROVIDER], disable_fallback=Mock())
    monkeypatch.setattr(onnx.ort, "InferenceSession", Mock(return_value=native))
    monkeypatch.setattr(onnx, "attach_webgpu_provider", Mock())
    session = onnx.ort_session("layout.onnx", model_kind="layout", selection=selected)
    assert session.get_providers() == [webgpu.WEBGPU_PROVIDER]
    native.disable_fallback.assert_called_once()
    assert yaml.safe_load(linux_config.read_text(encoding="utf-8"))["model"]["onnx_device"] == (
        "gpu" if source == "file" else "auto"
    )
    assert not Path(f"{linux_config}.onnx-calibration.json").exists()
    policy._selections.clear()
    assert policy.resolve_onnx_device().reason == ("configured_gpu" if source == "file" else "discrete_gpu")
    download.assert_not_called()
    onnx_calibration.run_calibration.assert_not_called()


@pytest.mark.parametrize("failure", ["binding", "initialization", "silent_cpu"])
def test_linux_discrete_session_failure_preserves_auto(
    linux_config: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """免校准独显发生绑定失败、初始化错误或静默 CPU 回退时，不保存 GPU 偏好。"""
    from mineru.model.runtime import onnx, onnx_calibration, vulkan

    descriptor = WebGPUDevice(0x10DE, 0x2489)
    integrated = with_gpu_classification(vulkan.VulkanDevice(0x1002, 1, "integrated"), classify_vulkan_gpu(1))
    discrete = with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "discrete"), classify_vulkan_gpu(2))
    monkeypatch.setattr(policy, "discover_webgpu_devices", lambda: (descriptor,))
    monkeypatch.setattr(
        vulkan, "enumerate_vulkan_devices", lambda: (integrated, discrete) if failure == "binding" else (discrete,)
    )
    selected = policy.resolve_onnx_device()
    assert selected.reason == "discrete_gpu"
    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(side_effect=[RuntimeError("GPU failed"), cpu] if failure == "initialization" else None, return_value=cpu)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    monkeypatch.setattr(onnx, "attach_webgpu_provider", Mock())
    assert onnx.ort_session("layout.onnx", model_kind="layout", selection=selected) is cpu
    assert factory.call_count == (1 if failure == "binding" else 2)
    assert yaml.safe_load(linux_config.read_text(encoding="utf-8"))["model"]["onnx_device"] == "auto"
    onnx_calibration.run_calibration.assert_not_called()


@pytest.mark.parametrize("state", ["unavailable", "duplicate", "virtual"])
def test_linux_unconfirmed_discrete_still_calibrates(linux_config: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    """类型未知、同型号多卡及虚拟 GPU 均不能获得独显免校准资格。"""
    from mineru.model.runtime import onnx_calibration, vulkan

    descriptor = WebGPUDevice(0x10DE, 0x2489, (("type", "discrete"),))
    physical = with_gpu_classification(
        vulkan.VulkanDevice(0x10DE, 0x2489, "GPU"), classify_vulkan_gpu(3 if state == "virtual" else 2)
    )
    monkeypatch.setattr(policy, "discover_webgpu_devices", lambda: (descriptor,))
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        Mock(
            side_effect=OSError("query failed") if state == "unavailable" else None,
            return_value=(physical, physical) if state == "duplicate" else (physical,),
        ),
    )
    assert policy.resolve_onnx_device().reason == "rec_calibration"
    onnx_calibration.run_calibration.assert_called_once()


def test_linux_hidden_discrete_does_not_exempt_visible_integrated(linux_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ORT 中仍可见但被 ICD 隐藏的独显，不能让实际可见的集显免校准。"""
    from mineru.model.runtime import onnx_calibration

    integrated = WebGPUDevice(0x1002, 1)
    monkeypatch.setattr(policy, "discover_webgpu_devices", lambda: (WebGPUDevice(0x10DE, 0x2489), integrated))
    selected = policy.resolve_onnx_device()
    assert selected.reason == "rec_calibration" and selected.adapter == with_gpu_classification(
        integrated, classify_vulkan_gpu(1)
    )
    onnx_calibration.run_calibration.assert_called_once()


@pytest.mark.parametrize("requested", ["cpu", "gpu"])
def test_linux_explicit_modes_do_not_calibrate(linux_config: Path, monkeypatch: pytest.MonkeyPatch, requested: str) -> None:
    from mineru.model.runtime import onnx_calibration

    monkeypatch.setattr(settings.get_config().model, "onnx_device", requested)
    assert policy.resolve_onnx_device().device == requested
    onnx_calibration.run_calibration.assert_not_called()
    assert policy.discover_webgpu_devices.call_count == (0 if requested == "cpu" else 1)


@pytest.mark.parametrize("error", [ImportError("plugin"), OSError("Vulkan"), TimeoutError("worker"), ValueError("outputs")])
def test_linux_failures_preserve_auto_and_do_not_retry(
    linux_config: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    from mineru.model.runtime import onnx_calibration

    failing = Mock(side_effect=error)
    if isinstance(error, (ImportError, OSError)) and not isinstance(error, TimeoutError):
        monkeypatch.setattr(policy, "discover_webgpu_devices", failing)
    else:
        monkeypatch.setattr(onnx_calibration, "run_calibration", failing)
    for _ in range(2):
        assert policy.resolve_onnx_device().device == "cpu"
    assert failing.call_count == 1
    assert yaml.safe_load(linux_config.read_text())["model"]["onnx_device"] == "auto"


def test_linux_environment_override_never_persists(linux_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "get_config_source", lambda key: "env")
    assert policy.resolve_onnx_device().device == "gpu"
    assert yaml.safe_load(linux_config.read_text())["model"]["onnx_device"] == "auto"


@pytest.mark.parametrize("state", ["no_devices", "pending_threshold"])
def test_linux_unavailable_state_preserves_auto(linux_config: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    from mineru.model.runtime import onnx_calibration

    if state == "no_devices":
        monkeypatch.setattr(policy, "discover_webgpu_devices", Mock(return_value=()))
    else:
        monkeypatch.setitem(policy.ONNX_GPU_TIME_RATIO_THRESHOLDS, "webgpu", None)
    assert policy.resolve_onnx_device().device == "cpu"
    onnx_calibration.run_calibration.assert_not_called()
    assert yaml.safe_load(linux_config.read_text())["model"]["onnx_device"] == "auto"


def test_linux_waiting_process_rechecks_saved_choice(linux_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class OtherProcessSaved:
        def acquire(self) -> None:
            linux_config.write_text("model:\n  onnx_device: cpu\n")

        def release(self) -> None:
            pass

    monkeypatch.setattr(policy, "FileLock", lambda *args, **kwargs: OtherProcessSaved())
    assert policy.resolve_onnx_device().device == "cpu"
    policy.discover_webgpu_devices.assert_not_called()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_unstable_calibration_keeps_auto(config_file: Path, monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    from mineru.model.runtime import onnx_calibration, vulkan, webgpu

    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(webgpu, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x7D51, "integrated"), classify_vulkan_gpu(1)),),
    )
    monkeypatch.setattr(policy, "discover_webgpu_devices", lambda: (WebGPUDevice(0x8086, 0x7D51),))
    measure = Mock(return_value={"decision_valid": False, "cpu_elapsed": 2.0, "gpu_elapsed": 1.0})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    selected = policy.resolve_onnx_device()
    assert selected.device == "cpu" and selected.reason == "calibration_unstable"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == "auto"
    assert policy.resolve_onnx_device() == selected
    measure.assert_called_once()
