"""Device policy, one-time persistence, and calibration failures without GPU hardware."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from mineru.config import Config
from mineru.model.runtime import onnx_device as policy
from mineru.model.runtime.windows_gpu import GpuAdapter, preferred_gpu

settings = importlib.import_module("mineru.config")


def adapter(*, index: int = 0, uma: bool | None = True, vendor: int = 0x8086, rank: int = 0) -> GpuAdapter:
    return GpuAdapter(index, f"luid-{index}", "test GPU", vendor, 1, uma, rank)


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "config.yaml"
    target.write_text("model:\n  onnx_device: auto\n  source: local\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_file_path", lambda: str(target))
    monkeypatch.setattr(settings, "get_config_source", lambda key: "file")
    monkeypatch.setattr(settings.config.model, "onnx_device", "auto")
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(policy, "_selections", {})
    monkeypatch.setattr(policy, "ONNX_GPU_TIME_RATIO_THRESHOLD", 0.75)
    monkeypatch.setattr(policy, "discover_gpus", lambda **_: (adapter(),))
    import onnxruntime
    from mineru.model import registry
    from mineru.model.runtime import onnx_calibration

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=lambda: "rec.onnx")))
    monkeypatch.setattr(onnx_calibration, "run_calibration", Mock(return_value={"cpu_elapsed": 2.0, "gpu_elapsed": 1.0}))
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
    monkeypatch.setattr(settings.config.model, "onnx_device", "gpu")
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
    monkeypatch.setattr(settings.config.model, "onnx_device", "cpu")
    discover = Mock(side_effect=AssertionError("discovery"))
    monkeypatch.setattr(policy, "discover_gpus", discover)
    assert policy.resolve_onnx_device().device == "cpu"
    discover.assert_not_called()


@pytest.mark.parametrize(("gpu_time", "expected"), [(1.0, "gpu"), (1.6, "cpu")])
def test_calibration_saved_once(config_file: Path, monkeypatch: pytest.MonkeyPatch, gpu_time: float, expected: str) -> None:
    from mineru.model.runtime import onnx_calibration

    measure = Mock(return_value={"cpu_elapsed": 2.0, "gpu_elapsed": gpu_time})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    assert policy.resolve_onnx_device().device == expected
    assert policy.resolve_onnx_device().device == expected
    assert measure.call_count == 1
    assert yaml.safe_load(config_file.read_text())["model"] == {"onnx_device": expected, "source": "local"}
    policy._selections.clear()  # A new process reads the persisted choice instead of benchmarking again.
    assert policy.resolve_onnx_device().device == expected
    assert measure.call_count == 1


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
        monkeypatch.setattr(policy, "ONNX_GPU_TIME_RATIO_THRESHOLD", None)
    else:
        monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux"))
    assert policy.resolve_onnx_device().device == "cpu"
    assert yaml.safe_load(config_file.read_text())["model"]["onnx_device"] == ("cpu" if mode == "no_gpu" else "auto")


def test_manual_gpu_retained_without_gpu(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.config.model, "onnx_device", "gpu")
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

    monkeypatch.setattr(settings.config.model, "onnx_device", requested)
    config_file.write_text(f"model:\n  onnx_device: {requested}\n  source: local\n", encoding="utf-8")
    monkeypatch.setenv("MINERU_TABLE_DEVICE", "cpu" if expected == "gpu" else "cuda")
    discover = Mock(return_value=(adapter(index=7, uma=uma),))
    monkeypatch.setattr(policy, "discover_gpus", discover)
    measure = Mock(return_value={"cpu_elapsed": 2.0, "gpu_elapsed": gpu_time})
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
