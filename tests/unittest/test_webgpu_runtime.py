"""Linux plugin lifecycle, real-provider checks, model routing and shared locking."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime.gpu import classify_vulkan_gpu, with_gpu_classification
from mineru.model.runtime import onnx, webgpu
from mineru.model.runtime.onnx_device import OnnxSelection


def device() -> webgpu.WebGPUDevice:
    return webgpu.WebGPUDevice(0x1002, 1, (("pci_bus_id", "0000:00:01.0"),))


def test_registration_is_lazy_once_per_process_and_filters_virtual_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = SimpleNamespace(get_library_path=lambda: "webgpu.so")
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setattr(webgpu, "_registered_pid", None)
    register = Mock()
    monkeypatch.setattr(onnx.ort, "register_execution_provider_library", register)
    gpu = SimpleNamespace(type=onnx.ort.OrtHardwareDeviceType.GPU, vendor_id=0x1002, device_id=1, metadata={})
    virtual = SimpleNamespace(type=onnx.ort.OrtHardwareDeviceType.GPU, vendor_id=0, device_id=0, metadata={})
    cpu = SimpleNamespace(type=onnx.ort.OrtHardwareDeviceType.CPU, vendor_id=1, device_id=0, metadata={})
    monkeypatch.setattr(
        onnx.ort,
        "get_ep_devices",
        lambda: [SimpleNamespace(ep_name=webgpu.WEBGPU_PROVIDER, device=item) for item in (virtual, cpu, gpu)],
    )
    # CPU creation never imports/registers the plugin or enumerates hardware.
    monkeypatch.setattr(onnx.ort, "InferenceSession", Mock(return_value=object()))
    onnx.create_ort_session("cpu.onnx")
    register.assert_not_called()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: webgpu.discover_webgpu_devices(), range(8)))
    assert register.call_count == 1
    assert all(row == (webgpu.WebGPUDevice(0x1002, 1),) for row in results)
    assert len({OnnxSelection("gpu", row[0], backend="webgpu") for row in results}) == 1
    monkeypatch.setattr(webgpu, "_registered_pid", -1)
    webgpu.discover_webgpu_devices()
    assert register.call_count == 2


def test_production_options_resolve_native_handles_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = SimpleNamespace(device=SimpleNamespace(vendor_id=0x1002, device_id=1, metadata=dict(device().metadata)))
    monkeypatch.setattr(webgpu, "_devices", lambda: [ep])
    options = SimpleNamespace(add_provider_for_devices=Mock())
    webgpu.attach_webgpu_provider(
        options, with_gpu_classification(replace(device(), vulkan_device=(0x1002, 1)), classify_vulkan_gpu(2))
    )
    options.add_provider_for_devices.assert_called_once_with([ep], webgpu.WEBGPU_OPTIONS)
    descriptor = json.loads(json.dumps(asdict(device())))
    assert descriptor == {
        "vendor_id": 0x1002,
        "device_id": 1,
        "metadata": [["pci_bus_id", "0000:00:01.0"]],
        "vulkan_device": None,
        "kind": "unknown",
        "kind_source": "unknown",
        "unified_memory": None,
    }
    with pytest.raises(RuntimeError, match="no longer available"):
        webgpu.attach_webgpu_provider(options, webgpu.WebGPUDevice(1, 2))


def test_registration_failure_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webgpu, "_registered_pid", None)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", SimpleNamespace(get_library_path=lambda: "missing.so"))
    register = Mock(side_effect=OSError("libvulkan missing"))
    monkeypatch.setattr(onnx.ort, "register_execution_provider_library", register)
    for _ in range(2):
        with pytest.raises(OSError, match="libvulkan"):
            webgpu.discover_webgpu_devices()
    assert register.call_count == 2 and webgpu._registered_pid is None


def test_factory_uses_attached_plugin_without_provider_override(monkeypatch: pytest.MonkeyPatch) -> None:
    attach = Mock()
    native = SimpleNamespace(get_providers=lambda: [webgpu.WEBGPU_PROVIDER, "CPUExecutionProvider"], disable_fallback=Mock())
    factory = Mock(return_value=native)
    monkeypatch.setattr(onnx, "attach_webgpu_provider", attach)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    session = onnx.create_ort_session("rec.onnx", adapter=device(), model_kind="ocr_rec")
    assert isinstance(session, onnx.WebGPUSession)
    assert "providers" not in factory.call_args.kwargs
    assert attach.call_args.args[1] == device()
    native.disable_fallback.assert_called_once()


@pytest.mark.parametrize("failure", ["plugin_missing", "vulkan_error", "exception", "silent"])
def test_gpu_failure_uses_clean_cpu_options(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    def attach(opts: object, descriptor: object) -> None:
        opts.add_session_config_entry("test.plugin", "attached")
        if failure == "plugin_missing":
            raise ImportError("missing plugin")
        if failure == "vulkan_error":
            raise OSError("libvulkan missing")

    cpu = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    factory = Mock(side_effect=[RuntimeError("device lost"), cpu] if failure == "exception" else None, return_value=cpu)
    monkeypatch.setattr(onnx, "attach_webgpu_provider", attach)
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    save = Mock()
    monkeypatch.setattr(onnx, "confirm_gpu_session", save)
    options = onnx.ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 3
    options.enable_cpu_mem_arena = False
    options.enable_profiling = True
    options.profile_file_prefix = "unet-profile"
    assert (
        onnx.ort_session(
            "unet.onnx", model_kind="unet", selection=OnnxSelection("gpu", device(), backend="webgpu"), sess_options=options
        )
        is cpu
    )
    fallback = factory.call_args.kwargs["sess_options"]
    assert fallback is not options
    assert (fallback.intra_op_num_threads, fallback.inter_op_num_threads) == (2, 3)
    assert not fallback.enable_cpu_mem_arena
    assert fallback.enable_profiling and fallback.profile_file_prefix == "unet-profile"
    with pytest.raises(RuntimeError):
        fallback.get_session_config_entry("test.plugin")
    save.assert_not_called()


def test_linux_routes_only_five_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx, "sys", SimpleNamespace(platform="linux"))
    select = Mock(return_value=OnnxSelection("gpu", device(), backend="webgpu"))
    factory = Mock(return_value=SimpleNamespace(get_providers=lambda: [webgpu.WEBGPU_PROVIDER]))
    monkeypatch.setattr(onnx, "resolve_onnx_device", select)
    monkeypatch.setattr(onnx, "create_ort_session", factory)
    for kind in ("layout", "mfr", "ocr_det", "ocr_rec", "unet"):
        onnx.ort_session("test.onnx", model_kind=kind)
        assert factory.call_args.kwargs["adapter"] == device()
    assert select.call_count == 5
    for kind in ("", "seal_det", "table", "slanet_plus", "mfr_tail_cpu"):
        onnx.ort_session("test.onnx", model_kind=kind)
        assert factory.call_args.kwargs["adapter"] is None
    assert select.call_count == 5


def test_webgpu_creation_and_two_session_runs_share_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    maximum = 0
    counter_lock = threading.Lock()

    def work(*args: object, **kwargs: object) -> list[int]:
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with counter_lock:
            active -= 1
        return [1]

    native = SimpleNamespace(run=work, get_providers=lambda: [webgpu.WEBGPU_PROVIDER], disable_fallback=Mock())
    sessions = [onnx.WebGPUSession(native), onnx.WebGPUSession(native)]
    monkeypatch.setattr(onnx, "attach_webgpu_provider", work)
    monkeypatch.setattr(onnx.ort, "InferenceSession", lambda *args, **kwargs: native)

    def task(index: int) -> None:
        if index % 3 == 0:
            onnx.create_ort_session("rec.onnx", adapter=device())
        else:
            assert sessions[index % 2].run(None, {}) == [1]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(task, range(12)))
    assert maximum == 1


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "DmlExecutionProvider", webgpu.WEBGPU_PROVIDER])
def test_mfr_tail_policy_uses_actual_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    from mineru.model.mfr import pp_formulanet_plus_m_onnx as mfr

    config = tmp_path / "model.yml"
    config.write_text("PostProcess:\n  character_dict: []\n")
    session = SimpleNamespace(get_providers=lambda: [provider], get_inputs=lambda: [SimpleNamespace(name="image")])
    monkeypatch.setattr(mfr, "ort_session", Mock(return_value=session))
    for name in ("UniMERNetDecode", "UniMERNetImgDecode", "UniMERNetTestTransform", "LatexImageFormat"):
        monkeypatch.setattr(mfr, name, Mock())
    model = mfr.PPFormulaNetPlusMONNX("mfr.onnx", str(config))
    assert model.cpu_tail_enabled is (provider != "CPUExecutionProvider")


@pytest.mark.parametrize(("lang", "kind"), [("ch", "ocr_det"), ("seal", "seal_det")])
def test_normal_and_seal_detectors_are_explicit(monkeypatch: pytest.MonkeyPatch, lang: str, kind: str) -> None:
    from mineru.model.ocr import pp_ocr_v6_onnx as ocr

    detector = Mock()
    monkeypatch.setattr(ocr, "TextDetectorONNX", detector)
    monkeypatch.setattr(ocr, "TextRecognizerONNX", Mock())
    ocr.PPOCRv6ONNX("det.onnx", "rec.onnx", "dict.txt", lang=lang)
    assert detector.call_args.kwargs["model_kind"] == kind
