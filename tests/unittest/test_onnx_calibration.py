"""Validate the portable benchmark protocol and data-derived threshold selection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.runtime import onnx_calibration as calibration
from mineru.model.runtime.windows_gpu import GpuAdapter
from mineru.model.runtime.webgpu import WEBGPU_PROVIDER, WebGPUDevice
from mineru.model.runtime.onnx_rec import CompactRecModel, REC_OUTPUT_FORMAT
from scripts.benchmark_onnx_selection import derive_threshold


def gpu() -> GpuAdapter:
    return GpuAdapter(2, "id", "test", 0x8086, 1, True, 0)


@pytest.fixture(autouse=True)
def fake_rec_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protocol tests mock sessions; real graph conversion is covered separately."""
    monkeypatch.setattr(calibration, "prepare_rec_model", lambda path: CompactRecModel("compact.onnx", 4, "raw", "compact"))


def test_installed_samples_are_fixed_and_complete() -> None:
    first, metadata = calibration.prepare_input()
    second, repeated = calibration.prepare_input()
    assert first.shape[:3] == (6, 3, 48)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert metadata == repeated
    manifest = json.loads((calibration.sample_directory() / "manifest.json").read_text(encoding="utf-8"))
    text = "".join(row["text"] for row in manifest["samples"])
    assert any("\u4e00" <= character <= "\u9fff" for character in text)
    assert any(character.isascii() and character.isalpha() for character in text)
    assert any(character.isdigit() for character in text)


def test_corrupt_sample_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads((calibration.sample_directory() / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / manifest["samples"][0]["file"]).write_bytes(b"corrupt")
    monkeypatch.setattr(calibration, "sample_directory", lambda: tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        calibration.prepare_input()


def test_measurement_reuses_input_and_excludes_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model.runtime import onnx

    tensor = np.zeros((6, 3, 48, 320), dtype=np.float32)
    feeds = []
    providers = []
    monkeypatch.setattr(calibration, "prepare_input", lambda: (tensor, {"shape": list(tensor.shape)}))
    ticks = iter([0, 4, 10, 110, 200, 202, 300, 303, 400, 408, 500, 600, 700, 701, 800, 802])
    monkeypatch.setattr(calibration.time, "perf_counter", lambda: next(ticks))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake weights")

    def create(path: str, **kwargs: object) -> SimpleNamespace:
        provider = "DmlExecutionProvider" if kwargs["adapter"] else "CPUExecutionProvider"
        providers.append(provider)

        def run(names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            feeds.append(feed["image"])
            assert names == ["mineru_rec_ids", "mineru_rec_scores"]
            return [np.ones((6, 3), dtype=np.int64), np.ones((6, 3), dtype=np.float32)]

        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="image")], run=run, get_providers=lambda: [provider])

    monkeypatch.setattr(onnx, "create_ort_session", create)
    result = calibration.measure_rec(str(model), gpu())
    assert providers == ["CPUExecutionProvider", "DmlExecutionProvider"]
    assert len(feeds) == 6 and all(feed is tensor for feed in feeds)
    assert result["cpu_elapsed"] == 5
    assert result["gpu_elapsed"] == 3
    assert result["cpu"]["first_s"] == result["gpu"]["first_s"] == 100
    assert result["cpu"]["init_s"] == 4 and result["gpu"]["init_s"] == 8
    assert result["output_format"] == REC_OUTPUT_FORMAT
    assert result["model_sha256"] == "compact" and result["source_model_sha256"] == "raw"


def test_worker_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Mock(side_effect=subprocess.TimeoutExpired("worker", 15))
    monkeypatch.setattr(calibration.subprocess, "run", runner)
    with pytest.raises(subprocess.TimeoutExpired):
        calibration.run_calibration("rec.onnx", gpu())
    assert runner.call_args.kwargs["timeout"] == 15
    assert runner.call_args.kwargs["env"]["MINERU_MODEL_ONNX_DEVICE"] == "cpu"


def test_worker_silent_fallback_is_not_a_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    output = {"cpu_elapsed": 1, "gpu_elapsed": 0.1, "outputs_match": True, "gpu": {"providers": ["CPUExecutionProvider"]}}
    monkeypatch.setattr(
        calibration.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(output)))
    )
    with pytest.raises(ValueError, match="DmlExecutionProvider"):
        calibration.run_calibration("rec.onnx", gpu())


def reports() -> list[dict]:
    result = []
    for label, ratios in (
        ("ultra5", [0.5, 0.52, 0.55, 0.53, 0.51]),
        ("i7", [1.2, 1.3, 1.1, 1.4, 1.3]),
        ("max395", [0.3, 0.32, 0.29, 0.33, 0.31]),
    ):
        runs = [
            {
                "cpu_elapsed": 1.0,
                "gpu_elapsed": ratio,
                "outputs_match": True,
                "gpu": {"providers": ["DmlExecutionProvider"]},
                "input": {"manifest_sha256": "samples", "input_sha256": "input"},
                "model_sha256": "model",
                "ort_version": "1.24.4",
                "intra_threads": 4,
                "inter_threads": 1,
            }
            for ratio in ratios
        ]
        result.append({"label": label, "runs": runs, "errors": [], "threshold": None})
    return result


def test_threshold_is_derived_from_all_five_runs() -> None:
    result = derive_threshold(reports())
    assert result["lower"] == 0.55
    assert result["upper"] == 1.0
    assert result["threshold"] == 0.775
    assert result["status"] == "awaiting_independent_validation"


@pytest.mark.parametrize("failure", ["missing_host", "missing_run", "different_input", "different_ort", "failed", "overlap"])
def test_invalid_calibration_set_is_rejected(failure: str) -> None:
    data = copy.deepcopy(reports())
    if failure == "missing_host":
        data.pop()
    elif failure == "missing_run":
        data[0]["runs"].pop()
    elif failure == "different_input":
        data[0]["runs"][0]["input"]["input_sha256"] = "other"
    elif failure == "different_ort":
        data[0]["runs"][0]["ort_version"] = "1.30.0"
    elif failure == "failed":
        data[0]["errors"] = ["failed"]
    else:
        data[0]["runs"][0]["gpu_elapsed"] = 1.5
    with pytest.raises(ValueError):
        derive_threshold(data)


@pytest.mark.parametrize("failure", [None, "fallback", "wrong_backend", "wrong_output_format", "nan", "mismatch"])
def test_webgpu_worker_protocol(monkeypatch: pytest.MonkeyPatch, failure: str | None) -> None:
    output = {
        "backend": "webgpu",
        "output_format": REC_OUTPUT_FORMAT,
        "cpu_elapsed": 1,
        "gpu_elapsed": 0.4,
        "outputs_match": True,
        "gpu": {"providers": [WEBGPU_PROVIDER, "CPUExecutionProvider"]},
        "cpu": {"providers": ["CPUExecutionProvider"]},
    }
    if failure == "fallback":
        output["gpu"]["providers"] = ["CPUExecutionProvider"]
    elif failure == "wrong_backend":
        output["backend"] = "dml"
    elif failure == "wrong_output_format":
        output["output_format"] = "full-probabilities"
    elif failure == "nan":
        output["gpu_elapsed"] = float("nan")
    elif failure == "mismatch":
        output["outputs_match"] = False
    runner = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(output)))
    monkeypatch.setattr(calibration.subprocess, "run", runner)
    descriptor = WebGPUDevice(1, 2, (("pci_bus_id", "test"),))
    if failure:
        with pytest.raises(ValueError):
            calibration.run_calibration("rec.onnx", descriptor)
    else:
        assert calibration.run_calibration("rec.onnx", descriptor)["backend"] == "webgpu"
    request = json.loads(runner.call_args.kwargs["input"])
    assert request["backend"] == "webgpu"
    assert request["adapter"]["metadata"] == [["pci_bus_id", "test"]]


def test_threshold_derivation_rejects_mixed_backends_and_ep_options() -> None:
    data = reports()
    data[0]["backend"] = "webgpu"
    with pytest.raises(ValueError, match="mix"):
        derive_threshold(data)
    for report in data:
        report["backend"] = "webgpu"
        report["expected"] = "cpu" if report["label"] == "i7" else "gpu"
        for run in report["runs"]:
            run.update(backend="webgpu", ep_version="0.4.0", webgpu_options={"preferredLayout": "NHWC"})
            run["gpu"]["providers"] = [WEBGPU_PROVIDER]
    assert derive_threshold(data)["backend"] == "webgpu"
    data[0]["runs"][0]["webgpu_options"] = {"preferredLayout": "NCHW"}
    with pytest.raises(ValueError, match="EP version/options"):
        derive_threshold(data)


def test_threshold_derivation_rejects_mixed_output_contracts() -> None:
    data = reports()
    data[0]["runs"][0]["output_format"] = REC_OUTPUT_FORMAT
    with pytest.raises(ValueError, match="differ"):
        derive_threshold(data)
