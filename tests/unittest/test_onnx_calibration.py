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

from mineru.model.runtime.gpu import classify_d3d12_gpu, with_gpu_classification
from mineru.model.runtime import onnx_calibration as calibration
from mineru.model.runtime.windows_gpu import GpuAdapter
from mineru.model.runtime.webgpu import WEBGPU_PROVIDER, WebGPUDevice
from mineru.model.runtime.onnx_rec import CompactRecModel, REC_OUTPUT_FORMAT
from scripts.benchmark_onnx_selection import derive_threshold


def gpu() -> GpuAdapter:
    return with_gpu_classification(GpuAdapter(2, "id", "test", 0x8086, 1, 0), classify_d3d12_gpu(True))


@pytest.fixture(autouse=True)
def fake_rec_preparation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """隔离模型准备和校准证据，避免协议测试触及用户配置。"""
    monkeypatch.setattr("mineru.model.runtime.onnx_calibration_cache.calibration_record_path", lambda: tmp_path / "record.json")
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


def measure_fake(
    monkeypatch: pytest.MonkeyPatch,
    durations: dict[str, list[float]],
    *,
    gpu_first: bool = False,
    changes: dict[str, int] | None = None,
    gpu_token: int = 1,
) -> tuple[dict, list[str], list[str]]:
    from mineru.model.runtime import onnx

    tensor = np.zeros((6, 3, 48, 320), dtype=np.float32)
    monkeypatch.setattr(calibration, "prepare_input", lambda: (tensor, {"shape": list(tensor.shape)}))
    clock = [0.0]
    monkeypatch.setattr(calibration.time, "perf_counter", lambda: clock[0])
    created: list[str] = []
    calls: list[str] = []

    def create(path: str, **kwargs: object) -> SimpleNamespace:
        device = "gpu" if kwargs["adapter"] else "cpu"
        created.append(device)
        clock[0] += 4 if device == "cpu" else 8
        provider = "DmlExecutionProvider" if device == "gpu" else "CPUExecutionProvider"
        count = 0

        def run(names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            nonlocal count
            assert names == ["mineru_rec_ids", "mineru_rec_scores"]
            assert feed["image"] is tensor
            clock[0] += durations[device][count]
            count += 1
            calls.append(device)
            token = gpu_token if device == "gpu" else 1
            if (changes or {}).get(device) == count:
                token = 2
            return [np.full((6, 3), token, dtype=np.int64), np.ones((6, 3), dtype=np.float32)]

        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="image")], run=run, get_providers=lambda: [provider])

    monkeypatch.setattr(onnx, "create_ort_session", create)
    return calibration.measure_rec("rec.onnx", gpu(), gpu_first=gpu_first), created, calls


@pytest.mark.parametrize("gpu_first", [False, True])
@pytest.mark.parametrize("ratio", [0.4, 1.2])
def test_stable_clear_winner_stops_after_one_plus_three(monkeypatch: pytest.MonkeyPatch, gpu_first: bool, ratio: float) -> None:
    result, created, calls = measure_fake(
        monkeypatch, {"cpu": [100, 1, 1, 1], "gpu": [100, ratio, ratio, ratio]}, gpu_first=gpu_first
    )
    order = ["gpu", "cpu"] if gpu_first else ["cpu", "gpu"]
    assert created == order
    assert calls == [device for device in order for _ in range(4)]
    assert result["sampling"]["stage"] == "initial"
    assert result["sampling"]["warmup_runs"] == 1 and result["sampling"]["timed_runs"] == 3
    assert result["sampling"]["extension_reasons"] == []
    assert result["ratio"] == pytest.approx(ratio)
    assert result["decision_valid"] is True
    assert result["cpu"]["warmup_s"] == pytest.approx([100])
    assert result["cpu"]["init_s"] == 4 and result["gpu"]["init_s"] == 8
    assert result["measurement"] == calibration.CALIBRATION_MEASUREMENT
    assert result["model_sha256"] == "compact" and result["source_model_sha256"] == "raw"


@pytest.mark.parametrize("gpu_first", [False, True])
def test_extension_reuses_sessions_and_reclassifies_first_three_calls(monkeypatch: pytest.MonkeyPatch, gpu_first: bool) -> None:
    durations = {"cpu": [100, 200, 300, 1, 1, 1, 1, 1, 1, 999], "gpu": [100, 200, 300, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 99]}
    result, created, calls = measure_fake(monkeypatch, durations, gpu_first=gpu_first)
    order = ["gpu", "cpu"] if gpu_first else ["cpu", "gpu"]
    assert created == order  # No sessions recreated when extending.
    assert calls == [device for count in [4, 6] for device in order for _ in range(count)]
    assert result["sampling"]["stage"] == "extended"
    assert result["sampling"]["warmup_runs"] == 3 and result["sampling"]["timed_runs"] == 7
    assert "cpu_spread" in result["sampling"]["extension_reasons"]
    assert result["cpu"]["warmup_s"] == pytest.approx(durations["cpu"][:3])
    assert result["cpu"]["timed_s"] == durations["cpu"][3:]
    assert result["gpu"]["timed_s"] == pytest.approx(durations["gpu"][3:])
    assert len(result["sampling"]["initial"]["cpu"]["timed_s"]) == 3
    assert result["cpu_elapsed"] == 1
    assert result["gpu_elapsed"] == pytest.approx(0.4)
    assert result["decision_valid"] is True  # One isolated outlier does not defeat the median.


@pytest.mark.parametrize("ratio", [0.7, 0.8, 0.9])
def test_near_threshold_always_extends(monkeypatch: pytest.MonkeyPatch, ratio: float) -> None:
    result, _, calls = measure_fake(monkeypatch, {"cpu": [2] + [1] * 9, "gpu": [2] + [ratio] * 9})
    assert len(calls) == 20
    assert "near_threshold" in result["sampling"]["extension_reasons"]
    assert result["decision_valid"] is True
    assert result["ratio"] == pytest.approx(ratio)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_persistent_drift_does_not_produce_a_valid_decision(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    durations = {"cpu": [10] + [1] * 9, "gpu": [10] + [0.4] * 9}
    durations[device] = [10, 1, 2, 1, 1, 1, 2, 2, 2, 2]
    result, _, calls = measure_fake(monkeypatch, durations)
    assert len(calls) == 20
    assert result["decision_valid"] is False
    assert result["outputs_match"] is True


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("change_at", [2, 4, 5, 10])
def test_text_checks_cover_initial_and_extended_runs(monkeypatch: pytest.MonkeyPatch, device: str, change_at: int) -> None:
    with pytest.raises(ValueError, match="changed between runs"):
        measure_fake(monkeypatch, {"cpu": [2] + [1] * 9, "gpu": [2] + [0.8] * 9}, changes={device: change_at})


def test_cpu_gpu_mismatch_rejected_before_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="CPU and GPU rec calibration text differ"):
        measure_fake(monkeypatch, {"cpu": [2, 1, 1, 1], "gpu": [2, 0.8, 0.8, 0.8]}, gpu_token=2)


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_each_timing_must_be_finite_positive(monkeypatch: pytest.MonkeyPatch, invalid: float) -> None:
    with pytest.raises(ValueError, match="invalid timings"):
        measure_fake(monkeypatch, {"cpu": [invalid], "gpu": []})


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


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "fallback",
        "wrong_backend",
        "wrong_output_format",
        "wrong_measurement",
        "nan",
        "mismatch",
        "counts",
        "median",
        "stability",
        "threshold",
    ],
)
def test_webgpu_worker_protocol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str | None) -> None:
    """新校准必须替换旧证据，协议或输出验证失败时不可保留旧结论。"""
    (tmp_path / "record.json").write_text('{"old": true}', encoding="utf-8")
    output = {
        "schema_version": 5,
        "backend": "webgpu",
        "sampling": {"stage": "initial", "warmup_runs": 1, "timed_runs": 3},
        "measurement": dict(calibration.CALIBRATION_MEASUREMENT),
        "decision_threshold": 0.8,
        "decision_valid": True,
        "output_format": REC_OUTPUT_FORMAT,
        "cpu_elapsed": 1,
        "gpu_elapsed": 0.4,
        "ratio": 0.4,
        "outputs_match": True,
        "intra_threads": 4,
        "inter_threads": 1,
        "model_sha256": "compact-model",
        "gpu": {"providers": [WEBGPU_PROVIDER, "CPUExecutionProvider"], "warmup_s": [2], "timed_s": [0.4] * 3, "median_s": 0.4},
        "cpu": {"providers": ["CPUExecutionProvider"], "warmup_s": [2], "timed_s": [1] * 3, "median_s": 1},
    }
    if failure == "counts":
        output["gpu"]["timed_s"].pop()
    elif failure == "median":
        output["gpu"]["median_s"] = 3
    elif failure == "stability":
        output["decision_valid"] = False
    elif failure == "threshold":
        output["decision_threshold"] = 0.9
    elif failure == "fallback":
        output["gpu"]["providers"] = ["CPUExecutionProvider"]
    elif failure == "wrong_backend":
        output["backend"] = "dml"
    elif failure == "wrong_output_format":
        output["output_format"] = "full-probabilities"
    elif failure == "wrong_measurement":
        output["measurement"] = {"warmup_runs": 1, "timed_runs": 2, "statistic": "sum"}
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
        assert not (tmp_path / "record.json").exists()
    else:
        assert calibration.run_calibration("rec.onnx", descriptor)["backend"] == "webgpu"
        saved = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
        assert saved["outputs_match"] is True and saved["decision_valid"] is True
        assert saved["fingerprint"]["adapter"]["device_id"] == 2
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


def test_threshold_derivation_rejects_mixed_measurement_protocols() -> None:
    data = reports()
    for report in data:
        for run in report["runs"]:
            run["measurement"] = dict(calibration.CALIBRATION_MEASUREMENT)
            run["decision_valid"] = True
            run["decision_threshold"] = 0.8
    assert derive_threshold(data)["threshold"] == 0.775
    data[0]["runs"][0].pop("measurement")
    with pytest.raises(ValueError, match="measurement protocols differ"):
        derive_threshold(data)


def test_threshold_derivation_rejects_unstable_adaptive_result() -> None:
    data = reports()
    data[0]["runs"][0]["decision_valid"] = False
    with pytest.raises(ValueError, match="Unstable"):
        derive_threshold(data)


def test_threshold_derivation_rejects_different_adaptive_decision_thresholds() -> None:
    data = reports()
    for report in data:
        for run in report["runs"]:
            run.update(measurement=dict(calibration.CALIBRATION_MEASUREMENT), decision_valid=True, decision_threshold=0.8)
    data[0]["runs"][0]["decision_threshold"] = 0.9
    with pytest.raises(ValueError, match="measurement protocols differ"):
        derive_threshold(data)
