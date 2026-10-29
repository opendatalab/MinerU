# Copyright (c) Opendatalab. All rights reserved.
"""Bounded rec calibration worker, shared by automatic selection and the release benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from .windows_gpu import GpuAdapter
from .webgpu import WEBGPU_OPTIONS, WEBGPU_PROVIDER, WebGPUDevice
from .onnx_rec import REC_IDS_OUTPUT, REC_OUTPUT_FORMAT, REC_SCORES_OUTPUT, prepare_rec_model, validate_rec_outputs

CALIBRATION_TIMEOUT_SECONDS = 15
CALIBRATION_INITIAL_WARMUP_RUNS = 1
CALIBRATION_INITIAL_TIMED_RUNS = 3
CALIBRATION_WARMUP_RUNS = 3
CALIBRATION_TIMED_RUNS = 7
CALIBRATION_MAX_RELATIVE_SPREAD = 0.15
CALIBRATION_THRESHOLD_MARGIN = 0.1
CALIBRATION_MEASUREMENT: dict[str, int | float | str] = {
    "strategy": "adaptive-v1",
    "initial_warmup_runs": CALIBRATION_INITIAL_WARMUP_RUNS,
    "initial_timed_runs": CALIBRATION_INITIAL_TIMED_RUNS,
    "extended_warmup_runs": CALIBRATION_WARMUP_RUNS,
    "extended_timed_runs": CALIBRATION_TIMED_RUNS,
    "max_relative_spread": CALIBRATION_MAX_RELATIVE_SPREAD,
    "threshold_margin": CALIBRATION_THRESHOLD_MARGIN,
    "extended_stability": "inclusive-iqr-and-first-last-three-median-drift",
    "statistic": "median",
}


def sample_directory() -> Path:
    """Locate installed package data, independently of the working directory."""
    return Path(__file__).resolve().parents[2] / "resources/onnx_calibration"


def prepare_input() -> tuple[Any, dict[str, Any]]:
    """Check bundled PNG hashes and prepare the exact production batch once."""
    import numpy as np
    from PIL import Image
    from ..ocr.image import resize_text_recognition_image

    root = sample_directory()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = manifest["samples"]
    if len(samples) != 6:
        raise ValueError("Calibration requires exactly six samples")
    images = []
    for sample in samples:
        path = root / sample["file"]
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("Calibration sample checksum mismatch")
        with Image.open(path) as image:
            images.append(np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1]))
    ratios = np.array([image.shape[1] / image.shape[0] for image in images])
    order = np.argsort(ratios)
    tensor = np.concatenate(
        [resize_text_recognition_image(images[index], float(ratios.max()), (3, 48, 320))[None] for index in order]
    ).astype(np.float32)
    return tensor, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
        "shape": list(tensor.shape),
        "sample_order": order.tolist(),
    }


def _tokens(outputs: Any, num_classes: int) -> list[list[int]]:
    import numpy as np

    ids, _ = validate_rec_outputs(outputs, batch_size=6, num_classes=num_classes)
    result = []
    for row in ids:
        keep = np.ones(len(row), dtype=bool)
        keep[1:] = row[1:] != row[:-1]
        keep &= row != 0
        result.append(row[keep].tolist())
    return result


def _statistics(samples: list[float]) -> dict[str, float]:
    median = statistics.median(samples)
    if len(samples) == CALIBRATION_INITIAL_TIMED_RUNS:
        spread = (max(samples) - min(samples)) / median
    else:
        lower, _, upper = statistics.quantiles(samples, n=4, method="inclusive")
        spread = (upper - lower) / median
    drift = abs(statistics.median(samples[:3]) - statistics.median(samples[-3:])) / median
    return {"median_s": median, "relative_spread": spread, "relative_drift": drift}


def _extension_reasons(cpu: list[float], gpu: list[float], threshold: float) -> list[str]:
    reasons = []
    for device, samples in (("cpu", cpu), ("gpu", gpu)):
        if _statistics(samples)["relative_spread"] > CALIBRATION_MAX_RELATIVE_SPREAD:
            reasons.append(f"{device}_spread")
    ratio = statistics.median(gpu) / statistics.median(cpu)
    if abs(ratio - threshold) <= CALIBRATION_THRESHOLD_MARGIN + 1e-12:
        reasons.append("near_threshold")
    # An observed range, not a statistical confidence interval.
    if min(gpu) / max(cpu) <= threshold <= max(gpu) / min(cpu):
        reasons.append("range_crosses_threshold")
    return reasons


def _threshold(adapter: GpuAdapter | WebGPUDevice, threshold: float | None) -> float:
    from .onnx_device import ONNX_GPU_TIME_RATIO_THRESHOLDS

    if threshold is None:
        threshold = ONNX_GPU_TIME_RATIO_THRESHOLDS["webgpu" if isinstance(adapter, WebGPUDevice) else "dml"]
    if threshold is None or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Calibration threshold must be finite and between zero and one")
    return threshold


@dataclass
class _RecMeasurement:
    device: str
    session: Any
    input_name: str
    init_s: float
    elapsed: list[float] = field(default_factory=list)
    tokens: list[list[int]] | None = None

    def run_until(self, total: int, tensor: Any, num_classes: int) -> None:
        for _ in range(total - len(self.elapsed)):
            started = time.perf_counter()
            outputs = self.session.run([REC_IDS_OUTPUT, REC_SCORES_OUTPUT], {self.input_name: tensor})
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Calibration returned invalid timings")
            tokens = _tokens(outputs, num_classes)
            if self.tokens is not None and tokens != self.tokens:
                raise ValueError(f"{self.device} rec calibration output changed between runs")
            self.tokens = tokens
            self.elapsed.append(elapsed)

    def report(self, warmup_runs: int) -> dict[str, Any]:
        timed = self.elapsed[warmup_runs:]
        return {
            "init_s": self.init_s,
            "first_s": self.elapsed[0],
            "warmup_s": self.elapsed[:warmup_runs],
            "timed_s": timed,
            **_statistics(timed),
            "providers": self.session.get_providers(),
        }


def measure_rec(
    model_path: str,
    adapter: GpuAdapter | WebGPUDevice,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
    gpu_first: bool = False,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Start with 1+3, extending the same sessions to 3+7 for uncertain measurements."""
    import onnxruntime as ort
    from .onnx import configure_ort_threads, create_ort_session
    from .onnx_device import metacommands_enabled

    threshold = _threshold(adapter, threshold)
    compact = prepare_rec_model(model_path)
    tensor, evidence = prepare_input()
    options = ort.SessionOptions()
    configure_ort_threads(options, intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads)
    backend = "webgpu" if isinstance(adapter, WebGPUDevice) else "dml"
    result: dict[str, Any] = {
        "schema_version": 5,
        "measurement": dict(CALIBRATION_MEASUREMENT),
        "decision_threshold": threshold,
        "backend": backend,
        "ep_version": version("onnxruntime-ep-webgpu") if backend == "webgpu" else ort.__version__,
        "webgpu_options": dict(WEBGPU_OPTIONS) if backend == "webgpu" else None,
        "adapter": asdict(adapter),
        "input": evidence,
        "model_sha256": compact.model_sha256,
        "source_model_sha256": compact.source_sha256,
        "output_format": REC_OUTPUT_FORMAT,
        "ort_version": ort.__version__,
        "ort_path": ort.__file__,
        "python": sys.version,
        "intra_threads": options.intra_op_num_threads,
        "inter_threads": options.inter_op_num_threads,
        "rec_metacommands": metacommands_enabled(adapter.vendor, "ocr_rec") if isinstance(adapter, GpuAdapter) else None,
        "gpu_first": gpu_first,
    }
    measurements: dict[str, _RecMeasurement] = {}
    for device in ("gpu", "cpu") if gpu_first else ("cpu", "gpu"):
        started = time.perf_counter()
        session = create_ort_session(
            compact.path,
            adapter=adapter if device == "gpu" else None,
            model_kind="ocr_rec",
            intra_op_num_threads=options.intra_op_num_threads,
            inter_op_num_threads=options.inter_op_num_threads,
        )
        init_s = time.perf_counter() - started
        expected = (
            (WEBGPU_PROVIDER if backend == "webgpu" else "DmlExecutionProvider") if device == "gpu" else "CPUExecutionProvider"
        )
        if session.get_providers()[:1] != [expected]:
            raise ValueError(f"Calibration requires {expected}")
        measurement = _RecMeasurement(device, session, session.get_inputs()[0].name, init_s)
        measurement.run_until(CALIBRATION_INITIAL_WARMUP_RUNS + CALIBRATION_INITIAL_TIMED_RUNS, tensor, compact.num_classes)
        measurements[device] = measurement
    if measurements["cpu"].tokens != measurements["gpu"].tokens:
        raise ValueError("CPU and GPU rec calibration text differ")
    initial = {device: measurement.report(CALIBRATION_INITIAL_WARMUP_RUNS) for device, measurement in measurements.items()}
    reasons = _extension_reasons(initial["cpu"]["timed_s"], initial["gpu"]["timed_s"], threshold)
    warmup_runs = CALIBRATION_INITIAL_WARMUP_RUNS
    if reasons:
        warmup_runs = CALIBRATION_WARMUP_RUNS
        # Keep both sessions alive; run serially. Calls 2 and 3 are reclassified as warm-ups,
        # call 4 is retained, and only six further calls per backend are needed.
        for measurement in measurements.values():
            measurement.run_until(CALIBRATION_WARMUP_RUNS + CALIBRATION_TIMED_RUNS, tensor, compact.num_classes)
    result["sampling"] = {
        "stage": "extended" if reasons else "initial",
        "warmup_runs": warmup_runs,
        "timed_runs": CALIBRATION_TIMED_RUNS if reasons else CALIBRATION_INITIAL_TIMED_RUNS,
        "extension_reasons": reasons,
        "initial": initial,
    }
    for device, measurement in measurements.items():
        result[device] = measurement.report(warmup_runs)
        result[f"{device}_elapsed"] = result[device]["median_s"]
    result["ratio"] = result["gpu_elapsed"] / result["cpu_elapsed"]
    result["outputs_match"] = True
    result["decision_valid"] = all(
        result[device][metric] <= CALIBRATION_MAX_RELATIVE_SPREAD
        for device in ("cpu", "gpu")
        for metric in ("relative_spread", "relative_drift")
    )
    return result


def _validate_sampling(result: dict[str, Any]) -> None:
    sampling = result.get("sampling", {})
    counts = {
        "initial": (CALIBRATION_INITIAL_WARMUP_RUNS, CALIBRATION_INITIAL_TIMED_RUNS),
        "extended": (CALIBRATION_WARMUP_RUNS, CALIBRATION_TIMED_RUNS),
    }
    if sampling.get("stage") not in counts:
        raise ValueError("Calibration worker returned invalid sampling stage")
    warmups, timed = counts[sampling["stage"]]
    if (sampling.get("warmup_runs"), sampling.get("timed_runs")) != (warmups, timed):
        raise ValueError("Calibration worker returned invalid sampling counts")
    stable = True
    for device in ("cpu", "gpu"):
        report = result[device]
        warmup_s, timed_s = report.get("warmup_s", []), report.get("timed_s", [])
        if len(warmup_s) != warmups or len(timed_s) != timed:
            raise ValueError("Calibration worker returned invalid sampling counts")
        if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in warmup_s + timed_s):
            raise ValueError("Calibration worker returned invalid timings")
        stats = _statistics(timed_s)
        if result[f"{device}_elapsed"] != stats["median_s"] or report.get("median_s") != stats["median_s"]:
            raise ValueError("Calibration worker returned inconsistent median")
        stable &= all(stats[metric] <= CALIBRATION_MAX_RELATIVE_SPREAD for metric in ("relative_spread", "relative_drift"))
    if result["decision_valid"] != stable:
        raise ValueError("Calibration worker returned inconsistent stability decision")
    if result.get("ratio") != result["gpu_elapsed"] / result["cpu_elapsed"]:
        raise ValueError("Calibration worker returned inconsistent ratio")
    if sampling["stage"] == "initial" and _extension_reasons(
        result["cpu"]["timed_s"], result["gpu"]["timed_s"], result["decision_threshold"]
    ):
        raise ValueError("Calibration worker skipped required extended sampling")


def run_calibration(
    model_path: str,
    adapter: GpuAdapter | WebGPUDevice,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
    gpu_first: bool = False,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run a fresh, hidden worker and kill it on timeout; failed measurements are never decisions."""
    from .onnx_calibration_cache import invalidate_calibration_record, save_calibration_record

    invalidate_calibration_record()
    # Build/hash the derived graph outside the bounded inference worker, like model download.
    threshold = _threshold(adapter, threshold)
    prepare_rec_model(model_path)
    backend = "webgpu" if isinstance(adapter, WebGPUDevice) else "dml"
    request = {
        "backend": backend,
        "model_path": model_path,
        "adapter": asdict(adapter),
        "intra_op_num_threads": intra_op_num_threads,
        "inter_op_num_threads": inter_op_num_threads,
        "gpu_first": gpu_first,
        "threshold": threshold,
    }
    environment = {**os.environ, "PYTHONUTF8": "1", "MINERU_MODEL_ONNX_DEVICE": "cpu"}
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-B", "-m", "mineru.model.runtime.onnx_calibration"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=CALIBRATION_TIMEOUT_SECONDS,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if process.returncode:
        raise RuntimeError(f"ONNX calibration worker failed ({process.returncode}): {process.stderr[-1500:]}")
    result = json.loads(process.stdout)
    for key in ("cpu_elapsed", "gpu_elapsed"):
        if not isinstance(result.get(key), (float, int)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Calibration worker returned invalid timings")
    provider = WEBGPU_PROVIDER if backend == "webgpu" else "DmlExecutionProvider"
    if (
        result.get("schema_version") != 5
        or result.get("backend") != backend
        or result.get("measurement") != CALIBRATION_MEASUREMENT
        or result.get("decision_threshold") != threshold
        or type(result.get("decision_valid")) is not bool
        or result.get("output_format") != REC_OUTPUT_FORMAT
        or result.get("outputs_match") is not True
        or result.get("gpu", {}).get("providers", [])[:1] != [provider]
        or result.get("cpu", {}).get("providers", [])[:1] != ["CPUExecutionProvider"]
    ):
        raise ValueError(f"Calibration worker did not confirm valid {provider} results")
    _validate_sampling(result)
    result["worker_wall_s"] = time.perf_counter() - started
    # 保存实际校准证据供下次服务启动读取，不将手动 cpu/gpu 偏好视为性能结论。
    save_calibration_record(adapter, result)
    return result


def main() -> None:
    """Private JSON-over-stdin worker; no user files or global configuration are modified."""
    request = json.load(sys.stdin)
    backend = request.pop("backend")
    descriptor = request.pop("adapter")
    if backend == "webgpu":
        descriptor["metadata"] = tuple(tuple(pair) for pair in descriptor["metadata"])
        if descriptor.get("vulkan_device") is not None:
            descriptor["vulkan_device"] = tuple(descriptor["vulkan_device"])
        adapter = WebGPUDevice(**descriptor)
    elif backend == "dml":
        adapter = GpuAdapter(**descriptor)
    else:
        raise ValueError(f"Unknown calibration backend: {backend}")
    result = measure_rec(adapter=adapter, **request)
    print(json.dumps(result, ensure_ascii=True))


__all__ = [
    "CALIBRATION_INITIAL_TIMED_RUNS",
    "CALIBRATION_INITIAL_WARMUP_RUNS",
    "CALIBRATION_MEASUREMENT",
    "CALIBRATION_TIMED_RUNS",
    "CALIBRATION_TIMEOUT_SECONDS",
    "CALIBRATION_WARMUP_RUNS",
    "measure_rec",
    "prepare_input",
    "run_calibration",
    "sample_directory",
]

if __name__ == "__main__":
    main()
