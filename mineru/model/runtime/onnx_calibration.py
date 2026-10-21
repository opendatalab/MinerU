"""Bounded rec calibration worker, shared by automatic selection and the release benchmark."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .windows_gpu import GpuAdapter
from .webgpu import WEBGPU_OPTIONS, WEBGPU_PROVIDER, WebGPUDevice
from .onnx_rec import REC_IDS_OUTPUT, REC_OUTPUT_FORMAT, REC_SCORES_OUTPUT, prepare_rec_model, validate_rec_outputs

CALIBRATION_TIMEOUT_SECONDS = 15


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


def measure_rec(
    model_path: str,
    adapter: GpuAdapter | WebGPUDevice,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
    gpu_first: bool = False,
) -> dict[str, Any]:
    """One warm-up and two timed runs per device; never consult the automatic selector."""
    import onnxruntime as ort
    from .onnx import configure_ort_threads, create_ort_session
    from .onnx_device import metacommands_enabled

    compact = prepare_rec_model(model_path)
    tensor, evidence = prepare_input()
    options = ort.SessionOptions()
    configure_ort_threads(options, intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads)
    backend = "webgpu" if isinstance(adapter, WebGPUDevice) else "dml"
    result: dict[str, Any] = {
        "schema_version": 3,
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
    decoded: dict[str, list[list[int]]] = {}
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
        name = session.get_inputs()[0].name
        elapsed: list[float] = []
        reference = None
        for _ in range(3):
            started = time.perf_counter()
            outputs = session.run([REC_IDS_OUTPUT, REC_SCORES_OUTPUT], {name: tensor})
            elapsed.append(time.perf_counter() - started)
            tokens = _tokens(outputs, compact.num_classes)
            if reference is not None and tokens != reference:
                raise ValueError(f"{device} rec calibration output changed between runs")
            reference = tokens
        decoded[device] = reference
        result[device] = {"init_s": init_s, "first_s": elapsed[0], "timed_s": elapsed[1:], "providers": session.get_providers()}
        result[f"{device}_elapsed"] = sum(elapsed[1:])
        del session
    if decoded["cpu"] != decoded["gpu"]:
        raise ValueError("CPU and GPU rec calibration text differ")
    result["ratio"] = result["gpu_elapsed"] / result["cpu_elapsed"]
    result["outputs_match"] = True
    return result


def run_calibration(
    model_path: str,
    adapter: GpuAdapter | WebGPUDevice,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
    gpu_first: bool = False,
) -> dict[str, Any]:
    """Run a fresh, hidden worker and kill it on timeout; failed measurements are never decisions."""
    # Build/hash the derived graph outside the bounded inference worker, like model download.
    prepare_rec_model(model_path)
    backend = "webgpu" if isinstance(adapter, WebGPUDevice) else "dml"
    request = {
        "backend": backend,
        "model_path": model_path,
        "adapter": asdict(adapter),
        "intra_op_num_threads": intra_op_num_threads,
        "inter_op_num_threads": inter_op_num_threads,
        "gpu_first": gpu_first,
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
        result.get("backend") != backend
        or result.get("output_format") != REC_OUTPUT_FORMAT
        or result.get("outputs_match") is not True
        or result.get("gpu", {}).get("providers", [])[:1] != [provider]
        or result.get("cpu", {}).get("providers", [])[:1] != ["CPUExecutionProvider"]
    ):
        raise ValueError(f"Calibration worker did not confirm valid {provider} results")
    result["worker_wall_s"] = time.perf_counter() - started
    return result


def main() -> None:
    """Private JSON-over-stdin worker; no user files or global configuration are modified."""
    request = json.load(sys.stdin)
    backend = request.pop("backend")
    descriptor = request.pop("adapter")
    if backend == "webgpu":
        descriptor["metadata"] = tuple(tuple(pair) for pair in descriptor["metadata"])
        adapter = WebGPUDevice(**descriptor)
    elif backend == "dml":
        adapter = GpuAdapter(**descriptor)
    else:
        raise ValueError(f"Unknown calibration backend: {backend}")
    result = measure_rec(adapter=adapter, **request)
    print(json.dumps(result, ensure_ascii=True))


__all__ = ["CALIBRATION_TIMEOUT_SECONDS", "measure_rec", "prepare_input", "run_calibration", "sample_directory"]

if __name__ == "__main__":
    main()
