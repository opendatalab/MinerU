"""Calibrate and independently validate the production six-line rec selection protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from mineru.model.runtime.onnx_calibration import run_calibration
from mineru.model.runtime.onnx_device import choose_gpu
from mineru.model.runtime.windows_gpu import discover_gpus, preferred_gpu


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment() -> dict[str, Any]:
    """Record host and power evidence outside the timed worker."""
    data: dict[str, Any] = {"host": platform.node(), "os": platform.platform(), "cpu": platform.processor()}
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            data["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        power = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
        )
        data["power_scheme"] = power.stdout.decode("utf-8", errors="replace").strip()
        drivers = []
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        ) as parent:
            for i in range(winreg.QueryInfoKey(parent)[0]):
                name = winreg.EnumKey(parent, i)
                if not name.isdigit():
                    continue
                with winreg.OpenKey(parent, name) as key:
                    try:
                        drivers.append(
                            {
                                field: winreg.QueryValueEx(key, field)[0]
                                for field in ("DriverDesc", "DriverVersion", "MatchingDeviceId")
                            }
                        )
                    except OSError:
                        continue
        data["display_drivers"] = drivers
    return data


def derive_threshold(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the midpoint of the separating interval; reject mismatched or failed protocols."""
    by_label = {report["label"]: report for report in reports}
    if len(reports) != 3 or set(by_label) != {"ultra5", "i7", "max395"}:
        raise ValueError("Exactly one ultra5, i7 and max395 report is required")
    signatures = set()
    ratios: dict[str, list[float]] = {}
    for label, report in by_label.items():
        if report.get("errors") or len(report["runs"]) != 5 or report.get("threshold") is not None:
            raise ValueError(f"{label} needs five successful calibration processes")
        ratios[label] = []
        for run in report["runs"]:
            if not run.get("outputs_match") or run["gpu"]["providers"][0] != "DmlExecutionProvider":
                raise ValueError("Missing DirectML execution evidence")
            signatures.add(
                (
                    run["input"]["manifest_sha256"],
                    run["input"]["input_sha256"],
                    run["model_sha256"],
                    run["ort_version"],
                    run["intra_threads"],
                    run["inter_threads"],
                )
            )
            ratio = run["gpu_elapsed"] / run["cpu_elapsed"]
            choose_gpu(run["cpu_elapsed"], run["gpu_elapsed"], 0.5)  # Validate finite positive timings only.
            ratios[label].append(ratio)
    if len(signatures) != 1:
        raise ValueError("Samples, weights, ORT or thread settings differ across calibration runs")
    lower = max(*ratios["ultra5"], *ratios["max395"])
    upper = min(*ratios["i7"], 1.0)
    if lower >= upper:
        raise ValueError(f"No separating interval: L={lower}, U={upper}; revise samples and rerun")
    return {
        "threshold": (lower + upper) / 2,
        "lower": lower,
        "upper": upper,
        "ratios": ratios,
        "signature": list(next(iter(signatures))),
        "status": "awaiting_independent_validation",
    }


def run(args: argparse.Namespace) -> None:
    adapters = discover_gpus()
    if args.luid:
        adapter = next((gpu for gpu in adapters if gpu.luid == args.luid), None)
    else:
        adapter = preferred_gpu(adapters)
    if adapter is None:
        raise ValueError("Requested hardware GPU was not found")
    if adapter.uma is not True:
        raise ValueError("Threshold calibration must explicitly target a confirmed integrated GPU")
    if args.model is None:
        from mineru.model.registry import MINERU_4_MODELS_ONNX

        model = MINERU_4_MODELS_ONNX.ocr_rec.ensure()
    else:
        model = args.model.resolve(strict=True)
    # Refuse to overwrite a completed or partial experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "label": args.label,
        "environment": environment(),
        "adapter": asdict(adapter),
        "threshold": args.threshold,
        "runs": [],
        "errors": [],
    }
    target = args.output / "report.json"
    count = 5 if args.threshold is None else 3
    for index in range(count):
        try:
            result = run_calibration(
                str(model),
                adapter,
                intra_op_num_threads=4,
                inter_op_num_threads=1,
                gpu_first=args.threshold is None and index % 2 == 1,
            )
            if args.threshold is not None:
                result["selected"] = (
                    "gpu" if choose_gpu(result["cpu_elapsed"], result["gpu_elapsed"], args.threshold) else "cpu"
                )
            report["runs"].append(result)
            print(
                f"{args.label} {index + 1}/{count}: ratio={result['ratio']:.6f}, wall={result['worker_wall_s']:.3f}s",
                flush=True,
            )
        except Exception as error:
            report["errors"].append({"process": index, "error": str(error)})
            print(f"{args.label} {index + 1}/{count}: {error}", flush=True)
        write_json(target, report)
    if args.threshold is not None:
        expected = "cpu" if args.label == "i7" else "gpu"
        report["validation_passed"] = not report["errors"] and all(row["selected"] == expected for row in report["runs"])
        write_json(target, report)
    if report["errors"] or report.get("validation_passed") is False:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices")
    measure = commands.add_parser("run")
    measure.add_argument("--label", choices=("ultra5", "i7", "max395"), required=True)
    measure.add_argument("--luid", help="Benchmark-only adapter identity; use devices to find the iGPU")
    measure.add_argument("--model", type=Path)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--threshold", type=float, help="Run three independent validation processes")
    derive = commands.add_parser("derive")
    derive.add_argument("reports", type=Path, nargs=3)
    derive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "devices":
        print(json.dumps([asdict(gpu) for gpu in discover_gpus()], ensure_ascii=False, indent=2))
    elif args.command == "run":
        run(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = derive_threshold([json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
