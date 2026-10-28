# Copyright (c) Opendatalab. All rights reserved.
"""隔离运行时导入和原生设备探测；不在模块导入时读取配置或初始化模型。"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import importlib.util
import json
import subprocess
import sys
from typing import Any

from .hardware_types import RuntimeProbe

PROBE_MARKER = "MINERU_HARDWARE_RESULT="
PROBE_TIMEOUT_SECONDS = 15


def _torch_probe() -> dict[str, Any]:
    """只查询 Torch 版本、可用性和设备属性，不创建张量或模型。"""
    import torch

    from .device import PROBE_TORCH_DEVICE_APIS, resolve_torch_device

    backends: dict[str, Any] = {}
    for name in ("cuda", "mps", "npu", *PROBE_TORCH_DEVICE_APIS):
        try:
            if name == "mps":
                available = torch.backends.mps.is_available()
                backends[name] = {"available": available}
                continue
            if name == "npu":
                if importlib.util.find_spec("torch_npu") is None:
                    continue
                api = importlib.import_module("torch_npu").npu
            else:
                api = getattr(torch, name, None)
            if api is None:
                continue
            available = bool(api.is_available())
            devices: list[dict[str, Any]] = []
            if available:
                for index in range(api.device_count()):
                    entry: dict[str, Any] = {"index": index}
                    if hasattr(api, "get_device_name"):
                        entry["name"] = api.get_device_name(index)
                    if hasattr(api, "get_device_properties"):
                        properties = api.get_device_properties(index)
                        entry["total_memory_bytes"] = getattr(properties, "total_memory", None)
                    devices.append(entry)
            backends[name] = {"available": available, "devices": devices}
        except Exception as error:
            backends[name] = {"available": None, "error": str(error)}
    selected = resolve_torch_device()
    return {
        "version": torch.__version__,
        "module_path": torch.__file__,
        "cuda_build": torch.version.cuda,
        "rocm_build": getattr(torch.version, "hip", None),
        "backends": backends,
        "selected_device": asdict(selected),
    }


def _onnx_probe() -> dict[str, Any]:
    """检查 ORT provider 和适配器库存，不调用会触发校准的设备选择。"""
    import onnxruntime as ort

    result: dict[str, Any] = {
        "version": ort.__version__,
        "module_path": ort.__file__,
        "providers": ort.get_available_providers(),
    }
    if sys.platform == "win32":
        from .windows_gpu import discover_gpus, preferred_gpu

        adapters = discover_gpus()
        selected = preferred_gpu(adapters)
        result["backend"] = "dml"
        result["adapters"] = [asdict(adapter) for adapter in adapters]
        result["selected_adapter"] = asdict(selected) if selected is not None else None
    elif sys.platform == "linux":
        from .webgpu import discover_webgpu_devices, select_webgpu_device

        try:
            adapters = discover_webgpu_devices()
            selected = select_webgpu_device(adapters)
            result.update(backend="webgpu", adapters=[asdict(adapter) for adapter in adapters])
            result["selected_adapter"] = asdict(selected) if selected is not None else None
        except Exception as error:
            result["adapter_error"] = str(error)
    return result


def _selection_probe() -> dict[str, Any]:
    """解析有效后端组合，只读配置而不进行推荐或模型选型。"""
    from ...config import get_config
    from ..vlm.selector import resolve_vlm_engine
    from .device import resolve_small_model_backend

    config = get_config()
    return {
        "small_backend": resolve_small_model_backend(),
        "vlm_engine": "http-client" if config.model.vlm.server_url else resolve_vlm_engine(config.model.vlm.engine),
    }


def probe_runtime(name: str) -> RuntimeProbe:
    """子进程内执行显式探测分支，将局部导入异常变成结构化结果。"""
    modules = {"torch": "torch", "onnx": "onnxruntime", "vllm": "vllm", "lmdeploy": "lmdeploy", "mlx-vlm": "mlx_vlm"}
    try:
        if name in modules and importlib.util.find_spec(modules[name]) is None:
            return RuntimeProbe(name, "not_installed")
        if name == "torch":
            data = _torch_probe()
        elif name == "onnx":
            data = _onnx_probe()
        elif name == "selection":
            data = _selection_probe()
        elif name == "llama-cpp":
            from .llama_probe import probe_devices

            if importlib.util.find_spec("mineru_llama_cpp") is None:
                return RuntimeProbe(name, "not_installed")
            if sys.platform == "darwin":
                module = importlib.import_module("mineru_llama_cpp")
                data = {"module_path": module.__file__, "backend": "platform_native"}
            else:
                data = {"devices": [asdict(device) for device in probe_devices()]}
        elif name == "vulkan":
            from .vulkan import enumerate_vulkan_devices

            data = {"devices": [asdict(device) for device in enumerate_vulkan_devices()]}
        elif name == "dxgi":
            from .windows_gpu import inspect_gpus

            data = {"devices": inspect_gpus()}
        elif name in modules:
            module = importlib.import_module(modules[name])
            data = {"module_path": module.__file__, "version": getattr(module, "__version__", None)}
        else:
            raise ValueError(f"Unknown hardware probe: {name}")
        return RuntimeProbe(name, "ok", data)
    except Exception as error:
        return RuntimeProbe(name, "error", error=f"{type(error).__name__}: {error}")


def _decode_result(output: str) -> dict[str, Any]:
    """从原生库输出中读取独立的结果标记，拒绝缺失或畸形的协议数据。"""
    line = next((line for line in reversed(output.splitlines()) if line.startswith(PROBE_MARKER)), None)
    if line is None:
        raise ValueError("Hardware probe did not return a result")
    result = json.loads(line.removeprefix(PROBE_MARKER))
    if not isinstance(result, dict):
        raise ValueError("Hardware probe returned an invalid result")
    return result


def run_probe(name: str) -> RuntimeProbe:
    """为每项运行时检查启动有超时保护的独立进程，失败不阻断其他信息采集。"""
    try:
        process = subprocess.run(
            [sys.executable, "-m", "mineru.model.runtime.hardware_probe", name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if process.returncode:
            return RuntimeProbe(name, "error", error=f"Probe exited {process.returncode}: {process.stderr[-1000:]}")
        return RuntimeProbe(**_decode_result(process.stdout))
    except subprocess.TimeoutExpired:
        return RuntimeProbe(name, "timeout", error=f"Probe exceeded {PROBE_TIMEOUT_SECONDS} seconds")
    except (OSError, ValueError, TypeError) as error:
        return RuntimeProbe(name, "error", error=str(error))


def run_detection() -> dict[str, Any]:
    """隔离命令的推荐过程，原生日志与下载进度进入 stderr，stdout 只交付结果。"""
    process = subprocess.run(
        [sys.executable, "-m", "mineru.model.runtime.hardware_probe", "detect"],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout.splitlines():
        if not line.startswith(PROBE_MARKER):
            print(line, file=sys.stderr)
    result = _decode_result(process.stdout)
    if process.returncode or "error" in result:
        raise ValueError(result.get("error", "Hardware detection failed"))
    return result


def main() -> None:
    """私有工作进程入口，结果标记之外的 Python 输出全部重定向到 stderr。"""
    name = sys.argv[1] if len(sys.argv) == 2 else ""
    failed = False
    with redirect_stdout(sys.stderr):
        if name == "detect":
            from .tier_recommendation import recommend_tiers

            try:
                result = asdict(recommend_tiers())
            except Exception as error:
                result = {"error": str(error)}
                failed = True
        else:
            result = asdict(probe_runtime(name))
    print(PROBE_MARKER + json.dumps(result, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


__all__ = ["PROBE_TIMEOUT_SECONDS", "probe_runtime", "run_detection", "run_probe"]

if __name__ == "__main__":
    main()
