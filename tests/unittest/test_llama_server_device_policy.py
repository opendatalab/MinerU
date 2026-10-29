"""Engine/server 共用 CPU/GPU 决策，显式设备参数原样交给原生运行时。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.kit.vlm_server import llama_cpp_server as server
from mineru.model.runtime import device, llama_device as policy, llama_probe, onnx_device
from mineru.model.runtime.llama_device import LlamaSelection, SERIALIZE_ENV
from mineru.model.runtime.onnx_device import OnnxSelection
from mineru.model.runtime.webgpu import WebGPUDevice


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(device, "resolve_small_model_backend", lambda: "onnx")
    for name in (
        *policy._BACKEND_ENV,
        "MINERU_LLAMA_CPP_BACKEND",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_N_GPU_LAYERS",
        "MTMD_BACKEND_DEVICE",
        "LLAMA_ARG_MMPROJ_OFFLOAD",
        "LLAMA_ARG_OFFLINE",
        "LLAMA_ARG_THREADS",
        "LLAMA_ARG_THREADS_BATCH",
    ):
        monkeypatch.delenv(name, raising=False)
    forbidden = Mock(side_effect=AssertionError("server must not probe llama devices"))
    monkeypatch.setattr(llama_probe, "probe_devices", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


@pytest.mark.parametrize(
    "args,layers,names",
    [
        ([], None, None),
        (["-ngl", "0"], 0, None),
        (["--n-gpu-layers=all"], -1, None),
        (["--device=Vulkan1"], None, ("Vulkan1",)),
        (["-ngl", "0", "--gpu-layers", "99", "-dev", "Vulkan1"], 99, ("Vulkan1",)),
    ],
)
def test_raw_selection_arguments_are_read_before_defaults(
    monkeypatch: pytest.MonkeyPatch, args: list[str], layers: int | None, names: tuple[str, ...] | None
) -> None:
    resolve = Mock(return_value=LlamaSelection())
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy(args)
    resolve.assert_called_once_with(n_gpu_layers=layers, device_names=names)


def test_native_environment_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "Vulkan1")
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "99")
    resolve = Mock(return_value=LlamaSelection())
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy([])
    resolve.assert_called_once_with(n_gpu_layers=99, device_names=("Vulkan1",))


def test_offline_prohibits_calibration_download(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = Mock(return_value=LlamaSelection())
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy(["--offline"])
    resolve.assert_called_once_with(n_gpu_layers=None, device_names=None, allow_download=False)


@pytest.mark.parametrize("source", ["cli", "env"])
@pytest.mark.parametrize("cpu_model", [False, True])
def test_explicit_projector_preserves_native_names_and_mixed_routing(
    monkeypatch: pytest.MonkeyPatch, source: str, cpu_model: bool
) -> None:
    args = ["-ngl", "0"] if cpu_model else ["--device", "Vulkan1"]
    if source == "cli":
        args += ["--mmproj-device", "Vulkan9"]
    else:
        monkeypatch.setenv("MTMD_BACKEND_DEVICE", "Vulkan9")
    select = Mock(side_effect=AssertionError("explicit projector must bypass ONNX"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", select)
    selected = server._device_policy(args)
    assert selected.device == "native" and not selected.environment
    server._apply_device_policy(args, selected)
    server._apply_thread_policy(args, selected)
    assert (server._flag_value(args, "--mmproj-device") or os.environ.get("MTMD_BACKEND_DEVICE")) == "Vulkan9"
    assert "--mmproj-offload" in args
    if cpu_model:
        assert server._flag_value(args, "--n-gpu-layers", "-ngl") == "0"
        assert "--no-op-offload" in args
        assert "--threads" not in args
    else:
        assert server._flag_value(args, "--device") == "Vulkan1"
        assert server._flag_value(args, "--threads") == "4"
    select.assert_not_called()


def test_disabled_projector_does_not_activate_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = Mock(return_value=LlamaSelection("cpu"))
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy(["-ngl", "0", "--mmproj-device", "Vulkan1", "--no-mmproj-offload"])
    resolve.assert_called_once_with(n_gpu_layers=0, device_names=None)


def test_cpu_policy_disables_model_projector_and_op_offload() -> None:
    args = []
    server._apply_device_policy(args, LlamaSelection("cpu", reason="onnx_rec_calibration"))
    assert server._flag_value(args, "--n-gpu-layers") == "0"
    assert server._flag_value(args, "--device") == "none"
    assert server._flag_value(args, "--mmproj-device") == "none"
    assert "--no-mmproj-offload" in args and "--no-op-offload" in args


def test_gpu_policy_does_not_inject_device_names() -> None:
    args = ["--model", "model.gguf"]
    original = list(args)
    server._apply_device_policy(args, LlamaSelection("gpu", reason="onnx_configured_gpu"))
    assert args == original


@pytest.mark.parametrize("fails", [False, True])
def test_exec_scopes_compatibility_and_preserves_user_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fails: bool
) -> None:
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(server, "llama_server_binary", lambda: binary)
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "Vulkan1")
    monkeypatch.setenv("MTMD_BACKEND_DEVICE", "Vulkan2")

    def execute(path: str, args: list[str]) -> None:
        assert os.environ[SERIALIZE_ENV] == "1"
        assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"
        assert os.environ["LLAMA_ARG_DEVICE"] == "Vulkan1"
        assert os.environ["MTMD_BACKEND_DEVICE"] == "Vulkan2"
        if fails:
            raise OSError("exec failed")

    monkeypatch.setattr(server.os, "execv", execute)
    selection = LlamaSelection("gpu", environment=((SERIALIZE_ENV, "1"),))
    if fails:
        with pytest.raises(OSError, match="exec failed"):
            server._exec_server([], selection)
    else:
        server._exec_server([], selection)
    assert SERIALIZE_ENV not in os.environ
    assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("explicit", [False, True])
def test_only_automatic_onnx_route_adds_intel_compatibility(
    monkeypatch: pytest.MonkeyPatch, platform: str, explicit: bool
) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    select = Mock(return_value=OnnxSelection("gpu", WebGPUDevice(0x8086, 0x7D51), "configured_gpu", "webgpu"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", select)
    args = ["--device", "Vulkan1"] if explicit else []
    selected = server._device_policy(args)
    assert dict(selected.environment).get(SERIALIZE_ENV) == ("1" if platform == "linux" and not explicit else None)
    server._apply_device_policy(args, selected)
    assert args == (["--device", "Vulkan1"] if explicit else [])
    assert select.call_count == int(not explicit)


@pytest.mark.parametrize("decision", ["cpu", "gpu"])
def test_server_main_applies_automatic_mode_before_gpu_defaults(monkeypatch: pytest.MonkeyPatch, decision: str) -> None:
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", lambda: OnnxSelection(decision, reason="configured_" + decision))
    monkeypatch.setattr(server.sys, "argv", ["llama-server", "-m", "model.gguf", "--mmproj", "mmproj.gguf"])
    monkeypatch.setattr(server, "_read_gguf_context_length", lambda path: 0)
    execute = Mock()
    monkeypatch.setattr(server, "_exec_server", execute)
    server.main()
    args, selection = execute.call_args.args
    assert selection.device == decision
    assert server._flag_value(args, "--n-gpu-layers") == ("0" if decision == "cpu" else "99")
    assert ("--no-mmproj-offload" in args) is (decision == "cpu")
    assert ("--no-op-offload" in args) is (decision == "cpu")
    assert server._flag_value(args, "--threads") == (None if decision == "cpu" else "4")


@pytest.mark.parametrize("source", ["short", "long", "inline", "env"])
@pytest.mark.parametrize("threads", ["6", "-1"])
def test_explicit_threads_override_gpu_default(monkeypatch: pytest.MonkeyPatch, source: str, threads: str) -> None:
    args = {
        "short": ["-t", threads],
        "long": ["--threads", threads],
        "inline": [f"--threads={threads}"],
        "env": [],
    }[source]
    if source == "env":
        monkeypatch.setenv("LLAMA_ARG_THREADS", threads)
    original = list(args)
    server._apply_thread_policy(args, LlamaSelection("gpu"))
    assert args == original


@pytest.mark.parametrize("source", ["cli", "env"])
def test_batch_threads_remain_independent(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    args = ["--threads-batch", "8"] if source == "cli" else []
    if source == "env":
        monkeypatch.setenv("LLAMA_ARG_THREADS_BATCH", "8")
    original = list(args)
    server._apply_thread_policy(args, LlamaSelection("gpu"))
    assert args == [*original, "--threads", "4"]


@pytest.mark.parametrize(
    "args,backend,expected",
    [
        (["--device", "Vulkan1"], "auto", "4"),
        (["-ngl", "99"], "auto", "4"),
        ([], "vulkan", "4"),
        ([], "cpu", None),
        (["-ngl", "0"], "auto", None),
        (["--device", "none"], "auto", None),
    ],
)
def test_manual_device_routes_apply_thread_defaults(
    monkeypatch: pytest.MonkeyPatch, args: list[str], backend: str, expected: str | None
) -> None:
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", backend)
    selected = server._device_policy(args)
    server._apply_device_policy(args, selected)
    server._apply_thread_policy(args, selected)
    assert server._flag_value(args, "--threads") == expected


def test_unresolved_native_route_retains_native_thread_default() -> None:
    args: list[str] = []
    server._apply_thread_policy(args, LlamaSelection(reason="platform_native"))
    assert not args
