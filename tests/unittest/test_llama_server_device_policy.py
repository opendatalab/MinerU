"""server 与进程内引擎遵循同一设备计划，参数别名与环境不泄漏。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.kit.vlm_server import llama_cpp_server as server
from mineru.model.runtime import llama_device as policy
from mineru.model.runtime.llama_device import LlamaSelection, SERIALIZE_ENV
from mineru.model.runtime.llama_probe import LlamaDevice


def plan() -> LlamaSelection:
    """显式选择原来的 Vulkan1 后，隔离原生设备索引会变为 Vulkan0。"""
    return LlamaSelection(
        "gpu",
        (LlamaDevice("Vulkan0", "Intel", "vulkan", 0x8086, 0x4680, 1, kind="integrated", kind_source="vulkan"),),
        "explicit_gpu",
        ((SERIALIZE_ENV, "1"), ("GGML_VK_VISIBLE_DEVICES", "1")),
        (("Vulkan1", "Vulkan0"),),
    )


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
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising=False)
    monkeypatch.delenv("LLAMA_ARG_N_GPU_LAYERS", raising=False)
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
    for name in ("LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_DEVICE", "MTMD_BACKEND_DEVICE", "LLAMA_ARG_MMPROJ_OFFLOAD"):
        monkeypatch.delenv(name, raising=False)
    resolve = Mock(return_value=LlamaSelection())
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy(["--offline"])
    resolve.assert_called_once_with(n_gpu_layers=None, device_names=None, allow_download=False)


def test_cpu_model_with_explicit_gpu_projector_still_gets_workaround(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = Mock(return_value=plan())
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    args = ["-ngl", "0", "--mmproj-device", "Vulkan1", "--mmproj-offload"]
    selected = server._device_policy(args)
    assert selected.cpu_model
    server._apply_device_policy(args, selected)
    assert server._flag_value(args, "--n-gpu-layers") == "0"
    assert server._flag_value(args, "--mmproj-device") == "Vulkan0"
    assert dict(selected.environment)[SERIALIZE_ENV] == "1"


def test_disabled_projector_does_not_activate_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = Mock(return_value=LlamaSelection("cpu"))
    monkeypatch.setattr(server, "resolve_llama_device", resolve)
    server._device_policy(["-ngl", "0", "--mmproj-device", "Vulkan1", "--no-mmproj-offload"])
    resolve.assert_called_once_with(n_gpu_layers=0, device_names=None)


def test_cpu_policy_disables_model_projector_and_op_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMA_ARG_MMPROJ_OFFLOAD", raising=False)
    args = []
    server._apply_device_policy(args, LlamaSelection("cpu", reason="onnx_rec_calibration"))
    assert server._flag_value(args, "--n-gpu-layers") == "0"
    assert server._flag_value(args, "--device") == "none"
    assert "--no-mmproj-offload" in args and "--no-op-offload" in args


def test_projector_device_remaps_with_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTMD_BACKEND_DEVICE", raising=False)
    args = ["--device", "Vulkan1", "--mmproj-device", "Vulkan1"]
    server._apply_device_policy(args, plan())
    assert server._flag_value(args, "--device") == "Vulkan0"
    assert server._flag_value(args, "--mmproj-device") == "Vulkan0"
    assert "Vulkan1" not in args
    assert args.count("--device") == 1 and args.count("--mmproj-device") == 1
    with pytest.raises(ValueError, match="GPU group"):
        server._apply_device_policy(["--mmproj-device", "Vulkan2"], plan())


@pytest.mark.parametrize("fails", [False, True])
def test_exec_scopes_workaround_and_restores_on_return(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fails: bool) -> None:
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(server, "llama_server_binary", lambda: binary)
    monkeypatch.delenv(SERIALIZE_ENV, raising=False)
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)

    def execute(path: str, args: list[str]) -> None:
        assert os.environ[SERIALIZE_ENV] == "1" and os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"
        if fails:
            raise OSError("exec failed")

    monkeypatch.setattr(server.os, "execv", execute)
    if fails:
        with pytest.raises(OSError, match="exec failed"):
            server._exec_server([], plan())
    else:
        server._exec_server([], plan())
    assert SERIALIZE_ENV not in os.environ and "GGML_VK_VISIBLE_DEVICES" not in os.environ


def test_exec_remaps_environment_before_native_parser_validates_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(server, "llama_server_binary", lambda: binary)
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "Vulkan1")
    monkeypatch.setenv("MTMD_BACKEND_DEVICE", "Vulkan1")
    args = []
    server._apply_device_policy(args, plan())

    def execute(path: str, argv: list[str]) -> None:
        assert os.environ["LLAMA_ARG_DEVICE"] == "Vulkan0"
        assert os.environ["MTMD_BACKEND_DEVICE"] == "Vulkan0"
        assert "Vulkan1" not in argv

    monkeypatch.setattr(server.os, "execv", execute)
    server._exec_server(args, plan())
    assert os.environ["LLAMA_ARG_DEVICE"] == "Vulkan1"
    assert os.environ["MTMD_BACKEND_DEVICE"] == "Vulkan1"


@pytest.mark.parametrize(
    "platform,chip,serialize",
    [("linux", 0x7D51, True), ("linux", 0xB640, True), ("linux", 0x56A0, False), ("win32", 0x7D51, False)],
)
@pytest.mark.parametrize("projector_only", [False, True])
def test_actual_device_policy_reaches_server_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, chip: int, serialize: bool, projector_only: bool
) -> None:
    """复用真实设备策略，验证模型或仅视觉编码选中 Xe-LPG 时的启动环境。"""
    for name in (
        *policy._BACKEND_ENV,
        "MINERU_LLAMA_CPP_BACKEND",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_N_GPU_LAYERS",
        "MTMD_BACKEND_DEVICE",
        "LLAMA_ARG_MMPROJ_OFFLOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform=platform))
    device = LlamaDevice(
        "Vulkan0", "Intel", "vulkan", 0x8086, chip, 0, kind="integrated" if chip != 0x56A0 else "discrete", kind_source="vulkan"
    )
    monkeypatch.setattr(policy, "discover_llama_devices", Mock(return_value=(device,)))
    args = ["-ngl", "0", "--mmproj-device", "Vulkan0", "--mmproj-offload"] if projector_only else ["--device", "Vulkan0"]
    selected = server._device_policy(args)
    assert selected.cpu_model is projector_only
    assert dict(selected.environment).get(SERIALIZE_ENV) == ("1" if serialize else None)
    server._apply_device_policy(args, selected)

    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(server, "llama_server_binary", lambda: binary)

    def execute(path: str, argv: list[str]) -> None:
        assert os.environ.get(SERIALIZE_ENV) == ("1" if serialize else None)
        assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "0"
        if projector_only:
            assert server._flag_value(argv, "--n-gpu-layers") == "0"
            assert server._flag_value(argv, "--mmproj-device") == "Vulkan0"

    monkeypatch.setattr(server.os, "execv", execute)
    server._exec_server(args, selected)
    assert SERIALIZE_ENV not in os.environ and "GGML_VK_VISIBLE_DEVICES" not in os.environ
