from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from mineru.kit.commands import vlm_server
from mineru.kit.vlm_server import mlx_vlm_server
from mineru_vl_utils import mlx_compat


@pytest.mark.parametrize("model_args", [["--model", "custom"], ["--model=custom"]])
def test_native_forwarding_and_cleanup(monkeypatch: pytest.MonkeyPatch, model_args: list[str]) -> None:
    """原生服务退出时恢复 argv 并结束兼容目录生命周期。"""
    events = []
    original = sys.argv

    @contextmanager
    def prepare(model: str) -> Iterator[Path]:
        """记录路径准备与清理顺序。"""
        events.append(model)
        try:
            yield Path("/prepared")
        finally:
            events.append("cleanup")

    def native() -> None:
        """模拟原生解析失败，确认异常不会泄漏资源。"""
        events.append(sys.argv[1:])
        raise SystemExit(2)

    monkeypatch.setattr(mlx_compat, "prepare_mlx_model_path", prepare)
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", SimpleNamespace(main=native))
    with pytest.raises(SystemExit, match="2"):
        mlx_vlm_server.main(args=[*model_args, "--port=1234", "--api-key", "test"], prog_name="test", standalone_mode=False)
    assert sys.argv is original
    assert events == [
        "custom",
        ["--model", "/prepared", "--host", "127.0.0.1", "--port", "1234", "--max-num-seqs", "8", "--api-key", "test"],
        "cleanup",
    ]


def test_help_skips_model_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """帮助直接委托原生入口，不解析或下载模型。"""
    seen = []
    monkeypatch.setattr(mlx_vlm_server, "_run_native", lambda *args: seen.append(args))
    mlx_vlm_server.main(args=["--help"], prog_name="test", standalone_mode=False)
    assert seen == [(["--help"], "test", False)]


@pytest.mark.parametrize("version, available", [("0.3.9", False), ("0.7.0", True), ("0.8.0", False)])
def test_version_gate(monkeypatch: pytest.MonkeyPatch, version: str, available: bool) -> None:
    """自动检测只接受已验证的版本范围。"""
    monkeypatch.setattr(vlm_server.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vlm_server.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(vlm_server, "is_mac_os_version_supported", lambda _: True)
    monkeypatch.setattr(vlm_server.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(vlm_server.importlib.util, "find_spec", lambda _: object())
    assert vlm_server._mlx_server_available() is available


@pytest.mark.parametrize("system,machine", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_platform_gate(monkeypatch: pytest.MonkeyPatch, system: str, machine: str) -> None:
    """不支持的平台给出平台提示，不尝试导入 MLX。"""
    monkeypatch.setattr(vlm_server.platform, "system", lambda: system)
    monkeypatch.setattr(vlm_server.platform, "machine", lambda: machine)
    assert "Apple Silicon" in vlm_server._mlx_server_error()


def test_default_model_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认模型来自注册表，原生调用期间兼容目录保持有效。"""
    seen = []

    @contextmanager
    def prepare(model: str) -> Iterator[Path]:
        """记录默认模型并提供临时模型路径。"""
        seen.append(model)
        yield Path("/prepared")

    monkeypatch.setattr(mlx_vlm_server, "MINERU_2_5_PRO_2605_1_2B", SimpleNamespace(ensure=lambda: Path("/default")))
    monkeypatch.setattr(mlx_compat, "prepare_mlx_model_path", prepare)
    monkeypatch.setattr(mlx_vlm_server, "_run_native", lambda args, *rest: seen.append(args))
    mlx_vlm_server.main(args=[], prog_name="test", standalone_mode=False)
    assert seen == ["/default", ["--model", "/prepared", "--host", "127.0.0.1", "--port", "8080", "--max-num-seqs", "8"]]


@pytest.mark.parametrize(
    "args,env,expected",
    [([], None, "8"), ([], "4", "4"), (["--max-num-seqs", "12"], "4", "12"), (["--max-num-seqs=16"], None, "16")],
)
def test_server_sequence_limit_precedence(
    monkeypatch: pytest.MonkeyPatch, args: list[str], env: str | None, expected: str
) -> None:
    """活跃并发默认八，显式 CLI 与环境配置按优先级覆盖默认值。"""
    from contextlib import nullcontext

    if env is None:
        monkeypatch.delenv("MLX_VLM_MAX_NUM_SEQS", raising=False)
    else:
        monkeypatch.setenv("MLX_VLM_MAX_NUM_SEQS", env)
    monkeypatch.setattr(mlx_compat, "prepare_mlx_model_path", lambda model: nullcontext(Path("/prepared")))
    seen = []
    monkeypatch.setattr(mlx_vlm_server, "_run_native", lambda argv, *rest: seen.extend(argv))
    mlx_vlm_server.main(args=["--model", "test", *args], prog_name="test", standalone_mode=False)
    assert seen.count("--max-num-seqs") == 1
    assert seen[seen.index("--max-num-seqs") + 1] == expected
