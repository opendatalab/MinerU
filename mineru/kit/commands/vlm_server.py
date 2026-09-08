from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata
import platform
import sys
from collections.abc import Callable
from typing import Literal

import typer
from loguru import logger

from docvortex.foundation.platform import is_mac_os_version_supported
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message

FORWARD_CONTEXT_SETTINGS = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _mlx_server_error() -> str | None:
    """检查平台与发布版本，不导入 MLX 重依赖。"""
    if platform.system() != "Darwin" or platform.machine() != "arm64" or not is_mac_os_version_supported("14.0"):
        return "MLX server requires Apple Silicon and macOS 14 or newer."
    try:
        from packaging.version import Version

        version = Version(importlib.metadata.version("mlx-vlm"))
        if not Version("0.7.0") <= version < Version("0.8.0"):
            return f"MLX server requires mlx-vlm>=0.7.0,<0.8.0; installed: {version}. Install 'mineru[full]'."
        if importlib.util.find_spec("mlx_vlm.server") is None:
            return "mlx_vlm.server is unavailable. Install 'mineru[full]'."
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
        return "MLX-VLM is not installed. Install 'mineru[full]'."
    return None


def _mlx_server_available() -> bool:
    """供自动引擎选择复用平台与依赖检查。"""
    return _mlx_server_error() is None


def _run_with_forwarded_argv(main_fn: Callable[[], None], args: list[str]) -> None:
    original_argv = sys.argv
    sys.argv = [sys.argv[0], *args]
    try:
        main_fn()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise typer.Exit(code) from None
    finally:
        sys.argv = original_argv


def _resolve_auto_engine() -> Literal["vllm", "lmdeploy", "mlx"]:
    if _module_available("vllm"):
        logger.info("Using vLLM as the inference engine for VLM server.")
        return "vllm"
    if _module_available("lmdeploy"):
        logger.info("Using LMDeploy as the inference engine for VLM server.")
        return "lmdeploy"
    if _mlx_server_available():
        logger.info("Using MLX-VLM as the inference engine for VLM server.")
        return "mlx"
    logger.info("vLLM/LMDeploy/MLX-VLM is not installed. Please install at least one of them.")
    raise typer.Exit(1) from None


def vlm_server_cmd(
    ctx: typer.Context,
    engine: str = typer.Option("auto", "--engine", help="VLM serving engine: auto, vllm, lmdeploy, mlx"),
) -> None:
    """Start the local VLM server with OpenAI-compatible chat completions."""
    if engine not in {"auto", "vllm", "lmdeploy", "mlx"}:
        exit_with_message("invalid_request", f"Unsupported engine '{engine}'.", "engine")
    extra_args = list(ctx.args)

    if engine == "auto":
        engine = _resolve_auto_engine()

    if engine == "vllm":
        if not _module_available("vllm"):
            logger.error("vLLM is not installed. Please install vLLM or choose lmdeploy/mlx as the engine.")
            raise typer.Exit(1) from None
        from ..vlm_server import vllm_server

        _run_with_forwarded_argv(vllm_server.main, extra_args)

    elif engine == "lmdeploy":
        if not _module_available("lmdeploy"):
            logger.error("LMDeploy is not installed. Please install LMDeploy or choose vllm/mlx as the engine.")
            raise typer.Exit(1) from None
        from ..vlm_server import lmdeploy_server

        _run_with_forwarded_argv(lmdeploy_server.main, extra_args)

    elif engine == "mlx":
        if not _mlx_server_available():
            logger.error(_mlx_server_error() or "MLX server is unavailable.")
            raise typer.Exit(1) from None
        from ..vlm_server import mlx_vlm_server

        mlx_vlm_server.main(
            args=extra_args,
            prog_name="mineru-kit vlm-server",
            standalone_mode=False,
        )


def main() -> None:
    """以独立命令启动 VLM 服务，复用 kit 的未知引擎参数透传规则。"""
    configure_standard_streams()
    app = typer.Typer(add_completion=False)
    app.command(context_settings=FORWARD_CONTEXT_SETTINGS)(vlm_server_cmd)
    app()


__all__ = ["FORWARD_CONTEXT_SETTINGS", "main", "vlm_server_cmd"]
