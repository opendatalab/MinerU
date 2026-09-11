from __future__ import annotations

import os

import typer

from ...config import config, get_config_file_exists, get_config_file_path, get_config_source
from ...model.download import (
    DOWNLOAD_MODEL_SOURCES,
    MODEL_SOURCE_ENV_VAR,
    download_model_repo,
    verify_model_repo,
)
from ...model.registry import (
    MODEL_REPOS,
    ModelRepo,
    get_model_repo,
    model_repo_names,
    model_repos_for_tier,
    validate_model_tier,
)
from ...model.runtime.device import resolve_small_model_backend
from ...model.vlm.selector import resolve_vlm_engine
from ...types import DEPLOYMENT_TIERS
from ...utils.stdio import configure_standard_streams
from ..errors import exit_with_message
from ..output import print_info, print_success

app = typer.Typer(help="Download, inspect, and verify local MinerU models.", no_args_is_help=True)


def _validate_download_source(source: str | None) -> str | None:
    """校验显式模型来源，未指定时保留自动选择。"""
    if source is None:
        return None
    normalized = source.strip().lower()
    if normalized not in DOWNLOAD_MODEL_SOURCES:
        expected = ", ".join(DOWNLOAD_MODEL_SOURCES)
        exit_with_message("invalid_request", f"Unsupported source '{source}'. Expected one of: {expected}.", "source")
    return normalized


def _select_target_repos(
    repo_name: str | None,
    tier: str | None,
    *,
    small_backend: str | None = None,
    vlm_engine: str | None = None,
) -> tuple[ModelRepo, ...]:
    """选择显式仓库或按独立后端组合档位资源。"""
    if repo_name and tier:
        exit_with_message("invalid_request", "Pass either a model repo name or --tier, not both.")
    if not repo_name and tier is None:
        exit_with_message("invalid_request", "Pass a model repo name or --tier.")

    if tier is not None:
        try:
            resolved_tier = validate_model_tier(tier)
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "tier")
        try:
            return model_repos_for_tier(resolved_tier, small_backend=small_backend, vlm_engine=vlm_engine)
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "backend")

    try:
        return (get_model_repo(repo_name or ""),)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "repo")


def _format_repo_status(repo: ModelRepo) -> str:
    """格式化仓库的本地就绪状态。"""
    result = verify_model_repo(repo)
    status = "ready" if result.ready else "missing"
    return f"{repo.name}: {status} ({repo.local_dir()})"


@app.command("download")
def download_cmd(
    repo: str | None = typer.Argument(None, help="Model repo: MinerU-4_models_torch, MinerU-4_models_onnx, or a VLM repo"),
    tier: str | None = typer.Option(None, "--tier", help="Model tier to prepare: basic or standard"),
    small_backend: str | None = typer.Option(
        None,
        "--small-backend",
        help="Small model backend: auto, onnx, torch. Ignored when REPO is given.",
    ),
    vlm_engine: str | None = typer.Option(
        None,
        "--vlm-engine",
        help="Local VLM engine: auto, llama-cpp, vllm, lmdeploy, mlx. Ignored when REPO is given.",
    ),
    source: str | None = typer.Option(None, "--source", "-s", help="Model source: auto, huggingface, or modelscope"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """下载显式仓库或当前后端组合所需的档位资源。"""
    normalized_source = _validate_download_source(source)
    repos = _select_target_repos(repo, tier, small_backend=small_backend, vlm_engine=vlm_engine)

    for target_repo in repos:
        try:
            root = download_model_repo(target_repo, source=normalized_source, local_as_auto=True)
        except Exception as exc:
            exit_with_message("api_error", f"Failed to download {target_repo.name}: {exc}")
        if verbose:
            print_info(f"{target_repo.name}: {root}")

    label = f"tier {tier}" if tier is not None else repos[0].name
    print_success(f"Downloaded models for {label}.")


@app.command("show")
def show_cmd(
    small_backend: str | None = typer.Option(
        None,
        "--small-backend",
        help="Small model backend: auto, onnx, torch. Ignored when REPO is given.",
    ),
    vlm_engine: str | None = typer.Option(
        None,
        "--vlm-engine",
        help="Local VLM engine: auto, llama-cpp, vllm, lmdeploy, mlx. Ignored when REPO is given.",
    ),
) -> None:
    """显示配置来源、有效后端与档位资源。"""
    try:
        effective_small_backend = resolve_small_model_backend(small_backend)
        effective_vlm_engine = (
            "http-client" if vlm_engine is None and config.model.vlm.server_url else resolve_vlm_engine(vlm_engine)
        )
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "backend")

    config_file = get_config_file_path()
    lines = [
        f"Config: {config_file}",
        f"Config exists: {str(get_config_file_exists()).lower()}",
        f"MINERU_MODEL_SOURCE={os.getenv(MODEL_SOURCE_ENV_VAR, '') or '(unset)'}",
        f"model.base_dir: {config.model.base_dir}",
        f"model.base_dir.source: {get_config_source('model.base_dir')}",
        f"model.source: {config.model.source}",
        f"model.source.source: {get_config_source('model.source')}",
        f"model.small_backend: {config.model.small_backend}",
        f"model.small_backend.source: {get_config_source('model.small_backend')}",
        f"model.vlm.engine: {config.model.vlm.engine}",
        f"model.vlm.engine.source: {get_config_source('model.vlm.engine')}",
        f"Effective small backend: {effective_small_backend}",
        f"Effective VLM engine: {effective_vlm_engine}",
        "Repos:",
    ]
    for line in lines:
        print_info(line)

    for repo in MODEL_REPOS:
        print_info(f"  {_format_repo_status(repo)}")

    print_info("Model tiers:")
    for tier in DEPLOYMENT_TIERS:
        repos = model_repos_for_tier(tier, small_backend=effective_small_backend, vlm_engine=vlm_engine)
        names = ", ".join(repo.name for repo in repos) or "(none)"
        print_info(f"  {tier}: {names}")


@app.command("verify")
def verify_cmd(
    repo: str | None = typer.Argument(None, help="Optional model repo name"),
    tier: str | None = typer.Option(None, "--tier", help="Optional model tier: basic or standard"),
    small_backend: str | None = typer.Option(
        None,
        "--small-backend",
        help="Small model backend: auto, onnx, torch. Ignored when REPO is given.",
    ),
    vlm_engine: str | None = typer.Option(
        None,
        "--vlm-engine",
        help="Local VLM engine: auto, llama-cpp, vllm, lmdeploy, mlx. Ignored when REPO is given.",
    ),
) -> None:
    """校验显式仓库或当前后端组合所需的本地资源。"""
    if repo is not None:
        repos = _select_target_repos(repo, tier)
    else:
        try:
            repos = model_repos_for_tier(tier or "standard", small_backend=small_backend, vlm_engine=vlm_engine)
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "backend")

    failures = 0
    for target_repo in repos:
        result = verify_model_repo(target_repo)
        if result.ready:
            print_success(f"{target_repo.name}: ok")
            continue
        failures += 1
        missing = ", ".join(result.missing_paths)
        print_info(f"{target_repo.name}: missing key paths: {missing}")

    if failures:
        raise typer.Exit(1)


def download_main() -> None:
    """配置标准流后直接运行模型下载命令，无需再输入 download 子命令。"""
    configure_standard_streams()
    typer.run(download_cmd)


__all__ = ["app", "download_cmd", "download_main", "show_cmd", "verify_cmd", "model_repo_names"]
