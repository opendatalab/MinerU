from __future__ import annotations

import os

import typer

from ...config import config, get_config_file_exists, get_config_file_path, get_config_source
from ...types import DEPLOYMENT_TIERS
from ...utils.model_registry import (
    MODEL_REPOS,
    ModelRepo,
    get_model_repo,
    model_repo_names,
    model_repos_for_tier,
    validate_model_tier,
)
from ...utils.models_download_utils import (
    DOWNLOAD_MODEL_SOURCES,
    MODEL_SOURCE_ENV_VAR,
    download_model_repo,
    verify_model_repo,
)
from ..errors import exit_with_message
from ..output import print_info, print_success

app = typer.Typer(help="Download, inspect, and verify local MinerU models.", no_args_is_help=True)


def _validate_download_source(source: str | None) -> str | None:
    if source is None:
        return None
    normalized = source.strip().lower()
    if normalized not in DOWNLOAD_MODEL_SOURCES:
        expected = ", ".join(DOWNLOAD_MODEL_SOURCES)
        exit_with_message("invalid_request", f"Unsupported source '{source}'. Expected one of: {expected}.", "source")
    return normalized


def _select_target_repos(repo_name: str | None, tier: str | None) -> tuple[ModelRepo, ...]:
    if repo_name and tier:
        exit_with_message("invalid_request", "Pass either a model repo name or --tier, not both.")
    if not repo_name and tier is None:
        exit_with_message("invalid_request", "Pass a model repo name or --tier.")

    if tier is not None:
        try:
            resolved_tier = validate_model_tier(tier)
        except ValueError as exc:
            exit_with_message("invalid_request", str(exc), "tier")
        return model_repos_for_tier(resolved_tier)

    try:
        return (get_model_repo(repo_name or ""),)
    except ValueError as exc:
        exit_with_message("invalid_request", str(exc), "repo")


def _format_repo_status(repo: ModelRepo) -> str:
    result = verify_model_repo(repo)
    status = "ready" if result.ready else "missing"
    return f"{repo.name}: {status} ({repo.local_dir()})"


@app.command("download")
def download_cmd(
    repo: str | None = typer.Argument(None, help="Model repo: PDF-Extract-Kit-1.0 or MinerU2.5-Pro-2605-1.2B"),
    tier: str | None = typer.Option(None, "--tier", help="Model tier to prepare: basic or standard"),
    source: str | None = typer.Option(None, "--source", "-s", help="Model source: auto, huggingface, or modelscope"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Download a model repo or the repos required by a tier."""
    normalized_source = _validate_download_source(source)
    repos = _select_target_repos(repo, tier)

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
def show_cmd() -> None:
    """Show current MinerU model configuration."""
    config_file = get_config_file_path()
    lines = [
        f"Config: {config_file}",
        f"Config exists: {str(get_config_file_exists()).lower()}",
        f"MINERU_MODEL_SOURCE={os.getenv(MODEL_SOURCE_ENV_VAR, '') or '(unset)'}",
        f"model.base_dir: {config.model.base_dir}",
        f"model.base_dir.source: {get_config_source('model.base_dir')}",
        f"model.source: {config.model.source}",
        f"model.source.source: {get_config_source('model.source')}",
        "Repos:",
    ]
    for line in lines:
        print_info(line)

    for repo in MODEL_REPOS:
        print_info(f"  {_format_repo_status(repo)}")

    print_info("Model tiers:")
    for tier in DEPLOYMENT_TIERS:
        repos = model_repos_for_tier(tier)
        names = ", ".join(repo.name for repo in repos) or "(none)"
        print_info(f"  {tier}: {names}")


@app.command("verify")
def verify_cmd(
    repo: str | None = typer.Argument(None, help="Optional model repo name"),
    tier: str | None = typer.Option(None, "--tier", help="Optional model tier: basic or standard"),
) -> None:
    """Verify local model repos and required paths."""
    repos = MODEL_REPOS if repo is None and tier is None else _select_target_repos(repo, tier)

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


__all__ = ["app", "download_cmd", "show_cmd", "verify_cmd", "model_repo_names"]
