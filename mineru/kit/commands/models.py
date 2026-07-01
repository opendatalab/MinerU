from __future__ import annotations

import os
from pathlib import Path

import typer
from loguru import logger

from ...utils.enum_class import ModelPath
from ...utils.models_download_utils import (
    MINERU_CONFIG_VERSION,
    auto_download_and_get_model_root_path,
    is_config_version_outdated,
    merge_config_dict,
    resolve_model_source,
)
from ..common import PIPELINE_MODEL_PATHS, VLM_MODEL_MARKERS, read_json_file, resolve_models_config_path, write_json_file
from ..errors import exit_with_message
from ..output import print_info, print_success

app = typer.Typer(help="Download, inspect, and verify local MinerU models.", no_args_is_help=True)

MODEL_SOURCE_ENV_VAR = "MINERU_MODEL_SOURCE"
REMOTE_MODEL_SOURCES = ("auto", "huggingface", "modelscope")


def _load_or_template_config() -> dict:
    config_path = resolve_models_config_path()
    payload = read_json_file(config_path)
    template_path = Path(__file__).resolve().parents[3] / "mineru.template.json"
    template_payload = read_json_file(template_path) or {"models-dir": {}, "config_version": MINERU_CONFIG_VERSION}
    if payload is not None:
        if not is_config_version_outdated(payload.get("config_version", "0.0.0")):
            return payload
        return merge_config_dict(template_payload, payload, skip_keys={"config_version"})
    return template_payload


def _update_models_dir(bundle: str, model_dir: str) -> Path:
    config_path = resolve_models_config_path()
    payload = _load_or_template_config()
    models_dir = payload.get("models-dir")
    if not isinstance(models_dir, dict):
        models_dir = {}
        payload["models-dir"] = models_dir
    models_dir[bundle] = model_dir
    model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
    if model_source in {"huggingface", "modelscope"}:
        payload["model-source"] = model_source
    write_json_file(config_path, payload)
    return config_path


def _get_effective_download_model_source(requested_model_source: str) -> str:
    current_model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
    if current_model_source == "local":
        logger.warning(
            f"{MODEL_SOURCE_ENV_VAR}=local means using pre-downloaded local models. "
            f"`mineru-kit models download` will temporarily use '{requested_model_source}' to perform a real download."
        )
        return resolve_model_source(requested_model_source, allow_auto=True)
    if current_model_source is None:
        return resolve_model_source(requested_model_source, allow_auto=True)
    return resolve_model_source(current_model_source)


def _download_pipeline_models() -> str:
    download_finish_path = ""
    for model_path in PIPELINE_MODEL_PATHS:
        logger.info(f"Downloading model: {model_path}")
        download_finish_path = auto_download_and_get_model_root_path(model_path, repo_mode="pipeline")
    return download_finish_path


def _download_vlm_models() -> str:
    return auto_download_and_get_model_root_path("/", repo_mode="vlm")


def _with_temporary_model_source(model_source: str):
    class _ModelSourceContext:
        def __enter__(self_inner) -> None:
            self_inner.original = os.getenv(MODEL_SOURCE_ENV_VAR)
            os.environ[MODEL_SOURCE_ENV_VAR] = model_source

        def __exit__(self_inner, exc_type, exc, tb) -> None:
            if self_inner.original is None:
                os.environ.pop(MODEL_SOURCE_ENV_VAR, None)
            else:
                os.environ[MODEL_SOURCE_ENV_VAR] = self_inner.original

    return _ModelSourceContext()


@app.command("download")
def download_cmd(
    bundle: str = typer.Argument(..., help="Model bundle: pipeline, vlm, all"),
    source: str = typer.Option("auto", "--source", "-s", help="Model source: auto, huggingface, or modelscope"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Download a model bundle and update mineru.json."""
    if bundle not in {"pipeline", "vlm", "all"}:
        exit_with_message("invalid_request", f"Unsupported bundle '{bundle}'. Expected pipeline, vlm, or all.", "bundle")
    if source not in REMOTE_MODEL_SOURCES:
        exit_with_message("invalid_request", f"Unsupported source '{source}'.", "source")
    effective_source = _get_effective_download_model_source(source)
    try:
        with _with_temporary_model_source(effective_source):
            if bundle in {"pipeline", "all"}:
                pipeline_dir = _download_pipeline_models()
                config_path = _update_models_dir("pipeline", pipeline_dir)
                if verbose:
                    print_info(f"Configured pipeline models in {config_path}")
            if bundle in {"vlm", "all"}:
                vlm_dir = _download_vlm_models()
                config_path = _update_models_dir("vlm", vlm_dir)
                if verbose:
                    print_info(f"Configured vlm models in {config_path}")
    except Exception as exc:
        exit_with_message("api_error", f"Failed to download models: {exc}")
    print_success(f"Downloaded {bundle} models from {effective_source}.")


@app.command("show")
def show_cmd() -> None:
    """Show current MinerU model configuration."""
    config_path = resolve_models_config_path()
    payload = read_json_file(config_path)
    if payload is None:
        print_info(f"Config: {config_path} (missing)")
        return
    models_dir = payload.get("models-dir", {})
    pipeline = models_dir.get("pipeline", "") if isinstance(models_dir, dict) else ""
    vlm = models_dir.get("vlm", "") if isinstance(models_dir, dict) else ""
    lines = [
        f"Config: {config_path}",
        f"MINERU_MODEL_SOURCE={os.getenv(MODEL_SOURCE_ENV_VAR, '') or '(unset)'}",
        f"model-source: {payload.get('model-source', '') or '(unset)'}",
        f"pipeline: {pipeline or '(unset)'}",
        f"pipeline.exists: {Path(pipeline).exists() if pipeline else False}",
        f"vlm: {vlm or '(unset)'}",
        f"vlm.exists: {Path(vlm).exists() if vlm else False}",
    ]
    for line in lines:
        print_info(line)


def _verify_pipeline(root: Path) -> list[str]:
    missing: list[str] = []
    for rel in PIPELINE_MODEL_PATHS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def _verify_vlm(root: Path) -> list[str]:
    missing: list[str] = []
    for rel in VLM_MODEL_MARKERS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


@app.command("verify")
def verify_cmd(
    bundle: str | None = typer.Argument(None, help="Optional bundle: pipeline, vlm, all"),
) -> None:
    """Verify configured model directories and key paths."""
    if bundle is not None and bundle not in {"pipeline", "vlm", "all"}:
        exit_with_message("invalid_request", f"Unsupported bundle '{bundle}'. Expected pipeline, vlm, or all.", "bundle")
    config_path = resolve_models_config_path()
    payload = read_json_file(config_path)
    if payload is None:
        exit_with_message("file_not_found", f"Config file not found: {config_path}")
    models_dir = payload.get("models-dir")
    if not isinstance(models_dir, dict):
        exit_with_message("invalid_request", f"'models-dir' not found in {config_path}")

    targets = ("pipeline", "vlm") if bundle is None or bundle == "all" else (bundle,)
    failures = 0
    for target in targets:
        raw = models_dir.get(target)
        if not raw:
            print_info(f"{target}: missing config")
            failures += 1
            continue
        root = Path(raw)
        if not root.exists():
            print_info(f"{target}: missing directory {root}")
            failures += 1
            continue
        missing = _verify_pipeline(root) if target == "pipeline" else _verify_vlm(root)
        if missing:
            print_info(f"{target}: missing key paths: {', '.join(missing)}")
            failures += 1
        else:
            print_success(f"{target}: ok")
    if failures:
        raise typer.Exit(1)
