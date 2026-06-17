# Copyright (c) Opendatalab. All rights reserved.
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from mineru.utils.engine_utils import get_vlm_engine

SERVICE_CONFIG_DEFAULTS: dict[str, Any] = {
    "enable_vlm_preload": False,
}


def split_service_and_model_config(
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_config = dict(config or {})
    service_config: dict[str, Any] = {}

    for key, default in SERVICE_CONFIG_DEFAULTS.items():
        service_config[key] = bool(raw_config.pop(key, default))

    return service_config, raw_config


def build_local_api_cli_args(
    extra_cli_args: Sequence[str],
    *,
    enable_vlm_preload: bool,
) -> tuple[str, ...]:
    args = tuple(extra_cli_args)
    if not enable_vlm_preload:
        return args

    if "--enable-vlm-preload" in args or any(
        arg.startswith("--enable-vlm-preload=") for arg in args
    ):
        return args

    return args + ("--enable-vlm-preload", "true")


def resolve_gradio_local_api_cli_args(
    extra_cli_args: Sequence[str],
    *,
    api_url: str | None,
    enable_vlm_preload: bool,
) -> tuple[str, ...]:
    if enable_vlm_preload and api_url:
        logger.warning(
            "Ignoring --enable-vlm-preload because --api-url points to an existing MinerU FastAPI service."
        )
        return tuple(extra_cli_args)

    return build_local_api_cli_args(
        extra_cli_args,
        enable_vlm_preload=enable_vlm_preload,
    )


def preload_vlm_model(*, model_kwargs: Mapping[str, Any] | None = None) -> str:
    vlm_engine = get_vlm_engine("auto", is_async=True)
    logger.info(f"Start init {vlm_engine}...")

    from mineru.backend.vlm.vlm_analyze import ModelSingleton

    model_singleton = ModelSingleton()
    model_singleton.get_model(
        vlm_engine,
        None,
        None,
        **dict(model_kwargs or {}),
    )
    logger.info(f"{vlm_engine} init successfully.")
    return vlm_engine


def maybe_preload_vlm_model(
    enable_vlm_preload: bool,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
) -> str | None:
    if not enable_vlm_preload:
        return None
    return preload_vlm_model(model_kwargs=model_kwargs)
