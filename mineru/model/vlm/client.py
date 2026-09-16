"""解析与预加载共享的 VLM 客户端构造入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import VlmConfig, config

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient

    from .async_runtime import AsyncVlmPredictor


def get_vlm_predictor(vlm_config: VlmConfig | None = None) -> tuple[MinerUClient | AsyncVlmPredictor, str]:
    """按连接配置选择远程客户端或本地引擎，预加载与真实解析复用相同缓存。"""
    settings = (vlm_config if vlm_config is not None else config.model.vlm).model_copy(deep=True)
    settings.validate_environment()
    from .runtime import ModelSingleton

    if settings.server_url:
        predictor = ModelSingleton().get_model(
            backend="http-client",
            model_path=None,
            server_url=settings.server_url,
            model_name=settings.model or None,
            server_headers={"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {},
            http_timeout=settings.http_timeout,
            max_concurrency=settings.max_concurrency,
        )
        return predictor, "http-client"

    from .selector import get_vlm_engine

    engine = get_vlm_engine(settings.engine, is_async=True)
    return ModelSingleton().get_model(
        backend=engine,
        model_path=None,
        server_url=None,
        max_concurrency=settings.max_concurrency,
    ), engine


def uses_native_async_vlm(vlm_config: VlmConfig | None = None) -> bool:
    """不加载模型即可判断当前连接是否支持本期原生异步编排。"""
    settings = vlm_config if vlm_config is not None else config.model.vlm
    if settings.server_url:
        return True
    from .selector import resolve_vlm_engine

    return resolve_vlm_engine(settings.engine) == "vllm"


__all__ = ["get_vlm_predictor", "uses_native_async_vlm"]
