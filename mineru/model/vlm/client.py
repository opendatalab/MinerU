"""解析与预加载共享的 VLM 客户端构造入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import VlmConfig, config

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient


def get_vlm_predictor(vlm_config: VlmConfig | None = None) -> tuple[MinerUClient, str]:
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

    engine = get_vlm_engine("auto", is_async=False)
    return ModelSingleton().get_model(backend=engine, model_path=None, server_url=None), engine


__all__ = ["get_vlm_predictor"]
