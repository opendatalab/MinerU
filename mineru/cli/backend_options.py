# Copyright (c) Opendatalab. All rights reserved.

BACKEND_PIPELINE = "pipeline"
BACKEND_VLM_ENGINE = "vlm-engine"
BACKEND_HYBRID_ENGINE = "hybrid-engine"
BACKEND_HYBRID_FLASH_ENGINE = "hybrid-flash-engine"
BACKEND_VLM_HTTP_CLIENT = "vlm-http-client"
BACKEND_HYBRID_HTTP_CLIENT = "hybrid-http-client"
BACKEND_HYBRID_FLASH_HTTP_CLIENT = "hybrid-flash-http-client"

DEFAULT_BACKEND = BACKEND_HYBRID_ENGINE

LOCAL_BACKEND_CHOICES = (
    BACKEND_PIPELINE,
    BACKEND_VLM_ENGINE,
    BACKEND_HYBRID_ENGINE,
    BACKEND_HYBRID_FLASH_ENGINE,
)
HTTP_CLIENT_BACKEND_CHOICES = (
    BACKEND_VLM_HTTP_CLIENT,
    BACKEND_HYBRID_HTTP_CLIENT,
    BACKEND_HYBRID_FLASH_HTTP_CLIENT,
)
PUBLIC_BACKEND_CHOICES = LOCAL_BACKEND_CHOICES + HTTP_CLIENT_BACKEND_CHOICES
BACKEND_SCHEMA_EXTRA = {"enum": list(PUBLIC_BACKEND_CHOICES)}
LEGACY_BACKEND_ALIASES = {
    "vlm-auto-engine": BACKEND_VLM_ENGINE,
    "hybrid-auto-engine": BACKEND_HYBRID_ENGINE,
    "hybrid-flash-auto-engine": BACKEND_HYBRID_FLASH_ENGINE,
}


def get_backend_choices(include_http_client: bool = True) -> list[str]:
    """按入口配置返回公开 backend 选项，避免各入口重复维护字符串列表。"""
    choices = list(LOCAL_BACKEND_CHOICES)
    if include_http_client:
        choices.extend(HTTP_CLIENT_BACKEND_CHOICES)
    return choices


def normalize_backend(backend: str) -> str:
    """将旧 backend 别名规范为当前公开名称，并校验最终名称是否合法。"""
    normalized_backend = LEGACY_BACKEND_ALIASES.get(backend, backend)
    if normalized_backend not in PUBLIC_BACKEND_CHOICES:
        allowed_values = ", ".join(PUBLIC_BACKEND_CHOICES)
        raise ValueError(f"Invalid backend. Allowed values: {allowed_values}")
    return normalized_backend


def validate_backend(backend: str) -> str:
    """校验公开入口允许的 backend 名称，并返回规范后的后端名称。"""
    return normalize_backend(backend)
