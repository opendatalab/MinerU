# Copyright (c) Opendatalab. All rights reserved.
"""LLM 辅助后处理共享的 OpenAI-compatible 请求客户端。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from ...config import LLMAidedConfig

ValidatedResult = TypeVar("ValidatedResult")

_MAX_LLM_RETRIES = 3


class LLMAidedClient:
    """复用单个异步 OpenAI-compatible 客户端并提供带校验的 JSON 请求。"""

    def __init__(self, config: LLMAidedConfig, client: Any | None = None) -> None:
        """保存强类型配置，并按需创建带共享并发限制的异步客户端。"""
        self._config = config
        self._client = client or AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self._owns_client = client is None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def request_validated_json(
        self,
        *,
        operation: str,
        prompt: str,
        validator: Callable[[Any], ValidatedResult | None],
        temperature: float,
    ) -> ValidatedResult | None:
        """异步请求 JSON，并在每次重试中执行调用方提供的结构校验。"""
        request_params: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self._config.enable_thinking is not None:
            request_params["extra_body"] = {"enable_thinking": self._config.enable_thinking}

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                async with self._semaphore:
                    completion = await self._client.chat.completions.create(**request_params)
                content = self._extract_message_content(completion)
                parsed = json_repair.loads(content)
                validated = validator(parsed)
                if validated is not None:
                    return validated
                logger.warning(
                    "LLM {} response validation failed on attempt {}/{}",
                    operation,
                    attempt,
                    _MAX_LLM_RETRIES,
                )
            except Exception as exc:
                logger.warning(
                    "LLM {} request failed on attempt {}/{}: {}",
                    operation,
                    attempt,
                    _MAX_LLM_RETRIES,
                    type(exc).__name__,
                )

        logger.error("LLM {} failed after {} attempts", operation, _MAX_LLM_RETRIES)
        return None

    async def close(self) -> None:
        """仅关闭由当前包装器自行创建的底层异步客户端。"""
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _extract_message_content(completion: Any) -> str:
        """从非流式 Chat Completions 响应中提取最终文本并移除思考前缀。"""
        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError("LLM response does not contain choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response content is empty")
        content = content.strip()
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1].strip()
        if not content:
            raise ValueError("LLM response content is empty after removing thinking output")
        return content


__all__ = ["LLMAidedClient"]
