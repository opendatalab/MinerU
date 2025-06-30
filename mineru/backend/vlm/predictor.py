# Copyright (c) Opendatalab. All rights reserved.

import time

from loguru import logger

from .base_predictor import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    BasePredictor,
)
from .sglang_client_predictor import SglangClientPredictor

hf_loaded = False
try:
    from .hf_predictor import HuggingfacePredictor

    hf_loaded = True
except ImportError as e:
    logger.warning("hf is not installed. If you are not using transformers, you can ignore this warning.")

engine_loaded = False
try:
    from sglang.srt.server_args import ServerArgs

    from .sglang_engine_predictor import SglangEnginePredictor

    engine_loaded = True
except Exception as e:
    logger.warning("sglang is not installed. If you are not using sglang, you can ignore this warning.")


def get_predictor(
    backend: str = "sglang-client",
    model_path: str | None = None,
    server_url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    http_timeout: int = 600,
    **kwargs,
) -> BasePredictor:
    start_time = time.time()

    if backend == "transformers":
        if not model_path:
            raise ValueError("model_path must be provided for transformers backend.")
        if not hf_loaded:
            raise ImportError(
                "transformers is not installed, so huggingface backend cannot be used. "
                "If you need to use huggingface backend, please install transformers first."
            )
        predictor = HuggingfacePredictor(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    elif backend == "sglang-engine":
        if not model_path:
            raise ValueError("model_path must be provided for sglang-engine backend.")
        if not engine_loaded:
            raise ImportError(
                "sglang is not installed, so sglang-engine backend cannot be used. "
                "If you need to use sglang-engine backend for inference, "
                "please install sglang[all]==0.4.8 or a newer version."
            )
        predictor = SglangEnginePredictor(
            server_args=ServerArgs(model_path, **kwargs),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
    elif backend == "sglang-client":
        if not server_url:
            raise ValueError("server_url must be provided for sglang-client backend.")
        predictor = SglangClientPredictor(
            server_url=server_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            http_timeout=http_timeout,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supports: transformers, sglang-engine, sglang-client.")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"get_predictor cost: {elapsed}s")
    return predictor
