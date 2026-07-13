from __future__ import annotations

import inspect
import time
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from mineru_vl_utils.mlx_compat import load_mlx_model

from ...utils.model_registry import MINERU_2_5_PRO_2605_1_2B


async def _resolve_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _default_model_id(configured_model: str | None) -> str:
    if configured_model:
        return configured_model
    return str(MINERU_2_5_PRO_2605_1_2B.ensure())


def _load_mlx_model_for_server(path_or_hf_repo: str, adapter_path: str | None = None, **kwargs: Any) -> Any:
    if adapter_path is not None:
        kwargs["adapter_path"] = adapter_path
    return load_mlx_model(path_or_hf_repo, **kwargs)


def _text_from_content_part(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    part_type = part.get("type")
    if part_type in {"text", "input_text", "output_text"}:
        text = part.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _sanitize_chat_template_prompt(prompt: Any) -> Any:
    if not isinstance(prompt, list):
        return prompt

    sanitized = []
    for message in prompt:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content")
        else:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "".join(_text_from_content_part(part) for part in content)
        sanitized.append({"role": role, "content": content})
    return sanitized


def _wrap_apply_chat_template(apply_chat_template: Any) -> Any:
    def _apply_chat_template(processor: Any, config: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        return apply_chat_template(processor, config, _sanitize_chat_template_prompt(prompt), *args, **kwargs)

    return _apply_chat_template


def create_app(default_model: str | None = None) -> FastAPI:
    try:
        from mlx_vlm import server as mlx_server
    except ImportError as exc:
        raise click.ClickException(
            "mlx-vlm is not installed. Install optional MLX dependencies in this environment, "
            "for example: pip install 'mineru[mlx]'."
        ) from exc

    mlx_server.load = _load_mlx_model_for_server
    mlx_server.apply_chat_template = _wrap_apply_chat_template(mlx_server.apply_chat_template)
    default_model_id = _default_model_id(default_model)

    app = FastAPI(
        title="MinerU MLX VLM Server",
        description="OpenAI-compatible adapter for mlx-vlm.",
        version=getattr(mlx_server, "__version__", "unknown"),
    )

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Any:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        if not payload.get("model"):
            payload["model"] = default_model_id
        chat_request = mlx_server.ChatRequest.model_validate(payload)
        return await _resolve_result(mlx_server.chat_completions_endpoint(chat_request))

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Any:
        return await v1_chat_completions(request)

    @app.get("/v1/models")
    async def v1_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": default_model_id,
                    "object": "model",
                    "created": int(time.time()),
                }
            ],
        }

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return await v1_models()

    @app.get("/health")
    async def health() -> Any:
        health_check = getattr(mlx_server, "health_check", None)
        if health_check is None:
            return {"status": "healthy", "loaded_model": None, "loaded_adapter": None}
        return await _resolve_result(health_check())

    @app.post("/unload")
    async def unload() -> Any:
        unload_model_endpoint = getattr(mlx_server, "unload_model_endpoint", None)
        if unload_model_endpoint is None:
            return {"status": "no_model_loaded", "message": "No model unload endpoint is available."}
        return await _resolve_result(unload_model_endpoint())

    return app


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--host", default="127.0.0.1", show_default=True, help="Host for the HTTP server.")
@click.option("--port", default=8080, show_default=True, type=int, help="Port for the HTTP server.")
@click.option("--model", default=None, help="Default MLX VLM model id or local model path.")
def _command(host: str, port: int, model: str | None) -> None:
    app = create_app(default_model=model)
    uvicorn.run(app, host=host, port=port, workers=1)


def main(*, args: list[str], prog_name: str, standalone_mode: bool) -> None:
    _command.main(args=args, prog_name=prog_name, standalone_mode=standalone_mode)


__all__ = ["create_app", "main"]
