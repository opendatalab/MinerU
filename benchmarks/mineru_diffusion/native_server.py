from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from benchmarks.mineru_diffusion.harness import extract_openai_image_and_prompt
from vllm.model_executor.models.mineru_diffusion import (
    MinerUDiffusionForConditionalGeneration,
)


STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
SYSTEM_PROMPT = "You are a helpful assistant."


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = 1.0
    block_size: int = Field(default=32, ge=1)
    dynamic_threshold: float = 0.95


def _round_up_to_block(length: int, block_size: int) -> int:
    return ((length + block_size - 1) // block_size) * block_size


class MinerUNativeEngine:

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        dtype: torch.dtype,
        max_length: int,
        remask_strategy: str,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        self.remask_strategy = remask_strategy
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.processor.tokenizer = self.tokenizer
        self.processor.chat_template = self.tokenizer.chat_template
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf_config,
                hf_text_config=hf_config.text_config,
                dtype=dtype,
            )
        )
        self.model = MinerUDiffusionForConditionalGeneration(
            vllm_config=vllm_config,
        ).eval()
        self._load_weights()
        self.model.to(device=self.device, dtype=dtype)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|MASK|>")

    def _load_weights(self) -> None:
        index_path = self.model_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        model_state = self.model.state_dict()
        loaded: set[str] = set()

        for shard_name in sorted(set(weight_map.values())):
            shard = load_file(str(self.model_path / shard_name), device="cpu")
            for name, tensor in shard.items():
                target = model_state.get(name)
                if target is None:
                    continue
                target.copy_(tensor.to(dtype=target.dtype))
                loaded.add(name)

        missing = sorted(set(model_state) - loaded)
        if missing:
            raise RuntimeError(f"missing native weights: {missing[:8]}")

    def _trim_response(self, text: str) -> str:
        for stop in STOP_STRINGS:
            text = text.split(stop, 1)[0]
        return text.strip()

    @torch.no_grad()
    def generate(
        self,
        *,
        image: str | None,
        prompt: str,
        max_tokens: int,
        block_size: int,
        temperature: float,
        dynamic_threshold: float,
    ) -> str:
        if image is None:
            raise ValueError("MinerU-Diffusion requests must include an image")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            images=[image],
            text=prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(torch.long).to(self.device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(torch.long).to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype).to(self.device)

        gen_length = _round_up_to_block(max_tokens, block_size)
        response_ids, _, _ = self.model.generate(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            mask_token_id=self.mask_token_id,
            denoising_steps=block_size,
            gen_length=gen_length,
            block_length=block_size,
            temperature=temperature,
            remasking_strategy=self.remask_strategy,
            dynamic_threshold=dynamic_threshold,
            tokenizer=self.tokenizer,
            stopping_criteria=list(STOP_STRINGS),
        )
        return self._trim_response(
            self.tokenizer.decode(response_ids[0], skip_special_tokens=False)
        )


def create_app(engine: MinerUNativeEngine) -> FastAPI:
    app = FastAPI(title="MinerU-Diffusion native compatibility server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            image, prompt = extract_openai_image_and_prompt(request.model_dump())
            output = engine.generate(
                image=image,
                prompt=prompt,
                max_tokens=request.max_tokens,
                block_size=request.block_size,
                temperature=request.temperature,
                dynamic_threshold=request.dynamic_threshold,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "id": f"mineru-native-{int(started * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "mineru-diffusion-native",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "mineru_metrics": {"latency_s": time.perf_counter() - started},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--remask-strategy", default="low_confidence_dynamic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = MinerUNativeEngine(
        Path(args.model_path).expanduser().resolve(),
        device=args.device,
        dtype=getattr(torch, args.dtype),
        max_length=args.max_length,
        remask_strategy=args.remask_strategy,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
