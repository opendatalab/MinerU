import asyncio
import json
import re
from base64 import b64encode
from typing import AsyncIterable, Iterable, List, Optional, Set, Tuple, Union

import httpx

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
from .utils import aio_load_resource, load_resource


class SglangClientPredictor(BasePredictor):
    def __init__(
        self,
        server_url: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        http_timeout: int = 600,
    ) -> None:
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        self.http_timeout = http_timeout

        base_url = self.get_base_url(server_url)
        self.check_server_health(base_url)
        self.model_path = self.get_model_path(base_url)
        self.server_url = f"{base_url}/generate"

    @staticmethod
    def get_base_url(server_url: str) -> str:
        matched = re.match(r"^(https?://[^/]+)", server_url)
        if not matched:
            raise ValueError(f"Invalid server URL: {server_url}")
        return matched.group(1)

    def check_server_health(self, base_url: str):
        try:
            response = httpx.get(f"{base_url}/health_generate", timeout=self.http_timeout)
        except httpx.ConnectError:
            raise RuntimeError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise RuntimeError(
                f"Server {base_url} is not healthy. Status code: {response.status_code}, response body: {response.text}"
            )

    def get_model_path(self, base_url: str) -> str:
        try:
            response = httpx.get(f"{base_url}/get_model_info", timeout=self.http_timeout)
        except httpx.ConnectError:
            raise RuntimeError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get model info from {base_url}. Status code: {response.status_code}, response body: {response.text}"
            )
        return response.json()["model_path"]

    def build_sampling_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        presence_penalty: Optional[float],
        no_repeat_ngram_size: Optional[int],
        max_new_tokens: Optional[int],
    ) -> dict:
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        if presence_penalty is None:
            presence_penalty = self.presence_penalty
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = self.no_repeat_ngram_size
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # see SamplingParams for more details
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "custom_params": {
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        }

    def build_request_body(
        self,
        image: bytes,
        prompt: str,
        sampling_params: dict,
    ) -> dict:
        image_base64 = b64encode(image).decode("utf-8")
        return {
            "text": prompt,
            "image_data": image_base64,
            "sampling_params": sampling_params,
            "modalities": ["image"],
        }

    def predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        prompt = self.build_prompt(prompt)

        sampling_params = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(image, str):
            image = load_resource(image)

        request_body = self.build_request_body(image, prompt, sampling_params)
        response = httpx.post(self.server_url, json=request_body, timeout=self.http_timeout)
        response_body = response.json()
        return response_body["text"]

    def batch_predict(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        max_concurrency: int = 100,
    ) -> List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_batch_predict(
            images=images,
            prompts=prompts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            max_concurrency=max_concurrency,
        )

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    def stream_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        prompt = self.build_prompt(prompt)

        sampling_params = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(image, str):
            image = load_resource(image)

        request_body = self.build_request_body(image, prompt, sampling_params)
        request_body["stream"] = True

        with httpx.stream(
            "POST",
            self.server_url,
            json=request_body,
            timeout=self.http_timeout,
        ) as response:
            pos = 0
            for chunk in response.iter_lines():
                if not (chunk or "").startswith("data:"):
                    continue
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                chunk_text = data["text"][pos:]
                # meta_info = data["meta_info"]
                pos += len(chunk_text)
                yield chunk_text

    async def aio_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        prompt = self.build_prompt(prompt)

        sampling_params = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(image, str):
            image = await aio_load_resource(image)

        request_body = self.build_request_body(image, prompt, sampling_params)

        if async_client is None:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.post(self.server_url, json=request_body)
                response_body = response.json()
        else:
            response = await async_client.post(self.server_url, json=request_body)
            response_body = response.json()

        return response_body["text"]

    async def aio_batch_predict(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        max_concurrency: int = 100,
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."

        semaphore = asyncio.Semaphore(max_concurrency)
        outputs = [""] * len(images)

        async def predict_with_semaphore(
            idx: int,
            image: str | bytes,
            prompt: str,
            async_client: httpx.AsyncClient,
        ):
            async with semaphore:
                output = await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_new_tokens=max_new_tokens,
                    async_client=async_client,
                )
                outputs[idx] = output

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            tasks = []
            for idx, (prompt, image) in enumerate(zip(prompts, images)):
                tasks.append(predict_with_semaphore(idx, image, prompt, client))
            await asyncio.gather(*tasks)

        return outputs

    async def aio_batch_predict_as_iter(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        max_concurrency: int = 100,
    ) -> AsyncIterable[Tuple[int, str]]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."

        semaphore = asyncio.Semaphore(max_concurrency)

        async def predict_with_semaphore(
            idx: int,
            image: str | bytes,
            prompt: str,
            async_client: httpx.AsyncClient,
        ):
            async with semaphore:
                output = await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_new_tokens=max_new_tokens,
                    async_client=async_client,
                )
                return (idx, output)

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            pending: Set[asyncio.Task[Tuple[int, str]]] = set()

            for idx, (prompt, image) in enumerate(zip(prompts, images)):
                pending.add(
                    asyncio.create_task(
                        predict_with_semaphore(idx, image, prompt, client),
                    )
                )

            while len(pending) > 0:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    yield task.result()

    async def aio_stream_predict(
        self,
        image: str | bytes,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> AsyncIterable[str]:
        prompt = self.build_prompt(prompt)

        sampling_params = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(image, str):
            image = await aio_load_resource(image)

        request_body = self.build_request_body(image, prompt, sampling_params)
        request_body["stream"] = True

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            async with client.stream(
                "POST",
                self.server_url,
                json=request_body,
            ) as response:
                pos = 0
                async for chunk in response.aiter_lines():
                    if not (chunk or "").startswith("data:"):
                        continue
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    chunk_text = data["text"][pos:]
                    # meta_info = data["meta_info"]
                    pos += len(chunk_text)
                    yield chunk_text
