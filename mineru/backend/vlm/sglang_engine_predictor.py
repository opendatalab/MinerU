from base64 import b64encode
from typing import AsyncIterable, Iterable, List, Optional, Union

from sglang.srt.server_args import ServerArgs

from ...model.vlm_sglang_model.engine import BatchEngine
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


class SglangEnginePredictor(BasePredictor):
    def __init__(
        self,
        server_args: ServerArgs,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
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
        self.engine = BatchEngine(server_args=server_args)

    def load_image_string(self, image: str | bytes) -> str:
        if not isinstance(image, (str, bytes)):
            raise ValueError("Image must be a string or bytes.")
        if isinstance(image, bytes):
            return b64encode(image).decode("utf-8")
        if image.startswith("file://"):
            return image[len("file://") :]
        return image

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
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )[0]

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
    ) -> List[str]:

        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        prompts = [self.build_prompt(prompt) for prompt in prompts]

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
        sampling_params = {
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

        image_strings = [self.load_image_string(img) for img in images]

        output = self.engine.generate(
            prompt=prompts,
            image_data=image_strings,
            sampling_params=sampling_params,
        )
        return [item["text"] for item in output]

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
        raise NotImplementedError("Streaming is not supported yet.")

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
    ) -> str:
        output = await self.aio_batch_predict(
            [image],  # type: ignore
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        return output[0]

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
    ) -> List[str]:

        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        prompts = [self.build_prompt(prompt) for prompt in prompts]

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
        sampling_params = {
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

        image_strings = [self.load_image_string(img) for img in images]

        output = await self.engine.async_generate(
            prompt=prompts,
            image_data=image_strings,
            sampling_params=sampling_params,
        )
        ret = []
        for item in output:  # type: ignore
            ret.append(item["text"])
        return ret

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
        raise NotImplementedError("Streaming is not supported yet.")

    def close(self):
        self.engine.shutdown()
