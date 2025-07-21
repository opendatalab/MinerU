import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterable, Iterable, List, Optional, Union

DEFAULT_SYSTEM_PROMPT = (
    "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
)
DEFAULT_USER_PROMPT = "Document Parsing:"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 100
DEFAULT_MAX_NEW_TOKENS = 16384


class BasePredictor(ABC):
    system_prompt = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens

    @abstractmethod
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
    ) -> str: ...

    @abstractmethod
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
    ) -> List[str]: ...

    @abstractmethod
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
    ) -> Iterable[str]: ...

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
        return await asyncio.to_thread(
            self.predict,
            image,
            prompt,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )

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
        return await asyncio.to_thread(
            self.batch_predict,
            images,
            prompts,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )

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
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def synced_predict():
            for chunk in self.stream_predict(
                image=image,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.create_task(
            asyncio.to_thread(synced_predict),
        )

        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            assert isinstance(chunk, str)
            yield chunk

    def build_prompt(self, prompt: str) -> str:
        if prompt.startswith("<|im_start|>"):
            return prompt
        if not prompt:
            prompt = DEFAULT_USER_PROMPT

        return f"<|im_start|>system\n{self.system_prompt}<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
        # Modify here. We add <|box_start|> at the end of the prompt to force the model to generate bounding box.
        # if "Document OCR" in prompt:
        #     return f"<|im_start|>system\n{self.system_prompt}<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n<|box_start|>"
        # else:
        #     return f"<|im_start|>system\n{self.system_prompt}<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"

    def close(self):
        pass
