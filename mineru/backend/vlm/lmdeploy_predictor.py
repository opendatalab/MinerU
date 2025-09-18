import torch
from functools import lru_cache

from typing import Iterable, List, Optional, Union
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig

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
from lmdeploy.vl import load_image
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

from mineru.model.vlm_lmdeploy_model.logit_processor import lmdeploy_custom_logits_processor
from mineru.model.vlm_lmdeploy_model.config import (
    LMDEPLOY_TP_SIZE, LMDEPLOY_SESSION_LEN, LMDEPLOY_MAX_PREFILL_TOKEN_NUM,
    LMDEPLOY_CACHE_MAX_ENTRY_COUNT, LMDEPLOY_SUPPORTED_DEVICE_TYPE,
    lmdeploy_block_size, lmdeploy_eager_mode, get_chat_template)


class LmdeployePredictor(BasePredictor):

    def __init__(
        self,
        model_path: str,
        device_map="auto",
        device="cuda",
        torch_dtype="auto",
        load_in_8bit=False,
        load_in_4bit=False,
        use_flash_attn=False,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        assert device in LMDEPLOY_SUPPORTED_DEVICE_TYPE, f"Lmdeploy backend only support {LMDEPLOY_SUPPORTED_DEVICE_TYPE} devices."

        self.pipeline = self.get_pipeline(model_path, device)
        self.gen_config = self.get_gen_config(top_k, top_p, temperature,
                                              repetition_penalty)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = 151645

    def get_pipeline(
        self,
        model_path: str,
        device: str,
    ):
        pipe = pipeline(
            model_path,
            chat_template_config=get_chat_template(),
            backend_config=PytorchEngineConfig(
                tp=LMDEPLOY_TP_SIZE,
                block_size=lmdeploy_block_size(device),
                eager_mode=lmdeploy_eager_mode(device),
                device_type=device,
                session_len=LMDEPLOY_SESSION_LEN,
                max_prefill_token_num=LMDEPLOY_MAX_PREFILL_TOKEN_NUM,
                cache_max_entry_count=LMDEPLOY_CACHE_MAX_ENTRY_COUNT,
            ))
        return pipe

    def get_gen_config(
        self,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
    ):
        gen_config = GenerationConfig(
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=16384,
            temperature=0,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            logits_processors=[lmdeploy_custom_logits_processor],
        )
        return gen_config

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
        **kwargs,
    ) -> str:
        raise NotImplementedError()

    def batch_predict(
        self,
        images: List[str] | List[bytes],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,  # not supported by hf
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(
            images), "Length of prompts and images must match."

        inputs = []
        outputs = []
        images = [load_image('data:image,' + image) for image in images]

        for image in images:
            inputs.append(("Document Parsing:", image))
        results = self.pipeline(inputs,
                                do_preprocess=True,
                                gen_config=self.gen_config)

        output_ids = [
            result.token_ids[:result.generate_token_len] for result in results
        ]

        for output_id in output_ids:
            output_id = [output_id]
            if len(output_id[0]) > 0 and output_id[0][-1] == self.eos_token_id:
                output_id = output_id[:, :-1]
            output = self.tokenizer.batch_decode(
                output_id,
                skip_special_tokens=False,
            )[0].strip()
            outputs.append(output)

        return outputs

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
