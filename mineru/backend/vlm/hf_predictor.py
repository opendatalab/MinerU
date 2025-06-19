from io import BytesIO
from typing import Iterable, List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from ...model.vlm_hf_model import Mineru2QwenForCausalLM
from ...model.vlm_hf_model.image_processing_mineru2 import process_images
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
from .utils import load_resource


class HuggingfacePredictor(BasePredictor):
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

        kwargs = {"device_map": device_map, **kwargs}

        if device != "cuda":
            kwargs["device_map"] = {"": device}

        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch_dtype

        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Mineru2QwenForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        setattr(self.model.config, "_name_or_path", model_path)
        self.model.eval()

        vision_tower = self.model.get_model().vision_tower
        if device_map != "auto":
            vision_tower.to(device=device_map, dtype=self.model.dtype)

        self.image_processor = vision_tower.image_processor
        self.eos_token_id = self.model.config.eos_token_id

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
        prompt = self.build_prompt(prompt)

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = self.no_repeat_ngram_size
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        do_sample = (temperature > 0.0) and (top_k > 1)

        generate_kwargs = {
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["top_k"] = top_k

        if isinstance(image, str):
            image = load_resource(image)

        image_obj = Image.open(BytesIO(image))
        image_tensor = process_images([image_obj], self.image_processor, self.model.config)
        image_tensor = image_tensor[0].unsqueeze(0)
        image_tensor = image_tensor.to(device=self.model.device, dtype=self.model.dtype)
        image_sizes = [[*image_obj.size]]

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device=self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=True,
                **generate_kwargs,
                **kwargs,
            )

        # Remove the last token if it is the eos_token_id
        if len(output_ids[0]) > 0 and output_ids[0, -1] == self.eos_token_id:
            output_ids = output_ids[:, :-1]

        output = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False,
        )[0].strip()

        return output

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

        assert len(prompts) == len(images), "Length of prompts and images must match."

        outputs = []
        for prompt, image in tqdm(zip(prompts, images), total=len(images), desc="Predict"):
            output = self.predict(
                image,
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
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
