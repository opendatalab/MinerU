import os
import warnings
from typing import Optional

import torch
from ftfy import fix_text
from loguru import logger

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import logger as base_model_logger

from .unimer_swin import UnimerSwinConfig, UnimerSwinModel, UnimerSwinImageProcessor
from .unimer_mbart import UnimerMBartConfig, UnimerMBartForCausalLM
from ...utils import latex_rm_whitespace

AutoConfig.register(UnimerSwinConfig.model_type, UnimerSwinConfig)
AutoConfig.register(UnimerMBartConfig.model_type, UnimerMBartConfig)
AutoModel.register(UnimerSwinConfig, UnimerSwinModel)
AutoModelForCausalLM.register(UnimerMBartConfig, UnimerMBartForCausalLM)


# TODO: rewrite tokenizer
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text, **kwargs):
        return self.tokenizer(
            text,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            **kwargs,
        )

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated_text = [fix_text(text) for text in generated_text]
        return generated_text

    def detokenize(self, tokens):
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]):
                    del toks[b][i]
        return toks

class UnimernetModel(VisionEncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        # VisionEncoderDecoderModel's checking log has bug, disable for temp.
        base_model_logger.disabled = True
        try:
            super().__init__(config, encoder, decoder)
        finally:
            base_model_logger.disabled = False

        if not config or not hasattr(config, "_name_or_path"):
            raise RuntimeError("config._name_or_path is required by UnimernetModel.")

        model_path = config._name_or_path
        self.transform = UnimerSwinImageProcessor()
        self.tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(model_path))
        self._post_check()
    
    def _post_check(self):
        tokenizer = self.tokenizer

        if tokenizer.tokenizer.model_max_length != self.config.decoder.max_position_embeddings:
            warnings.warn(
                f"decoder.max_position_embeddings={self.config.decoder.max_position_embeddings}," +
                f" but tokenizer.model_max_length={tokenizer.tokenizer.model_max_length}, will set" +
                f" tokenizer.model_max_length to {self.config.decoder.max_position_embeddings}.")
            tokenizer.tokenizer.model_max_length = self.config.decoder.max_position_embeddings

        assert self.config.decoder.vocab_size == len(tokenizer)
        assert self.config.decoder_start_token_id == tokenizer.bos_token_id
        assert self.config.pad_token_id == tokenizer.pad_token_id

    @classmethod
    def from_checkpoint(cls, model_path: str, model_filename: str = "pytorch_model.pth", state_dict_strip_prefix="model.model."):
        config = VisionEncoderDecoderConfig.from_pretrained(model_path)
        config._name_or_path = model_path
        config.encoder = UnimerSwinConfig(**vars(config.encoder))
        config.decoder = UnimerMBartConfig(**vars(config.decoder))

        encoder = UnimerSwinModel(config.encoder)
        decoder = UnimerMBartForCausalLM(config.decoder)
        model = cls(config, encoder, decoder)

        # load model weights
        model_file_path = os.path.join(model_path, model_filename)
        checkpoint = torch.load(model_file_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        if not state_dict:
            raise RuntimeError("state_dict is empty.")
        if state_dict_strip_prefix:
            state_dict = {
                k[len(state_dict_strip_prefix):] if k.startswith(state_dict_strip_prefix) else k: v
                for k, v in state_dict.items()
            }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(unexpected_keys) > 0:
            warnings.warn("Unexpected key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            raise RuntimeError("Missing key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in missing_keys)))
        return model

    def forward_bak(self, samples):
        pixel_values, text = samples["image"], samples["text_input"]

        text_inputs = self.tokenizer.tokenize(text).to(pixel_values.device)
        decoder_input_ids, decoder_attention_mask = text_inputs["input_ids"], text_inputs["attention_mask"]

        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        labels = decoder_input_ids * 1
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        loss = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1],
            labels=labels[:, 1:],
        ).loss
        return {"loss": loss}

    def generate(self, samples, do_sample: bool = False, temperature: float = 0.2, top_p: float = 0.95, batch_size=64):
        pixel_values = samples["image"]
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        
        kwargs = {}
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        if self.tokenizer.tokenizer.model_max_length > 1152:
            if batch_size <= 32:
                self.tokenizer.tokenizer.model_max_length = 1152  # 6g
            else:
                self.tokenizer.tokenizer.model_max_length = 1344  # 8g

        outputs = super().generate(
            pixel_values=pixel_values,
            max_new_tokens=self.tokenizer.tokenizer.model_max_length, # required
            decoder_start_token_id=self.tokenizer.tokenizer.bos_token_id,
            do_sample=do_sample,
            **kwargs,
        )

        outputs = outputs[:, 1:].cpu().numpy()
        pred_tokens = self.tokenizer.detokenize(outputs)
        pred_str = self.tokenizer.token2str(outputs)
        fixed_str = [latex_rm_whitespace(s) for s in pred_str]
        return {"pred_ids": outputs, "pred_tokens": pred_tokens, "pred_str": pred_str, "fixed_str": fixed_str}

