from functools import lru_cache
from typing import Optional

from lmdeploy.model import MODELS, BaseChatTemplate, ChatTemplateConfig

LMDEPLOY_TP_SIZE = 1
LMDEPLOY_SESSION_LEN = 20240
LMDEPLOY_MAX_PREFILL_TOKEN_NUM = 15200
LMDEPLOY_CACHE_MAX_ENTRY_COUNT = 0.8

# devices
LMDEPLOY_SUPPORTED_DEVICE_TYPE = ["ascend"]

# ascend
LMDEPLOY_ASCEND_BLOCK_SIZE = 128
LMDEPLOY_ASCEND_EAGER_MODE = True


def lmdeploy_block_size(device: str) -> int:
    if device == "ascend":
        return LMDEPLOY_ASCEND_BLOCK_SIZE
    raise RuntimeError(f"unsupported device type: {device}! ")


def lmdeploy_eager_mode(device: str) -> int:
    if device == "ascend":
        return LMDEPLOY_ASCEND_EAGER_MODE
    raise RuntimeError(f"unsupported device type: {device}! ")


@lru_cache
def set_chat_template():

    @MODELS.register_module(name='mineru2')
    class Mineru(BaseChatTemplate):
        """Chat template of mineru model."""

        def __init__(
                self,
                meta_instruction="""A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
                system='<|im_start|>system\n',
                user='<|im_start|>user\n',
                assistant='<|im_start|>assistant\n',
                eosys='<|im_end|>',
                eoh='<|im_end|>',
                eoa='<|im_end|>\n',
                separator='\n',
                **kwargs):
            super().__init__(meta_instruction=meta_instruction,
                             system=system,
                             user=user,
                             assistant=assistant,
                             eosys=eosys,
                             eoh=eoh,
                             eoa=eoa,
                             separator=separator,
                             **kwargs)

        @classmethod
        def match(cls, model_path: str) -> Optional[str]:
            """Return the model_name that was registered to MODELS.

            Args:
                model_path (str): the model path used for matching.
            """
            path = model_path.lower()
            if 'mineru2' in path:
                return 'mineru2'
            return 'mineru2'


def get_chat_template():
    set_chat_template()
    return ChatTemplateConfig("mineru2")
