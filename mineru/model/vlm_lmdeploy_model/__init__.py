from . import model
from . import patch
from .image_processor import MinerUModel

from typing import Optional

from lmdeploy.pytorch.models.module_map import DEVICE_SPECIAL_MODULE_MAP
from lmdeploy.model import MODELS, BaseChatTemplate

DEVICE_SPECIAL_MODULE_MAP['ascend'].update({
    'Mineru2QwenForCausalLM':
    'mineru.model.vlm_lmdeploy_model.model.Mineru2QwenForCausalLM',
})