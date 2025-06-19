from sglang.srt.configs.model_config import multimodal_model_archs
from sglang.srt.models.registry import ModelRegistry

try:
    # sglang==0.4.5.post3
    from sglang.srt.managers.multimodal_processor import (
        PROCESSOR_MAPPING as PROCESSOR_MAPPING,
    )
except ImportError:
    # sglang==0.4.4.post1
    from sglang.srt.managers.image_processor import (
        IMAGE_PROCESSOR_MAPPING as PROCESSOR_MAPPING,
    )

from .. import vlm_hf_model as _
from .image_processor import Mineru2ImageProcessor
from .model import Mineru2QwenForCausalLM

ModelRegistry.models[Mineru2QwenForCausalLM.__name__] = Mineru2QwenForCausalLM
PROCESSOR_MAPPING[Mineru2QwenForCausalLM] = Mineru2ImageProcessor
multimodal_model_archs.append(Mineru2QwenForCausalLM.__name__)
