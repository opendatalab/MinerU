import os

from loguru import logger
from packaging import version

from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram


def enable_custom_logits_processors() -> bool:
    import torch
    from vllm import __version__ as vllm_version

    if not torch.cuda.is_available():
        logger.info("CUDA not available, disabling custom_logits_processors")
        return False

    major, minor = torch.cuda.get_device_capability()
    # 正确计算Compute Capability
    compute_capability = f"{major}.{minor}"

    # 安全地处理环境变量
    vllm_use_v1_str = os.getenv('VLLM_USE_V1', "1")
    if vllm_use_v1_str.isdigit():
        vllm_use_v1 = int(vllm_use_v1_str)
    else:
        vllm_use_v1 = 1

    if vllm_use_v1 == 0:
        logger.info("VLLM_USE_V1 is set to 0, disabling custom_logits_processors")
        return False
    elif version.parse(vllm_version) < version.parse("0.10.1"):
        logger.info(f"vllm version: {vllm_version} < 0.10.1, disable custom_logits_processors")
        return False
    elif version.parse(compute_capability) < version.parse("8.0"):
        if version.parse(vllm_version) >= version.parse("0.10.2"):
            logger.info(f"compute_capability: {compute_capability} < 8.0, but vllm version: {vllm_version} >= 0.10.2, enable custom_logits_processors")
            return True
        else:
            logger.info(f"compute_capability: {compute_capability} < 8.0 and vllm version: {vllm_version} < 0.10.2, disable custom_logits_processors")
            return False
    else:
        logger.info(f"compute_capability: {compute_capability} >= 8.0 and vllm version: {vllm_version} >= 0.10.1, enable custom_logits_processors")
        return True


def set_default_gpu_memory_utilization() -> float:
    from vllm import __version__ as vllm_version
    if version.parse(vllm_version) >= version.parse("0.11.0"):
        return 0.7
    else:
        return 0.5


def set_default_batch_size() -> int:
    try:
        device = get_device()
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('MINERU_VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_size = 8
            elif gpu_memory >= 8:
                batch_size = 4
            else:
                batch_size = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_size: {batch_size}')
        else:
            # Default batch_ratio when VRAM can't be determined
            batch_size = 1
            logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_size}')
    except Exception as e:
        logger.warning(f'Error determining VRAM: {e}, using default batch_ratio: 1')
        batch_size = 1
    return batch_size