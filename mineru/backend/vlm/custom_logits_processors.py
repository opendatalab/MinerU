import os

from loguru import logger
from packaging import version


def enable_custom_logits_processors():
    import torch
    from vllm import __version__ as vllm_version

    if not torch.cuda.is_available():
        logger.info("CUDA not available, disabling custom_logits_processors")
        return False

    major, minor = torch.cuda.get_device_capability()
    # 正确计算Compute Capability
    compute_capability = f"{major}.{minor}"
    compute_capability = float(compute_capability)

    # 安全地处理环境变量
    try:
        vllm_use_v1 = int(os.getenv('VLLM_USE_V1', "1"))
    except (ValueError, TypeError):
        vllm_use_v1 = 1
        logger.warning("Invalid VLLM_USE_V1 value")

    if vllm_use_v1 == 0:
        logger.info("VLLM_USE_V1 is set to 0, disabling custom_logits_processors")
        return False
    elif version.parse(vllm_version) < version.parse("0.10.1"):
        logger.info(f"vllm version: {vllm_version} < 0.10.1, disable custom_logits_processors")
        return False
    elif compute_capability < 8.0:
        if version.parse(vllm_version) >= version.parse("0.10.2"):
            logger.info(f"compute_capability: {compute_capability} < 8.0, but vllm version: {vllm_version} >= 0.10.2, enable custom_logits_processors")
            return True
        else:
            logger.info(f"compute_capability: {compute_capability} < 8.0 and vllm version: {vllm_version} < 0.10.2, disable custom_logits_processors")
            return False
    else:
        logger.info(f"compute_capability: {compute_capability} >= 8.0 and vllm version: {vllm_version} >= 0.10.1, enable custom_logits_processors")
        return True