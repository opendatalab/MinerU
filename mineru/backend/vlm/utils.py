import os

from loguru import logger
from packaging import version

from mineru.utils.check_sys_env import is_windows_environment, is_linux_environment
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram


def enable_custom_logits_processors() -> bool:
    import torch
    from vllm import __version__ as vllm_version

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # 正确计算Compute Capability
        compute_capability = f"{major}.{minor}"
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        compute_capability = "8.0"
    else:
        logger.info("CUDA not available, disabling custom_logits_processors")
        return False

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


def set_lmdeploy_backend(device_type: str) -> str:
    if device_type.lower() in ["ascend", "maca", "camb"]:
        lmdeploy_backend = "pytorch"
    elif device_type.lower() in ["cuda"]:
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        if is_windows_environment():
            lmdeploy_backend = "turbomind"
        elif is_linux_environment():
            major, minor = torch.cuda.get_device_capability()
            compute_capability = f"{major}.{minor}"
            if version.parse(compute_capability) >= version.parse("8.0"):
                lmdeploy_backend = "pytorch"
            else:
                lmdeploy_backend = "turbomind"
        else:
            raise ValueError("Unsupported operating system.")
    else:
        raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
    return lmdeploy_backend


def set_default_gpu_memory_utilization() -> float:
    from vllm import __version__ as vllm_version
    device = get_device()
    gpu_memory = get_vram(device)
    if version.parse(vllm_version) >= version.parse("0.11.0") and gpu_memory <= 8:
        return 0.7
    else:
        return 0.5


def set_default_batch_size() -> int:
    try:
        device = get_device()
        gpu_memory = get_vram(device)

        if gpu_memory >= 16:
            batch_size = 8
        elif gpu_memory >= 8:
            batch_size = 4
        else:
            batch_size = 1
        logger.info(f'gpu_memory: {gpu_memory} GB, batch_size: {batch_size}')

    except Exception as e:
        logger.warning(f'Error determining VRAM: {e}, using default batch_ratio: 1')
        batch_size = 1
    return batch_size