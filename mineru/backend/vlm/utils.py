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
    elif hasattr(torch, 'gcu') and torch.gcu.is_available():
        compute_capability = "8.0"
    elif hasattr(torch, 'musa') and torch.musa.is_available():
        compute_capability = "8.0"
    elif hasattr(torch, 'mlu') and torch.mlu.is_available():
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


def _get_device_config(device_type: str) -> dict | None:
    """获取不同设备类型的配置参数"""

    # 各设备类型的配置定义
    DEVICE_CONFIGS = {
        # "musa": {
        #     "compilation_config_dict": {
        #         "cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30],
        #         "simple_cuda_graph": True
        #     },
        #     "block_size": 32,
        # },
        "corex": {
            "compilation_config_dict": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "level": 0
            },
        },
        "kxpu": {
            "compilation_config_dict": {
                "splitting_ops": [
                    "vllm.unified_attention", "vllm.unified_attention_with_output",
                    "vllm.unified_attention_with_output_kunlun", "vllm.mamba_mixer2",
                    "vllm.mamba_mixer", "vllm.short_conv", "vllm.linear_attention",
                    "vllm.plamo2_mamba_mixer", "vllm.gdn_attention", "vllm.sparse_attn_indexer"
                ]
            },
            "block_size": 128,
            "dtype": "float16",
            "distributed_executor_backend": "mp",
        },
    }

    return DEVICE_CONFIGS.get(device_type.lower())


def _check_server_arg_exists(args: list, arg_name: str) -> bool:
    """检查命令行参数列表中是否已存在指定参数"""
    return any(arg == f"--{arg_name}" or arg.startswith(f"--{arg_name}=") for arg in args)


def _add_server_arg_if_missing(args: list, arg_name: str, value: str) -> None:
    """如果参数不存在，则添加到命令行参数列表"""
    if not _check_server_arg_exists(args, arg_name):
        args.extend([f"--{arg_name}", value])


def _add_engine_kwarg_if_missing(kwargs: dict, key: str, value) -> None:
    """如果参数不存在，则添加到 kwargs 字典"""
    if key not in kwargs:
        kwargs[key] = value


def mod_kwargs_by_device_type(kwargs_or_args: dict | list, vllm_mode: str) -> dict | list:
    """根据设备类型修改 vllm 配置参数

    Args:
        kwargs_or_args: 配置参数，server 模式为 list，engine 模式为 dict
        vllm_mode: vllm 运行模式 ("server", "sync_engine", "async_engine")

    Returns:
        修改后的配置参数
    """
    device_type = os.getenv("MINERU_VLLM_DEVICE", "")
    config = _get_device_config(device_type)

    if config is None:
        return kwargs_or_args

    if vllm_mode == "server":
        _apply_server_config(kwargs_or_args, config)
    else:
        _apply_engine_config(kwargs_or_args, config, vllm_mode)

    return kwargs_or_args


def _apply_server_config(args: list, config: dict) -> None:
    """应用 server 模式的配置"""
    import json

    if "compilation_config_dict" in config:
        _add_server_arg_if_missing(
            args, "compilation-config",
            json.dumps(config["compilation_config_dict"], separators=(',', ':'))
        )

    for key in ["block_size", "dtype", "distributed_executor_backend"]:
        if key in config:
            # 转换 key 格式: block_size -> block-size
            arg_name = key.replace("_", "-")
            _add_server_arg_if_missing(args, arg_name, str(config[key]))


def _apply_engine_config(kwargs: dict, config: dict, vllm_mode: str) -> None:
    """应用 engine 模式的配置"""
    try:
        from vllm.config import CompilationConfig
    except ImportError:
        raise ImportError("Please install vllm to use the vllm-async-engine backend.")

    if "compilation_config_dict" in config:
        config_dict = config["compilation_config_dict"]
        if vllm_mode == "sync_engine":
            compilation_config = config_dict
        elif vllm_mode == "async_engine":
            compilation_config = CompilationConfig(**config_dict)
        else:
            return
        _add_engine_kwarg_if_missing(kwargs, "compilation_config", compilation_config)

    for key in ["block_size", "dtype", "distributed_executor_backend"]:
        if key in config:
            _add_engine_kwarg_if_missing(kwargs, key, config[key])
