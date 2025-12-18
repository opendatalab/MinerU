#  Copyright (c) Opendatalab. All rights reserved.
from loguru import logger

from mineru.utils.check_sys_env import is_mac_os_version_supported


def get_vlm_engine(inference_engine, is_async: bool = False) -> str:
    if inference_engine == 'auto':
        try:
            import vllm
            if is_async:
                inference_engine = 'vllm-async'
            else:
                inference_engine = 'vllm'
        except ImportError:
            try:
                import lmdeploy
                inference_engine = 'lmdeploy'
            except ImportError:
                try:
                    from mlx_vlm import load as mlx_load
                    if is_mac_os_version_supported():
                        inference_engine = 'mlx'
                    else:
                        inference_engine = 'transformers'
                except ImportError:
                    inference_engine = 'transformers'
    if inference_engine != 'transformers':
        inference_engine = f"{inference_engine}-engine"
    logger.info(f"Using {inference_engine} as the inference engine for VLM.")
    return inference_engine