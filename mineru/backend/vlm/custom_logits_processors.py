from loguru import logger


def enable_custom_logits_processors():
    import torch
    compute_capability = 0.0
    custom_logits_processors = False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        compute_capability = float(major) + (float(minor) / 10.0)
    if compute_capability >= 8.0:
        logger.info(f"compute_capability: {compute_capability}, enable custom_logits_processors")
        custom_logits_processors = True
    return custom_logits_processors