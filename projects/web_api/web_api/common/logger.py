import os
from loguru import logger
from pathlib import Path
from datetime import datetime


def setup_log(config):
    """
    Setup logging
    :param config:  config file
    :return:
    """
    log_path = os.path.join(Path(__file__).parent.parent, "log")
    if not Path(log_path).exists():
        Path(log_path).mkdir(parents=True, exist_ok=True)
    log_level = config.get("LOG_LEVEL")
    log_name = f'log_{datetime.now().strftime("%Y-%m-%d")}.log'
    log_file_path = os.path.join(log_path, log_name)
    logger.add(str(log_file_path), rotation='00:00', encoding='utf-8', level=log_level, enqueue=True)
