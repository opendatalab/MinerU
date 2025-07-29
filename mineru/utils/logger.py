# -*- encoding: utf-8 -*-
# @Author: Jocker1212
# @Contact: xinyijianggo@gmail.com
import logging
from functools import lru_cache


@lru_cache(maxsize=32)
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    format_str = logging.Formatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    logger.addHandler(sh)
    sh.setFormatter(format_str)
    return logger
