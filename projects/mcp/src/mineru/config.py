"""MinerU File转Markdown转换服务的配置工具。"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# API 配置
MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net")
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")

# 本地API配置
USE_LOCAL_API = os.getenv("USE_LOCAL_API", "").lower() in ["true", "1", "yes"]
LOCAL_MINERU_API_BASE = os.getenv("LOCAL_MINERU_API_BASE", "http://localhost:8080")

# 转换后文件的默认输出目录
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./downloads")


# 设置日志系统
def setup_logging():
    """
    设置日志系统，根据环境变量配置日志级别。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    # 获取环境变量中的日志级别设置
    log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
    debug_mode = os.getenv("MINERU_DEBUG", "").lower() in ["true", "1", "yes"]

    # 如果设置了debug_mode，则覆盖log_level
    if debug_mode:
        log_level = "DEBUG"

    # 确保log_level是有效的
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"

    # 设置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 配置日志
    logging.basicConfig(level=getattr(logging, log_level), format=log_format)

    logger = logging.getLogger("mineru")
    logger.setLevel(getattr(logging, log_level))

    # 输出日志级别信息
    logger.info(f"日志级别设置为: {log_level}")

    return logger


# 创建默认的日志记录器
logger = setup_logging()


# 如果输出目录不存在，则创建它
def ensure_output_dir(output_dir=None):
    """
    确保输出目录存在。

    Args:
        output_dir: 输出目录的可选路径。如果为 None，则使用 DEFAULT_OUTPUT_DIR。

    Returns:
        表示输出目录的 Path 对象。
    """
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# 验证 API 配置
def validate_api_config():
    """
    验证是否已设置所需的 API 配置。

    Returns:
        dict: 配置状态。
    """
    return {
        "api_base": MINERU_API_BASE,
        "api_key_set": bool(MINERU_API_KEY),
        "output_dir": DEFAULT_OUTPUT_DIR,
    }
