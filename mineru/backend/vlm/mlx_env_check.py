import sys
import platform
import psutil
from loguru import logger
from packaging import version

# 移除默认的logger handler，以自定义格式
logger.remove()
# 添加一个新的handler，设置日志格式，使其更加清晰易读
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def check_mlx_vlm_environment():
    """
    检查当前环境是否满足在Apple Silicon Mac上运行mlx-vlm的要求。

    检查点:
    1. 操作系统是否为 macOS ('darwin')
    2. CPU架构是否为 Apple Silicon ('arm64')
    3. 系统内存是否 >= 8 GB
    4. 是否已安装 'mlx-vlm' 库
    5. 'mlx-vlm' 库的版本是否 >= '0.3.3'

    Returns:
        bool: 如果所有条件都满足，则返回 True，否则返回 False。
    """
    # 1. 检查操作系统是否为macOS
    if sys.platform != 'darwin':
        logger.error(f"操作系统检查失败: 当前系统为 '{sys.platform}'，需要 'darwin' (macOS)。")
        return False
    logger.info("✅ 操作系统检查通过: 当前为 macOS。")

    # 2. 检查CPU架构是否为Apple Silicon (M系列芯片)
    if platform.machine() != 'arm64':
        logger.error(f"CPU架构检查失败: 当前架构为 '{platform.machine()}'，需要 'arm64' (Apple Silicon)。")
        return False
    logger.info("✅ CPU架构检查通过: 当前为 Apple Silicon (arm64)。")

    # 3. 检查内存大小
    required_memory_gb = 8
    total_memory_bytes = psutil.virtual_memory().total
    total_memory_gb = total_memory_bytes / (1024 ** 3)  # 字节转换为GB

    if total_memory_gb < required_memory_gb:
        logger.error(f"内存检查失败: 系统总内存为 {total_memory_gb:.2f} GB，需要 >= {required_memory_gb} GB。")
        return False
    logger.info(f"✅ 内存检查通过: 系统总内存为 {total_memory_gb:.2f} GB。")

    # 4. & 5. 检查依赖库 'mlx-vlm' 及其版本号
    required_package = 'mlx-vlm'
    minimum_version = '0.3.3'
    
    try:
        from importlib import metadata
        installed_version_str = metadata.version(required_package)
        logger.info(f"✅ 依赖库检查通过: 已安装 '{required_package}'，版本为 {installed_version_str}。")
        
        # 检查版本号是否大于等于最低要求
        if version.parse(installed_version_str) < version.parse(minimum_version):
            logger.error(f"版本号不满足要求: '{required_package}' 的版本为 {installed_version_str}，但需要 >= '{minimum_version}'。")
            return False
        logger.info(f"✅ 版本号满足要求: '{required_package}' 版本正确。")

    except ImportError:
        logger.error("无法导入 'importlib.metadata'。请确保您的 Python 版本 >= 3.8。")
        return False
    except metadata.PackageNotFoundError:
        logger.error(f"依赖库检查失败: 未安装 '{required_package}'。")
        return False

    # 如果所有检查都通过
    logger.success("🎉 恭喜！您的环境完全满足运行要求。")
    return True

if __name__ == "__main__":
    logger.info("--- 开始进行 Mac MLX 环境符合性检查 ---")
    is_ready = check_mlx_vlm_environment()
    if is_ready:
        print("\n检查结果: 环境就绪 (Environment Ready)")
    else:
        print("\n检查结果: 环境不符合要求 (Environment Not Ready)")
    logger.info("--- 环境检查结束 ---")

