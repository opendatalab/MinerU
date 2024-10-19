
from loguru import logger


def ImportPIL(f):
    try:
        import PIL  # noqa: F401
    except ImportError:
        logger.error('Pillow not installed, please install by pip.')
        exit(1)
    return f
