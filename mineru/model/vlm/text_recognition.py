"""行内切片文本识别的结果契约与局部失败隔离。"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from mineru_vl_utils.vlm_client.base_client import ClientClosedError, ServerError

from .errors import is_fatal_engine_error

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient
    from PIL.Image import Image

TextRecognitionResult: TypeAlias = str | None | Exception


def raise_unrecoverable_text_error(error: Exception) -> None:
    """关闭、引擎死亡及程序契约错误必须传播，不能降级成单片识别失败。"""
    if (
        is_fatal_engine_error(error)
        or isinstance(
            error,
            (ClientClosedError, MemoryError, AssertionError, AttributeError, TypeError, LookupError, NotImplementedError),
        )
        or (isinstance(error, RuntimeError) and not isinstance(error, ServerError))
    ):
        raise error


def validate_text_result(value: object) -> str | None:
    """保留合法空响应供宿主回退，拒绝无法保持输入输出契约的非文本结果。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"VLM text recognition returned {type(value).__name__}, expected str or None")
    return str(value)


def recognize_text_batch(client: MinerUClient, images: list[Image]) -> list[TextRecognitionResult]:
    """正常路径保留本地合批；可恢复批次异常时逐片补试一次，保留成功结果。"""
    if not images:
        return []
    try:
        outputs = client.batch_content_extract(images, types="text")
    except Exception as error:
        raise_unrecoverable_text_error(error)
        results: list[TextRecognitionResult] = []
        for image in images:
            try:
                output = client.content_extract(image, type="text")
            except Exception as item_error:
                raise_unrecoverable_text_error(item_error)
                results.append(item_error)
            else:
                results.append(validate_text_result(output))
        return results
    if len(outputs) != len(images):
        raise ValueError(f"VLM text recognition count mismatch: inputs={len(images)}, outputs={len(outputs)}")
    return [validate_text_result(output) for output in outputs]


__all__ = ["TextRecognitionResult", "raise_unrecoverable_text_error", "recognize_text_batch", "validate_text_result"]
