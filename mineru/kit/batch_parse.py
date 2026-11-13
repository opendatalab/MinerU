"""Kit 的输入边界和批次错误策略；不依赖文件名推断 Office 锁文件。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Literal

from ..errors import InvalidRequestError, MineruError
from ..filetypes import PARSEABLE_EXTENSIONS
from ..integrations.docvortex import translate_document_errors
from ..model.vlm.errors import is_fatal_engine_error
from ..parser.api_client import _V1APIError
from ..parser.file_type import guess_suffix_by_path
from ..types import FILE_SUFFIXES
from ..utils.i18n import t

_INPUT_FAILURE_CODES = frozenset(
    {"file_corrupted", "file_encrypted", "file_too_large", "resource_limit", "file_not_found", "file_permission_denied"}
)


@dataclass
class BatchParseSummary:
    """记录真实的逐项处理结果，终止时保留尚未处理的数量。"""

    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    aborted: bool = False

    def message(self, total: int) -> str:
        """以统一摘要替代可能掩盖失败的全部成功提示。"""
        return t(
            "Batch finished: succeeded={succeeded}, skipped={skipped}, failed={failed}, unprocessed={unprocessed}.",
            succeeded=self.succeeded,
            skipped=self.skipped,
            failed=self.failed,
            unprocessed=total - self.succeeded - self.skipped - self.failed,
        )

    @property
    def exit_code(self) -> int:
        """只有至少一个成功且无失败、无系统性中止的批次才成功退出。"""
        return int(self.succeeded == 0 or self.failed > 0 or self.aborted)


def validate_parse_input(path: Path) -> None:
    """本地与远程分流前验证内容类型，名称只用于候选发现和无签名格式提示。"""
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise InvalidRequestError("file_type_unsupported", t("Unsupported file type: {path}", path=path), "input")
        if path.suffix.lower().lstrip(".") not in PARSEABLE_EXTENSIONS:
            raise InvalidRequestError("file_type_unsupported", t("Unsupported file type: {path}", path=path), "input")
        with translate_document_errors(native_input=True):
            suffix = guess_suffix_by_path(path)
        if suffix not in FILE_SUFFIXES and suffix not in PARSEABLE_EXTENSIONS:
            raise InvalidRequestError("file_type_unsupported", t("Unsupported file type: {path}", path=path), "input")
    except FileNotFoundError as exc:
        raise InvalidRequestError("file_not_found", str(exc), "input") from exc
    except PermissionError as exc:
        raise InvalidRequestError("file_permission_denied", str(exc), "input") from exc


def classify_parse_error(error: Exception) -> tuple[Literal["skipped", "failed", "abort"], str, str]:
    """只隔离明确的输入错误；未知异常、引擎和远程服务故障一律终止批次。"""
    if is_fatal_engine_error(error):
        return "abort", "engine_dead", str(error)
    if isinstance(error, (MineruError, _V1APIError)):
        if error.code == "file_type_unsupported":
            return "skipped", error.code, error.message
        if error.code in _INPUT_FAILURE_CODES:
            return "failed", error.code, error.message
        return "abort", error.code, error.message
    return "abort", "parse_failed", str(error)


__all__ = ["BatchParseSummary", "classify_parse_error", "validate_parse_input"]
