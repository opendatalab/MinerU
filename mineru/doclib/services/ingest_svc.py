"""Doclib 源文件刷新、去重入库、元数据和初始解析入队。"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ...errors import InvalidRequestError, MineruError
from ...filetypes import (
    CSV_EXTENSIONS,
    IMAGE_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    PAGE_RANGE_PARSE_EXTENSIONS,
    TEXT_EXTENSIONS,
    TIERED_PARSE_EXTENSIONS,
    file_type_for_extension,
    is_office_temp_lock_file,
)
from ...parser.page_range import expand_page_range
from ...types import Tier
from ..core.db import DatabaseManager
from ..parse_server_state import ParseServerHealth
from ..core.file_io import FileStat, MetadataExtractionError, compute_sha256, extract_metadata, get_file_stat
from ..core.fts import FTSManager, truncate_head_tail
from ..rows import FileRow, Sha256Row, ShortIdRow, WatchTargetRow
from ..types import (
    FILE_STATUS_ACTIVE,
    FILE_STATUS_DELETED,
    FILE_STATUS_UNREACHABLE,
    PARSE_STATUS_PENDING,
    RULE_TYPE_PARSING_RULE,
    WATCH_STATUS_UNREACHABLE,
    FileInfo,
)
from ..utils.path_utils import normalize_doclib_path
from . import parse_policy
from .config_svc import ConfigService

if TYPE_CHECKING:
    from ..telemetry import TelemetryService


def _now_ms() -> int:
    """返回当前毫秒时间，保持文件状态与解析队列的时间单位。"""
    return int(time.time() * 1000)


FileRefreshStatus = Literal["known", "new", "changed", "missing", "deleted", "unreachable", "unsupported", "error"]


@dataclass(frozen=True)
class FileRefreshResult:
    """记录文件刷新分类和入库错误，供扫描、搜索与解析请求共同消费。"""

    file: FileInfo | None
    status: FileRefreshStatus
    error_code: str | None = None
    error_msg: str | None = None

    @property
    def needs_ingest(self) -> bool:
        """文件已登记但尚未绑定内容哈希时需要入库。"""
        return self.file is not None and self.file.sha256 is None


def raise_refresh_error(refreshed: FileRefreshResult) -> None:
    """保持刷新失败的错误码、消息和路径参数语义。"""
    code = (
        (refreshed.file.error_code if refreshed.file and refreshed.file.error_code else None)
        or refreshed.error_code
        or "ingest_failed"
    )
    message = (
        (refreshed.file.error_msg if refreshed.file and refreshed.file.error_msg else None)
        or refreshed.error_msg
        or "File could not be ingested."
    )
    if code == "file_permission_denied":
        raise InvalidRequestError(code, message, "path")
    raise MineruError(code, message, "path")


async def ensure_doc_record(
    db: DatabaseManager,
    *,
    sha256: str,
    size_bytes: int,
    file_type: str,
    page_count: int | None,
    title: str | None,
    author: str | None,
    subject: str | None,
    keywords: str | None,
    is_image_based: int = 0,
    language: str | None = None,
    error_code: str | None,
    error_msg: str | None,
    first_seen_at: int,
    updated_at: int,
) -> None:
    """按内容哈希只创建一次文档，短标识冲突时逐字符扩展。"""
    existing = cast(ShortIdRow | None, await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha256,)))
    if existing is not None:
        return

    for length in range(7, len(sha256) + 1):
        short_id = sha256[:length]
        await db.execute(
            "INSERT OR IGNORE INTO docs (sha256, short_id, size_bytes, file_type, page_count, "
            "title, author, subject, keywords, is_image_based, error_code, error_msg, first_seen_at, updated_at, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                short_id,
                size_bytes,
                file_type,
                page_count,
                title,
                author,
                subject,
                keywords,
                is_image_based,
                error_code,
                error_msg,
                first_seen_at,
                updated_at,
                language,
            ),
        )
        inserted = cast(ShortIdRow | None, await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha256,)))
        if inserted is not None:
            return
        conflict = cast(Sha256Row | None, await db.fetchone("SELECT sha256 FROM docs WHERE short_id=?", (short_id,)))
        if conflict is not None and conflict["sha256"] != sha256:
            continue
        break

    raise RuntimeError(f"Failed to allocate unique short_id for document {sha256}")


class IngestService:
    """独立管理源文件与文档入库，解析执行通过单独服务处理。"""

    def __init__(
        self,
        db: DatabaseManager,
        fts: FTSManager,
        config_svc: ConfigService,
        *,
        health: ParseServerHealth,
        telemetry_svc: TelemetryService | None = None,
    ) -> None:
        """保存显式数据库、索引、规则配置和可选指标记录依赖。"""
        self.health = health
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.telemetry_svc = telemetry_svc

    async def refresh_file(
        self,
        path: str,
        watch_id: int | None = None,
        *,
        ensure_ingested: bool = False,
        allow_images: bool = False,
    ) -> FileRefreshResult:
        """刷新路径对应的文件状态，按需同步完成入库。"""
        path = normalize_doclib_path(path)
        ext = Path(path).suffix.lower().lstrip(".")
        existing = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))
        if ext not in INGESTIBLE_EXTENSIONS or (ext in IMAGE_EXTENSIONS and not allow_images) or is_office_temp_lock_file(path):
            return FileRefreshResult(file=_file_info(existing), status="unsupported")

        try:
            stat = await get_file_stat(path)
        except FileNotFoundError:
            return await self._refresh_missing_file(path, existing)
        except PermissionError as exc:
            return await self._refresh_stat_error(existing, "file_permission_denied", str(exc))
        except OSError as exc:
            return await self._refresh_stat_error(existing, "stat_failed", str(exc))

        result = await self._refresh_existing_file_with_stat(path, ext, stat, existing, watch_id)
        if ensure_ingested and result.needs_ingest:
            row = await self.ingest_file(path, watch_id=watch_id, allow_images=allow_images)
            return FileRefreshResult(file=_file_info(row), status=result.status)
        return result

    async def _refresh_existing_file_with_stat(
        self,
        path: str,
        ext: str,
        stat: FileStat,
        existing: FileRow | None,
        watch_id: int | None,
    ) -> FileRefreshResult:
        """依据大小、修改时间和历史状态登记新文件或标记变化。"""
        now = _now_ms()
        filename = Path(path).name

        if existing and existing["status"] == FILE_STATUS_ACTIVE:
            unchanged = existing["mtime_ms"] == stat.mtime_ms and existing["size_bytes"] == stat.size_bytes
            if unchanged:
                if watch_id is not None and existing["watch_id"] is None:
                    await self.db.execute(
                        "UPDATE files SET watch_id=?, updated_at=? WHERE id=? AND watch_id IS NULL",
                        (watch_id, now, existing["id"]),
                    )
                    existing = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
                return FileRefreshResult(file=_file_info(existing), status="known")

            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                "sha256=NULL, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                "WHERE id=?",
                (
                    filename,
                    ext,
                    stat.size_bytes,
                    stat.mtime_ms,
                    watch_id,
                    now,
                    existing["id"],
                ),
            )
            row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
            return FileRefreshResult(file=_file_info(row), status="changed")

        if existing:
            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                "sha256=NULL, watch_id=COALESCE(?, watch_id), status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, "
                "deleted_at=NULL, updated_at=? WHERE id=?",
                (
                    filename,
                    ext,
                    stat.size_bytes,
                    stat.mtime_ms,
                    watch_id,
                    FILE_STATUS_ACTIVE,
                    now,
                    existing["id"],
                ),
            )
            row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
            return FileRefreshResult(file=_file_info(row), status="changed")

        file_id = await self.db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat.size_bytes,
                stat.mtime_ms,
                watch_id,
                now,
                now,
            ),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (file_id,)))
        return FileRefreshResult(file=_file_info(row), status="new")

    async def _refresh_missing_file(self, path: str, existing: FileRow | None) -> FileRefreshResult:
        """区分已删除路径与不可达的可移动监控目录。"""
        if existing is None:
            return FileRefreshResult(file=None, status="missing")

        now = _now_ms()
        file_status = FILE_STATUS_DELETED
        deleted_at: int | None = now
        if existing["status"] != FILE_STATUS_DELETED and await self._is_missing_due_to_unreachable_watch(existing):
            file_status = FILE_STATUS_UNREACHABLE
            deleted_at = None

        await self.db.execute(
            "UPDATE files SET status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, deleted_at=?, updated_at=? WHERE id=?",
            (file_status, deleted_at, now, existing["id"]),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
        refresh_status: FileRefreshStatus = "unreachable" if file_status == FILE_STATUS_UNREACHABLE else "deleted"
        return FileRefreshResult(file=_file_info(row), status=refresh_status)

    async def _is_missing_due_to_unreachable_watch(self, file_row: FileRow) -> bool:
        """检查缺失路径是否来自暂不可达的可移动目录。"""
        watch_id = file_row.get("watch_id")
        if watch_id is None:
            return False
        watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE id=?", (watch_id,)))
        if not watch or not watch.get("removable"):
            return False
        if watch["status"] == WATCH_STATUS_UNREACHABLE:
            return True
        try:
            await get_file_stat(watch["path"])
        except OSError:
            if self.config_svc is not None:
                await self.config_svc.update_watch_status(watch_id, WATCH_STATUS_UNREACHABLE)
            return True
        return False

    async def _refresh_stat_error(self, existing: FileRow | None, error_code: str, error_msg: str) -> FileRefreshResult:
        """保留原有权限和 stat 失败状态，并解除文件锁与哈希绑定。"""
        if existing is None:
            return FileRefreshResult(file=None, status="error", error_code=error_code, error_msg=error_msg)

        now = _now_ms()
        await self.db.execute(
            "UPDATE files SET sha256=NULL, locked_at=NULL, error_code=?, error_msg=?, updated_at=? WHERE id=?",
            (error_code, error_msg[:500], now, existing["id"]),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
        return FileRefreshResult(file=_file_info(row), status="error", error_code=error_code, error_msg=error_msg)

    async def ensure_ingested(self, path: str, watch_id: int | None = None, *, allow_images: bool = False) -> FileRow | None:
        """确保路径已刷新并入库，保持不支持格式和错误的原有边界。"""
        path = normalize_doclib_path(path)
        refreshed = await self.refresh_file(path, watch_id=watch_id, ensure_ingested=True, allow_images=allow_images)
        if refreshed.status == "error":
            raise_refresh_error(refreshed)
        if refreshed.file is None or refreshed.status == "unsupported":
            return None
        return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

    async def ingest_file(
        self,
        path: str,
        watch_id: int | None = None,
        *,
        trigger: str = "parse",
        allow_images: bool = False,
    ) -> FileRow | None:
        """记录入库计数和耗时，将权限错误映射为稳定路径错误。"""
        path = normalize_doclib_path(path)
        start_ms = _now_ms()
        status = "succeeded"
        try:
            return await self._ingest_file(path, watch_id=watch_id, allow_images=allow_images)
        except PermissionError as exc:
            status = "failed"
            await self._mark_file_error(path, "file_permission_denied", str(exc))
            raise InvalidRequestError("file_permission_denied", str(exc), "path") from None
        except Exception:
            status = "failed"
            raise
        finally:
            dimensions = {"status": status, "trigger": trigger}
            await self._record_count("ingest.finished.count", dimensions=dimensions)
            await self._record_duration("ingest.duration_bucket.count", start_ms, dimensions=dimensions)

    async def _mark_file_error(self, path: str, error_code: str, error_msg: str) -> None:
        """保存文件错误并解除入库锁，限制持久化消息长度。"""
        path = normalize_doclib_path(path)
        now = _now_ms()
        await self.db.execute(
            "UPDATE files SET sha256=NULL, locked_at=NULL, error_code=?, error_msg=?, updated_at=? WHERE path=?",
            (error_code, error_msg[:500], now, path),
        )

    async def _ingest_file(self, path: str, watch_id: int | None = None, *, allow_images: bool = False) -> FileRow | None:
        """计算哈希并复用文档，提取元数据和索引，再按规则创建初始解析。"""
        ext = Path(path).suffix.lower().lstrip(".")
        if ext not in INGESTIBLE_EXTENSIONS or (ext in IMAGE_EXTENSIONS and not allow_images) or is_office_temp_lock_file(path):
            return None

        stat = await get_file_stat(path)
        sha256 = await compute_sha256(path)
        now = _now_ms()
        filename = Path(path).name

        existing_path = cast(
            FileRow | None,
            await self.db.fetchone("SELECT * FROM files WHERE path=? AND status=?", (path, FILE_STATUS_ACTIVE)),
        )
        if existing_path and existing_path["sha256"] == sha256:
            return existing_path

        # check if same content (sha256) is already tracked by another path
        existing_sha = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE sha256=?", (sha256,)))
        if existing_sha:
            # same content, possibly a new or changed path — bind this file row
            # to the existing doc without duplicating docs/parses.
            if existing_path:
                await self.db.execute(
                    "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                    "sha256=?, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                    "WHERE id=?",
                    (
                        filename,
                        ext,
                        stat.size_bytes,
                        stat.mtime_ms,
                        sha256,
                        watch_id,
                        now,
                        existing_path["id"],
                    ),
                )
            else:
                await self.db.execute(
                    "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
                    "sha256, watch_id, first_seen_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        path,
                        filename,
                        ext,
                        stat.size_bytes,
                        stat.mtime_ms,
                        sha256,
                        watch_id,
                        now,
                        now,
                    ),
                )
            file_row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))
            if file_row:
                await self.fts.upsert_filename(file_row["id"], Path(path).stem, ext)
            # the source may have been moved here from another tracked path;
            # retire stale duplicates whose files no longer exist.
            await self._mark_moved_duplicate_files_deleted(sha256, path)
            return file_row

        # brand new document
        metadata_error_code = None
        metadata_error_msg = None
        try:
            metadata = await extract_metadata(path)
            metadata_error_code = metadata.get("error_code")
            metadata_error_msg = metadata.get("error_msg")
        except MetadataExtractionError as exc:
            metadata_error_code = exc.code
            metadata_error_msg = str(exc)[:500] or "Failed to extract document metadata"
            metadata = {
                "page_count": None,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }
        except Exception as exc:
            metadata_error_code = "open_failed"
            metadata_error_msg = str(exc)[:500] or "Failed to extract document metadata"
            metadata = {
                "page_count": None,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }

        # 无固有分页或元数据暂不可读的结构化格式默认按一页处理。
        page_count = metadata["page_count"]
        if page_count is None:
            if ext in ("doc", "docx", "rtf"):
                page_count = 1  # reflow docs
            elif ext in ("ppt", "pptx", "xls", "xlsx"):
                page_count = 1  # will be updated by metadata extraction
            elif ext in CSV_EXTENSIONS:
                page_count = 1

        now = _now_ms()
        await self.db.execute(
            "INSERT OR IGNORE INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat.size_bytes,
                stat.mtime_ms,
                watch_id,
                now,
                now,
            ),
        )

        await ensure_doc_record(
            self.db,
            sha256=sha256,
            size_bytes=stat.size_bytes,
            file_type=file_type_for_extension(ext),
            page_count=page_count,
            title=metadata["title"],
            author=metadata["author"],
            subject=metadata["subject"],
            keywords=metadata["keywords"],
            is_image_based=int(metadata.get("is_image_based") or 0),
            language=metadata.get("language"),
            error_code=metadata_error_code,
            error_msg=metadata_error_msg,
            first_seen_at=now,
            updated_at=now,
        )

        # FTS filename index
        file_row = await self.db.fetchone("SELECT id FROM files WHERE path=?", (path,))
        if file_row:
            await self.fts.upsert_filename(
                file_row["id"],
                Path(path).stem,
                ext,
            )

        if ext in TEXT_EXTENSIONS:
            await self.db.execute(
                "UPDATE files SET sha256=?, locked_at=NULL, updated_at=? WHERE path=?",
                (sha256, now, path),
            )
            text = await asyncio.to_thread(Path(path).read_text, encoding="utf-8", errors="replace")
            await self.fts.replace(
                sha256=sha256,
                tier=None,
                text=truncate_head_tail(text),
                title=metadata["title"] or "",
                author=metadata["author"] or "",
            )
            return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

        # determine tier and page_range for initial parse
        tier: Tier = "flash"
        privacy = "local"
        initial_page_range = (
            parse_policy.default_parse_range(page_count)
            if ext in PAGE_RANGE_PARSE_EXTENSIONS
            else expand_page_range(None, page_count or 1)
        )

        # check parsing-rules
        matched = await self.config_svc.match_rules(path, RULE_TYPE_PARSING_RULE)
        if matched:
            rule = matched[0]
            rule_tier = cast(Tier | None, rule.get("tier"))
            rule_remote = bool(rule.get("remote", False))
            if ext in TIERED_PARSE_EXTENSIONS:
                tier = rule_tier or parse_policy.resolve_parsing_rule_default_tier(health=self.health, remote=rule_remote)
                privacy = "remote" if rule_remote and tier != "flash" else "local"
            else:
                tier = "flash"
                privacy = "local"
            rule_page_range = rule.get("page_range")
            if rule_page_range and ext in PAGE_RANGE_PARSE_EXTENSIONS:
                initial_page_range = expand_page_range(rule_page_range, page_count or 1)

        # insert parse batch
        parse_page_range = expand_page_range(initial_page_range, page_count or 1)
        await self.db.execute(
            "UPDATE files SET sha256=?, locked_at=NULL, updated_at=? WHERE path=?",
            (sha256, now, path),
        )
        await self.db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (sha256, tier, parse_page_range, PARSE_STATUS_PENDING, privacy, now, now),
        )
        await self._record_count("parse_task.created.count", dimensions={"tier": tier})

        return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

    async def _mark_moved_duplicate_files_deleted(self, sha256: str, keep_path: str) -> None:
        """同哈希的新路径入库后，将已不存在的旧路径标为删除。"""
        rows = cast(
            list[FileRow],
            await self.db.fetchall(
                "SELECT id, path FROM files WHERE sha256=? AND status=? AND path<>?",
                (sha256, FILE_STATUS_ACTIVE, keep_path),
            ),
        )
        now = _now_ms()
        for row in rows:
            if not os.path.exists(row["path"]):
                await self.db.execute(
                    "UPDATE files SET status=?, deleted_at=?, updated_at=? WHERE id=?",
                    (FILE_STATUS_DELETED, now, now, row["id"]),
                )

    async def _record_count(self, metric_name: str, *, value: int = 1, dimensions: dict[str, str] | None = None) -> None:
        """通过显式指标接口记录入库和初始解析计数。"""
        if self.telemetry_svc is not None:
            await self.telemetry_svc.record_count(metric_name, value=value, dimensions=dimensions)

    async def _record_duration(self, metric_name: str, start_ms: int, *, dimensions: dict[str, str] | None = None) -> None:
        """使用原有非负耗时和维度记录入库耗时桶。"""
        if self.telemetry_svc is not None:
            await self.telemetry_svc.record_duration_bucket(
                metric_name, duration_ms=max(0, _now_ms() - start_ms), dimensions=dimensions
            )


def _file_info(row: FileRow | None) -> FileInfo | None:
    """将文件数据库行转换为对外文件信息。"""
    return FileInfo.model_validate(row) if row is not None else None


__all__ = ["FileRefreshResult", "FileRefreshStatus", "IngestService", "ensure_doc_record", "raise_refresh_error"]
