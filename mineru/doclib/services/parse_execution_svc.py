"""Doclib 已抢占解析批次的执行、产物和索引更新。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

from ...filetypes import (
    OFD_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PAGE_RANGE_PARSE_EXTENSIONS,
    is_office_temp_lock_file,
)
from ...parser.api_client import _APITransportError, _V1APIError
from ...parser.base import ParseResult
from ...parser.page_range import (
    format_page_range as _page_numbers_to_range_str,
)
from ...parser.page_range import (
    parse_page_range_set,
)
from ...types import TIER_ORDER, Tier
from ..core import fts as fts_utils
from ..core.db import DatabaseManager
from ..parse_server_state import ParseServerHealth
from ..core.file_io import (
    metadata_to_doclib,
)
from ..core.fts import FTSManager
from ..core.parse_artifacts import (
    accessible_file_for_sha256,
    parse_artifact_directory,
    parse_batch_json_path,
    write_parse_payload,
)
from ..remote_api import resolve_remote_api_key
from ..rows import FileRow, ParseRow
from ..types import (
    PARSE_STATUS_DONE,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PARSING,
)
from . import parse_policy
from .config_svc import ConfigService
from .parse_errors import ParseFailure

logger = logging.getLogger("mineru.parse")


if TYPE_CHECKING:
    from ..telemetry import TelemetryService


def _now_ms() -> int:
    """返回当前毫秒时间，保持数据库状态与锁超时的原有单位。"""
    return int(time.time() * 1000)


class ParseExecutionService:
    """执行已抢占的解析批次，拥有传输、产物写出和索引更新。"""

    def __init__(
        self,
        db: DatabaseManager,
        fts: FTSManager,
        config_svc: ConfigService,
        data_dir: str,
        *,
        health: ParseServerHealth,
        telemetry_svc: TelemetryService | None = None,
    ) -> None:
        """保存此服务所需的显式依赖，不隐式组装另一侧服务。"""
        self.health = health
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)
        self.telemetry_svc = telemetry_svc

    async def process_doc(self, task: ParseRow) -> bool:
        """执行已抢占的批次并写出产物，按原有边界更新元数据、索引和终态。"""
        task_start_ms = _now_ms()
        sha256 = task["sha256"]
        tier = task["tier"]
        page_range = task["page_range"]
        privacy = task.get("privacy", "local")

        # guard
        current = await self.db.fetchone("SELECT status FROM parses WHERE id=?", (task["id"],))
        if current is None or current["status"] != PARSE_STATUS_PARSING:
            return False

        # find the file
        file_row = await accessible_file_for_sha256(self.db, sha256)
        if file_row is None:
            await self._fail_task(task["id"], "no_accessible_file", "No active file found for this document")
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="no_accessible_file",
            )
            return False
        if is_office_temp_lock_file(file_row["path"]):
            await self._fail_task(task["id"], "file_type_unsupported", "Office temporary lock files are not supported")
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="file_type_unsupported",
            )
            return False
        file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")

        output_dir = parse_artifact_directory(self.data_dir, sha256, tier)

        # route parse based on tier
        via = "local"
        execute_start_ms = _now_ms()
        server_dim = "local(flash)" if tier == "flash" else "unknown"
        try:
            if tier == "flash":
                logger.debug(
                    "Local flash parse started task_id=%s path=%s page_range=%s",
                    task["id"],
                    file_row["path"],
                    page_range,
                )
                result = await self._parse_via_local(file_row, tier, page_range)
                logger.debug(
                    "Local flash parse completed task_id=%s elapsed_ms=%d",
                    task["id"],
                    _now_ms() - execute_start_ms,
                )
            else:
                result, via = await self._parse_via_api(file_row, tier, page_range, privacy)
                server_dim = await self._telemetry_server_dim(via=via, tier=tier)
        except ParseFailure as exc:
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code=exc.code,
            )
            await self._fail_task(task["id"], exc.code, exc.message)
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code=exc.code)
            return False
        except _V1APIError as exc:
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code=exc.code,
            )
            await self._fail_task(task["id"], exc.code, exc.message[:500])
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code=exc.code)
            return False
        except Exception as exc:
            logger.error(
                "Parse execution failed for task_id=%s path=%s tier=%s page_range=%s: %s",
                task["id"],
                file_row["path"],
                tier,
                page_range,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code="parse_failed",
            )
            await self._fail_task(task["id"], "parse_failed", str(exc)[:500])
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code="parse_failed")
            return False
        await self._record_execute(execute_start_ms, tier=tier, server=server_dim, status="succeeded")

        export_payload = result.to_dict(skip_defaults=True)
        new_pages = export_payload["pages"]
        if not new_pages:
            await self._fail_task(task["id"], "parse_empty", "Parse completed but returned no pages")
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code="parse_failed")
            return False

        persisted_page_range = page_range
        full_document_page_count: int | None = None
        if file_ext not in PAGE_RANGE_PARSE_EXTENSIONS:
            actual_page_numbers = {page.page_idx + 1 for page in result.pages}
            persisted_page_range = _page_numbers_to_range_str(actual_page_numbers)
            full_document_page_count = max(actual_page_numbers)

        # save per-batch JSON (markdown is generated on read from /docs/{doc_ref}/content)
        done_at_ms = _now_ms()
        json_path = parse_batch_json_path(self.data_dir, sha256, tier, persisted_page_range, done_at_ms)
        write_start_ms = _now_ms()
        try:
            os.makedirs(output_dir, exist_ok=True)
            write_parse_payload(json_path, export_payload)
        except Exception as exc:
            logger.error(
                "Parse post-write finalization failed for task_id=%s path=%s tier=%s page_range=%s: %s",
                task["id"],
                file_row["path"],
                tier,
                persisted_page_range,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await self._fail_task(task["id"], "parse_json_write_failed", str(exc)[:500])
            await self._record_write(
                write_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            return False

        try:
            # generate markdown for FTS (not persisted, only for search indexing)
            md_text = ""
            md = result.markdown(add_markers=False) if hasattr(result, "markdown") else ""
            md_text += md + "\n"

            # 原文件属性不受推理档位限制，先更新数据库和已有索引，再处理正文索引。
            await self._update_source_metadata(sha256, result)
            refreshed_doc = await self.db.fetchone("SELECT title, author FROM docs WHERE sha256=?", (sha256,))
            if refreshed_doc:
                file_row = cast(FileRow, {**file_row, **refreshed_doc})

            # update fts_contents (tier-gated)
            await self._maybe_update_fts(sha256, tier, md_text, file_row)

            # update docs metadata (tier-gated)
            await self._maybe_update_docs_meta(sha256, tier)
            if full_document_page_count is not None:
                await self.db.execute(
                    "UPDATE docs SET page_count=?, updated_at=? WHERE sha256=?",
                    (full_document_page_count, done_at_ms, sha256),
                )

            # mark done
            await self.db.execute(
                "UPDATE parses SET status=?, page_range=?, done_at=?, locked_at=NULL, via=?, updated_at=? WHERE id=?",
                (PARSE_STATUS_DONE, persisted_page_range, done_at_ms, via, done_at_ms, task["id"]),
            )
        except Exception as exc:
            await self._fail_task(task["id"], "parse_json_write_failed", str(exc)[:500])
            await self._record_write(
                write_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            return False

        await self._record_write(write_start_ms, tier=tier, status="succeeded")
        await self._record_count("parse_task.files.count", dimensions={"tier": tier})
        await self._record_count("parse_task.pages.count", value=len(new_pages), dimensions={"tier": tier})
        await self._record_parse_task_finished(task_start_ms, tier=tier, status="succeeded")
        return True

    async def _parse_via_local(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
    ) -> ParseResult:
        """调用本地解析器，整本格式不传递内部缓存页范围。"""
        from ...parser import parse

        file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")
        parser_page_range = page_range if file_ext in PAGE_RANGE_PARSE_EXTENSIONS else ""
        result = await asyncio.to_thread(
            parse,
            file_row["path"],
            tier=tier,
            page_range=parser_page_range,
        )
        return result

    async def _parse_via_api(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
        privacy: str,
    ) -> tuple[ParseResult, str]:
        """调用已选择的本地或远端 API，保留传输错误及源页号映射语义。"""
        from ...parser.api_client import MinerUApiParser

        base_url, api_key, via = await self._resolve_api_target(privacy, tier)

        # resolve and validate tier against server capabilities
        resolved_tier = self._resolve_tier(tier, via)

        parser = MinerUApiParser(
            api_url=base_url,
            api_key=api_key,
            tier=resolved_tier,
            include_images=Path(file_row["path"]).suffix.lower().lstrip(".") in (OFFICE_EXTENSIONS | OFD_EXTENSIONS),
        )
        try:
            file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")
            parser_page_range = page_range if file_ext in PAGE_RANGE_PARSE_EXTENSIONS else ""
            result = await parser.parse_async(file_row["path"], page_range=parser_page_range)
        except _APITransportError as exc:
            if via == "remote":
                code = "remote_timeout" if exc.timed_out else "remote_unreachable"
                target = "Remote API"
            else:
                code = "parse_server_unavailable"
                target = "Local parse-server"
            raise ParseFailure(
                code,
                f"{target} transport failed during {exc.stage} after {exc.attempts} attempt(s) ({type(exc.cause).__name__}).",
            ) from exc
        if file_ext in PAGE_RANGE_PARSE_EXTENSIONS:
            _remap_api_result_pages_to_page_range(result, page_range)
        return result, via

    def _resolve_tier(self, tier: Tier, via: str) -> Tier:
        """校验目标服务已公布的档位能力，不静默切换用户请求。"""
        health = self.health
        if via == "remote":
            supported = health.remote.probe.tiers
        else:
            supported = health.local.probe.tiers

        if not supported:
            raise ParseFailure("engine_unavailable", "Parse-server health status unknown, cannot validate tier")

        if tier not in supported:
            raise ParseFailure("tier_mismatch", f"Tier '{tier}' not supported by parse-server. Available: {supported}")
        return tier

    async def _resolve_api_target(self, privacy: str, tier: Tier) -> tuple[str, str | None, str]:
        """根据显式隐私设置和健康状态选择目标，保持远端转本地的既有回退。"""
        health = self.health

        if privacy == "remote":
            url = cast(str, await self.config_svc.get("parse_server.remote.url"))
            api_key = (await resolve_remote_api_key(self.config_svc)).value
            if not health.remote.probe.healthy:
                if health.remote.probe.error_code == "invalid_api_key":
                    raise ParseFailure(
                        "invalid_api_key",
                        parse_policy.probe_error_message(
                            "Remote parse-server authentication failed", health.remote.probe.error_msg
                        ),
                        "parse_server.remote.api_key",
                    )
                # try fallback to local
                local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
                if local_mode != "disabled" and health.local.probe.healthy:
                    local_url = _local_parse_server_url(local_mode, health)
                    if local_url:
                        return local_url, await _local_parse_server_api_key(self.config_svc), "local"
                raise ParseFailure("parse_server_unavailable", "Remote parse-server unavailable and no local fallback")
            return url, api_key, "remote"

        # privacy == local
        local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
        if local_mode == "disabled":
            raise ParseFailure("no_engine", "Local parse-server is disabled. Use --tier flash or --remote.")

        local_url = _local_parse_server_url(local_mode, health)
        if local_url is None:
            raise ParseFailure("engine_unavailable", "Local parse-server URL not configured.")

        if not health.local.probe.healthy:
            if local_mode == "self_hosted" and health.local.probe.error_code == "invalid_api_key":
                raise ParseFailure(
                    "invalid_api_key",
                    parse_policy.probe_error_message(
                        "Local self-hosted parse-server authentication failed", health.local.probe.error_msg
                    ),
                    "parse_server.local.self_hosted_api_key",
                )
            raise ParseFailure("engine_unavailable", "Local parse-server is not ready. Please wait or check server status.")

        api_key = await _local_parse_server_api_key(self.config_svc)
        return local_url, api_key, "local"

    async def _fail_task(self, task_id: int, code: str, message: str) -> None:
        """记录执行失败并解除批次锁，保持原有错误码与消息。"""
        now = _now_ms()
        await self.db.execute(
            "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL, updated_at=? WHERE id=?",
            (PARSE_STATUS_FAILED, code, message, now, task_id),
        )

    async def _telemetry_server_dim(self, *, via: str, tier: Tier) -> str:
        """根据实际执行目标和档位生成既有服务维度。"""
        if tier == "flash":
            return "local(flash)"
        if via == "local":
            local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
            if local_mode == "managed":
                return "local(managed)"
            if local_mode == "self_hosted":
                return "local(self-hosted)"
            return "unknown"
        if via == "remote":
            url = cast(str | None, await self.config_svc.get("parse_server.remote.url"))
            return "remote(official)" if _is_official_remote_url(url or "") else "remote(custom)"
        return "unknown"

    async def _record_parse_task_finished(
        self,
        start_ms: int,
        *,
        tier: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        """记录批次结束计数、耗时与错误维度。"""
        dimensions = {"status": status, "tier": tier}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.finished.count", dimensions=dimensions)
        await self._record_duration("parse_task.duration_bucket.count", start_ms, dimensions=dimensions)

    async def _record_execute(
        self,
        start_ms: int,
        *,
        tier: str,
        server: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        """记录解析执行阶段的计数和耗时。"""
        dimensions = {"status": status, "tier": tier, "server": server}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.execute.count", dimensions=dimensions)
        await self._record_duration("parse_task.execute_duration_bucket.count", start_ms, dimensions=dimensions)

    async def _record_write(
        self,
        start_ms: int,
        *,
        tier: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        """记录产物写出阶段的计数和耗时。"""
        dimensions = {"status": status, "tier": tier}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.write.count", dimensions=dimensions)
        await self._record_duration("parse_task.write_duration_bucket.count", start_ms, dimensions=dimensions)

    async def _maybe_update_fts(self, sha256: str, tier: Tier, text: str, file_row: FileRow) -> None:
        """仅以相同或更高档位的正文更新索引，并保留首尾截断。"""
        existing_tier = await self.fts.get_tier(sha256)
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            return  # current FTS data is from a higher or equal tier

        text = fts_utils.truncate_head_tail(text)
        await self.fts.replace(
            sha256=sha256,
            tier=tier,
            text=text,
            title=file_row.get("title") or "",
            author=file_row.get("author") or "",
        )

    async def _update_source_metadata(self, sha256: str, result: ParseResult) -> None:
        """使用原文档属性补齐数据库与已有索引，空字段不清除旧值。"""
        values = metadata_to_doclib(result.middle_json.metadata)
        await self.db.execute(
            "UPDATE docs SET title=COALESCE(?,title), author=COALESCE(?,author), "
            "subject=COALESCE(?,subject), keywords=COALESCE(?,keywords), language=COALESCE(?,language), "
            "updated_at=? WHERE sha256=?",
            (values["title"], values["author"], values["subject"], values["keywords"], values["language"], _now_ms(), sha256),
        )
        await self.db.execute(
            "UPDATE fts_contents SET title=COALESCE((SELECT title FROM docs WHERE sha256=?),''), "
            "author=COALESCE((SELECT author FROM docs WHERE sha256=?),'') WHERE sha256=?",
            (sha256, sha256, sha256),
        )
        if result.middle_json.metadata.file_suffix == "pdf" and values["page_count"] is not None:
            await self.db.execute("UPDATE docs SET page_count=? WHERE sha256=?", (values["page_count"], sha256))

    async def _maybe_update_docs_meta(self, sha256: str, tier: Tier) -> None:
        """按档位更新文档元数据标记并清理已恢复的错误。"""
        doc = await self.db.fetchone("SELECT meta_tier FROM docs WHERE sha256=?", (sha256,))
        if doc is None:
            return
        existing_tier = doc.get("meta_tier")
        now = _now_ms()
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            await self.db.execute(
                "UPDATE docs SET error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
                (now, sha256),
            )
            return

        # for now just update meta_tier; full metadata update comes when
        # engine provides richer output
        await self.db.execute(
            "UPDATE docs SET meta_tier=?, error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
            (tier, now, sha256),
        )

    async def _record_count(self, metric_name: str, *, value: int = 1, dimensions: dict[str, str] | None = None) -> None:
        """通过显式指标接口记录计数，未配置时跳过。"""
        if self.telemetry_svc is not None:
            await self.telemetry_svc.record_count(metric_name, value=value, dimensions=dimensions)

    async def _record_duration(self, metric_name: str, start_ms: int, *, dimensions: dict[str, str] | None = None) -> None:
        """通过显式指标接口记录非负毫秒耗时。"""
        if self.telemetry_svc is not None:
            await self.telemetry_svc.record_duration_bucket(
                metric_name, duration_ms=max(0, _now_ms() - start_ms), dimensions=dimensions
            )


def _remap_api_result_pages_to_page_range(result: ParseResult, page_range: str) -> None:
    """先校验全部映射，再同时更新正文与布局几何的源页号。"""
    if not result.pages:
        return
    requested_page_numbers = sorted(parse_page_range_set(page_range))
    actual_page_numbers = [page.page_idx + 1 for page in result.pages]
    if actual_page_numbers == requested_page_numbers:
        return
    if len(requested_page_numbers) != len(result.pages):
        raise ParseFailure(
            "parse_page_remap_failed",
            (
                "Parse result page count does not match requested page_range: "
                f"requested={page_range}, returned={actual_page_numbers}"
            ),
        )
    from docvortex.document.pdf.layout import LAYOUT_EXTENSION, remap_layout_geometry

    page_index_map = [page_no - 1 for page_no in requested_page_numbers]
    extensions = result.middle_json.extensions
    geometry = None
    if LAYOUT_EXTENSION in extensions:
        try:
            geometry = remap_layout_geometry(extensions[LAYOUT_EXTENSION], page_index_map)
        except (ValueError, TypeError, KeyError) as exc:
            raise ParseFailure(
                "parse_page_remap_failed",
                f"Cannot remap PDF layout geometry for page_range={page_range}: {exc}",
            ) from exc

    # 所有可能失败的映射先完成，避免异常时正文与几何只更新一部分。
    for page, page_idx in zip(result.pages, page_index_map, strict=True):
        page.page_idx = page_idx
    if geometry is not None:
        extensions[LAYOUT_EXTENSION] = geometry


def _local_parse_server_url(mode: str, health: object) -> str | None:
    """按本地运行模式选取既有服务地址。"""
    if mode == "managed":
        return getattr(health, "managed_url", None)
    if mode == "self_hosted":
        return getattr(health, "self_hosted_url", None)
    return None


async def _local_parse_server_api_key(config_svc: ConfigService) -> str | None:
    """读取本地自托管认证配置，托管服务沿用空认证。"""
    return (await config_svc.get("parse_server.local.self_hosted_api_key")) or None


def _is_official_remote_url(url: str) -> bool:
    """仅按官方主机及子域判断遥测目标分类。"""
    host = urlparse(url).hostname or ""
    host = host.lower().rstrip(".")
    return host in {"mineru.net", "mineru.org.cn"} or host.endswith(".mineru.net") or host.endswith(".mineru.org.cn")


__all__ = ["ParseExecutionService"]
