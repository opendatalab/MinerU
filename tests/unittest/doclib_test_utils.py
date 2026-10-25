"""Doclib 测试显式组装共享入库、队列和执行依赖。"""

from __future__ import annotations

from dataclasses import dataclass

from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.ingest_svc import IngestService
from mineru.doclib.services.parse_execution_svc import ParseExecutionService
from mineru.doclib.services.parse_queue_svc import ParseQueueService
from mineru.doclib.telemetry import TelemetryService


@dataclass
class ParseServices:
    """只保存真实服务对象，不代理已经移除的组合服务接口。"""

    ingest: IngestService
    queue: ParseQueueService
    execution: ParseExecutionService


def create_parse_services(
    db: DatabaseManager,
    fts: FTSManager,
    config_svc: ConfigService,
    data_dir: str,
    *,
    parse_lock_timeout_sec: int,
    telemetry_svc: TelemetryService | None = None,
    execution_type: type[ParseExecutionService] = ParseExecutionService,
) -> ParseServices:
    """构造共享数据库、索引、配置和指标的平级队列及执行服务。"""
    ingest = IngestService(db, fts, config_svc, telemetry_svc=telemetry_svc)
    queue = ParseQueueService(
        db, fts, data_dir, ingest_svc=ingest, parse_lock_timeout_sec=parse_lock_timeout_sec, telemetry_svc=telemetry_svc
    )
    execution = execution_type(db, fts, config_svc, data_dir, telemetry_svc=telemetry_svc)
    return ParseServices(ingest, queue, execution)
