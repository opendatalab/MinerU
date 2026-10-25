"""Doclib 测试显式组装共享入库与解析依赖。"""

from __future__ import annotations

from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.ingest_svc import IngestService
from mineru.doclib.services.parse_svc import ParseService
from mineru.doclib.telemetry import TelemetryService


def create_parse_service(
    db: DatabaseManager,
    fts: FTSManager,
    config_svc: ConfigService,
    data_dir: str,
    *,
    parse_lock_timeout_sec: int,
    telemetry_svc: TelemetryService | None = None,
    service_type: type[ParseService] = ParseService,
) -> ParseService:
    """为测试构造共享数据库、索引、规则和指标的两个真实服务。"""
    ingest_svc = IngestService(db, fts, config_svc, telemetry_svc=telemetry_svc)
    return service_type(
        db,
        fts,
        config_svc,
        data_dir,
        ingest_svc=ingest_svc,
        parse_lock_timeout_sec=parse_lock_timeout_sec,
        telemetry_svc=telemetry_svc,
    )
