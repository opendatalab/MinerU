"""Doclib 测试显式组装共享入库、队列和执行依赖。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mineru.config import Config, LogConfig, ManagedParseServerConfig, get_config
from mineru.doclib.background.parse_server_health import ParseServerHealthCheck
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.parse_server_state import ParseServerHealth
from mineru.doclib.server import DoclibServer
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.content_read_svc import ContentReadService
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
    health: ParseServerHealth
    config: Config


def create_parse_services(
    db: DatabaseManager,
    fts: FTSManager,
    config_svc: ConfigService,
    data_dir: str,
    *,
    parse_lock_timeout_sec: int,
    telemetry_svc: TelemetryService | None = None,
    execution_type: type[ParseExecutionService] = ParseExecutionService,
    health: ParseServerHealth | None = None,
    config: Config | None = None,
) -> ParseServices:
    """构造共享数据库、索引、配置和指标的平级队列及执行服务。"""
    health = ParseServerHealth() if health is None else health
    config = (get_config() if config is None else config).model_copy(deep=True)
    ingest = IngestService(db, fts, config_svc, health=health, telemetry_svc=telemetry_svc)
    queue = ParseQueueService(
        db,
        fts,
        data_dir,
        ingest_svc=ingest,
        health=health,
        latex_delimiters=config.render.latex_delimiters,
        parse_lock_timeout_sec=parse_lock_timeout_sec,
        telemetry_svc=telemetry_svc,
    )
    execution = execution_type(db, fts, config_svc, data_dir, health=health, telemetry_svc=telemetry_svc)
    return ParseServices(ingest, queue, execution, health, config)


def create_doclib_server(
    state: Any, *, title: str = "MinerU Doclib API", config: Config | None = None, health: ParseServerHealth | None = None
) -> DoclibServer:
    """测试组装边界显式选取配置和健康对象，不在生产服务中隐式回落。"""
    from mineru.config import get_config
    from mineru.doclib.parse_server_state import ParseServerHealth
    from mineru.doclib.server import DoclibServer

    return DoclibServer(
        state,
        title=title,
        config=(get_config() if config is None else config).model_copy(deep=True),
        health=ParseServerHealth() if health is None else health,
    )


def create_content_read_service(db: DatabaseManager, *, data_dir: str, config: Config | None = None) -> ContentReadService:
    """为内容读取测试提供显式公式配置。"""
    from mineru.config import get_config
    from mineru.doclib.services.content_read_svc import ContentReadService

    cfg = get_config() if config is None else config
    return ContentReadService(db, data_dir=data_dir, latex_delimiters=cfg.render.latex_delimiters)


def create_health_check(
    config_svc: ConfigService,
    *,
    interval_sec: int,
    probe_timeout_sec: int,
    startup_grace_sec: int,
    stop_timeout_sec: int,
    health: ParseServerHealth | None = None,
    startup_timeout_sec: int | None = None,
    managed_parse_server: ManagedParseServerConfig | None = None,
    log_cfg: LogConfig | None = None,
) -> ParseServerHealthCheck:
    """为探测测试组装完整配置，测试替身无需依赖模块级健康对象。"""
    from mineru.config import get_config
    from mineru.doclib.background.parse_server_health import ParseServerHealthCheck
    from mineru.doclib.parse_server_state import ParseServerHealth

    cfg = get_config().doclib
    return ParseServerHealthCheck(
        config_svc,
        interval_sec=interval_sec,
        probe_timeout_sec=probe_timeout_sec,
        startup_grace_sec=startup_grace_sec,
        stop_timeout_sec=stop_timeout_sec,
        health=ParseServerHealth() if health is None else health,
        startup_timeout_sec=cfg.parse_server_startup_timeout_sec if startup_timeout_sec is None else startup_timeout_sec,
        managed_parse_server=cfg.managed_parse_server if managed_parse_server is None else managed_parse_server,
        log_cfg=cfg.log if log_cfg is None else log_cfg,
    )
