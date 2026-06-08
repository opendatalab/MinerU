import os
import platform
import socket
import tempfile

from pydantic import BaseModel


def _uds_available() -> bool:
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
        return True
    except Exception:
        return False


def _default_uds_path() -> str:
    system = platform.system().lower()
    if system not in ("windows", "darwin", "linux"):
        raise RuntimeError(f"System [{system}] is not supported.")
    if not _uds_available():
        raise RuntimeError("Unix domain socket is not available.")
    TEMP = tempfile.gettempdir() if system == "windows" else "/tmp"
    return os.path.join(TEMP, "mineru.sock")


def _default_data_path() -> str:
    HOME = os.path.expanduser("~")
    return os.path.join(HOME, "MinerU")


def _default_log_path() -> str:
    HOME = os.path.expanduser("~")
    return os.path.join(HOME, "MinerU", "mineru.log")


def _default_db_path() -> str:
    HOME = os.path.expanduser("~")
    return os.path.join(HOME, "MinerU", "mineru.db")


class UDSConfig(BaseModel):
    path: str = _default_uds_path()
    permission: int = 0o600


class HTTPConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 15980
    strict_port: bool = False
    backlog: int = 128
    timeout: int = 600


class LogConfig(BaseModel):
    path: str = _default_log_path()
    level: str = "info"
    # max_bytes: 5242880         # 5MB 轮转
    # backup_count: 3


class ServerConfig(BaseModel):
    uds: UDSConfig = UDSConfig()
    http: HTTPConfig = HTTPConfig()
    log: LogConfig = LogConfig()
    data_dir: str = _default_data_path()
    ingest_workers: int = 2
    parse_workers: int = 2
    compaction_interval_sec: int = 3600


class SQLiteConfig(BaseModel):
    path: str = _default_db_path()
    mmap_size: int = 268435456  # 256 MB
    cache_size: int = -20000  # 20 MB (negative = KB)
    wal_autocheckpoint: int = 1000
    journal_size_limit: int = 33_554_432  # 32 MB
    temp_store: str = "memory"
    synchronous: str = "NORMAL"


class Config(BaseModel):
    """Configurations mainly needed before server was started."""

    server: ServerConfig = ServerConfig()
    sqlite: SQLiteConfig = SQLiteConfig()
