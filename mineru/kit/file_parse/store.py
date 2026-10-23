"""独立于 Doclib 的 SQLite 恢复记录与临时文件管理。"""

from __future__ import annotations

import asyncio
import json

import aiosqlite
from filelock import FileLock

from .models import Record
from .settings import FileParseSettings


class RecordStore:
    """每个状态目录只允许一个进程使用，各上游配置拥有独立数据空间。"""

    def __init__(self, settings: FileParseSettings) -> None:
        """只保存路径和锁对象，实际创建资源由应用启动阶段负责。"""
        assert settings.state_dir is not None
        self.root = settings.state_dir
        self.directory = self.root / settings.namespace
        self.sources = self.directory / "sources"
        self.results = self.directory / "results"
        self.lock = FileLock(str(self.root / ".lock"), thread_local=False)
        self.connection: aiosqlite.Connection | None = None
        self._writes = asyncio.Lock()

    async def open(self) -> list[Record]:
        """获取独占锁并加载记录，启动时清理上次进程遗留的结果临时文件。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock.acquire(timeout=0)
        try:
            self.sources.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)
            for path in self.results.iterdir():
                if path.is_file():
                    path.unlink()
            self.connection = await aiosqlite.connect(self.directory / "requests.sqlite3")
            await self.connection.execute("PRAGMA journal_mode=WAL")
            await self.connection.execute("PRAGMA synchronous=FULL")
            await self.connection.execute(
                "CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            await self.connection.commit()
            async with self.connection.execute("SELECT data FROM requests") as cursor:
                records = [Record(**json.loads(row[0])) for row in await cursor.fetchall()]
            retained = {record.source_path for record in records if record.source_path}
            for path in self.sources.iterdir():
                if path.is_file() and str(path) not in retained:
                    path.unlink()
            return records
        except BaseException:
            await self.close()
            raise

    async def save(self, record: Record) -> None:
        """原子提交单个记录快照，任务 ID 确认后立即落盘。"""
        assert self.connection is not None
        data = record.to_json()
        async with self._writes:
            await self.connection.execute(
                "INSERT INTO requests(request_id, data) VALUES (?, ?) ON CONFLICT(request_id) DO UPDATE SET data=excluded.data",
                (record.request_id, data),
            )
            await self.connection.commit()

    async def delete(self, request_id: str) -> None:
        """回收过期恢复记录，保留其他请求及其他上游空间。"""
        assert self.connection is not None
        async with self._writes:
            await self.connection.execute("DELETE FROM requests WHERE request_id=?", (request_id,))
            await self.connection.commit()

    async def close(self) -> None:
        """关闭数据库后释放目录独占锁，允许下次启动恢复。"""
        try:
            if self.connection is not None:
                await self.connection.close()
                self.connection = None
        finally:
            self.lock.release()


__all__ = ["RecordStore"]
