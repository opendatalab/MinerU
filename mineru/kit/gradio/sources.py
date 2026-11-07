"""上传级输入准备及文件租期；解析结果不拥有第二份预览文档。"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...filetypes import IMAGE_EXTENSIONS, PDF_EXTENSIONS
from .conversion import await_task_completion, run_sync_output

if TYPE_CHECKING:
    from .document_io import GradioDocumentIO


@dataclass(frozen=True)
class PreparedSource:
    """只保存不可变路径和原始身份，允许宿主通过 CPU sidecar 序列化传递。"""

    source_id: str
    root: Path
    original_path: Path
    source_path: Path
    parse_path: Path
    preview_pdf_path: Path | None
    file_suffix: str

    def as_state(self) -> dict[str, str]:
        """转成纯字符串记录，不传递 PDFium 句柄或文件内容。"""
        return {key: str(value) if value is not None else "" for key, value in asdict(self).items()}

    @classmethod
    def from_state(cls, value: dict[str, Any], *, source_root: Path) -> PreparedSource:
        """限定准备目录身份及其所有派生文件，原始上传路径只作为身份使用。"""
        source_id = value["source_id"]
        if not isinstance(source_id, str) or uuid.UUID(source_id).hex != source_id:
            raise ValueError("Invalid prepared source identity")
        root = Path(value["root"]).resolve()
        if root != source_root.resolve() / source_id:
            raise ValueError("Prepared source escapes source root")
        paths = {key: Path(value[key]).resolve() for key in ("source_path", "parse_path")}
        preview = Path(value["preview_pdf_path"]).resolve() if value["preview_pdf_path"] else None
        for path in (*paths.values(), *([preview] if preview else [])):
            if not path.is_relative_to(root):
                raise ValueError("Prepared file escapes source directory")
        return cls(
            source_id, root, Path(value["original_path"]), **paths, preview_pdf_path=preview, file_suffix=value["file_suffix"]
        )


def prepare_source(path: Path, *, source_id: str, output_root: Path) -> PreparedSource:
    """保存稳定输入；图片仅在此转换一次，预览和解析共用同一 PDF 字节。"""
    if uuid.UUID(source_id).hex != source_id:
        raise ValueError("Invalid prepared source identity")
    path = path.resolve()
    root = output_root.resolve() / "gradio" / "sources" / source_id
    root.mkdir(parents=True, exist_ok=False)
    try:
        source = root / path.name
        shutil.copyfile(path, source)
        suffix = path.suffix.lower().lstrip(".")
        preview = source if suffix in PDF_EXTENSIONS else None
        if suffix in IMAGE_EXTENSIONS:
            from docvortex.document.pdf import PDFDocument

            preview = root / f"{path.stem}.pdf"
            with PDFDocument.from_image(source.read_bytes()) as document:
                preview.write_bytes(document.bytes)
        return PreparedSource(source_id, root, path, source, preview or source, preview, suffix)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


@dataclass
class _SourceLease:
    """准备任务和解析共享租期，失效后只清理无成功结果引用的文件。"""

    original_path: Path
    task: asyncio.Task[PreparedSource]
    users: int = 0
    retained: bool = False
    retired: bool = False


class SessionSources:
    """按应用、会话和选择身份隔离准备任务，合并同一输入的并发准备。"""

    def __init__(self, document_io: GradioDocumentIO, output_root: Path) -> None:
        """只初始化注册表，实际文件操作均在显式准备和释放时执行。"""
        self.document_io = document_io
        self.output_root = output_root
        self.current: dict[str, tuple[str, _SourceLease]] = {}
        self.revisions: dict[str, int] = {}
        self.cleanups: set[asyncio.Task[None]] = set()

    async def prepare(self, session: str, path: Path, source_id: str, revision: int) -> PreparedSource:
        """接收单调选择修订号，拒绝晚到准备请求并复用本轮已创建的任务。"""
        path = path.resolve()
        previous = self.current.get(session)
        if previous and previous[0] == source_id:
            if previous[1].original_path != path:
                raise ValueError("Source identity does not match uploaded file")
            return await asyncio.shield(previous[1].task)
        if revision <= self.revisions.get(session, 0):
            raise ValueError("Source preparation has expired")
        self.revisions[session] = revision
        if previous:
            previous[1].retired = True
        task = asyncio.create_task(
            run_sync_output(self.document_io.prepare_source, path, source_id=source_id, output_root=self.output_root)
        )
        lease = _SourceLease(path, task)
        self.current[session] = (source_id, lease)
        try:
            source = await asyncio.shield(task)
            if lease.retired:
                raise ValueError("Source preparation has expired")
            return source
        finally:
            if previous:
                self._schedule_cleanup(previous[1])
            await self._cleanup(lease)

    def _schedule_cleanup(self, lease: _SourceLease) -> None:
        """旧同步准备在后台安全退出，不阻塞已经就绪的新文件回执。"""
        task = asyncio.create_task(self._cleanup(lease))
        self.cleanups.add(task)
        task.add_done_callback(self.cleanups.discard)

    async def acquire(self, session: str, path: Path, *, source_id: str = "") -> tuple[PreparedSource, _SourceLease]:
        """浏览器使用预检记录；公开 API 在转换前走同一准备流程。"""
        current = self.current.get(session)
        if source_id and (not current or current[0] != source_id or current[1].original_path != path.resolve()):
            raise ValueError("Source preparation has expired")
        if not current or current[1].original_path != path.resolve():
            await self.prepare(session, path, uuid.uuid4().hex, self.revisions.get(session, 0) + 1)
        lease = self.current[session][1]
        lease.users += 1
        try:
            prepared = await asyncio.shield(lease.task)
            if lease.retired:
                raise ValueError("Source preparation has expired")
            return prepared, lease
        except BaseException:
            await self.release(lease)
            raise

    async def release(self, lease: _SourceLease, *, retain: bool = False) -> None:
        """成功结果保留其文件引用；最后一个使用者退出后才清理过期输入。"""
        lease.retained |= retain
        lease.users -= 1
        await self._cleanup(lease)

    async def retire(self, session: str, *, revision: int | None = None) -> None:
        """清除和关闭时使准备失效，不强制终止正在运行的同步转换。"""
        if revision is not None:
            if revision <= self.revisions.get(session, 0):
                return
            self.revisions[session] = revision
        previous = self.current.pop(session, None)
        if previous:
            previous[1].retired = True
            await self._cleanup(previous[1])

    async def _cleanup(self, lease: _SourceLease) -> None:
        """等待同步准备真正结束后回收未引用目录，失败任务已自行清理。"""
        if not lease.retired or lease.retained or lease.users:
            return
        try:
            source = await await_task_completion(lease.task)
        except (Exception, asyncio.CancelledError):
            return
        if not lease.users and not lease.retained:
            await asyncio.to_thread(shutil.rmtree, source.root, True)


__all__ = ["PreparedSource", "SessionSources", "prepare_source"]
