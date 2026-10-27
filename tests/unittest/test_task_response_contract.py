"""跨部署验证逐文件响应与实际档位，阻止内部请求字段再次进入公共协议。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from test_task_api import _WorkerTransport, _archive, _submit, _terminal, _upload

from mineru.integrations.docvortex import build_metadata
from mineru.kit.router import RouterSettings, create_app as create_router
from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson, PageInfo, Tier


def _mixed_inputs() -> list[tuple[str, bytes]]:
    """生成有效 PDF、HTML、DOCX 和 PNG，原生格式执行真实解析和导出。"""
    pdf = io.BytesIO()
    canvas = Canvas(pdf)
    canvas.drawString(60, 720, "Mixed task PDF")
    canvas.save()
    word = io.BytesIO()
    document = Document()
    document.add_paragraph("Mixed task DOCX")
    document.save(word)
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "PNG")
    return [
        ("input.pdf", pdf.getvalue()),
        ("input.html", b"<html><body><p>Mixed task HTML</p></body></html>"),
        ("input.docx", word.getvalue()),
        ("input.png", image.getvalue()),
    ]


def _service(tmp_path: Path, deployment: str) -> FastAPI:
    """创建完整本地或双 worker Router 服务，复用真实上传和后台执行流程。"""
    if deployment == "api":
        return api_server.create_app(upload_dir=str(tmp_path / "api"), tiers=("flash", "basic", "standard", "advanced"))
    workers = {
        name: api_server.create_app(upload_dir=str(tmp_path / name), tiers=("flash", "basic", "standard", "advanced"))
        for name in ("worker-a", "worker-b")
    }
    return create_router(
        RouterSettings(
            upstream_urls=tuple(f"http://{name}" for name in workers), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=_WorkerTransport(workers),
    )


@pytest.mark.parametrize("deployment", ["api", "router"])
@pytest.mark.parametrize(
    "requested,actual",
    [("flash", "flash"), ("basic", "basic"), ("standard", "standard"), ("advanced", "advanced"), ("advanced", "basic")],
)
def test_mixed_file_tiers_come_from_completed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment: str, requested: Tier, actual: Tier
) -> None:
    """混合任务逐文件展示实际档位，最后一组刻意让推理结果与请求不同。"""
    monkeypatch.setattr(api_server, "_preflight_tier_dependencies", lambda *_args, **_kwargs: None)
    original = api_server.parse_async
    calls: list[tuple[str, Tier]] = []

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """仅替代 PDF 和图片推理，原生文档仍走真实解析，记录实际路由参数。"""
        suffix = Path(path).suffix
        calls.append((suffix, kwargs["tier"]))
        if suffix in {".html", ".docx"}:
            return await original(path, **kwargs)
        effort = {"flash": "flash", "basic": "medium", "standard": "high", "advanced": "xhigh"}[actual]
        return ParseResult(
            MiddleJson(
                pages=[PageInfo(page_idx=0)],
                is_full_document=True,
                metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
                extensions=build_metadata(effort=effort, parse_mode="txt"),
            )
        )

    monkeypatch.setattr(api_server, "parse_async", infer)
    inputs = _mixed_inputs()
    with TestClient(_service(tmp_path, deployment)) as client:
        task = _submit(client, inputs, tier=requested)
        assert {"status_url", "tier", "pending_uploads"}.isdisjoint(task)
        for item in task["files"]:
            assert set(item) == {
                "file_id",
                "file_name",
                "bytes",
                "upload",
                "status",
            }
            assert item.get("parse") is None
            assert set(item.get("upload")) == {"url", "headers", "expires_at"}
        _upload(client, task, inputs)
        completed = _terminal(client, task)
        assert completed["status"] == "completed", completed
        assert [item.get("parse")["tier"] for item in completed["files"]] == [actual, "flash", "flash", actual]
        assert dict(calls) == {".pdf": requested, ".html": "flash", ".docx": "flash", ".png": requested}
        for index, item in enumerate(completed["files"]):
            assert item.get("upload") is None
            assert set(item.get("parse")) == {"tier", "duration_ms", "parser_version"}
            assert item.get("parse")["duration_ms"] >= 0
            middle = json.loads(_archive(client, completed, index)["middle_json.json"])
            assert middle["extensions"]["mineru"]["tier"] == item.get("parse")["tier"]
        assert b"Mixed task HTML" in _archive(client, completed, 1)["markdown.md"]
        assert b"Mixed task DOCX" in _archive(client, completed, 2)["markdown.md"]


def test_missing_actual_metadata_does_not_fabricate_requested_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """结果缺少真实档位时解析失败，不能以请求默认值伪造完成统计。"""

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """返回缺少产品扩展的结果，覆盖 worker 生产者违反契约的边界。"""
        return ParseResult(
            MiddleJson(pages=[], metadata={"file_suffix": "html", "producer": {"name": "test", "version": "test"}})
        )

    monkeypatch.setattr(api_server, "parse_async", infer)
    with TestClient(api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",))) as client:
        inputs = [("input.html", b"<p>missing metadata</p>")]
        task = _submit(client, inputs)
        _upload(client, task, inputs)
        result = _terminal(client, task)
        assert result["status"] == "failed"
        assert result["files"][0].get("parse") is None and result["files"][0].get("result_url") is None
