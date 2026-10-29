"""Tasks 默认选档、原生格式归一化和逐跳档位固定的跨部署回归。"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from test_task_response_contract import _mixed_inputs

from mineru.integrations.docvortex import build_metadata
from mineru.kit.router import RouterSettings, create_app as create_router
from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.parser.tier import runtime_options_for_tier
from mineru.tasks.contracts import TaskFileInput, UploadTaskOptions
from mineru.tasks.errors import ApiServerError
from mineru.tasks.service import TERMINAL_TASK_STATUSES, resolve_task_tier
from mineru.types import TIERS, MiddleJson, PageInfo, Tier, select_default_task_tier


class _RecordingTransport(httpx.AsyncBaseTransport):
    """使用真实 ASGI 服务，并记录 Router 在每一跳发送的文件档位。"""

    def __init__(self, apps: dict[str, FastAPI], submissions: list[tuple[str, dict[str, Any]]]) -> None:
        """共享提交记录，应用生命周期由测试统一管理。"""
        self.transports = {name: httpx.ASGITransport(app=app) for name, app in apps.items()}
        self.submissions = submissions

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """按主机转发真实协议，保留请求正文供逐文件断言使用。"""
        if request.method == "POST" and request.url.path == "/v1/tasks":
            self.submissions.append((request.url.host, json.loads(await request.aread())))
        return await self.transports[request.url.host].handle_async_request(request)


def _set_tiers(app: FastAPI, tiers: tuple[Tier, ...]) -> None:
    """显式设置测试部署能力，避免初始化真实推理模型或依赖启动档位展开。"""
    app.state.tier_runtime_options = {tier: runtime_options_for_tier(tier) for tier in tiers}
    app.state.tiers = api_server._tiers_for_server_tiers(list(tiers))
    app.state.default_tier = select_default_task_tier(tiers)


@asynccontextmanager
async def _deployment(
    tmp_path: Path, deployment: str, capabilities: dict[str, tuple[Tier, ...]]
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI, list[FastAPI], list[tuple[str, dict[str, Any]]]]]:
    """在同一事件循环启动本地、Router 或级联服务，确保任务完整关闭。"""
    submissions: list[tuple[str, dict[str, Any]]] = []
    apps: list[FastAPI] = []
    async with AsyncExitStack() as stack:
        workers: dict[str, FastAPI] = {}
        for name, tiers in capabilities.items():
            with patch.object(api_server, "ensure_tier_runtime_dependencies"):
                worker = api_server.create_app(upload_dir=str(tmp_path / name), tiers=tiers)
            await stack.enter_async_context(worker.router.lifespan_context(worker))
            workers[name] = worker
            apps.append(worker)
        app = next(iter(workers.values()))
        if deployment != "api":
            app = create_router(
                RouterSettings(
                    upstream_urls=tuple(f"http://{name}" for name in workers),
                    local_gpus="none",
                    worker_refresh_interval_seconds=0,
                ),
                transport=_RecordingTransport(workers, submissions),
            )
            await stack.enter_async_context(app.router.lifespan_context(app))
            apps.append(app)
        if deployment == "cascade":
            app = create_router(
                RouterSettings(upstream_urls=("http://inner",), local_gpus="none", worker_refresh_interval_seconds=0),
                transport=_RecordingTransport({"inner": app}, submissions),
            )
            await stack.enter_async_context(app.router.lifespan_context(app))
            apps.append(app)
        client = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://entry")
        )
        yield client, app, apps, submissions


def _request_body(inputs: list[tuple[str, bytes]], options: dict[str, Any]) -> dict[str, Any]:
    """保留 tier 缺省与 null 的区别，避免通用测试助手自动补上 flash。"""
    return {"files": [{"file_name": name, "bytes": len(data)} for name, data in inputs], **options}


async def _complete(client: httpx.AsyncClient, task: dict[str, Any], inputs: list[tuple[str, bytes]]) -> dict[str, Any]:
    """上传真实源文件并等待终态，执行和结果获取均经过公共接口。"""
    for item, (_name, data) in zip(task["files"], inputs):
        response = await client.put(item["upload"]["url"], content=data, headers=item["upload"]["headers"])
        assert response.status_code == 200, response.text
    async with asyncio.timeout(20):
        while True:
            response = await client.get(f"/v1/tasks/{task['task_id']}")
            assert response.status_code == 200, response.text
            result = response.json()
            if result["status"] in TERMINAL_TASK_STATUSES:
                return result
            await asyncio.sleep(0.01)


async def _archive(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, bytes]:
    """下载并读取真实产物，用元数据和文本验证最终执行路径。"""
    response = await client.get(item["result_url"])
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _stub_inference(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Tier]]:
    """仅替代 PDF/图片推理，原生文档继续使用真实解析和导出。"""
    original = api_server.parse_async
    calls: list[tuple[str, Tier]] = []

    async def infer(path: str, **kwargs: Any) -> ParseResult:
        """记录实际入参并生成对应档位结果，不将请求值填入响应层。"""
        suffix = Path(path).suffix
        tier: Tier = kwargs["tier"]
        calls.append((suffix, tier))
        if suffix not in {".pdf", ".png"}:
            return await original(path, **kwargs)
        effort = {"flash": "flash", "basic": "medium", "standard": "high", "advanced": "xhigh"}[tier]
        return ParseResult(
            MiddleJson(
                pages=[PageInfo(page_idx=0)],
                is_full_document=True,
                metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
                extensions=build_metadata(effort=effort, parse_mode="txt"),
            )
        )

    monkeypatch.setattr(api_server, "parse_async", infer)
    return calls


@pytest.mark.parametrize(
    "tiers,expected",
    [
        (TIERS, "standard"),
        (("standard",), "standard"),
        (("basic", "standard"), "standard"),
        (("basic", "advanced"), "basic"),
        (("flash", "basic"), "basic"),
        (("flash",), "flash"),
        (("flash", "advanced"), "flash"),
        (("advanced",), "advanced"),
        ((), None),
    ],
)
def test_task_default_tier_matrix(tiers: tuple[Tier, ...], expected: Tier | None) -> None:
    """覆盖非连续能力集合，仅剩 Advanced 时才默认选择该档位。"""
    inputs = [TaskFileInput(file_name="input.pdf", bytes=1)]
    if expected is None:
        with pytest.raises(ApiServerError) as error:
            resolve_task_tier(UploadTaskOptions(), inputs, set(tiers))
        assert error.value.status_code == 503
        assert error.value.error.code == "quality_tier_unavailable"
        assert "No parsing tier is available" in str(error.value)
    else:
        assert resolve_task_tier(UploadTaskOptions(), inputs, set(tiers)) == expected


@pytest.mark.parametrize("tier", TIERS)
def test_task_explicit_tier_is_not_upgraded_or_downgraded(tier: Tier) -> None:
    """显式档位只校验是否存在，原生文档则固定归一为 flash。"""
    inputs = [TaskFileInput(file_name="input.pdf", bytes=1)]
    assert resolve_task_tier(UploadTaskOptions(tier=tier), inputs, set(TIERS)) == tier
    assert resolve_task_tier(UploadTaskOptions(tier=tier), inputs, {tier}) == tier
    with pytest.raises(ApiServerError) as error:
        resolve_task_tier(UploadTaskOptions(tier=tier), inputs, set(TIERS) - {tier})
    assert error.value.status_code == 400 and error.value.error.param == "tier"
    native = [TaskFileInput(file_name="input.docx", bytes=1)]
    assert resolve_task_tier(UploadTaskOptions(tier=tier), native, {"flash"}) == "flash"


@pytest.mark.parametrize("deployment", ["api", "router"])
@pytest.mark.parametrize(
    "tiers,options,names,status,param",
    [
        (("flash",), {}, ["input.pdf", "input.docx"], 202, None),
        (("flash",), {"tier": None}, ["input.pdf"], 202, None),
        (("flash",), {"tier": "advanced"}, ["input.pdf", "input.docx"], 400, "tier"),
        (("flash",), {"tier": ""}, ["input.docx"], 400, "tier"),
        (("flash",), {"tier": "unknown"}, ["input.docx"], 400, "tier"),
        (("advanced",), {}, ["input.pdf"], 202, None),
        (("standard",), {}, ["input.pdf", "input.docx"], 202, None),
        (("standard",), {"tier": "flash"}, ["input.pdf", "input.docx"], 400, "tier"),
        (("standard",), {"tier": "basic"}, ["input.png"], 400, "tier"),
        (("standard",), {"tier": "advanced"}, ["input.pdf"], 400, "tier"),
    ],
)
def test_task_tier_preflight_and_schema(
    tmp_path: Path,
    deployment: str,
    tiers: tuple[Tier, ...],
    options: dict[str, Any],
    names: list[str],
    status: int,
    param: str | None,
) -> None:
    """真实创建接口在分配任务和上传链接前拒绝非法或不可执行的请求。"""

    async def scenario() -> None:
        """同时检查运行时行为和 OpenAPI 的可选字段描述。"""
        async with _deployment(tmp_path, deployment, {"worker": tiers}) as (client, app, apps, submissions):
            schema = (await client.get("/openapi.json")).json()
            request = schema["paths"]["/v1/tasks"]["post"]["requestBody"]["content"]["application/json"]["schema"]
            assert "tier" not in request["required"]
            assert "standard, basic, flash" in request["properties"]["tier"]["description"]
            response = await client.post("/v1/tasks", json=_request_body([(name, b"x") for name in names], options))
            assert response.status_code == status, response.text
            if status == 202:
                record = app.state.task_store._records[response.json()["task_id"]]
                assert record.options.tier == select_default_task_tier(tiers)
            else:
                assert "type" not in response.json()["error"]
                if param is None:
                    assert "param" not in response.json()["error"]
                else:
                    assert response.json()["error"]["param"] == param
                assert response.json()["error"]["code"] == ("quality_tier_unavailable" if status == 503 else "invalid_request")
                assert all(not service.state.task_store._records for service in apps)
                assert not submissions
                assert "files" not in response.json()

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
@pytest.mark.parametrize(
    "tiers,expected",
    [(("advanced", "flash"), "flash"), (("advanced", "basic"), "basic"), (("advanced",), "advanced")],
)
@pytest.mark.parametrize("options", [{}, {"tier": None}])
def test_non_contiguous_default_tier_across_deployments(
    tmp_path: Path, deployment: str, tiers: tuple[Tier, ...], expected: Tier, options: dict[str, None]
) -> None:
    """真实 API 与两层 Router 接单时共享缺档默认顺序，不把 Advanced 提前。"""

    async def scenario() -> None:
        """通过公开接口创建任务并检查持久到接单记录中的实际档位。"""
        async with _deployment(tmp_path, deployment, {"worker": tiers}) as (client, app, _apps, _calls):
            response = await client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
            assert response.status_code == 202, response.text
            assert app.state.task_store._records[response.json()["task_id"]].options.tier == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
@pytest.mark.parametrize(
    "options,expected", [({}, "standard"), ({"tier": None}, "standard"), ({"tier": "advanced"}, "advanced")]
)
def test_mixed_task_tiers_across_deployments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment: str, options: dict[str, Any], expected: Tier
) -> None:
    """异构 worker 和级联 Router 对混合文件保留同一选档，原生产物执行真实解析。"""
    calls = _stub_inference(monkeypatch)

    async def scenario() -> None:
        """检查接单记录、各跳请求、推理参数和最终产物中的档位。"""
        capabilities = (
            {"worker": ("standard", "advanced")}
            if deployment == "api"
            else {"quality": ("standard", "advanced"), "native": ("basic",)}
        )
        async with _deployment(tmp_path, deployment, capabilities) as (client, app, apps, submissions):
            inputs = _mixed_inputs()
            response = await client.post("/v1/tasks", json=_request_body(inputs, options))
            assert response.status_code == 202, response.text
            task = response.json()
            assert app.state.task_store._records[task["task_id"]].options.tier == expected
            result = await _complete(client, task, inputs)
            assert result["status"] == "completed", result
            assert "tier" not in result
            assert dict(calls) == {".pdf": expected, ".png": expected, ".html": "flash", ".docx": "flash"}
            assert [item["parse"]["tier"] for item in result["files"]] == [expected, "flash", "flash", expected]
            for index, item in enumerate(result["files"]):
                archive = await _archive(client, item)
                middle = json.loads(archive["middle_json.json"])
                assert middle["extensions"]["mineru"]["tier"] == item["parse"]["tier"]
                if index in {1, 2}:
                    assert (b"Mixed task HTML" if index == 1 else b"Mixed task DOCX") in archive["markdown.md"]
            assert len(submissions) == {"api": 0, "router": 4, "cascade": 8}[deployment]
            for host, body in submissions:
                native = Path(body["files"][0]["file_name"]).suffix in {".html", ".docx"}
                assert body.get("tier") == (None if native else expected)
                assert host == "inner" or host in ({"native", "quality"} if native else {"quality"})
            for service in apps:
                if hasattr(service.state, "execution_store"):
                    for execution in service.state.execution_store._executions.values():
                        native = Path(execution.request.file_name).suffix in {".html", ".docx"}
                        assert execution.request.tier == ("flash" if native else expected)

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
def test_native_only_task_ignores_unavailable_explicit_tier(tmp_path: Path, deployment: str) -> None:
    """纯原生文档携带未开放的 advanced 时仍按 flash 完成，并保存归一化选项。"""

    async def scenario() -> None:
        """使用真实 HTML 和 DOCX 验证单一 flash 部署的参数作用域。"""
        async with _deployment(tmp_path, deployment, {"native": ("standard",)}) as (client, app, _apps, submissions):
            inputs = _mixed_inputs()[1:3]
            response = await client.post("/v1/tasks", json=_request_body(inputs, {"tier": "advanced"}))
            assert response.status_code == 202, response.text
            task = response.json()
            assert app.state.task_store._records[task["task_id"]].options.tier == "flash"
            result = await _complete(client, task, inputs)
            assert result["status"] == "completed", result
            assert all(item["parse"]["tier"] == "flash" for item in result["files"])
            assert all("tier" not in body for _host, body in submissions)

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router", "cascade"])
@pytest.mark.parametrize("remove_selected", [False, True])
def test_accepted_task_keeps_tier_when_capabilities_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deployment: str, remove_selected: bool
) -> None:
    """接单后的能力变化不重新选档；原档位失效时按文件失败而不自动切换。"""
    calls = _stub_inference(monkeypatch)

    async def scenario() -> None:
        """在等待上传期间增加 standard 或移除 basic，复现后续转发的选择边界。"""
        async with _deployment(tmp_path, deployment, {"worker": ("flash", "basic")}) as (client, app, apps, _submissions):
            inputs = [_mixed_inputs()[0]]
            response = await client.post("/v1/tasks", json=_request_body(inputs, {}))
            assert response.status_code == 202, response.text
            task = response.json()
            record = app.state.task_store._records[task["task_id"]]
            assert record.options.tier == "basic"
            changed: tuple[Tier, ...] = ("flash", "standard") if remove_selected else ("flash", "basic", "standard")
            for service in apps:
                if hasattr(service.state, "execution_store"):
                    _set_tiers(service, changed)
                else:
                    await service.state.worker_pool.refresh_all()
            result = await _complete(client, task, inputs)
            assert record.options.tier == "basic"
            assert result["status"] == ("failed" if remove_selected else "completed"), result
            if remove_selected:
                assert not calls and "parse" not in result["files"][0]
            else:
                assert calls == [(".pdf", "basic")]
                assert result["files"][0]["parse"]["tier"] == "basic"

    asyncio.run(scenario())


@pytest.mark.parametrize("deployment", ["api", "router"])
def test_flash_only_server_parses_real_pdf_without_tier(tmp_path: Path, deployment: str) -> None:
    """真实文本 PDF 不指定 tier 即可完成 flash 解析、轮询和产物下载。"""

    async def scenario() -> None:
        """检查下载后的正文和产品元数据，不替换 PDF 解析入口。"""
        async with _deployment(tmp_path, deployment, {"worker": ("flash",)}) as (client, _app, _apps, _submissions):
            inputs = [_mixed_inputs()[0]]
            response = await client.post("/v1/tasks", json=_request_body(inputs, {"ocr_mode": "txt"}))
            assert response.status_code == 202, response.text
            result = await _complete(client, response.json(), inputs)
            assert result["status"] == "completed", result
            item = result["files"][0]
            assert item["parse"]["tier"] == "flash"
            archive = await _archive(client, item)
            assert b"Mixed task PDF" in archive["markdown.md"]
            assert json.loads(archive["middle_json.json"])["extensions"]["mineru"]["tier"] == "flash"

    asyncio.run(scenario())
