"""当前 Router 进程的远程 upstream 管理接口，与解析凭证和下层管理操作隔离。"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends, Response, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, field_validator

from ...types import Tier
from .proxy import RouterProxyError
from .resources import ResourceRegistry
from .workers import WorkerPool, WorkerState, normalize_base_url


class RegisterWorkerRequest(BaseModel):
    """注册输入只接受 base URL，不接受下层凭证或能力覆盖。"""

    model_config = ConfigDict(extra="forbid")
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """与启动配置共用 URL 校验和去重规则。"""
        return normalize_base_url(value)


class WorkerResponse(BaseModel):
    """提供运维判断可用性和注销进度所需的最小节点快照。"""

    worker_id: str
    url: str
    state: Literal["active", "draining"]
    healthy: bool
    tiers: list[Tier]
    active_jobs: int
    retained_routes: int


class WorkerListResponse(BaseModel):
    """返回本层远程节点，不递归展开下层 Router 的节点或本地托管进程。"""

    object: Literal["list"] = "list"
    data: list[WorkerResponse]


def build_admin_router(
    pool: WorkerPool,
    registry: ResourceRegistry,
    *,
    api_key: str,
    collect_resources: Callable[[], None],
) -> APIRouter:
    """为显式启用管理凭证的 Router 构建独立认证及成员管理路由。"""
    bearer = HTTPBearer(auto_error=False, scheme_name="RouterAdmin", description="Independent Router management key")

    async def require_admin(credentials: HTTPAuthorizationCredentials | None = Security(bearer)) -> None:
        """只校验本层管理凭证，不调用或携带该凭证探测任何 upstream。"""
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), api_key.encode()):
            raise RouterProxyError(401, "invalid_api_key", "Invalid Router management API key")

    router = APIRouter(prefix="/v1/router/workers", tags=["Router management"], dependencies=[Depends(require_admin)])

    def describe(worker: WorkerState) -> WorkerResponse:
        """只返回公共管理字段，避免序列化内部连接、启动设置和认证信息。"""
        return WorkerResponse(
            worker_id=worker.worker_id,
            url=worker.base_url,
            state=worker.state,
            healthy=worker.healthy,
            tiers=sorted(worker.tiers),
            active_jobs=worker.active_jobs,
            retained_routes=registry.worker_route_count(worker.worker_id),
        )

    @router.get("", response_model=WorkerListResponse)
    async def list_workers() -> WorkerListResponse:
        """列出启动参数与运行期注册的所有远程节点，包括 draining 和不健康节点。"""
        return WorkerListResponse(data=[describe(worker) for worker in pool.workers if worker.source == "remote"])

    @router.post("", response_model=WorkerResponse, status_code=201, responses={200: {"model": WorkerResponse}})
    async def register_worker(body: RegisterWorkerRequest, response: Response) -> WorkerResponse:
        """先同步登记或恢复节点，再用无管理凭证的现有探测更新其健康与能力。"""
        worker, created = pool.register_remote(body.url)
        response.status_code = 201 if created else 200
        await pool.refresh(worker)
        return describe(worker)

    @router.delete(
        "/{worker_id}",
        response_model=WorkerResponse,
        status_code=202,
        responses={204: {"description": "Upstream removed after all references were released"}},
    )
    async def unregister_worker(worker_id: str) -> WorkerResponse | Response:
        """停止新分配后检查全部引用；不取消任务、不缩短保留期、不管理下层节点。"""
        try:
            worker = pool.drain_remote(worker_id)
        except KeyError as exc:
            raise RouterProxyError(404, "worker_not_found", "Remote worker not found", "worker_id") from exc
        collect_resources()
        if worker not in pool.workers:
            return Response(status_code=204)
        return describe(worker)

    return router


__all__ = ["RegisterWorkerRequest", "WorkerListResponse", "WorkerResponse", "build_admin_router"]
