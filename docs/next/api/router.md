# Router 远程 upstream 管理

实现：`mineru/kit/router/admin.py`、`workers.py`、`resources.py`。该接口是 Router 的可选部署管理能力，不属于普通 API Server 或 SaaS 必须实现的 Tasks 契约。

公开请求字段、响应和 curl/级联示例维护在[中文 Tasks 文档](https://opendatalab.github.io/MinerU/zh/usage/tasks/#upstream)与[英文 Tasks 文档](https://opendatalab.github.io/MinerU/usage/tasks/#dynamic-router-upstreams-and-cascading)。

## 接口与配置

| 接口 | 语义 |
|---|---|
| `GET /v1/router/workers` | 列出本层启动配置和动态注册的远程节点。 |
| `POST /v1/router/workers` | 按规范化 URL 幂等登记并探测；重复登记恢复 draining 节点。 |
| `DELETE /v1/router/workers/{worker_id}` | 停止新预占；无引用返回 204，等待引用释放返回 202。 |

`--admin-api-key` 优先于 `MINERU_ROUTER_ADMIN_API_KEY`；Python 工厂通过 `RouterSettings(admin_api_key=...)` 配置。reload 工厂恢复相同凭证。未配置时不挂载管理路由，启用后所有操作要求独立 Bearer Key。管理凭证不参与下层探测或解析认证。

## 生命周期边界

- 管理状态 `active/draining` 与健康状态独立。调度、预检及 health/models/tiers 仅聚合健康的 active 节点；按 ID 访问旧任务仍允许 draining 节点。
- 探测阶段校验 tier 数据后才整体发布能力快照；非法响应将节点标记为不健康，保留上一份合法快照，管理查询仍可使用。
- 登记、停止接单与执行预占在首次网络等待前完成，避免并发重复登记或注销后新增预占。已删除 ID 在当前进程内不复用。
- 注销必须等待活动执行、进行中的 HTTP 请求与流式响应、执行/文件路由、反向别名和租约释放。回收沿用既有结果保留期和五分钟扫描；健康检查不会重新激活 draining。
- 永久保留结果可以永久阻止节点删除；不提供强制删除。节点从本层移除不意味着可以关闭仍服务其他调用方的下层进程。
- 待上传或尚未分派的文件继续选择剩余节点；无可用节点使用原有 `quality_tier_unavailable` 失败语义，允许节点池清空后重新注册。
- 动态清单不落盘、不修改启动参数，重启恢复 `--upstream-url`；本地托管 worker 不在接口范围内。
- 级联只复用现有 V1 协议及逐跳结果租约，管理操作不向下传播。不增加全局负载统计、共享状态或故障接管；部署保持无环。
