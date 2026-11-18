# 部署与管理 Router

多个解析 worker 可以共用一个 V1 地址。客户端向 Router 提交普通 [Tasks](../usage/tasks.md) 请求；部署者可以在本机启动 GPU worker，或接入已运行的远程 API 服务。

## 启动与检查 {#startup}

以下命令在本机可见的 GPU 上启动托管 worker。

```bash
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus auto --worker-tier standard
```

仅聚合已有服务时，使用 `--local-gpus none`，并为每个地址传入 `--upstream-url`。先访问 Router 的 `/v1/health` 和 `/v1/tiers`，再向它发送[第一次 HTTP 请求](../usage/http_api.md)。`--preload-models` 作用于托管 worker，远程节点使用自己的启动配置。

## 文件如何分配 {#dispatch}

Router 按就绪文件独立选择 worker 并预占负载，同一父 Task 可使用多个 worker。
URL 交给 worker 下载，客户端上传字节按 worker 的缓存预检结果传输。
Router 的源缓存使用按调用方隔离的私有引用，不等待或转发 worker 的源 Files ID；结果仍通过 Files 映射下载。私有源引用在父任务完成清理后开始保留期，过期独立回收，活动传输继续保护字节。
健康接口的 `features.task_output_formats` 声明 Tasks 格式，`sources` 为 `upload/url`。
Router 创建任务前使用候选 worker 的受保护 `/v1/usage` 校验认证，不调用列表接口，不预占解析名额。
不声明新 Tasks 能力的旧 worker 不参与 Tasks 调度，更新时应先更新 worker，再更新 Router 和客户端。
Router 使用有界、可续租的内部结果租约保护 worker 产物，上传窗口不受结果租约影响；Router 丢失后租约会自行结束。已完成任务按 worker 并行续租，每轮网络等待最多 10 秒，避免失联 worker 阻塞其他 worker 的结果保留。

## 注册、移除与级联 {#upstreams}

Router 可以在运行期间注册、查询和安全注销远程 upstream。upstream 可以是独立的 V1 API Server，也可以是另一层 Router；操作只影响接收请求的这一层，不会注销下层的 worker 或关闭远程进程。本地托管 worker 不在管理接口范围内。

配置独立管理 Key 后启用接口，未配置时这些路径返回 404。`--admin-api-key` 优先于 `MINERU_ROUTER_ADMIN_API_KEY`，也支持 `--reload`。管理 Key 不等于解析 API Key，不会被转发给下层；解析请求继续使用原有认证。

```bash
# 空节点池也可启动；注册健康节点前 /v1/health 返回 503，管理接口仍可访问。
export MINERU_ROUTER_ADMIN_API_KEY='replace-with-a-management-secret'
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus none

# 在另一个终端设置相同管理 Key 后操作。
curl -X POST http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"url":"http://worker-a:8000"}'
curl http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
# 使用注册或列表响应中的 worker_id。
curl -i -X DELETE http://127.0.0.1:8002/v1/router/workers/remote-1 \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
```

| 接口 | 响应与行为 |
|---|---|
| `GET /v1/router/workers` | `200`，返回 `{"object":"list","data":[...]}`，包含启动参数和运行期注册的远程节点。 |
| `POST /v1/router/workers` | 请求只接受 `url`。新增返回 `201`，同一规范化 URL 重复注册返回 `200`；保留原 ID 和已有引用，draining 节点恢复接单。 |
| `DELETE /v1/router/workers/{worker_id}` | 无引用时删除并返回 `204`；仍有引用时返回 `202` 与 draining 快照；未知或本地托管节点返回 `404 worker_not_found`。 |

节点快照包含 `worker_id`、`url`、`state`（`active/draining`）、`healthy`、`tiers`、`active_jobs` 和 `retained_routes`。注册 URL 须为 HTTP/HTTPS base URL，允许路径前缀，禁止内嵌凭证、查询参数和 fragment。非法输入返回 `400 invalid_request`，缺失或错误管理凭证返回 `401 invalid_api_key`，错误均使用 `error.code/message` 和可选 `param`。

注册后执行一次健康与能力探测，暂时离线或返回非法 tier 数据的节点仍保留并返回 `healthy=false`，后续由周期检查重试。非法能力不替换上一份合法快照，也不影响管理列表查询。只有健康的 active 节点参与新任务调度及 health/models/tiers 能力聚合；健康恢复不会撤销 draining。

注销立即停止新预占，已有预占、上传交接、查询、取消、结果续租与下载继续执行。节点必须等执行预占、请求/流式传输、执行与文件路由及租约全部释放后才会移除；`active_jobs=0` 不能单独作为关闭远程服务的依据。结果默认保留 24 小时，节点可能在这段时间内保持 draining；后台复用五分钟扫描完成回收，也可重复 DELETE 检查。`--retention-seconds 0` 下长期结果引用可以一直阻止移除，无强制删除接口。

尚未分派的文件使用剩余可用节点，没有匹配节点时按原契约返回 `quality_tier_unavailable`，不自动降档或新增等待队列。允许注销最后一个节点。列表中节点消失只表示本层已释放引用；下层可能仍在为其他调用方服务。

两层部署时，顶层直接接入下层 Router 的地址：

```bash
mineru-kit router --local-gpus none \
  --upstream-url http://router-a:8002 \
  --upstream-url http://router-b:8002
```

也可通过上述 POST 动态加入 Router C。建议各下层管理独立 worker 池并保持无环拓扑；本次不提供全局负载统计或跨 Router 任务接管。

**动态清单仅当前进程有效。** 注册和注销不会修改启动配置；重启后恢复 `--upstream-url` 中的节点，动态节点须重新注册，worker ID 不能跨重启作为稳定标识。顶层仍有单点，任一 Router 重启后的旧任务恢复不在保证范围内。
