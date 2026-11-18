# Deploy and manage a Router

A Router exposes a single V1 address for multiple parsing workers. Clients submit regular [Tasks](../usage/tasks.md) requests. Operators can start local GPU workers or connect existing remote API servers.

## Start and check the service {#startup}

This command starts managed workers on the GPUs visible to the local process.

```bash
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus auto --worker-tier standard
```

To aggregate existing services only, use `--local-gpus none` and supply `--upstream-url` once per address. Check the Router at `/v1/health` and `/v1/tiers`, then send the [first HTTP request](../usage/http_api.md). `--preload-models` applies to managed workers. Remote nodes keep their own startup settings.

## File distribution {#dispatch}

The Router selects a worker for each ready file and reserves its load. Files in one parent task can run on different workers. Workers fetch URL inputs. For uploaded files, the Router checks each worker’s cache before sending bytes. Clients use the result link returned by Tasks.

Candidate workers must advertise the Tasks capabilities and pass an authenticated `/v1/usage` check. During an upgrade, update workers before the Router and clients. Internal leases keep worker results available while the Router holds references. The worker releases expired leases if the Router disappears.

## Register, drain, and cascade upstreams {#upstreams}

A running Router can register, inspect, and safely deregister remote upstreams. An upstream can be a V1 API Server or another Router. Management affects only the receiving Router; it does not deregister downstream workers or stop remote processes. Locally managed workers are excluded.

Set an independent management key to enable the endpoints. Without a key, these paths return 404. `--admin-api-key` overrides `MINERU_ROUTER_ADMIN_API_KEY` and also works with `--reload`. This key is separate from parsing credentials and is never forwarded downstream; parsing authentication is unchanged.

```bash
# An empty pool can start: health returns 503 until a healthy upstream is added,
# while the management endpoints remain available.
export MINERU_ROUTER_ADMIN_API_KEY='replace-with-a-management-secret'
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus none

# Set the same key in another terminal before running these requests.
curl -X POST http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"url":"http://worker-a:8000"}'
curl http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
# Use the worker_id returned by registration or listing.
curl -i -X DELETE http://127.0.0.1:8002/v1/router/workers/remote-1 \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
```

| Endpoint | Response and behavior |
|---|---|
| `GET /v1/router/workers` | `200`, with `{"object":"list","data":[...]}` containing both startup-configured and dynamically registered remote nodes. |
| `POST /v1/router/workers` | Accepts only `url`. Returns `201` for a new URL or `200` for an existing normalized URL. Re-registration preserves its ID and references and resumes a draining node. |
| `DELETE /v1/router/workers/{worker_id}` | Returns `204` after removal if no references remain, or `202` with a draining snapshot otherwise. Unknown and locally managed nodes return `404 worker_not_found`. |

Each node snapshot contains `worker_id`, `url`, `state` (`active/draining`), `healthy`, `tiers`, `active_jobs`, and `retained_routes`. URLs must be HTTP/HTTPS base URLs; path prefixes are supported, while embedded credentials, query strings, and fragments are rejected. Invalid input returns `400 invalid_request`; missing or incorrect management credentials return `401 invalid_api_key`. Errors use `error.code/message` and optional `param`.

Registration performs an initial health and capability probe. An unreachable node or one returning invalid tier data remains registered with `healthy=false` and is retried by the existing periodic checks. Invalid capabilities do not replace the last valid snapshot or break management listing. Only healthy, active nodes participate in new scheduling and health/models/tiers aggregation. A successful health check does not undo draining.

Deregistration stops new reservations immediately. Existing reservations, upload handoffs, queries, cancellation, result leases, and downloads continue. Removal waits for execution reservations, requests/streams, execution and file routes, and leases to be released; `active_jobs=0` alone does not mean the remote service can stop. Results default to 24-hour retention, so a node may remain draining during that period. The five-minute collection loop removes eligible nodes; repeating DELETE also checks eligibility. With `--retention-seconds 0`, permanent result references can prevent removal indefinitely. There is no force-delete operation.

Files that have not yet been dispatched use the remaining eligible nodes. If none qualify, they fail with the existing `quality_tier_unavailable` error; no implicit tier downgrade or new waiting queue is introduced. Removing the last node is allowed. Absence from this Router's list only means this Router has released its references; the downstream service may still serve other callers.

For a two-level deployment, configure the outer Router with inner Router URLs:

```bash
mineru-kit router --local-gpus none \
  --upstream-url http://router-a:8002 \
  --upstream-url http://router-b:8002
```

The same POST endpoint can add Router C at runtime. Use independent worker pools below each inner Router and an acyclic topology. Global load accounting and cross-Router task takeover are not provided.

**Dynamic membership lasts only for the current process.** Registration and deregistration do not modify startup configuration. Restarting restores the `--upstream-url` list; dynamic nodes must be registered again, and worker IDs are not stable across restarts. The outer Router remains a single point of failure, and existing tasks are not guaranteed to recover after any Router restarts.
