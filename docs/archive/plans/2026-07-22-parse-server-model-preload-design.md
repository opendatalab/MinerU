# Parse-server model preload

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

## Context

The local parse-server currently advertises tiers as soon as its HTTP routes are available. Basic and Standard models are initialized
lazily by their Singleton caches, so a server can report a tier as healthy before the selected device, inference engine, and model files
have been validated through actual initialization.

## Design

Keep lazy loading for SDK, one-shot parser calls, and self-hosted parse-servers by default. A parse-server started with
`--preload-models` prepares its configured startup tier before the FastAPI lifespan starts serving requests. Doclib managed
parse-servers always pass this option so they validate their advertised models before accepting work. The preload implementation stays
private to `api_server.py` because it is part of server startup rather than a reusable model-management API.

- Basic prepares the shared local Hybrid model context and the conditional table/seal model families that would otherwise remain lazy
  until a matching document is parsed.
- Standard prepares the platform-selected asynchronous VLM engine and the local Hybrid model context because it advertises Basic,
  Standard, and Advanced request tiers.
- Preparation uses the existing Singleton factories, so successfully loaded models are reused by the first parse.
- Flash ignores `--preload-models` because it has no local models.
- Health, model/tier advertisement, and job submission return a structured 503 after model preload fails. The documented successful
  health response schema remains unchanged.
- Managed supervision logs after the startup grace period, applies a separate hard startup timeout, and does not restart deterministic
  dependency, model-file, or device failures. Unclassified preload failures use the bounded restart policy.

This deliberately avoids a CUDA-only preflight. The selected production runtime remains authoritative across CPU, CUDA, MLX, MPS, NPU,
and other supported environments.
