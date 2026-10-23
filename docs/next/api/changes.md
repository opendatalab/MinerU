# Tasks 唯一异步接口迁移

这是破坏性 HTTP 接口升级：独立 `/v1/uploads` 全部路由及 `/v1/parse/jobs` 创建、列表、查询和取消路由已移除，返回 404；不保留别名或客户端自动回退。

| 旧方式 | 新方式 |
|---|---|
| Tasks 创建请求 `manifest` | `files`，仍为 1–100 项数组；旧字段或新旧混传返回 400。 |
| Tasks / Files JSON `filename` | `file_name`；下载附件命名规则保持不变。 |
| 创建请求必填 `client_file_id` | 可省略或为 `null`；服务端按输入顺序补全最小可用的 `fN`，避开所有显式 ID，响应仍必填。 |
| 创建 Upload、PUT、complete | POST `/v1/tasks` 提交清单，只 PUT 缺失文件；上传成功自动排队。 |
| 创建、查询、取消 Job | POST/GET/DELETE Tasks；GET `/v1/tasks` 列出父任务。 |
| `files[].output_files` | `files[].result_url`，每个输入一个完整产物链接。 |
| `markdown/middle_json/structured_content/zip` 输出选项 | `default` 默认 ZIP；另可组合 `docx/html/latex`。 |
| `local/inline/file_id` 输入 | 文件上传或 URL；同机先计算 hash，缓存未命中也需上传。 |
| 默认 ZIP 中的模型输出 | 显式 `include_model_output=true`，并包含 `default` 格式。 |
| Health `output_formats` | `task_output_formats`；sources 为 upload/url，task_model_output 声明附加模型输出能力。 |

`--allow-local-source` 和 `--max-inline-bytes` 同时删除。Files CRUD、Health、Tiers、Models、Usage 保留；同步 `file_parse` 继续作为独立包装服务运行。
SDK 保留 `MinerUApiParser` 方法和返回类型，状态类型改为 `ApiTaskStatus`，保留 `ApiJobStatus` Python 别名。直接本地 parser 与 HF CPU/ZeroGPU 内嵌解析不经过 HTTP。

停收旧请求、等待在途任务完成，再协调升级 worker、Router、SDK、file_parse 包装服务和应用。SaaS Go 服务需部署方按同一契约配套更新；进程内任务和缓存索引不保证跨重启恢复。
