# 当前端到端工作流

本页按现有入口串联调用链。详细参数分别维护在 [CLI](cli/README.md)、[SDK](sdk/README.md) 和 [API](api/README.md)。

## 无状态本地解析

`mineru-kit parse` → `mineru.parser.parse()` → `MinerUParser` → `doc_analyze()` → ParseResult → 输出写入。
本地批量不使用 Doclib 缓存；PDF 默认整本。单文件 Markdown 输出是文件路径，批量输出是目录。
`--format` 当前为 markdown/middle_json/zip；render 的九种能力不自动成为该命令的格式选项。

## HTTP 解析

1. POST `/v1/tasks` 提交 JSON files 清单，保存创建回执的 task_id/file_id。
2. 仅对有 upload 动作的文件 PUT 原始字节；没有 complete 请求。
3. GET 固定任务路径取得完整快照；就绪文件独立执行，不等待其他文件上传。
4. 单个文件 completed 后即可下载它的 result_url，不要求整个父任务终结。
5. 需要取消时 POST `/v1/tasks/{task_id}/cancel`；成功文件保留产物。

创建、查询、取消、上传确认使用不同响应模型。Tasks 输出为 default/docx/html/latex，
一个输入对应一个结果链接；参数、状态、认证和期限以 [Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/) 为准。
Router 复用此协议并把就绪文件分派给 worker。公共 file_id 与下游 ID 可不同，不由客户端推导存储路径。

## 同步包装与 WebUI

file-parse 先持久化源文件和 request_id，再创建／上传／等待 Tasks。
取得上游 task_id/file_id 后先保存再查询；等待超时返回 202，客户端断连不自动取消。
状态目录只恢复其协议空间内的记录，不保证上游进程重启后还能找回任务。

Gradio 未指定 api-url 时托管本地 V1 API；指定后使用该服务的能力发现。
清空、换文件、同会话重提和正常关闭会使旧任务失效并执行相应取消流程，已进入不可中断计算的工作允许收尾。

## 文档库解析和读取

`mineru parse` → DoclibClient → DoclibServer → IngestService / ParseQueueService → ParseWorkerPool → ParseExecutionService。
缓存覆盖足够时复用；否则按档位和配置使用本地 Flash 或 managed/self-hosted/remote API。
`force`、`invalidate`、`forget` 与文件删除含义不同，见 [文档库命令](cli/mineru-library.md)。

`mineru read` → ContentReadService → 已缓存 Middle JSON → locator、内容预算和续读请求。
请求页范围、stdout 输出预算、完整文件导出和客户端等待分别控制不同阶段。
`mineru parse` 默认前 10 页；`--pages all` 选择整本，`-o` 导出请求页的完整渲染内容。
等待超时不取消服务端解析，read 本身不触发新的质量解析。

## 扫描、watch 与索引

scan/watch 发现路径，经过入库更新文件身份与可达性，再按现有规则排队或维护索引。
主动 PDF/图片质量解析、parsing rule 默认选档和已有缓存读取各有顺序，不能共用一个默认值。
纯文本可以直接入库和读取；原生结构化格式使用 Flash。实际扩展名集合在 `mineru/filetypes.py`。

## 协议和验证

通用 SDK、HTTP/ZIP 读取当前共享协议；只有 Doclib 缓存读取兼容受支持历史结果。
保存素材与序列化必须在源资源关闭后仍可消费。验证应检查实际 Markdown、结构、图片和导出，
不能仅看任务 completed 或测试数量。状态、用量和素材分别遵守所属入口契约。
