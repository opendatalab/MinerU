# 参考资料 {#_1}

## SDK 与 HTTP 接口 {#_2}

| 调用方式 | 何时使用 | 参考 |
| --- | --- | --- |
| Python SDK | 在 Python 中直接解析或连接已有服务，得到 `ParseResult` | [示例与参数](../usage/sdk_api.md) |
| Tasks API | 提交一个或多个文件，逐个查询、取消或下载结果 | [第一次 HTTP 请求](../usage/http_api.md) · [字段、状态与错误](../usage/tasks.md) |
| file-parse | 向独立服务上传单文件，等待附件或凭请求 ID 恢复 | [表单字段与恢复流程](../usage/file_parse.md) |

## 输出与开发参考 {#_3}

[输出格式与结果协议](output_files.md){#_4}列出各入口的下载内容、ZIP 成员和素材保存方式。[开发参考](development.md)链接到 Middle JSON 规范、CLI/SDK 详细契约、架构与维护决策。

## 版本与迁移 {#_5}

查阅[更新日志](changelog.md)、[3.x → 4.0](migration_4.md)、[4.0.x → 4.1](migration_4_1.md)及[旧平台适配](../usage/compatibility.md)。
