# 使用指南 {#_1}

## 按任务选择入口 {#_2}

| 想得到的结果 | 使用方式 |
| --- | --- |
| 一个文件或整个目录的 Markdown、Middle JSON 或 ZIP | 用 `mineru-kit parse` 做[第一次转换](../quick_start/index.md)。命令默认处理 PDF 全部页，将产物写到 `-o` 指定的位置。 |
| 在浏览器里对照原文检查识别结果 | 打开 [WebUI](webui.md)，选择文件和页范围，点击原文或解析结果双向定位。 |
| 保存解析缓存、搜索文档并继续阅读 | 用 `mineru parse` 将文件加入[文档库](quick_usage.md)，再用 `mineru search` 和 `mineru read` 访问已有内容。 |
| 在程序中处理文档 | 选择[本地或远程 Python SDK](sdk_api.md)、[异步 Tasks](tasks.md) 或[单文件同步接口](file_parse.md)。 |
| 把结果交给现有应用或工作流 | 在[第三方集成](integrations.md)查看 n8n 下载模板和各历史教程的适用范围。 |

## 参数和排障 {#_3}

查看[命令行工具](cli_tools.md)了解导出和等待参数，查看[输出格式](../reference/output_files.md)确定需要的文件。部署模型、API 或多 GPU 服务请从[部署与运维](../deployment/index.md)开始。[常见问题](../faq/index.md)
