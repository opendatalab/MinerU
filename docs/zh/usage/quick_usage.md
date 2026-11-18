# 文档库与 Agent 阅读 {#mineru}

文档库适合需要反复搜索、引用或继续阅读同一份文档的场景。只想把文件或目录一次性导出为 Markdown 或 ZIP 时，使用[快速入门](../quick_start/index.md)里的 `mineru-kit parse`。

## 解析并保存到文档库 {#_2}

先启动本地文档库服务，再用[快速入门](../quick_start/index.md)中的 `report.html` 试一次。解析结果存入文档库，命令的 JSON 响应写入 `parsed.json`。原生 HTML 整份解析，使用 Flash。

```bash
mineru server start
mineru parse report.html --wait 600 --json > parsed.json
```

完成后的 `content` 包含这次返回的正文和 `content_ranges`。每个范围的 `start` 和 `end` 都是可读取的定位符，包含真实的文档 ID、档位和内容位置。需要完整 Markdown 文件时使用 `-o`。[解析、导出与等待](cli_tools.md)

## 搜索和读取已有内容 {#_3}

```bash
mineru search "MinerU" --json
```

搜索结果给出文档的 `short_id`、命中片段和可用路径。要读取刚解析的具体位置，可以复制 `parsed.json` 中的 `content.content_ranges[0].start`，将其作为 `mineru read` 的第一个参数。使用 Bash 和 `jq` 时可以将这个定位符直接传给 CLI。

```bash
locator=$(jq -er '.content.content_ranges[0].start' parsed.json) &&
  mineru read "$locator" --json
```

这会读取第一个已返回范围的起始位置。`read` 消费当前缓存。若读取响应中有 `next_request`，使用其中的 `locator` 继续读，`after` 有值时一并传给 `--after`。文本输出中的 `<!-- Next: ... -->` 给出了可执行的下一条命令。

## 阅读更多 PDF 页面 {#pdf}

`mineru parse document.pdf` 默认请求前 10 页。如果需要第 11 页，先请求它，再使用响应中的定位符。要导出整份 PDF，显式请求全部页。

```bash
mineru parse document.pdf --pages 11 --wait 600 --json
mineru parse document.pdf --pages all -o document.md --wait 600
```

PDF 页码从 1 开始，`1-5,8,r1` 表示前五页、第八页和最后一页。`--pages` 决定请求的页面，`--limit` 限制一次返回终端的文字量，`-o` 写出所请求范围的完整内容。等待超时后，使用同一条解析命令继续等待。[PDF 页范围](../../next/page-ranges.md)

## 服务与配置 {#related-settings}

- [下载与离线准备模型](model_source.md){#_1}，或查看[模型配置文件](model_source.md){#mineru_1}中的路径和后端设置。
- [部署解析服务](../deployment/index.md){#apiwebui}，或[在 WebUI 中使用已有服务](webui.md)。
- [配置标题分级和跨页表格续接](model_source.md#llm-post-processing){#configyaml-llm}。
- 文档库初始启用使用与诊断遥测。`mineru telemetry preview` 可检查数据，`mineru telemetry disable` 可停止采集和发送；服务启动前也可设置 `MINERU_TELEMETRY_DISABLED=1`。[遥测范围与设置](../../next/telemetry-minimum.md)
