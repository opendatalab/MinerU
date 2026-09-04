# PDF 页码范围规范

状态: Implemented

CLI 的 `--pages`、Python/Doclib/API 的 `page_range` 和 Gradio 页码输入使用同一套语法。
页码表示 PDF 文件内的页面顺序，从 1 开始，不使用正文印刷页码或 PDF page label。
内部 `page_idx` / `page_indices` 仍从 0 开始，抽页后保留原始页索引。

## 输入语法

| 表达式 | 含义 |
|---|---|
| `5` | 第五页 |
| `1-5` | 第一到第五页，包含两端 |
| `1-3,7,10-12` | 多个区间和单页 |
| `all` | 全部页面 |
| `r1` | 最后一页 |
| `r5-r1` | 最后五页 |
| `3-r1` | 第三页到最后一页 |

端点使用 ASCII 正整数（`[1-9][0-9]*`），或小写 `r` 加该正整数。`all` 只能独立使用。
允许首尾、逗号和连字符两侧空白，例如 ` 1 - 3 , r1 `。不接受前导零、正负号、中文分隔符、
大写 `R` / `ALL`、空选项和缺失端点；`1:5` 不是受支持的切片语法。

先以文档总页数求值 `rN`，再检查区间方向；任何倒序区间均报错。
`5-3`、`r1-r5` 无效；五页文档的 `4-r3` 也无效，不能被其他有效选项掩盖。

有效页面取区间与文档的交集，再去重、升序和合并相邻页：

- 十页文档的 `8-15` 得到 `8-10`。
- 十页文档的 `r20-r8` 得到 `1-3`。
- `3,1-3,2` 得到 `1-3`，不会重排或复制页面。
- 整个表达式选不到页面时，例如十页文档的 `20`，返回 `page_range_invalid`。

## 未指定与全部

空字符串、纯空白和接口允许的 `null` 都表示未指定，默认策略由入口选择：

| 入口 | 未指定 PDF 页码时 |
|---|---|
| `mineru parse` / Doclib 解析 | 首次前 10 页，不足 10 页则全部 |
| Doclib 内容读取 | 首次前 10 页；带续读游标时保留现有窗口策略 |
| Doclib 内容导出 | 导出已有缓存内容 |
| Python `parse` / `parse_async` | 全部 |
| V1 Parse Jobs / `MinerUApiParser` | 全部 |
| `mineru-kit parse` / Gradio | 全部 |

显式 `all` 选择全部 PDF 页面；内容读取/导出仍受实际缓存可用性约束。
非 PDF 的解析限制不变：不能显式设置页码（包括 `all`）；Gradio 对非 PDF 自动清空并禁用此控件。

## 输入与输出

```bash
mineru parse report.pdf --pages "1-3,7,r2-r1"
mineru parse report.pdf --pages all
mineru-kit parse report.pdf -o result.md --pages "r5-r1"
```

```python
from mineru.parser import parse

result = parse("report.pdf", tier="flash", page_range="1,3,r1")
```

```json
{
  "tier": "flash",
  "files": [{"source": {"type": "file_id", "file_id": "file-example"}, "page_range": "1-3,r1"}],
  "output_formats": ["middle_json"]
}
```

已求值的 API/Doclib 范围、覆盖状态、续读命令、解析记录及新缓存文件名使用正整数连字符格式。
例如十页文档的 `1-3,r1` 输出为 `1-3,10`，`5-5` 输出为 `5`，空覆盖集合输出空字符串。
parsing rules 和排队中的 V1 请求尚未取得总页数时允许保留 `rN` / `all`；实际解析时再求值。

## 错误与实现边界

统一错误码为 `page_range_invalid`，类型为 `invalid_request_error`。Python 抛出
`InvalidRequestError`；CLI 和 Gradio 保留该错误含义。V1 创建任务的语法错误返回 HTTP 400，
`error.param` 定位到 `files.<index>.page_range`。总页数求值错误通过异步任务的文件级错误返回。

共享实现位于 `mineru/parser/page_range.py`，负责语法、求值、格式化和页数统计。缓存内部的空范围
表示空集合，不表示用户输入的“未指定”。总页数未知的缓存内容导出可使用绝对页码或全部缓存，
不能用已有缓存的最大页码推测 `r1`。

## 不兼容升级

本次不支持旧输入 `1~5`、`-1`、`-5~-1`，也不提供旧页码缓存迁移或旧服务端回退。
更新调用方、SDK、Doclib 和 Parse Server，重新创建使用旧语法的 parsing rules。

已有 Doclib 数据可保留备份。升级后使用新的数据目录重新入库、解析：

1. 停止旧 Doclib 实例，在独立配置文件中把 `doclib.data_dir` 配置为新的空目录。
2. 按[配置说明](config.md)同步配置独立 DB/endpoint 路径，以新配置启动 Doclib；客户端使用同一配置。
3. 重新添加 watch / parsing rules，重新扫描或执行 `mineru parse`，生成新格式记录与缓存。

程序不会在此次升级中自动删除旧数据。不要把旧数据库或旧 parsed 目录复制到新实例继续使用。
