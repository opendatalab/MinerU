# mineru

状态: Draft
读者: 普通用户、Agent skill 作者、核心开发者
范围: `mineru` 命令的定位、子命令和本地文档库边界
非目标: 批处理专家参数；无状态解析工具参数
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru` 是用户和 Agent 的本地文档能力中心。它不是单纯的文件转换命令，而是在解析之上叠加本地文档库、SHA256 去重、缓存、搜索、watch、配置和远端切换。

`mineru` 适合：

- Agent 需要读取本地文件并逐步扩展上下文。
- 普通用户需要解析少量文档。
- 用户希望默认保留隐私，本地优先处理文档。
- 用户希望已经解析过的文档可复用、可搜索。

## 2. 子命令

| 子命令 | 作用 | 文档 |
|--------|------|------|
| `mineru parse` | 主动解析单个文档 | [mineru parse](mineru-parse.md) |
| `mineru read` | 按 locator 继续读取已有解析结果 | [mineru read](mineru-read.md) |
| `mineru scan` | 一次性扫描文件或目录并刷新 doclib 状态 | [mineru library](mineru-library.md) |
| `mineru watch` | 管理持续监控目录和 watch rescan | [mineru library](mineru-library.md) |
| `mineru server` | 管理本地 doclib 服务 | [mineru server](mineru-server.md) |
| `mineru search` | 搜索本地文档库内容 | [mineru library](mineru-library.md) |
| `mineru find` | 搜索或定位文件 | [mineru library](mineru-library.md) |
| `mineru list` | 列出 parses、scans、files、docs | [mineru library](mineru-library.md) |
| `mineru show` | 查看 file、doc、parse、scan 等资源详情 | [mineru library](mineru-library.md) |
| `mineru config` | 管理本地配置、watch、rules | [mineru library](mineru-library.md) |
| `mineru invalidate` | 标记已完成解析结果为 superseded | [mineru library](mineru-library.md) |
| `mineru forget` | 忘记本地 path 记录，不删除源文件 | [mineru library](mineru-library.md) |
| `mineru cleanup` | 清理 deleted file、orphan doc 和临时文件 | [mineru library](mineru-library.md) |
| `mineru telemetry` | 查看、预览、启停和 flush doclib telemetry | [Telemetry 设计](../telemetry.md) |
| `mineru version` | 输出 MinerU 与 Python 版本 | 本页 |

## 3. 本地文档库

默认 MinerU home 是 `~/.mineru`，可通过 `MINERU_HOME` 调整。

```text
~/.mineru/
  config.yaml
  doclib.endpoint.json
  doclib.sock
  doclib.db
  logs/
    doclib.log
    doclib.access.log
    doclib.stdout.log
    doclib.stderr.log
  doclib/
    parsed/
    temp/
```

`doclib.endpoint.json` 记录当前本地 doclib server 实际可用的 transport。UDS 可用时通常会有 `doclib.sock`；在 Windows Python runtime 不支持 UDS 时，doclib 可以只通过 TCP loopback 工作，此时不要求存在可用 socket 文件。

`doclib.db` 记录文件路径、文档 SHA256、解析任务、缓存状态、FTS 索引、watch 目录和配置。详细数据模型见 [系统架构](../architecture.md)。

## 4. 处理模型

`mineru` 的处理过程可以概括为：

```text
发现文件
  -> 检查路径与类型
  -> 计算 SHA256
  -> 命中文档缓存则复用
  -> 未命中则按 tier 解析
  -> 写入本地文档库和解析产物
  -> 输出到 STDOUT 或文件
```

纯文本文件可以直接读取；Office 和 HTML 通常可本地 CPU 解析；PDF 和图片按 tier 路由到默认选择、`flash`、`medium` 或 `high`。

## 5. parse 与 read

`mineru` 中有两条读取链路：

```text
parse(path) = ensure document is parsed, then read default content
read(locator) = read existing parsed content by stable locator
```

- `mineru parse` 适合第一次从文件 path 出发读取文档。
- `mineru read` 适合基于 `short_id`、page、block 或 char locator 继续阅读已有解析结果。

典型流程：

1. `mineru parse report.pdf`
2. 从输出或 JSON 响应中拿到后续 locator
3. `mineru read doc:ab12cd3/tier:medium/page:11`

## 6. 与 mineru-kit 的关系

`mineru` 使用更少、更稳定的参数集合。复杂批处理、目录输入、local/remote 显式切换和文件产物输出等能力放在 `mineru-kit parse`。

如果一个参数只服务解析内核调试或批处理工程化，默认不进入 `mineru`。
