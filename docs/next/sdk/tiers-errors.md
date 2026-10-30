# SDK 的档位与错误边界

核对入口：`mineru/parser/tier.py`、`mineru/parser/mineru_parser.py`、`mineru/kit/commands/parse.py`、
`mineru/types.py`、`mineru/doclib/services/parse_queue_svc.py`。

## 默认值按入口区分

| 入口 | PDF／图片 | 原生 Office、HTML/MHTML、CSV/TSV、EPUB、OFD |
|---|---|---|
| Python `parse()` / `MinerUParser` | 默认 standard；tier 必须是合法四档值 | 原生实现记录 flash/txt，合法质量档位不切换原生实现 |
| `mineru-kit parse` 本地单文件 | 未指定为 standard | 未指定为 flash；显式质量档位报错 |
| `mineru-kit parse` 本地批量 | 未指定为 standard | 每个原生文件归一为 flash |
| Tasks / `MinerUApiParser` / file-parse | 省略后由服务按 standard → basic → flash → advanced 选择 | 固定 flash，忽略合法显式档位 |
| Doclib 主动解析 | 结合规则与服务能力选择 standard → basic；不静默用 flash 替代质量解析 | 原生 Flash；Doclib 请求约束独立于 Tasks |

Tasks 只有在 Advanced 是唯一可用档位时才自动选择它。显式不支持的 PDF/图片档位返回错误。
`MinerUApiParser(tier=None)` 表示省略 HTTP 字段；直接本地 `parse(tier=None)` 不具有这个语义。
档位与硬件推荐不是同一件事，见 [产品档位说明](../tiers.md)。

## 错误与恢复

直接本地解析可能抛出 `FileNotFoundError`、`ValueError`、`MineruError` 或底层依赖异常，
不能承诺所有失败都属于一个 SDK 异常基类。
`MinerUApiParser` 对 HTTP/任务错误使用 API 错误对象；Doclib 通过自身错误映射保留本地产品分类。
具体边界见 [错误说明](../errors.md)。

任务创建结果未知时不要自动重放；保存 task_id/request_id，查询原任务。
等待超时不表示取消，取消需显式调用对应接口。上传重试应先查询是否仍有有效上传动作。
不要把 `retryable/user_action/docs_url` 等历史扩展提案当作当前响应字段。

没有 `backend` 公开解析参数，也没有 `mineru-kit parse --backend` 专家覆盖。
小模型后端和 VLM 引擎分别由 `model.small_backend`、`model.vlm` 配置。
