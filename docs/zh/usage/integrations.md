# 第三方集成

本页保留第三方集成入口。旧教程已移入仓库历史档案，不作为当前平台操作说明，也不代表本地 MinerU 4.1 API 兼容。

本地异步集成使用 [Tasks](tasks.md)，multipart 单文件集成使用独立 [file-parse 服务](file_parse.md)。升级前确认插件版本及它实际调用的接口；旧 `/file_parse`、Jobs 和上传 complete 流程不能直接复用。

| 集成 | 教程范围 |
|---|---|
| [BISHENG](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/BISHENG.md) | 历史教程，未验证当前兼容性 |
| [Cherry Studio](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Cherry_Studio.md) | 含云／本地配置；本地 4.1 需独立验证 |
| [Coze](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Coze.md) | 历史教程，未验证当前兼容性 |
| [DataFlow](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/DataFlow.md) | 历史教程，未验证当前兼容性 |
| [Dify](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Dify.md) | 含云／本地配置；本地 4.1 需独立验证 |
| [DingTalk](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/DingTalk.md) | 历史教程，未验证当前兼容性 |
| [FastGPT](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/FastGPT.md) | 历史教程，未验证当前兼容性 |
| [ModelWhale](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/ModelWhale.md) | 历史教程，未验证当前兼容性 |
| [RagFlow](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/RagFlow.md) | 含云／本地配置；本地 4.1 需独立验证 |
| [Sider](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Sider.md) | 历史教程，未验证当前兼容性 |
| [n8n](plugin/n8n.md) | 内置节点 ZIP 模板；尚需实例验收 |

[n8n ZIP 下载与解压模板](plugin/n8n.md#zip-workflow)只负责已有结果的下载和解压，不依赖社区解析节点。
