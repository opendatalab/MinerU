# 第三方集成

文档解析可接入 [Tasks](tasks.md) 处理异步批次，也可接入 [file-parse](file_parse.md) 上传单文件并等待附件。选择插件前，确认它使用的是官网服务还是自部署 API，以及实际调用的接口版本。

## n8n 结果下载

[n8n ZIP 工作流](plugin/n8n.md#zip-workflow)使用内置节点下载并解压已有结果。它需要已经完成的任务及有效的 `result_url`。模板已核对节点参数，仍待真实 n8n 实例验收。

## 旧版集成教程

以下链接指向归档时的中文教程，记录当时的平台界面和接口。自部署 4.x 客户端的迁移见 [4.0](../reference/migration_4.md) 和 [4.1](../reference/migration_4_1.md) 指南。

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
