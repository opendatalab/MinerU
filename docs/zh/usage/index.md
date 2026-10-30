# 使用指南

本章节提供了项目的完整使用说明。我们将通过以下几个部分，帮助您从基础到进阶逐步掌握项目的使用方法：

## 目录

- [档位与运行环境](tiers.md)
- [Python SDK](sdk_api.md)
- [V1 HTTP API](http_api.md)
- [Tasks 异步接口](tasks.md)
- [file-parse 同步接口](file_parse.md)
- [4.0.x → 4.1 迁移](../reference/migration_4_1.md)
- [3.x → 4.0 迁移](../reference/migration_4.md)
- 本地部署
    * [基础使用](./quick_usage.md) - 快速上手和基本使用
    * [模型源配置](./model_source.md) - 模型源的详细配置说明  
    * [命令行工具](./cli_tools.md) - 命令行工具的详细参数说明
    * [命令行进阶参数](./advanced_cli_parameters.md) - 一些适配命令行工具的进阶参数说明
- [旧平台适配（MinerU <4）](compatibility.md) - 国产卡保留原适配
- 插件与生态（保留原集成说明，不代表已支持 4.0 V1 API）
    * [Cherry Studio](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Cherry_Studio.md)
    * [Sider](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Sider.md)
    * [Dify](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Dify.md)
    * [n8n](plugin/n8n.md)
    * [Coze](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/Coze.md)
    * [FastGPT](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/FastGPT.md)
    * [ModelWhale](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/ModelWhale.md)
    * [DingTalk](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/DingTalk.md)
    * [DataFlow](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/DataFlow.md)
    * [BISHENG](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/BISHENG.md)
    * [RagFlow](https://github.com/opendatalab/MinerU/blob/master/docs/archive/integrations/RagFlow.md)

## 开始使用

`mineru` 是面向交互和 Agent 工作流的文档库 CLI；无状态批处理、V1 API Server、Gradio、模型管理和多服务 Router 使用 `mineru-kit`。

建议按照上述顺序阅读文档，这样可以帮助您更好地理解和使用项目功能。

如果您在使用过程中遇到问题，请查看 [FAQ](../faq/index.md)
