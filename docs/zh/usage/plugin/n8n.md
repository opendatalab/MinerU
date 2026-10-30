# n8n 下载和解压解析结果

本页提供一个使用 n8n 内置节点的下载模板，不依赖旧社区解析插件。
解析任务须先通过 [Tasks](../tasks.md) 或其他已验证的集成完成。

## ZIP 工作流 {#zip-workflow}

下载 [工作流 JSON](../../../assets/workflows/n8n-result-zip.json)，通过 n8n 的 Import from File 导入。
模板使用 Manual Trigger、HTTP Request 4.2 和 Compression 1 节点，默认未启用。

1. 在 Tasks 中请求 `default` 或其他 ZIP 输出，取得该文件的 `result_url`。
2. 在 Download result ZIP 节点中替换示例 URL。签名链接不附加 MinerU API Key；受保护的自部署 Files 地址使用 n8n 凭据，并只向对应服务发送认证。
3. 执行后 HTTP Request 把附件放入二进制字段 `data`，Compression 解压为 `result_` 前缀的字段。
4. 检查输出中的 Markdown、JSON 和图片。单独 DOCX/HTML 输出不经过 ZIP 解压节点。

模板不创建、上传或轮询任务，也不续期签名链接。链接过期时查询原任务，再更新下载地址。
节点参数已按 n8n 官方 [HTTP Request](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/) 和
[Compression](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.compression/) 定义核对，尚未在真实 n8n 实例执行端到端验收。
第三方插件的 4.1 兼容性见 [集成边界](../integrations.md)。
