# 实现不变量

## 用户选择与隐私

没有显式远端配置，不把文档上传到远程解析服务。显式请求的 PDF/图片档位不受支持时返回错误，
不自动改档。省略档位、硬件推荐与缓存读取的规则分别以 [档位说明](tiers.md) 为准。

## 共享文档协议与归属

DocVortex 唯一拥有原生文档解析、PDF 基础访问、schema、确定性后处理、素材与七种通用渲染和导出。
MinerU 保留模型推理、产品策略、可选 LLM 增强和 Content List V1/V2，不导入 DocVortex 私有实现。
所有 PDFium 调用遵守共享锁和资源管理；跨进程只传递字节及物化数据。

Middle JSON 的 schema 身份、2.0 版本、metadata 和 Block/InlineSpan 约束见
[现行规范](middle-json/README.md)。仅 Doclib 已持久化缓存允许历史转换。
渲染与写出不得修改原语义树，也不能依赖已关闭的源 PDF。

## API 与错误

Tasks、file-parse、CLI/Doclib 分别遵守自己的响应结构；公共 HTTP 错误没有 type。
已发布的错误码含义不能随意改变。协议变化同步 [迁移说明](https://opendatalab.github.io/MinerU/zh/reference/migration_4_1/)、
[API 参考](api/README.md)、消费者及相应测试。未实现字段不得先成为文档中的必需字段。

## 调度与生命周期

任务状态、上传、结果租约和清理使用各自的类型及作用域。取消后迟到结果不能恢复为成功。
清理只释放本实例拥有的资源；启动失败也执行已获得资源的清理，不因后续清理异常覆盖原始错误。
跨实例配置通过显式参数传递，不依赖进程默认配置的隐式副作用。

## 定位符与缓存

已发布的 page/block/char 定位符保持可解析，short_id 保持稳定；语法以
`mineru/doclib/locators.py` 和 [ADR-0012](decisions/0012-doclib-block-locator.md) 为准。
缓存协议或合并逻辑变化同时验证历史读取、页范围、素材和 compaction。

## 可静态追踪的依赖

MinerU 子模块使用相对导入；模块边界显式导出 `__all__`，注册和分派使用显式映射。
公开 API 的 import 不启动服务、读取环境配置、下载模型或导入重推理依赖。
