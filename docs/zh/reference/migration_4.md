# 从 3.x 迁移到 4.0

4.0 调整了命令入口、依赖组合、配置和 API。先使用独立环境验证，再迁移生产调用。AMD 和国产加速卡使用[旧平台指南](../usage/compatibility.md)，继续限制 `mineru<4`。

## 入口与安装

| 原用法 | 4.0 用法 |
| --- | --- |
| `mineru -p input.pdf -o output` | 文档库：`mineru parse input.pdf --pages all -o output.md`；一次性转换：`mineru-kit parse input.pdf -o output.md` |
| 目录批处理 | `mineru-kit parse ./documents -o ./output --format zip` |
| `mineru-gradio` | `mineru-webui` / `mineru-kit webui` |
| 旧模型下载参数 `-m all` | `mineru-kit models download --tier standard` |
| 旧 API 调用 | V1 上传 → 解析任务 → 文件产物 |
| `mineru[core]`、`mineru[pipeline]` 等 extras | 基础包、`mineru[torch]` 或 `mineru[full]` |

正式版安装使用 `"mineru>=4.0,<5"`（需要 extras 时加在包名后）。Python 包要求 `>=3.10,<3.15`；可选引擎按实际 wheel 和驱动选择版本。升级前停止原环境中的服务，升级后检查 `mineru version --json`。不要把旧平台的镜像和安装命令替换成主线 4.0。

## 档位、页码与输出

- 用 `flash/basic/standard/advanced` 选择解析档位；Basic 对应基础小模型流程，Standard / Advanced 使用 VLM。不要用旧 backend 的名称推断新默认质量。
- `mineru parse` 使用文档库、缓存和继续阅读协议，PDF 默认前 10 页；`mineru-kit parse` 与 Python SDK 默认全部页。
- PDF 页范围使用 `1-5,8,r3-r1`，`r1` 表示最后一页，`all` 表示全部；非 PDF 原生文档整本解析。
- `-o` 的文件/目录含义依入口而异。`mineru-kit parse` 单文件默认 Markdown；目录或多文件输入输出到目录。
- 旧结果中的 `_backend`、`pdf_info` 等字段不能作为 4.0 文档结构模板。按[当前输出协议](output_files.md)读取实际 JSON；历史产物只在明确支持的兼容读取路径中使用，不承诺所有旧 JSON 可直接传给 `ParseResult.from_dict()`。

历史结果的兼容范围如下：

| 数据情况 | 4.0 的处理方式 |
| --- | --- |
| 带 `schema` 身份的当前共享协议（`2.0`） | 由 `ParseResult`、HTTP/ZIP 与 Gradio 严格读取；携带当前身份但校验失败的数据不会回退尝试旧协议 |
| 可识别的 Doclib 历史缓存（3.4.5 `pdf_info` 页面、`schema_version` 为 `1.0`、或无 `schema` 键的 `2.0`） | 仅由 Doclib 专用缓存读取器在内存中转换；不代表其他入口兼容，也不改写历史文件 |
| Doclib 之外保存的旧结果 JSON | 不承诺 `ParseResult.from_dict()` 可以读取；请重新解析源文档 |
| 缺少素材或新版几何信息的旧产物 | 重新解析生成；仅修改版本号不等于完成迁移 |

「能读取」（Doclib 缓存）、「自动迁移」（不会改写任何文件）和「能重新导出全部格式」（需要当前素材与几何）是三个不同的承诺，请按上表区分，不要互相推断。

## 配置与模型

配置使用 `$MINERU_HOME/config.yaml`，可由 `MINERU_CONFIG` 指定；旧 `mineru.json` 配置不再读取。`model.stack`、`MINERU_MODEL_STACK` 和 `--stack` 已移除，改为独立配置：

```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
  small_backend: auto
  vlm:
    engine: auto
```

3.x 的两个旧环境变量仍会被识别并在冲突时报错：当配置了远程 VLM（`model.vlm.server_url` 非空）时，`MINERU_VL_API_KEY` 或 `MINERU_VL_MODEL_NAME` 非空且与新配置不一致会直接报错，要求清除旧变量。请将取值迁移到新位置：

| 旧变量 | 4.0 配置字段 | 4.0 环境变量 |
| --- | --- | --- |
| `MINERU_VL_API_KEY` | `model.vlm.api_key` | `MINERU_MODEL_VLM_API_KEY` |
| `MINERU_VL_MODEL_NAME` | `model.vlm.model` | `MINERU_MODEL_VLM_MODEL` |

默认按平台选择后端。显式 `onnx + llama-cpp` 可用于较轻的运行环境；Torch 与 VLM 引擎也可自由组合，但必须满足对应依赖。Apple Silicon 默认 VLM 是 llama.cpp。

变更配置后重启相关服务，并使用相同后端执行模型下载和验证。4.0 使用独立的小模型包，不能仅凭旧模型目录存在就认定已准备完成，详见[模型源说明](../usage/model_source.md)。

## API 与 WebUI

V1 API 使用 `/v1/health`、`/v1/tiers`、`/v1/uploads`、`/v1/parse/jobs` 和 `/v1/files`；旧 `/file_parse`、`/tasks` 不再提供。SDK 客户端和 WebUI 必须连接 V1 服务。文档解析 API 与 OpenAI 兼容 VLM 服务是不同接口，地址不可互换。

原 Gradio 客户端也需按新版事件接口调整，不能仅替换命令名称。第三方插件的旧适配文档保留历史说明，不表示已支持 V1 API。

迁移验证：先转换一个原生文档，再转换一份带文本层的 PDF；检查页数、Markdown、结构化 JSON 和素材。然后在实际目标硬件上验证所需质量档位和 API 调用。见[SDK 与 API](../usage/sdk_api.md)。
