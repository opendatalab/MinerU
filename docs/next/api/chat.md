# Chat 与 Responses

状态: Draft
读者: API 使用者、SDK 开发者、服务端开发者
范围: OpenAI-compatible 文档对话接口
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 定位

Chat Completions 和 Responses 面向细粒度文档理解场景，例如单张图片 OCR、单个表格提取或一段文本识别。完整 PDF、DOCX、PPTX 等多页文档解析应使用 [Parse Jobs](parse-jobs.md)。

两套接口都只支持单轮请求，不维护服务端会话历史。

## POST `/v1/chat/completions`

创建 Chat Completion。请求形态兼容 OpenAI Chat Completions 的核心字段。

```json
{
  "model": "MinerU2.5-Pro-2605-1.2B",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo..."
          }
        },
        {
          "type": "text",
          "text": "Text Recognition:"
        }
      ]
    }
  ],
  "stream": false,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_completion_tokens": 4096,
  "n": 1
}
```

请求字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `messages` | array | 是 | 消息列表。 |
| `model` | string | 是 | 模型 ID，来自 `GET /v1/models`。 |
| `stream` | bool | 否 | 是否流式输出，默认 `false`。 |
| `stream_options.include_usage` | bool | 否 | `stream=true` 时，最终 chunk 是否包含 usage。 |
| `temperature` | number | 否 | 采样温度，范围 `[0, 2]`。 |
| `top_p` | number | 否 | 核采样，范围 `[0, 1]`。 |
| `max_tokens` | integer | 否 | 兼容字段，建议使用 `max_completion_tokens`。 |
| `max_completion_tokens` | integer | 否 | 最大输出 token 数。 |
| `stop` | string 或 array | 否 | 最多 4 个停止序列。 |
| `seed` | integer | 否 | 尽量确定性采样。 |
| `frequency_penalty` | number | 否 | 兼容 OpenAI 字段。 |
| `presence_penalty` | number | 否 | 兼容 OpenAI 字段。 |
| `reasoning_effort` | string | 否 | 接受但可忽略，保留给未来模型。 |
| `n` | integer | 否 | 当前仅允许 `1`。 |

### Message

支持 role:

| role | content |
|------|---------|
| `system` | string 或 text part 数组。 |
| `developer` | string 或 text part 数组，与 `system` 等价处理。 |
| `user` | string 或多模态 part 数组。 |

不支持 `assistant`、`tool`、`function` role。

`user` content part:

| part | 结构 | 说明 |
|------|------|------|
| `text` | `{"type":"text","text":"..."}` | 文本指令。 |
| `image_url` | `{"type":"image_url","image_url":{"url":"..."}}` | 图片 URL 或 base64 data URI。`detail` 可传但忽略。 |
| `file` | `{"type":"file","file":{"file_id":"file-..."}}` | 引用 `purpose:"input_image"` 的 File。 |

输入限制:

| 限制项 | 值 |
|--------|----|
| 图片或 file 数量 | 每条消息最多 1 个，二选一。 |
| 图片文件大小 | 小于 10 MiB。 |
| 图片分辨率 | 小于 3500 x 3500。 |

### 非流式响应

```json
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "MinerU2.5-Pro-2605-1.2B",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "# 文本识别结果\n\n这是从图片中识别出的文字内容..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 86,
    "total_tokens": 206
  }
}
```

响应字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | Completion ID。 |
| `object` | string | 固定为 `"chat.completion"`。 |
| `created` | integer | Unix 秒级时间戳。 |
| `model` | string | 实际使用的模型。 |
| `system_fingerprint` | string | 后端配置指纹。 |
| `choices[].index` | integer | 候选索引。 |
| `choices[].message.role` | string | 固定为 `"assistant"`。 |
| `choices[].message.content` | string 或 null | 输出文本。 |
| `choices[].finish_reason` | string | `stop`、`length` 或 `content_filter`。 |
| `usage` | object | token 用量。 |

### 流式响应

`stream=true` 时返回 SSE，每行 `data:` 后是一个 `chat.completion.chunk` JSON 对象，最后以 `data: [DONE]` 结束。

```text
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2605-1.2B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"MinerU2.5-Pro-2605-1.2B","choices":[{"index":0,"delta":{"content":"# 文本"},"finish_reason":null}]}

data: [DONE]
```

如果 `stream_options.include_usage=true`，最终 JSON chunk 可以包含 `usage`，且 `choices` 为空数组。

### 不支持的 Chat 参数

以下参数不支持。实现可以返回 `400 unsupported_parameter`，也可以在明确兼容策略下忽略，但 SDK 不应依赖它们生效:

- `tools`、`tool_choice`、`function_call`、`functions`、`parallel_tool_calls`
- `audio`、`modalities`
- `web_search_options`
- `prediction`
- `response_format`
- `logprobs`、`top_logprobs`
- `logit_bias`
- `service_tier`
- `store`、`metadata`
- `safety_identifier`、`prompt_cache_key`、`prompt_cache_retention`、`user`、`verbosity`

Chat 错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_request` | 消息结构非法、role 不支持、缺少模型等。 |
| 400 | `unsupported_parameter` | 传入当前不支持的参数。 |
| 404 | `model_not_found` | 模型不存在。 |
| 413 | `content_too_large` | 图片或输入超过限制。 |
| 429 | `rate_limit_exceeded` | 触发 chat 类限流。 |
| 503 | `service_unavailable` | 模型服务不可用。 |

## POST `/v1/responses`

创建 Response。该接口与 Chat Completions 功能重叠，但输入输出采用 typed item list。

```json
{
  "model": "MinerU2.5-Pro-2605-1.2B",
  "input": [
    {
      "role": "user",
      "content": [
        { "type": "input_image", "image_url": "https://example.com/page.png" },
        { "type": "input_text", "text": "Text Recognition:" }
      ]
    }
  ],
  "stream": false,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_output_tokens": 4096,
  "store": false
}
```

请求字段:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `input` | string 或 array | 是 | 字符串或 EasyInputMessage 数组。 |
| `instructions` | string | 否 | 系统指令，等价于 Chat 的 system message。 |
| `model` | string | 是 | 模型 ID。 |
| `stream` | bool | 否 | 是否流式输出。 |
| `temperature` | number | 否 | 采样温度。 |
| `top_p` | number | 否 | 核采样。 |
| `max_output_tokens` | integer | 否 | 最大输出 token 数。 |
| `reasoning.effort` | string | 否 | 接受但可忽略。 |
| `reasoning.summary` | string | 否 | 接受但可忽略。 |
| `store` | bool | 否 | 当前仅允许 `false`。 |

EasyInputMessage:

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | string | `system`、`developer` 或 `user`。 |
| `content` | string 或 array | 文本或多模态 part。 |

Content part:

| part | 结构 | 说明 |
|------|------|------|
| `input_text` | `{"type":"input_text","text":"..."}` | 文本。 |
| `input_image` | `{"type":"input_image","image_url":"..."}` 或 `{"type":"input_image","file_id":"file-..."}` | 图片 URL、data URI 或 File。 |
| `input_file` | `{"type":"input_file","file_id":"file-..."}` | 仅支持 `purpose:"input_image"` 的 File。 |

不支持 assistant role、tool/function item、reasoning item、conversation 状态 item。

### 非流式响应

```json
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "model": "MinerU2.5-Pro-2605-1.2B",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "该文档主要介绍了...",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 36,
    "output_tokens": 87,
    "total_tokens": 123
  }
}
```

响应字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | Response ID。 |
| `object` | string | 固定为 `"response"`。 |
| `created_at` | integer | Unix 秒级时间戳。 |
| `status` | string | `completed`、`failed` 或 `incomplete`。 |
| `model` | string | 实际使用模型。 |
| `output[]` | array | 输出 item。当前仅返回 message。 |
| `output[].content[].type` | string | 当前仅返回 `output_text`。 |
| `usage` | object | token 用量。 |
| `incomplete_details` | object 或 null | `status:"incomplete"` 时说明原因。 |

### Responses 流式事件

`stream=true` 时返回 SSE，使用 `event:` + `data:` 双行格式。

事件类型:

| 事件 | 含义 |
|------|------|
| `response.created` | Response 已创建。 |
| `response.in_progress` | 开始处理。 |
| `response.output_item.added` | 新增 message item。 |
| `response.content_part.added` | 新增 output_text part。 |
| `response.output_text.delta` | 文本增量。 |
| `response.output_text.done` | 当前 output_text 完成。 |
| `response.content_part.done` | 当前 part 完成。 |
| `response.output_item.done` | 当前 item 完成。 |
| `response.completed` | 整体完成，携带完整 response 和 usage。 |

示例:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_...","output_index":0,"content_index":0,"delta":"# 文本"}
```

### 不支持的 Responses 参数

- `tools`、`tool_choice`、`parallel_tool_calls`、`max_tool_calls`
- `background`
- `conversation`、`previous_response_id`
- `include`、`prompt`、`top_logprobs`、`text`、`truncation`、`context_management`
- `metadata`、`safety_identifier`、`prompt_cache_key`、`prompt_cache_retention`、`user`、`service_tier`

Responses 错误:

| HTTP | code | 场景 |
|------|------|------|
| 400 | `invalid_request` | input 结构非法、role 不支持、缺少模型等。 |
| 400 | `unsupported_parameter` | `store=true` 或其他不支持参数。 |
| 404 | `model_not_found` | 模型不存在。 |
| 413 | `content_too_large` | 输入超过限制。 |
| 429 | `rate_limit_exceeded` | 触发 chat 类限流。 |
| 503 | `service_unavailable` | 模型服务不可用。 |

## 本地 Server 差异

Local Parse Server 可以暴露同样的 Chat 和 Responses endpoint，但实际可用模型取决于本地 `GET /v1/models`。

本地实现约束:

- 如果没有可服务的 VLM 或指定模型，应返回 `404 model_not_found` 或 `503 service_unavailable`。
- 不得静默切换到 `flash`。
- 本地部署通常不做 anonymous/registered 的功能售卖差异；启用 API Key 后，通过认证的请求可以使用本地 server 暴露的全部模型能力。
- 如果本地不实现 Chat/Responses，应在 `GET /v1/models` 中不要暴露对应模型，并对 endpoint 返回明确错误。
