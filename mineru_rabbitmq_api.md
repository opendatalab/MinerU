# MinerU RabbitMQ 异步解析服务接口文档

本文档详细介绍了如何通过 RabbitMQ 与 MinerU 文档解析服务进行交互。该服务允许您异步提交文件（如 PDF、Office 文档、图片等），并在处理完成后从指定队列获取结构化的解析结果。

## 1. 概述

服务流程如下：
1.  **客户端**：将文件内容编码为 Base64，并构建一个 JSON 消息（可包含 correlation_id 用于任务跟踪）。
2.  **客户端**：将消息发布到指定的 **解析队列**。
3.  **MinerU 服务端**：从解析队列中消费消息，执行文档解析。
4.  **MinerU 服务端**：将解析结果（包含 Markdown、JSON 等）作为一条新的 JSON 消息，发布到 **结果队列**，并保持相同的 correlation_id。
5.  **客户端**：监听结果队列，通过 correlation_id 匹配对应的处理结果。

---

## 2. 环境配置

在与服务交互之前，请确保以下环境变量已正确设置：

- `RABBITMQ_HOST`: RabbitMQ 服务的主机地址。默认值为 `localhost`。
- `RABBITMQ_QUEUE`: 解析队列的名称。默认值为 `mineru_parse_queue`。
- `RABBITMQ_RESULT_QUEUE`: 结果队列的名称。默认值为 `mineru_results_queue`。

---

## 3. 如何发送解析任务

要提交一个文件进行解析，您需要向解析队列发送一条消息。

- **目标队列**: `mineru_parse_queue` (或由 `RABBITMQ_QUEUE` 环境变量指定)
- **消息格式**: JSON
- **消息属性**: 消息应被标记为持久性 (`delivery_mode=2`)，以确保在服务重启后任务不会丢失。

### 输入消息格式

```json
{
  "file_name": "string",
  "file_content_base64": "string",
  "parse_method": "string (optional)",
  "correlation_id": "string (optional)"
}
```

#### 字段说明

| 字段 | 类型 | 是否必须 | 描述 |
| --- | --- | --- | --- |
| `file_name` | String | 是 | 原始文件的名称，包含扩展名。例如: `"document.pdf"` |
| `file_content_base64` | String | 是 | 文件内容的 Base64 编码字符串。 |
| `parse_method` | String | 否 | 解析方法。可选值为: `"auto"`, `"ocr"`, `"txt"`。默认为 `"auto"`。 |
| `correlation_id` | String | 否 | 任务关联ID，用于在结果中识别对应的请求。建议使用UUID。 |

---

## 4. 如何获取解析结果

处理完成后，结果会发送到结果队列。您需要监听此队列以接收结果。

- **源队列**: `mineru_results_queue` (或由 `RABBITMQ_RESULT_QUEUE` 环境变量指定)
- **队列声明**: 客户端在监听前应确保队列已声明为持久队列 (`durable=True`)。

### 输出消息格式

```json
{
  "file_name": "string",
  "md": "string",
  "content_list": "list",
  "middle_json": "dict",
  "correlation_id": "string (if provided in input)"
}
```

#### 字段说明

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `file_name` | String | 原始文件的名称。 |
| `md` | String | 从文档解析出的 Markdown 格式内容。 |
| `content_list` | List | 结构化的内容列表，每个元素代表文档中的一个内容块。 |
| `middle_json` | Dict | 解析过程中生成的中间结构化数据。 |
| `correlation_id` | String | 任务关联ID（如果在输入中提供）。 |

---

## 5. Python 示例代码

以下是一个完整的 Python 脚本，演示了如何发送一个 PDF 文件进行解析，并等待接收处理结果。

```python
import pika
import json
import base64
import uuid
import os

def send_and_receive_parse_result(file_path: str):
    """
    发送文件到 RabbitMQ 进行解析，并等待接收结果。

    Args:
        file_path: 要解析的文件的本地路径。
    """
    # -- RabbitMQ 配置 --
    host = os.getenv("RABBITMQ_HOST", "localhost")
    parse_queue = os.getenv("RABBITMQ_QUEUE", "mineru_parse_queue")
    result_queue = os.getenv("RABBITMQ_RESULT_QUEUE", "mineru_results_queue")

    # 生成唯一的 correlation_id 用于任务跟踪
    correlation_id = str(uuid.uuid4())

    # -- 1. 连接到 RabbitMQ --
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        channel = connection.channel()

        # 声明解析队列和结果队列（确保它们存在且是持久的）
        channel.queue_declare(queue=parse_queue, durable=True)
        channel.queue_declare(queue=result_queue, durable=True)
        print(f"Successfully connected to RabbitMQ on host '{host}'.")

    except pika.exceptions.AMQPConnectionError as e:
        print(f"Error: Failed to connect to RabbitMQ at '{host}'. Please ensure it is running and accessible.")
        print(f"Details: {e}")
        return

    # -- 2. 读取并编码文件 --
    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
        file_content_base64 = base64.b64encode(file_content).decode("utf-8")
        file_name = os.path.basename(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        connection.close()
        return

    # -- 3. 构建并发送消息 --
    message = {
        "file_name": file_name,
        "file_content_base64": file_content_base64,
        "parse_method": "auto", # 可以是 "auto", "ocr", 或 "txt"
        "correlation_id": correlation_id  # 添加任务关联ID
    }
    body = json.dumps(message)

    channel.basic_publish(
        exchange="",
        routing_key=parse_queue,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,  # 使消息持久化
            correlation_id=correlation_id,  # 在消息属性中也设置 correlation_id
        ),
    )
    print(f" [x] Sent '{file_name}' to queue '{parse_queue}' with correlation_id: {correlation_id}")

    # -- 4. 等待并接收结果 --
    print(f" [*] Waiting for a result on queue '{result_queue}' with correlation_id: {correlation_id}")
    
    result_body = None
    try:
        # 使用 consume 从结果队列获取消息
        # inactivity_timeout 设置一个超时时间（秒），防止无限等待
        for method_frame, properties, body in channel.consume(result_queue, inactivity_timeout=120):
            if method_frame:
                # 检查 correlation_id 是否匹配
                if properties.correlation_id == correlation_id:
                    # 确认消息已被接收
                    channel.basic_ack(method_frame.delivery_tag)
                    result_body = body
                    break
                else:
                    # 如果 correlation_id 不匹配，拒绝消息并重新排队
                    channel.basic_nack(method_frame.delivery_tag, requeue=True)
                    print(f" [!] Received message with different correlation_id: {properties.correlation_id}, expecting: {correlation_id}")
        
        if result_body:
            print("\n[+] Received result:")
            result_data = json.loads(result_body)
            # 验证结果中的 correlation_id
            if result_data.get("correlation_id") == correlation_id:
                print(f"  - Correlation ID: {result_data.get('correlation_id')}")
                print(f"  - File Name: {result_data.get('file_name')}")
                print(f"  - MD Content Length: {len(result_data.get('md', ''))} characters")
                print(f"  - Content List Items: {len(result_data.get('content_list', []))} items")
                # 您可以在这里添加更复杂的逻辑来处理结果
                # 例如: with open(f"result_{correlation_id}.json", "w") as f_out:
                #           json.dump(result_data, f_out, indent=2)
            else:
                print(f"  [!] Warning: correlation_id mismatch in result data")

        else:
            print(f"\n[-] Did not receive a result for correlation_id {correlation_id} within the timeout period (120 seconds).")

    except KeyboardInterrupt:
        print("\n[*] Aborted by user.")
    except Exception as e:
        print(f"\n[!] An error occurred while waiting for the result: {e}")
    finally:
        # -- 5. 关闭连接 --
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    # 使用示例：将下面的路径替换为您要测试的文件的实际路径
    # path_to_your_file = "path/to/your/document.pdf"
    path_to_your_file = r"D:\Coding\Web\MinerU_webapi\tests\unittest\test_data\assets\pdfs\test_01.pdf"
    
    if not os.path.exists(path_to_your_file):
         print(f"Error: Test file not found at '{path_to_your_file}'. Please update the 'path_to_your_file' variable.")
    else:
        send_and_receive_parse_result(path_to_your_file)

```

## 6. 多任务并发处理

使用 `correlation_id` 的主要优势是支持多任务并发处理。以下是一个处理多个文件的示例：

```python
import threading
import time

def process_multiple_files(file_paths: list):
    """
    并发处理多个文件
    
    Args:
        file_paths: 文件路径列表
    """
    threads = []
    
    for file_path in file_paths:
        thread = threading.Thread(target=send_and_receive_parse_result, args=(file_path,))
        threads.append(thread)
        thread.start()
        time.sleep(1)  # 稍微延迟以避免同时发送大量请求
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("All files processed.")

# 使用示例
# file_list = ["file1.pdf", "file2.pdf", "file3.pdf"]
# process_multiple_files(file_list)
```

通过使用 `correlation_id`，您可以确保即使在高并发环境下，每个请求都能正确匹配到对应的响应结果。 