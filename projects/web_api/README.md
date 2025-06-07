# 基于MinerU的PDF解析API

- MinerU的GPU镜像构建
- 基于FastAPI的PDF解析接口

## 构建方式

```
docker build -t mineru-api .
```

或者使用代理：

```
docker build --build-arg http_proxy=http://127.0.0.1:7890 --build-arg https_proxy=http://127.0.0.1:7890 -t mineru-api .
```

## 启动命令

```
docker run --rm -it --gpus=all -p 8000:8000 mineru-api
```

## 测试参数

访问地址：

```
http://localhost:8000/docs
http://127.0.0.1:8000/docs
```

## RabbitMQ消费者

使用 `rabbitmq_consumer.py` 可以从 RabbitMQ 队列读取待解析文件路径并生成解析结果，示例：
```bash
export RABBITMQ_HOST=localhost
export RABBITMQ_QUEUE=parse_queue
python rabbitmq_consumer.py
```

消息体需为 JSON 格式，至少包含 `file_path` 字段。
