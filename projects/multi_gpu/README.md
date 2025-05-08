## 项目简介
本项目提供基于 LitServe 的多 GPU 并行处理方案。LitServe 是一个简便且灵活的 AI 模型服务引擎，基于 FastAPI 构建。它为 FastAPI 增强了批处理、流式传输和 GPU 自动扩展等功能，无需为每个模型单独重建 FastAPI 服务器。

## 环境配置
请使用以下命令配置所需的环境：
```bash
pip install -U magic-pdf[full] litserve python-multipart filetype
```

## 快速使用
### 1. 启动服务端
以下示例展示了如何启动服务端，支持自定义设置：
```python
server = ls.LitServer(
    MinerUAPI(output_dir='/tmp'),  # 可自定义输出文件夹
    accelerator='cuda',  # 启用 GPU 加速
    devices='auto',  # "auto" 使用所有 GPU
    workers_per_device=1,  # 每个 GPU 启动一个服务实例
    timeout=False  # 设置为 False 以禁用超时
)
server.run(port=8000)  # 设定服务端口为 8000
```

启动服务端命令：
```bash
python server.py
```

### 2. 启动客户端
以下代码展示了客户端的使用方式，可根据需求修改配置：
```python
files = ['demo/small_ocr.pdf']  # 替换为文件路径，支持 pdf、jpg/jpeg、png、doc、docx、ppt、pptx 文件
n_jobs = np.clip(len(files), 1, 8)  # 设置并发线程数，此处最大为 8，可根据自身修改
results = Parallel(n_jobs, prefer='threads', verbose=10)(
    delayed(do_parse)(p) for p in files
)
print(results)
```

启动客户端命令：
```bash
python client.py
```
好了，你的文件会自动在多个 GPU 上并行处理！🍻🍻🍻
