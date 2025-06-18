# MinerU v2.0 多GPU服务器

[English](README.md)

这是一个精简的多GPU服务器实现。

## 快速开始

### 1. 安装 MinerU

```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[core]"
uv pip install litserve aiohttp loguru
```

### 2. 启动服务器

```bash
python server.py
```

### 3. 启动客户端

```bash
python client.py
```

现在，`[demo](../../demo/)` 文件夹下的PDF文件将并行处理。假设您有2个GPU，如果您将 `workers_per_device` 更改为 `2`，则可以同时处理4个PDF文件！

## 自定义

### 服务器

以下示例展示了如何启动带有自定义设置的服务器：
```python
server = ls.LitServer(
    MinerUAPI(output_dir='/tmp/mineru_output'),  # 自定义输出文件夹
    accelerator='auto',  # 您可以指定 'cuda'
    devices='auto',  # "auto" 使用所有可用的GPU
    workers_per_device=1,  # 每个GPU启动一个工作实例
    timeout=False  # 禁用超时，用于长时间处理
)
server.run(port=8000, generate_client_file=False)
```

### 客户端

客户端支持同步和异步处理：

```python
import asyncio
import aiohttp
from client import mineru_parse_async

async def process_documents():
    async with aiohttp.ClientSession() as session:
        # 基本用法
        result = await mineru_parse_async(session, 'document.pdf')
        
        # 带自定义选项
        result = await mineru_parse_async(
            session, 
            'document.pdf',
            backend='pipeline',
            lang='ch',
            formula_enable=True,
            table_enable=True
        )

# 运行异步处理
asyncio.run(process_documents())
```

### 并行处理
同时处理多个文件：
```python
async def process_multiple_files():
    files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
    
    async with aiohttp.ClientSession() as session:
        tasks = [mineru_parse_async(session, file) for file in files]
        results = await asyncio.gather(*tasks)
    
    return results
```