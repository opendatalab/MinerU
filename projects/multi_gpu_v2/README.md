# MinerU v2.0 Multi-GPU Server

[简体中文](README_zh.md)

A streamlined multi-GPU server implementation.

## Quick Start

### 1. install MinerU

```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[core]"
uv pip install litserve aiohttp loguru
```

### 2. Start the Server
```bash
python server.py
```

### 3. Start the Client
```bash
python client.py
```

Now, pdf files under folder [demo](../../demo/) will be processed in parallel. Assuming you have 2 gpus, if you change the `workers_per_device` to `2`, 4 pdf files will be processed at the same time!

## Customize

### Server 

Example showing how to start the server with custom settings:
```python
server = ls.LitServer(
    MinerUAPI(output_dir='/tmp/mineru_output'),
    accelerator='auto',  # You can specify 'cuda'
    devices='auto',  # "auto" uses all available GPUs
    workers_per_device=1,  # One worker instance per GPU
    timeout=False  # Disable timeout for long processing
)
server.run(port=8000, generate_client_file=False)
```

### Client 

The client supports both synchronous and asynchronous processing:

```python
import asyncio
import aiohttp
from client import mineru_parse_async

async def process_documents():
    async with aiohttp.ClientSession() as session:
        # Basic usage
        result = await mineru_parse_async(session, 'document.pdf')
        
        # With custom options
        result = await mineru_parse_async(
            session, 
            'document.pdf',
            backend='pipeline',
            lang='ch',
            formula_enable=True,
            table_enable=True
        )

# Run async processing
asyncio.run(process_documents())
```

### Concurrent Processing
Process multiple files simultaneously:
```python
async def process_multiple_files():
    files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
    
    async with aiohttp.ClientSession() as session:
        tasks = [mineru_parse_async(session, file) for file in files]
        results = await asyncio.gather(*tasks)
    
    return results
```
