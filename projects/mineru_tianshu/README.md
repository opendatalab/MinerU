# MinerU Tianshu (天枢)

> 天枢 - 企业级多GPU文档解析服务  
> 结合 SQLite 任务队列 + LitServe GPU负载均衡的最佳方案

## 🌟 核心特性

- ✅ **异步处理** - 客户端立即响应（~100ms），无需等待处理完成
- ✅ **任务持久化** - SQLite 存储，服务重启任务不丢失
- ✅ **GPU 负载均衡** - LitServe 自动调度，资源利用最优
- ✅ **优先级队列** - 重要任务优先处理
- ✅ **实时查询** - 随时查看任务进度和状态
- ✅ **RESTful API** - 支持任何编程语言接入
- ✅ **智能解析器** - PDF/图片用 MinerU，其他所有格式用 MarkItDown
- ✅ **内容获取** - 获取解析后的 Markdown 内容，支持图片上传到 MinIO

## 🏗️ 系统架构

```
客户端请求 → FastAPI Server (立即返回 task_id)
                    ↓
              SQLite 任务队列
                    ↓
            Task Scheduler (调度器)
                    ↓
         LitServe Worker Pool (GPU自动负载均衡)
                    ↓
              MinerU 核心处理
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd projects/mineru_tianshu
pip install -r requirements.txt
```

> **支持的文件格式**:
> - 📄 **PDF 和图片** (.pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp) - 使用 MinerU 解析（GPU 加速）
> - 📊 **其他所有格式** (Office、HTML、文本等) - 使用 MarkItDown 解析（快速处理）
>   - Office: .docx, .doc, .xlsx, .xls, .pptx, .ppt
>   - 网页: .html, .htm
>   - 文本: .txt, .md, .csv, .json, .xml 等

### 2. 启动服务

```bash
# 一键启动所有服务（推荐）
python start_all.py

# 或自定义配置
python start_all.py --workers-per-device 2 --devices 0,1
```

> **Windows 用户注意**: 项目已针对 Windows 的 multiprocessing 进行优化，可直接运行。

### 3. 使用 API

**方式A: 浏览器访问 API 文档**
```
http://localhost:8000/docs
```

**方式B: Python 客户端**
```python
python client_example.py
```

**方式C: cURL 命令**
```bash
# 提交任务
curl -X POST http://localhost:8000/api/v1/tasks/submit \
  -F "file=@document.pdf" \
  -F "lang=ch"

# 查询状态（任务完成后自动返回解析内容）
curl http://localhost:8000/api/v1/tasks/{task_id}

# 查询状态并上传图片到MinIO
curl http://localhost:8000/api/v1/tasks/{task_id}?upload_images=true
```

## 📁 项目结构

```
mineru_tianshu/
├── task_db.py              # 数据库管理
├── api_server.py           # API 服务器
├── litserve_worker.py      # Worker Pool (MinerU + MarkItDown)
├── task_scheduler.py       # 任务调度器
├── start_all.py            # 启动脚本
├── client_example.py       # 客户端示例
└── requirements.txt        # 依赖配置
```

## 📚 使用示例

### 示例 1: 提交任务并等待结果

```python
import requests
import time

# 提交文档
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/tasks/submit',
        files={'file': f},
        data={'lang': 'ch', 'priority': 0}
    )
    task_id = response.json()['task_id']
    print(f"✅ 任务已提交: {task_id}")

# 轮询等待完成
while True:
    response = requests.get(f'http://localhost:8000/api/v1/tasks/{task_id}')
    result = response.json()
    
    if result['status'] == 'completed':
        # 任务完成，自动返回解析内容
        if result.get('data'):
            content = result['data']['content']
            print(f"✅ 解析完成，内容长度: {len(content)} 字符")
            
            # 保存结果
            with open('output.md', 'w', encoding='utf-8') as f:
                f.write(content)
        break
    elif result['status'] == 'failed':
        print(f"❌ 失败: {result['error_message']}")
        break
    
    print(f"⏳ 处理中... 状态: {result['status']}")
    time.sleep(2)
```

### 示例 2: 图片上传到 MinIO

```python
import requests

task_id = "your-task-id"

# 查询状态并上传图片到 MinIO
response = requests.get(
    f'http://localhost:8000/api/v1/tasks/{task_id}',
    params={'upload_images': True}
)

result = response.json()
if result['status'] == 'completed' and result.get('data'):
    # 图片已替换为 MinIO URL
    content = result['data']['content']
    print(f"✅ 图片已上传: {result['data']['images_uploaded']}")
    
    with open('output_with_cloud_images.md', 'w', encoding='utf-8') as f:
        f.write(content)
```

### 示例 3: 批量处理

```python
import requests
import concurrent.futures

files = ['doc1.pdf', 'report.docx', 'data.xlsx']

def process_file(file_path):
    # 提交任务
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/v1/tasks/submit',
            files={'file': f}
        )
    return response.json()['task_id']

# 并发提交
with concurrent.futures.ThreadPoolExecutor() as executor:
    task_ids = list(executor.map(process_file, files))
    print(f"✅ 已提交 {len(task_ids)} 个任务")
```

### 示例 4: 使用内置客户端

```bash
# 运行完整示例
python client_example.py

# 运行特定示例
python client_example.py single   # 单任务
python client_example.py batch    # 批量任务
python client_example.py priority # 优先级队列
```

## ⚙️ 配置说明

### 启动参数

```bash
python start_all.py [选项]

选项:
  --output-dir PATH         输出目录 (默认: /tmp/mineru_tianshu_output)
  --api-port PORT          API端口 (默认: 8000)
  --worker-port PORT       Worker端口 (默认: 9000)
  --accelerator TYPE       加速器类型: auto/cuda/cpu/mps (默认: auto)
  --workers-per-device N   每个GPU的worker数 (默认: 1)
  --devices DEVICES        使用的GPU设备 (默认: auto，使用所有GPU)
```

### 配置示例

```bash
# CPU模式（无GPU或测试）
python start_all.py --accelerator cpu

# GPU模式: 24GB显卡，每卡2个worker
python start_all.py --accelerator cuda --workers-per-device 2

# 指定GPU: 只使用GPU 0和1
python start_all.py --accelerator cuda --devices 0,1

# 自定义端口
python start_all.py --api-port 8080 --worker-port 9090

# Mac M系列芯片
python start_all.py --accelerator mps
```

### MinIO 配置（可选）

如需使用图片上传到 MinIO 功能：

```bash
export MINIO_ENDPOINT="your-endpoint.com"
export MINIO_ACCESS_KEY="your-access-key"
export MINIO_SECRET_KEY="your-secret-key"
export MINIO_BUCKET="your-bucket"
```

### 硬件要求

| 后端 | 显存要求 | 推荐配置 |
|------|---------|---------|
| pipeline | 6GB+ | RTX 2060 以上 |
| vlm-transformers | 8GB+ | RTX 3060 以上 |
| vlm-vllm-engine | 8GB+ | RTX 4070 以上 |

## 📡 API 接口

> 完整文档: http://localhost:8000/docs

### 1. 提交任务
```http
POST /api/v1/tasks/submit

参数:
  file: 文件 (必需)
  backend: pipeline | vlm-transformers | vlm-vllm-engine (默认: pipeline)
  lang: ch | en | korean | japan | ... (默认: ch)
  priority: 0-100 (数字越大越优先，默认: 0)
```

### 2. 查询任务
```http
GET /api/v1/tasks/{task_id}?upload_images=false

参数:
  upload_images: 是否上传图片到 MinIO (默认: false)

返回:
  - status: pending | processing | completed | failed
  - data: 任务完成后返回 Markdown 内容
```

### 3. 队列统计
```http
GET /api/v1/queue/stats

返回: 各状态任务数量统计
```

### 4. 取消任务
```http
DELETE /api/v1/tasks/{task_id}

只能取消 pending 状态的任务
```

## 🔧 故障排查

### 问题1: Worker 无法启动

**检查GPU**
```bash
nvidia-smi  # 应显示GPU信息
```

**检查依赖**
```bash
pip list | grep -E "(mineru|litserve|torch)"
```

### 问题2: 任务一直 pending

**检查调度器**
```bash
ps aux | grep task_scheduler.py
```

**手动触发**
```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"action":"poll"}'
```

### 问题3: 显存不足

**减少worker数量**
```bash
python start_all.py --workers-per-device 1
```

**设置显存限制**
```bash
export MINERU_VIRTUAL_VRAM_SIZE=6
python start_all.py
```

### 问题4: 端口被占用

**查看占用**
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

**使用其他端口**
```bash
python start_all.py --api-port 8080
```

## 🛠️ 技术栈

- **Web**: FastAPI + Uvicorn
- **解析器**: MinerU (PDF/图片) + MarkItDown (Office/文本)
- **GPU 调度**: LitServe
- **存储**: SQLite + MinIO (可选)
- **日志**: Loguru

## 📝 核心依赖

```txt
mineru[core]>=2.5.0      # MinerU 核心
fastapi>=0.115.0         # Web 框架
litserve>=0.2.0          # GPU 负载均衡
markitdown>=0.1.3        # Office 文档解析
minio>=7.2.0             # MinIO 对象存储
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

遵循 MinerU 主项目许可证

---

**天枢 (Tianshu)** - 企业级多 GPU 文档解析服务 ⚡️

*北斗第一星，寓意核心调度能力*

