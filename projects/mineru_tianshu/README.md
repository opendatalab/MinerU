# MinerU Tianshu (天枢)

> 天枢 - 企业级多GPU文档解析服务  
> 结合 SQLite 任务队列 + LitServe GPU负载均衡的最佳方案

## 🌟 核心特性

### 高性能架构
- ✅ **Worker 主动拉取** - 0.5秒响应速度,无需调度器触发
- ✅ **并发安全** - 原子操作防止任务重复,支持多Worker并发
- ✅ **GPU 负载均衡** - LitServe 自动调度,避免显存冲突
- ✅ **多GPU隔离** - 每个进程只使用分配的GPU,彻底解决多卡占用

### 企业级功能
- ✅ **异步处理** - 客户端立即响应（~100ms）,无需等待处理完成
- ✅ **任务持久化** - SQLite 存储,服务重启任务不丢失
- ✅ **优先级队列** - 重要任务优先处理
- ✅ **自动清理** - 定期清理旧结果文件,保留数据库记录

### 智能解析
- ✅ **双解析器** - PDF/图片用 MinerU(GPU加速), Office/HTML等用 MarkItDown(快速)
- ✅ **内容获取** - API自动返回 Markdown 内容,支持图片上传到 MinIO
- ✅ **RESTful API** - 支持任何编程语言接入
- ✅ **实时查询** - 随时查看任务进度和状态

## 🏗️ 系统架构

```
客户端请求 → FastAPI Server (立即返回 task_id)
                    ↓
              SQLite 任务队列 (并发安全)
                    ↓
         LitServe Worker Pool (主动拉取 + GPU自动负载均衡)
                    ↓
              MinerU / MarkItDown 解析
                    ↓
         Task Scheduler (可选监控组件)
```

**架构特点**:
- ✅ **Worker 主动模式**: Workers 持续循环拉取任务,无需调度器触发
- ✅ **并发安全**: SQLite 使用原子操作防止任务重复处理
- ✅ **自动负载均衡**: LitServe 自动分配任务到空闲 GPU
- ✅ **智能解析**: PDF/图片用 MinerU,其他格式用 MarkItDown

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
├── task_db.py              # 数据库管理 (并发安全,支持清理)
├── api_server.py           # API 服务器 (自动返回内容)
├── litserve_worker.py      # Worker Pool (主动拉取 + 双解析器)
├── task_scheduler.py       # 任务调度器 (可选监控)
├── start_all.py            # 启动脚本
├── client_example.py       # 客户端示例
└── requirements.txt        # 依赖配置
```

**核心组件说明**:
- `task_db.py`: 使用原子操作保证并发安全,支持旧任务清理
- `api_server.py`: 查询接口自动返回Markdown内容,支持MinIO图片上传
- `litserve_worker.py`: Worker主动循环拉取任务,支持MinerU和MarkItDown双解析
- `task_scheduler.py`: 可选组件,仅用于监控和健康检查(默认5分钟监控,15分钟健康检查)

## 📚 使用示例

### 示例 1: 提交任务并等待结果 (新版本 - 自动返回内容)

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
        # v2.0 新特性: 任务完成后自动返回解析内容
        if result.get('data'):
            content = result['data']['content']
            print(f"✅ 解析完成，内容长度: {len(content)} 字符")
            print(f"   解析方法: {result['data'].get('parser', 'Unknown')}")
            
            # 保存结果
            with open('output.md', 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # 结果文件已被清理
            print(f"⚠️  任务完成但结果文件已清理: {result.get('message', '')}")
        break
    elif result['status'] == 'failed':
        print(f"❌ 失败: {result['error_message']}")
        break
    
    print(f"⏳ 处理中... 状态: {result['status']}")
    time.sleep(2)
```

### 示例 2: 图片上传到 MinIO (可选功能)

```python
import requests

task_id = "your-task-id"

# v2.0: 查询时自动返回内容,同时可选上传图片到 MinIO
response = requests.get(
    f'http://localhost:8000/api/v1/tasks/{task_id}',
    params={'upload_images': True}  # 启用图片上传
)

result = response.json()
if result['status'] == 'completed' and result.get('data'):
    # 图片已替换为 MinIO URL (HTML img 标签格式)
    content = result['data']['content']
    images_uploaded = result['data']['images_uploaded']
    
    print(f"✅ 图片已上传到 MinIO: {images_uploaded}")
    print(f"   内容长度: {len(content)} 字符")
    
    # 保存包含 MinIO 图片链接的 Markdown
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
  --output-dir PATH                 输出目录 (默认: /tmp/mineru_tianshu_output)
  --api-port PORT                   API端口 (默认: 8000)
  --worker-port PORT                Worker端口 (默认: 9000)
  --accelerator TYPE                加速器类型: auto/cuda/cpu/mps (默认: auto)
  --workers-per-device N            每个GPU的worker数 (默认: 1)
  --devices DEVICES                 使用的GPU设备 (默认: auto，使用所有GPU)
  --poll-interval SECONDS           Worker拉取任务间隔 (默认: 0.5秒)
  --enable-scheduler                启用可选的任务调度器 (默认: 不启动)
  --monitor-interval SECONDS        调度器监控间隔 (默认: 300秒=5分钟)
  --cleanup-old-files-days N        清理N天前的结果文件 (默认: 7天, 0=禁用)
```

**新增功能说明**:
- `--poll-interval`: Worker空闲时拉取任务的频率,默认0.5秒响应极快
- `--enable-scheduler`: 是否启动调度器(可选),仅用于监控和健康检查
- `--monitor-interval`: 调度器日志输出频率,建议5-10分钟避免刷屏
- `--cleanup-old-files-days`: 自动清理旧结果文件但保留数据库记录

### 配置示例

```bash
# 基础启动（推荐）
python start_all.py

# CPU模式（无GPU或测试）
python start_all.py --accelerator cpu

# GPU模式: 24GB显卡，每卡2个worker
python start_all.py --accelerator cuda --workers-per-device 2

# 指定GPU: 只使用GPU 0和1
python start_all.py --accelerator cuda --devices 0,1

# 启用监控调度器（可选）
python start_all.py --enable-scheduler --monitor-interval 300

# 调整Worker拉取频率（高负载场景）
python start_all.py --poll-interval 1.0

# 禁用旧文件清理（保留所有结果）
python start_all.py --cleanup-old-files-days 0

# 完整配置示例
python start_all.py \
  --accelerator cuda \
  --devices 0,1 \
  --workers-per-device 2 \
  --poll-interval 0.5 \
  --enable-scheduler \
  --monitor-interval 300 \
  --cleanup-old-files-days 7

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
  - data: 任务完成后**自动返回** Markdown 内容
    - markdown_file: 文件名
    - content: 完整的 Markdown 内容
    - images_uploaded: 是否已上传图片
    - has_images: 是否包含图片
  - message: 如果结果文件已清理会提示
  
注意:
  - v2.0 新特性: 完成的任务会自动返回内容,无需额外请求
  - 如果结果文件已被清理(超过保留期),data 为 null 但任务记录仍可查询
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

### 5. 管理接口

**重置超时任务**
```http
POST /api/v1/admin/reset-stale?timeout_minutes=60

将超时的 processing 任务重置为 pending
```

**清理旧任务**
```http
POST /api/v1/admin/cleanup?days=7

仅用于手动触发清理(自动清理会每24小时执行一次)
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

> ⚠️ **重要**: Worker 现在是主动拉取模式,不需要调度器触发!

**检查 Worker 是否运行**
```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep litserve_worker
```

**检查 Worker 健康状态**
```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"action":"health"}'
```

**查看数据库状态**
```bash
python -c "from task_db import TaskDB; db = TaskDB(); print(db.get_queue_stats())"
```

### 问题3: 显存不足或多卡占用

**减少worker数量**
```bash
python start_all.py --workers-per-device 1
```

**设置显存限制**
```bash
export MINERU_VIRTUAL_VRAM_SIZE=6
python start_all.py
```

**指定特定GPU**
```bash
# 只使用GPU 0
python start_all.py --devices 0
```

> 💡 **提示**: 新版本已修复多卡显存占用问题,通过设置 `CUDA_VISIBLE_DEVICES` 确保每个进程只使用分配的GPU

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
python start_all.py --api-port 8080 --worker-port 9090
```

### 问题5: 结果文件丢失

**查询任务状态**
```bash
curl http://localhost:8000/api/v1/tasks/{task_id}
```

**说明**: 如果返回 `result files have been cleaned up`,说明结果文件已被清理(默认7天后)

**解决方案**:
```bash
# 延长保留时间为30天
python start_all.py --cleanup-old-files-days 30

# 或禁用自动清理
python start_all.py --cleanup-old-files-days 0
```

### 问题6: 任务重复处理

**症状**: 同一个任务被多个 worker 处理

**原因**: 这不应该发生,数据库使用了原子操作防止重复

**排查**:
```bash
# 检查是否有多个 TaskDB 实例连接不同的数据库文件
# 确保所有组件使用同一个 mineru_tianshu.db
```

## 🛠️ 技术栈

- **Web**: FastAPI + Uvicorn
- **解析器**: MinerU (PDF/图片) + MarkItDown (Office/文本/HTML等)
- **GPU 调度**: LitServe (自动负载均衡)
- **存储**: SQLite (并发安全) + MinIO (可选)
- **日志**: Loguru
- **并发模型**: Worker主动拉取 + 原子操作

## 🆕 版本更新说明

### v2.0 重大改进

**1. Worker 主动拉取模式**
- ✅ Workers 持续循环拉取任务,无需调度器触发
- ✅ 默认 0.5 秒拉取间隔,响应速度极快
- ✅ 空闲时自动休眠,不占用CPU资源

**2. 数据库并发安全增强**
- ✅ 使用 `BEGIN IMMEDIATE` 和原子操作
- ✅ 防止任务重复处理
- ✅ 支持多 Worker 并发拉取

**3. 调度器变为可选**
- ✅ 不再是必需组件,Workers 可独立运行
- ✅ 仅用于系统监控和健康检查
- ✅ 默认不启动,减少系统开销

**4. 结果文件清理功能**
- ✅ 自动清理旧结果文件(默认7天)
- ✅ 保留数据库记录供查询
- ✅ 可配置清理周期或禁用

**5. API 自动返回内容**
- ✅ 查询接口自动返回 Markdown 内容
- ✅ 无需额外请求获取结果
- ✅ 支持图片上传到 MinIO

**6. 多GPU显存优化**
- ✅ 修复多卡显存占用问题
- ✅ 每个进程只使用分配的GPU
- ✅ 通过 `CUDA_VISIBLE_DEVICES` 隔离

### 迁移指南 (v1.x → v2.0)

**无需修改代码**,只需注意:
1. 调度器现在是可选的,不启动也能正常工作
2. 结果文件默认7天后清理,如需保留请设置 `--cleanup-old-files-days 0`
3. API 查询接口现在会返回 `data` 字段包含完整内容

### 性能提升

| 指标 | v1.x | v2.0 | 提升 |
|-----|------|------|-----|
| 任务响应延迟<sup>※</sup> | 5-10秒 (调度器轮询) | 0.5秒 (Worker主动拉取) | **10-20倍** |
| 并发安全性 | 基础锁机制 | 原子操作 + 状态检查 | **可靠性提升** |
| 多GPU效率 | 有时会出现显存冲突 | 完全隔离,无冲突 | **稳定性提升** |
| 系统开销 | 调度器持续运行 | 可选监控(5分钟) | **资源节省** |

※ 任务响应延迟指任务添加到被 Worker 开始处理的时间间隔。v1.x 主要受调度器轮询间隔影响，非测量端到端处理时间。实际端到端响应时间还包括任务类型和系统负载所有因子。

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

