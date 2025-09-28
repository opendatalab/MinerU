# MinerU PDF批量OCR处理工具

## 功能概述

这是一个基于MinerU的高性能PDF批量OCR处理工具，支持多GPU并行处理，具有完善的进程监控和任务管理功能。

## 主要特性

### 🚀 高性能并行处理
- **多GPU支持**：可同时使用多张GPU卡进行并行处理
- **多进程架构**：每个GPU可运行多个工作进程
- **任务负载均衡**：智能分发任务到各个GPU进程池

### 📊 智能监控系统
- **实时进程监控**：监控每个工作进程的状态和心跳
- **任务超时管理**：自动检测并重启超时的工作进程
- **性能统计**：记录处理页数、耗时等统计信息
- **详细日志**：可选的进程监控日志记录

### 🔧 灵活配置
- **自定义显存大小**：可配置每个进程使用的GPU显存
- **语言支持**：支持多种OCR语言（默认英文）
- **公式和表格识别**：支持数学公式和表格结构识别
- **输出格式**：结果以压缩JSON格式保存

## 核心组件

### 1. PDF处理函数 (`infer_one_pdf`)
- 读取PDF文件并转换格式
- 执行文档分析和OCR识别
- 生成中间JSON和模型JSON结果
- 支持公式和表格识别

### 2. GPU进程池 (`GPUProcessPool`)
- 管理单个GPU的多个工作进程
- 监控进程健康状态和任务执行时间
- 自动重启故障进程
- 记录性能统计数据

### 3. 多GPU管理器 (`MultiGPUProcessManager`)
- 协调多个GPU进程池
- 分发任务到各个GPU
- 收集所有GPU的处理结果
- 统一的任务进度监控

### 4. 工作进程 (`gpu_specific_worker`)
- 绑定特定GPU设备
- 处理单个PDF文件
- 发送心跳和状态更新
- 错误处理和资源清理

## 使用方法

### 基本命令
```bash
python ocr_pdf_with_mineru.py \
  --input-dir /path/to/pdf/files \
  --output-dir /path/to/output \
  --cuda-devices "0,1,2" \
  --num-processes 2 \
  --vram-size-gb 24
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | str | 必需 | 输入PDF文件目录 |
| `--output-dir` | str | 必需 | 输出结果目录 |
| `--vram-size-gb` | int | 24 | 每个进程使用的GPU显存大小(GB) |
| `--cuda-devices` | str | "0" | 使用的GPU设备号，多个用逗号分隔 |
| `--num-processes` | int | 1 | 每个GPU的工作进程数 |
| `--log-dir` | str | "/tmp" | 日志输出目录 |
| `--task-timeout` | int | 1800 | 任务超时时间(秒) |
| `--max-task-duration` | int | 1800 | 单个任务最大执行时间(秒) |
| `--monitor-log-path` | str | None | 进程监控日志路径 |

### 使用示例

#### 单GPU处理
```bash
python ocr_pdf_with_mineru.py \
  --input-dir ./pdfs \
  --output-dir ./results \
  --cuda-devices "0" \
  --num-processes 1
```

#### 多GPU并行处理
```bash
python ocr_pdf_with_mineru.py \
  --input-dir ./pdfs \
  --output-dir ./results \
  --cuda-devices "0,1,2,3" \
  --num-processes 2 \
  --vram-size-gb 16 \
  --monitor-log-path ./monitor_logs
```

## 输出格式

### 处理结果
每个PDF文件处理完成后会生成一个压缩的JSON文件：
- 文件名：`原PDF文件名.json.zip`
- 内容：包含OCR结果的JSON数据
- 结构：包含middle_json、model_json和原文件路径

### 日志文件
- **主日志**：包含处理统计和错误信息
- **监控日志**：记录每个进程的性能数据
- **页数统计**：在logs/count_page.txt中记录每个PDF的页数
- **异常事件日志**：在logs/anomaly_events.txt中记录所有异常事件
- **定时监控日志**：在logs/monitoring_status.txt中记录定时进程状态

#### 异常事件日志格式 (anomaly_events.txt)
异常事件日志记录所有超时、重启和异常检测事件：

**超时事件：**
```json
{
  "timestamp": "2025-01-15 14:30:25",
  "unix_timestamp": 1705312225.123,
  "event_type": "timeout",
  "gpu_id": 0,
  "worker_id": 1,
  "pid": 12345,
  "timeout_type": "task_timeout",
  "duration_seconds": 1825.5,
  "task_file": "/path/to/document.pdf",
  "action": "kill_and_restart"
}
```

**进程重启事件：**
```json
{
  "timestamp": "2025-01-15 14:32:10",
  "unix_timestamp": 1705312330.456,
  "event_type": "restart",
  "gpu_id": 0,
  "old_worker_id": 1,
  "old_pid": 12345,
  "new_worker_id": 5,
  "new_pid": 12567,
  "restart_reason": "task_timeout",
  "restart_count": 3,
  "action": "process_restarted"
}
```

**异常检测事件：**
```json
{
  "timestamp": "2025-01-15 14:35:00",
  "unix_timestamp": 1705312500.789,
  "event_type": "anomaly_detection",
  "gpu_id": 0,
  "active_processes": 1,
  "expected_processes": 2,
  "anomalies": ["Process count mismatch: 1/2", "Long running tasks: ['Worker 3: 1850.2s (/path/to/large.pdf)']"],
  "action": "monitoring_alert"
}
```

#### 定时监控日志格式 (monitoring_status.txt)
定时监控日志每分钟记录一次进程状态：

**正常监控记录：**
```json
{
  "timestamp": "2025-01-15 14:33:00",
  "unix_timestamp": 1705312380.123,
  "gpu_id": 0,
  "type": "periodic_monitoring",
  "active_processes": 2,
  "expected_processes": 2,
  "idle_processes": 1,
  "busy_processes": 1,
  "restart_count_total": 0
}
```

**程序关闭记录：**
```json
{
  "timestamp": "2025-01-15 15:45:30",
  "unix_timestamp": 1705316730.456,
  "gpu_id": 0,
  "type": "shutdown_complete",
  "workers_terminated": 2,
  "total_restarts": 3,
  "action": "process_pool_stopped"
}
```

## 环境配置

### 必需依赖
```bash
# 安装MinerU
pip install mineru

# 其他依赖
pip install loguru PyMuPDF psutil torch
```

### 环境变量
- `CUDA_VISIBLE_DEVICES`：自动设置为指定的GPU设备
- `MINERU_VIRTUAL_VRAM_SIZE`：GPU虚拟显存大小
- `MINERU_MODEL_SOURCE`：模型源设置为"modelscope"
- `PYTORCH_CUDA_ALLOC_CONF`：GPU内存分配策略

## 性能优化建议

### GPU配置
- **显存管理**：根据GPU显存大小调整`--vram-size-gb`参数
- **进程数量**：一般建议每个GPU运行1-2个进程
- **超时设置**：根据PDF复杂度调整超时时间

### 系统资源
- **内存**：确保系统有足够内存支持多进程运行
- **存储**：使用SSD可显著提升I/O性能
- **CPU**：多核CPU有助于并行处理

## 监控和调试

### 实时监控
- 终端会显示任务分发和完成进度
- 每10秒输出处理进度百分比
- 每10分钟输出各工作进程的性能统计

### 自动重启机制
工具具备完善的进程监控和自动重启功能：

#### 监控检测
- **进程死亡检测**：每3秒检查所有工作进程的存活状态
- **心跳超时检测**：监控进程心跳，检测无响应的进程
- **任务超时检测**：监控单个任务的执行时间，防止长时间卡死

#### 自动重启流程
1. **检测异常**：发现进程死亡、超时或无响应
2. **记录日志**：将超时/重启事件写入日志文件
3. **终止进程**：安全终止故障进程，清理GPU内存
4. **创建新进程**：立即创建新的工作进程替代
5. **恢复处理**：新进程自动接管任务队列继续处理

#### 重启原因
- `process_died`: 进程意外死亡或崩溃
- `task_timeout`: 单个任务执行时间超过限制
- `heartbeat_timeout`: 进程失去响应（心跳超时）

### 错误处理
- 自动重试失败的PDF文件
- 记录详细的错误堆栈信息
- 智能进程重启，维持处理能力

### 性能分析
- 记录每个PDF的处理时间
- 统计平均处理速度（页/小时）
- 提供成功率和失败原因分析

## 注意事项

1. **GPU内存**：确保GPU有足够显存运行OCR模型
2. **文件权限**：确保对输入输出目录有读写权限
3. **模型下载**：首次运行时会自动下载OCR模型
4. **并发控制**：避免同时运行多个实例处理相同目录
5. **磁盘空间**：确保输出目录有足够空间存储结果

## 故障排除

### 常见问题
- **GPU内存不足**：减少进程数或降低显存使用量
- **进程死锁**：检查PDF文件是否损坏或过大
- **模型加载失败**：检查网络连接和模型源配置

### 日志分析
- 查看主日志了解整体处理状态
- 检查监控日志分析性能瓶颈
- 关注错误日志定位具体问题

这个工具适用于大规模PDF文档的批量OCR处理，特别是在需要高性能和可靠性的生产环境中。通过合理配置参数，可以充分利用多GPU资源实现高效的文档处理。