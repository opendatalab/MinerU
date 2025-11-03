"""
MinerU FastAPI主文件 - 提供PDF解析的Web API服务

PDF解析流程：
1. 用户通过POST /submit_task上传PDF文件
2. 文件被转换为Task对象并加入到任务队列
3. 后台工作线程从队列中取出任务，调用do_parse()函数进行解析
4. do_parse()函数位于mineru.cli.common模块中，包含完整的解析逻辑
5. 解析结果被打包为ZIP文件供用户下载
"""

import uuid
import os
import sys

# 在任何可能初始化 GPU 的库导入之前，从配置文件读取 GPU 设置并应用
from pathlib import Path
import yaml

def _apply_gpu_env_from_config():
    try:
        cfg_path = Path(__file__).parent / "fast_api_hyper-parameter.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gpu_cfg = ((cfg.get("system") or {}).get("gpu") or {})
            cuda_order = gpu_cfg.get("cuda_device_order")
            if cuda_order:
                os.environ["CUDA_DEVICE_ORDER"] = str(cuda_order)
            cuda_visible = gpu_cfg.get("cuda_visible_devices")
            if cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
                os.environ["NVIDIA_VISIBLE_DEVICES"] = str(cuda_visible)
            # 设置 PyTorch CUDA 架构列表，若未配置则默认 3090 的 8.6
            torch_arch = gpu_cfg.get("torch_cuda_arch_list")
            if torch_arch is None or str(torch_arch).strip() == "":
                torch_arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or "8.6"
            os.environ["TORCH_CUDA_ARCH_LIST"] = str(torch_arch)
            os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
    except Exception:
        # 配置读取失败时不阻断启动
        pass

_apply_gpu_env_from_config()

def _apply_gpu_env_from_cfg_dict(cfg) -> None:
    """根据已加载的配置对象应用 GPU 环境变量（用于 CLI 指定自定义配置文件的场景）。"""
    try:
        gpu_cfg = ((cfg.get("system") or {}).get("gpu") or {})
        cuda_order = gpu_cfg.get("cuda_device_order")
        if cuda_order:
            os.environ["CUDA_DEVICE_ORDER"] = str(cuda_order)
        cuda_visible = gpu_cfg.get("cuda_visible_devices")
        if cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
            os.environ["NVIDIA_VISIBLE_DEVICES"] = str(cuda_visible)
        # 设置 PyTorch CUDA 架构列表，若未配置则默认 3090 的 8.6
        torch_arch = gpu_cfg.get("torch_cuda_arch_list")
        if torch_arch is None or str(torch_arch).strip() == "":
            torch_arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or "8.6"
        os.environ["TORCH_CUDA_ARCH_LIST"] = str(torch_arch)
        os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
    except Exception:
        # 不阻断启动
        pass

# 确保优先使用当前工作区源码包（避免误用 site-packages 中的已安装版本）
# 期望将 ".../Mineru/MinerU" 加入 sys.path，使得 `import mineru` 指向本地源码
workspace_pkg_root = Path(__file__).resolve().parents[2]
if (workspace_pkg_root / "mineru").exists():
    sys.path.insert(0, str(workspace_pkg_root))

import uvicorn
import click
import threading
import zipfile
import tempfile
import shutil
from pathlib import Path
from glob import glob
from queue import Queue
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Optional, Dict, Any
from loguru import logger
from base64 import b64encode
import yaml
import json
from typing import Dict, Any, Optional
import multiprocessing as mp
import multiprocessing.context as mp_ctx

from mineru.cli.common import read_and_convert_to_pdf_bytes, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.version import __version__
from mineru.cli.task_queue_manager import TaskQueueManager, TaskPersistenceManager, Task, TaskStatus, FileData

# 全局配置
config = {}


def load_config(config_file: str = "fast_api_hyper-parameter.yaml") -> Dict[str, Any]:
    """
    从 YAML 加载配置；若不存在或无效则抛错，不提供默认配置。

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    config_path = Path(__file__).parent / config_file

    if not config_path.exists():
        raise RuntimeError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or not cfg:
            raise RuntimeError("Configuration file is empty or not a valid mapping")
        logger.info(f"Configuration loaded from {config_path}")
        return cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

# 加载配置
config = load_config()

# Lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    global task_manager

    # 启动逻辑：创建并启动任务管理器
    # 注意：参数处理函数已内置在 TaskQueueManager 模块中
    if task_manager is None:
        task_manager = TaskQueueManager(
            config,
            task_persistence_manager=task_persistence,
        )
    task_manager.start_workers()
    print("Task queue workers started")

    yield

    # 关闭逻辑
    if task_manager:
        task_manager.stop_workers()
        print("Task queue workers stopped")

# 当服务部署在 Nginx 子路径下（例如通过 /3403/ 访问）时，
# 需要为 FastAPI 指定 root_path 与显式的 docs/openapi 路径，否则 Swagger UI 会在浏览器端请求根路径 /openapi.json 导致 404。
# 这里采用端口号作为前缀以与 Nginx 路由保持一致：/3403/
_prefix = f"/{config['system']['server']['port']}"

app = FastAPI(
    title="MinerU FastAPI Service",
    description="基于 vLLM 引擎的异步文档解析服务",
    version=__version__,
    lifespan=lifespan,
    root_path=_prefix,
    # 注意：在设置了 root_path 后，这里的路径应使用相对根路径，
    # 最终对外将自动呈现为 `${root_path}/docs` 等
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# 统一响应构造与全局异常处理
def _json_response(code: int, msg: str, data: Any = None, status_code: int = 200) -> JSONResponse:
    payload: Dict[str, Any] = {"code": int(code), "msg": str(msg)}
    if data is not None:
        payload["data"] = data
    # 使用 jsonable_encoder 确保所有内容均可被 JSON 序列化（例如 ValueError 等异常对象）
    safe_content = jsonable_encoder(payload, exclude_none=False)
    return JSONResponse(status_code=status_code, content=safe_content)


def api_ok(msg: str = "ok", data: Any = None, status_code: int = 200) -> JSONResponse:
    return _json_response(0, msg, data, status_code)


def api_error(msg: str, data: Any = None, status_code: int = 400) -> JSONResponse:
    return _json_response(1, msg, data, status_code)


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(request: Request, exc: RequestValidationError):
    logger.exception(f"Validation error on {request.url}: {exc}")
    # 使用 jsonable_encoder 处理可能包含异常对象的 ctx 字段
    return api_error("Validation error", data={"errors": jsonable_encoder(exc.errors())}, status_code=422)


@app.exception_handler(HTTPException)
async def _handle_http_exception(request: Request, exc: HTTPException):
    try:
        msg = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    except Exception:
        msg = str(exc.detail)
    # 不隐藏错误，返回统一结构并保留 HTTP 状态码
    return api_error(msg, data={"path": str(request.url.path)}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def _handle_unexpected_exception(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.url}: {exc}")
    return api_error("Internal server error", data={"path": str(request.url.path)}, status_code=500)







# 全局任务持久化管理器
task_persistence = TaskPersistenceManager(config.get("persistence", {}).get("log_file", "fast_api_log.json"))

# VLM 推理现在在独立的引擎子进程中进行，不再需要主进程的推理锁


# ============================
# 引入拆分的 VLM 引擎子进程助手
# ============================
from mineru.cli.vlm_engine_helper import VLMEngineProcess

def get_supported_file_types():
    """获取支持的文件类型列表：强制仅允许 '.pdf'。"""
    return [".pdf"]

def is_file_type_supported(filename: str) -> bool:
    """检查文件类型是否支持：强制仅允许 '.pdf'。"""
    file_path = Path(filename)
    return file_path.suffix.lower() == ".pdf"





# 全局任务队列管理器（将在main函数中初始化）
task_manager = None

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.post("/submit_task")
async def submit_task(
        file: UploadFile = File(...),
):
    """提交任务到队列进行异步处理（仅允许一个 PDF）"""

    # 验证文件类型（仅 .pdf）
    if not is_file_type_supported(file.filename):
        file_path = Path(file.filename)
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path.suffix}. Allowed types: ['.pdf']")

    # 检查队列是否已满
    if task_manager.task_queue.full():
        raise HTTPException(status_code=429, detail="Task queue is full. Please try again later.")

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 准备任务特定参数（只保留真正任务特定的参数）
    # 注意：全局共享参数（formula_enable, table_enable等）已在TaskQueueManager中预计算
    defaults = config["task_defaults"]
    params = {
        # 任务特定：页面范围（允许每个任务不同）
        "start_page_id": defaults.get("start_page_id", 0),
        "end_page_id": defaults.get("end_page_id", 99999),
        # 其他任务特定参数可以在这里添加
    }

    # 在提交阶段立即读取文件内容，避免UploadFile生命周期问题
    content = await file.read()
    file_data_list = [FileData(file.filename, content)]

    # 创建任务，包含超时设置（优先 task_defaults，其次 system.timeout）
    final_timeout = defaults.get("timeout_minutes") or config["system"]["timeout"]["task_timeout_minutes"]
    task = Task(task_id, file_data_list, params, final_timeout)

    # 添加到持久化日志
    task_persistence.add_task(task)

    # 添加到队列
    success = task_manager.add_task(task)
    if not success:
        raise HTTPException(status_code=429, detail="Failed to add task to queue")

    return api_ok(
        msg="Task submitted successfully",
        data={
            "task_id": task_id,
            "status": "accepted",
            "queue_position": task_manager.task_queue.qsize(),
        },
        status_code=202,
    )


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    # 首先尝试从内存中获取任务（当前运行的任务）
    task = task_manager.get_task(task_id)

    if task:
        # 如果是内存中的任务，返回实时状态
        response = {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "queue_position": 0  # 简化实现，实际应该计算队列位置
        }

        # 返回实时进度
        response["current_page"] = task.current_page
        response["total_pages"] = task.total_pages

        if task.started_at:
            response["started_at"] = task.started_at.isoformat()

        if task.completed_at:
            response["completed_at"] = task.completed_at.isoformat()

        if task.processing_duration_seconds is not None:
            response["processing_duration_seconds"] = task.processing_duration_seconds

        if task.error_message:
            response["error_message"] = task.error_message

        # 超时相关（仅针对 processing 阶段计时）
        response["timeout_minutes"] = task.timeout_minutes
        if task.started_at:
            _elapsed_sec = int((datetime.now() - task.started_at).total_seconds())
            response["processing_elapsed_seconds"] = _elapsed_sec
            if task.status == TaskStatus.PROCESSING and isinstance(task.timeout_minutes, (int, float)):
                response["processing_remaining_seconds"] = max(0, int(task.timeout_minutes * 60 - _elapsed_sec))

        if task.status == TaskStatus.COMPLETED and not task.downloaded:
            response["download_url"] = f"/download/{task_id}"

        return api_ok("ok", response, 200)

    # 如果内存中没有找到，尝试从持久化日志中获取历史任务
    task_data = task_persistence.get_task_history(task_id)
    if task_data:
        # 从持久化日志返回任务信息，确保datetime对象转换为字符串
        response = {
            "task_id": task_data.get("task_id", task_id),
            "status": task_data.get("status", "unknown"),
            "downloaded": task_data.get("downloaded", False),
            "is_historical": True  # 标记这是历史任务
        }

        # 进度字段（历史任务）
        if "current_page" in task_data:
            response["current_page"] = task_data.get("current_page")
        if "total_pages" in task_data:
            response["total_pages"] = task_data.get("total_pages")

        # 处理created_at字段
        if task_data.get("created_at"):
            if isinstance(task_data["created_at"], datetime):
                response["created_at"] = task_data["created_at"].isoformat()
            else:
                response["created_at"] = str(task_data["created_at"])

        # 处理started_at字段
        if task_data.get("started_at"):
            if isinstance(task_data["started_at"], datetime):
                response["started_at"] = task_data["started_at"].isoformat()
            else:
                response["started_at"] = str(task_data["started_at"])

        # 处理completed_at字段
        if task_data.get("completed_at"):
            if isinstance(task_data["completed_at"], datetime):
                response["completed_at"] = task_data["completed_at"].isoformat()
            else:
                response["completed_at"] = str(task_data["completed_at"])

        # 处理processing_duration_seconds字段
        if task_data.get("processing_duration_seconds") is not None:
            response["processing_duration_seconds"] = task_data["processing_duration_seconds"]

        if task_data.get("error_message"):
            response["error_message"] = task_data["error_message"]

        # 历史任务的超时/进度信息（仅在 processing 阶段计时）
        tm = task_data.get("timeout_minutes")
        tm_int = None
        if isinstance(tm, (int, float)):
            tm_int = int(tm)
        elif isinstance(tm, str) and tm.strip().isdigit():
            tm_int = int(tm.strip())
        if tm_int is not None:
            response["timeout_minutes"] = tm_int

        if response.get("status") == "processing" and task_data.get("started_at"):
            _started = task_data["started_at"]
            if isinstance(_started, datetime):
                _elapsed_sec = int((datetime.now() - _started).total_seconds())
                response["processing_elapsed_seconds"] = _elapsed_sec
                if tm_int is not None:
                    response["processing_remaining_seconds"] = max(0, int(tm_int * 60 - _elapsed_sec))

        # 历史任务不提供下载链接，因为文件可能已被清理
        # 如果需要下载，应该提示用户任务结果可能已被清理

        return api_ok("ok", response, 200)

    # 任务不存在
    raise HTTPException(
        status_code=404,
        detail="Task not found"
    )


@app.get("/download/{task_id}")
async def download_task_result(task_id: str):
    """下载任务结果文件"""
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task.status.value}"
        )

    if not task.result_path or not os.path.exists(task.result_path):
        raise HTTPException(
            status_code=404,
            detail="Result file not found"
        )

    # 安全检查：确保文件路径在临时目录内，防止路径遍历攻击
    result_path = Path(task.result_path).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()

    # 检查文件是否在临时目录内
    try:
        result_path.relative_to(temp_dir)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied: Result file not in expected location"
        )

    # 额外检查：确保文件名格式正确
    # 期望压缩包名调整为 {task_id}.zip
    expected_filename = f"{task_id}.zip"
    if result_path.name != expected_filename:
        raise HTTPException(
            status_code=403,
            detail="Access denied: Invalid result file name"
        )

    # 标记任务为已下载
    task.downloaded = True
    task_manager.mark_task_downloaded(task_id)
    task_persistence.update_task(task)  # 持久化下载状态

    logger.info(f"Task {task_id} marked as downloaded, will be cleaned up by background thread")

    return FileResponse(
        path=task.result_path,
        media_type='application/zip',
        filename=f"{task_id}.zip"
    )


@app.get("/queue_status")
async def get_queue_status():
    """获取队列状态"""
    try:
        engine_max = (config.get("advanced", {}) or {}).get("model", {}) or {}
        # 优先读取 max_num_seqs（vLLM标准字段），回退到 max_running_requests（旧字段）
        server_args = (engine_max.get("server_args", {}) or {})
        engine_max = server_args.get("max_num_seqs", server_args.get("max_running_requests", 1))
    except Exception:
        engine_max = 1

    try:
        max_running_pdf = (config.get("system", {}) or {}).get("queue", {}).get("max_running_pdf", 1)
    except Exception:
        max_running_pdf = 1

    try:
        active_tasks = len(getattr(task_manager, "_active_task_ids", set()))
    except Exception:
        active_tasks = 0

    return api_ok(
        data={
            "queue_size": task_manager.task_queue.qsize(),
            "max_queue_size": task_manager.max_queue_size,
            "engine_max_running_requests": engine_max,
            "max_running_pdf": max_running_pdf,
            "active_running_pdf": active_tasks,
            "pending_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.PENDING]),
            "processing_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.PROCESSING]),
            "completed_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.FAILED]),
            "timeout_tasks": len([t for t in task_manager.tasks.values() if t.status == TaskStatus.TIMEOUT]),
            "downloaded_tasks": len([t for t in task_manager.tasks.values() if t.downloaded])
        },
        status_code=200,
    )


@app.get("/config")
async def get_config():
    """获取当前配置信息"""
    # 返回配置的副本，但隐藏敏感信息
    safe_config = config.copy()

    # 可以在这里添加敏感信息过滤逻辑
    # 例如：移除密码、密钥等敏感字段

    # 有效超时（仅在 processing 阶段计时）
    effective_timeout = ((config.get("task_defaults", {}) or {}).get("timeout_minutes")
                         or ((config.get("system", {}) or {}).get("timeout", {}) or {}).get("task_timeout_minutes"))

    return api_ok(
        data={
            "config": safe_config,
            "config_file": getattr(app.state, "config_file_path", "fast_api_hyper-parameter.yaml"),
            "last_loaded": "startup",
            "effective_task_timeout_minutes": effective_timeout,
            "timeout_applies_to": "processing"
        },
        status_code=200,
    )


@app.get("/task_history")
async def get_task_history(task_id: str = None):
    """获取任务历史记录"""
    if task_id:
        task_data = task_persistence.get_task_history(task_id)
        if not task_data:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found in history"
            )
        return api_ok("ok", task_data, 200)
    else:
        return api_ok("ok", task_persistence.get_task_history(), 200)


@app.get("/task_stats")
async def get_task_stats():
    """获取任务统计信息"""
    default_days = config.get("persistence", {}).get("history_cleanup_days", 7)
    return api_ok(
        data={
            "stats": task_persistence.get_task_stats(),
            "log_file": str(task_persistence.log_file),
            "total_logged_tasks": len(task_persistence.tasks_log),
            "history_cleanup_days_default": default_days
        },
        status_code=200,
    )


@app.post("/cleanup_history")
async def cleanup_task_history(days: int = None):
    """清理指定天数之前的任务历史记录"""
    if days is None:
        days = config.get("persistence", {}).get("history_cleanup_days", 7)
    task_persistence.cleanup_old_tasks(days)
    return api_ok(
        data={
            "message": f"Cleaned up task history older than {days} days",
            "remaining_tasks": len(task_persistence.tasks_log)
        },
        status_code=200,
    )





@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', help='Server host (overrides config file)')
@click.option('--port', type=int, help='Server port (overrides config file)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
@click.option('--max-queue-size', type=int, help='Maximum queue size (overrides config file)')
@click.option('--config-file', default='fast_api_hyper-parameter.yaml', help='Configuration file path')
@click.option('--enable-torch-compile', is_flag=True, help='Enable torch.compile (overrides config)')
def main(ctx, host, port, reload, max_queue_size, config_file, enable_torch_compile, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 重新加载配置（无条件使用传入的配置文件）
    global config
    config = load_config(config_file)
    # 依据已加载配置对象，重新应用 GPU 环境（覆盖默认 YAML 的设置）
    _apply_gpu_env_from_cfg_dict(config)

    # 将配置参数与配置文件路径存储到应用状态中
    app.state.config = kwargs
    app.state.config_file_path = config_file

    # 使用命令行参数覆盖配置文件
    final_host = host or config["system"]["server"]["host"]
    final_port = port or config["system"]["server"]["port"]
    final_reload = reload or config["system"]["server"]["reload"]
    final_max_queue_size = max_queue_size or config["system"]["queue"]["max_queue_size"]

    # 将最终值写回配置对象，确保后续组件读取到覆盖后的参数
    config["system"]["server"]["host"] = final_host
    config["system"]["server"]["port"] = final_port
    config["system"]["server"]["reload"] = bool(final_reload)
    config["system"]["queue"]["max_queue_size"] = final_max_queue_size

    # 覆盖/开启 Torch Compile（若通过 CLI 指定）
    if enable_torch_compile:
        try:
            if "advanced" not in config:
                config["advanced"] = {}
            if "model" not in config["advanced"]:
                config["advanced"]["model"] = {}
            if "server_args" not in config["advanced"]["model"]:
                config["advanced"]["model"]["server_args"] = {}
            config["advanced"]["model"]["server_args"]["enable_torch_compile"] = True
        except Exception as e:
            # 显式打印，避免悄然忽略（开发阶段需要了解问题）
            print(f"Failed to set enable_torch_compile via CLI: {e}")

    # 如果使用了自定义配置文件，重建任务持久化管理器以使 log_file 生效
    global task_persistence
    task_persistence = TaskPersistenceManager(
        config.get("persistence", {}).get("log_file", "fast_api_log.json")
    )

    """启动MinerU FastAPI服务器的命令行入口"""
    print("=== MinerU FastAPI Service with vLLM-Engine Backend ===")
    print(f"Configuration file: {config_file}")
    print(f"Server URL: http://{final_host}:{final_port}")
    print(f"Queue Size: {final_max_queue_size}")
    try:
        print(f"Max Running PDF (task concurrency): {config['system']['queue']['max_running_pdf']}")
    except Exception:
        pass
    try:
        engine_max = (config.get("advanced", {}) or {}).get("model", {}) or {}
        # 优先读取 max_num_seqs（vLLM标准字段），回退到 max_running_requests（旧字段）
        server_args = (engine_max.get("server_args", {}) or {})
        engine_max = server_args.get("max_num_seqs", server_args.get("max_running_requests", 1))
    except Exception:
        engine_max = 1
    print(f"Engine Max Running Requests: {engine_max}")
    try:
        per_pdf = (((config.get('advanced', {}) or {}).get('model', {}) or {}).get('inference', {}) or {}).get('max_concurrency_per_pdf')
        if per_pdf is not None:
            print(f"Per-PDF Page Concurrency: {per_pdf}")
    except Exception:
        pass
    print(f"Task Timeout: {config['system']['timeout']['task_timeout_minutes']} minutes")
    print()
    print("API Endpoints:")
    print(f"- Submit Task: POST http://{final_host}:{final_port}/submit_task")
    print(f"- Task Status: GET http://{final_host}:{final_port}/task_status/{{task_id}}")
    print(f"- Download Result: GET http://{final_host}:{final_port}/download/{{task_id}}")
    print(f"- Queue Status: GET http://{final_host}:{final_port}/queue_status")
    print(f"- Config Info: GET http://{final_host}:{final_port}/config")
    print(f"- Task History: GET http://{final_host}:{final_port}/task_history")
    print(f"- Task Stats: GET http://{final_host}:{final_port}/task_stats")
    print(f"- Cleanup History: POST http://{final_host}:{final_port}/cleanup_history")
    print()
    print("Task Persistence:")
    print(f"- Log File: {task_persistence.log_file}")
    print(f"- Loaded Tasks: {len(task_persistence.tasks_log)}")
    print()
    print("API Documentation:")
    print(f"- Swagger UI: http://{final_host}:{final_port}/docs")
    print(f"- ReDoc: http://{final_host}:{final_port}/redoc")

    # 直接传入 app 对象，确保 "python fast_api.py" 也能正常运行
    # 额外打印：处理阶段超时（processing-only）有效取值
    effective_timeout = ((config.get('task_defaults', {}) or {}).get('timeout_minutes')
                         or ((config.get('system', {}) or {}).get('timeout', {}) or {}).get('task_timeout_minutes'))
    if effective_timeout is not None:
        print(f"Effective Task Timeout (processing-only): {effective_timeout} minutes")

    uvicorn.run(
        app,
        host=final_host,
        port=final_port,
        reload=final_reload
    )


if __name__ == "__main__":
    main()

