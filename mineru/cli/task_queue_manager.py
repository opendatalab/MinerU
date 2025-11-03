import uuid
import os
import threading
import zipfile
import tempfile
import shutil
from pathlib import Path
from queue import Queue
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger
import json

from mineru.cli.common import read_and_convert_to_pdf_bytes, pdf_suffixes, image_suffixes
from .vlm_engine_helper import VLMEngineProcess


# 任务状态枚举
class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# 文件数据结构
class FileData:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content = content


# 任务数据结构
class Task:
    def __init__(self, task_id: str, file_data_list: List[FileData], params: Dict[str, Any], timeout_minutes: int = 40):
        self.task_id = task_id
        self.file_data_list = file_data_list  # 存储文件数据而不是UploadFile对象
        self.params = params
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.processing_duration_seconds = None  # 处理时间（从processing到完成的时间差，单位：秒）
        self.result_path = None
        self.error_message = None
        self.timeout_minutes = timeout_minutes
        self.downloaded = False  # 标记是否已被下载
        self.temp_dir = None  # 临时目录路径，用于清理
        # 进度信息：当前已完成页数 / 总页数（VLM 模式按页）
        self.current_page = 0
        self.total_pages = 0


# 任务日志持久化管理器
class TaskPersistenceManager:
    def __init__(self, log_file: str = "fast_api_log.json"):
        self.log_file = Path(__file__).parent / log_file
        self.tasks_log = {}
        self._lock = threading.Lock()  # 添加线程锁保护并发写入
        self._load_log()

    def _load_log(self):
        """加载任务日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换时间字符串回datetime对象
                    for task_id, task_data in data.items():
                        if 'created_at' in task_data and task_data['created_at']:
                            task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
                        if 'started_at' in task_data and task_data['started_at']:
                            task_data['started_at'] = datetime.fromisoformat(task_data['started_at'])
                        if 'completed_at' in task_data and task_data['completed_at']:
                            task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at'])
                    self.tasks_log = data
                logger.info(f"Loaded {len(self.tasks_log)} tasks from log file")
            except Exception as e:
                logger.warning(f"Failed to load task log: {e}")
                self.tasks_log = {}
        else:
            logger.info("Task log file not found, starting with empty log")
            self.tasks_log = {}

    def _save_log(self):
        """保存任务日志"""
        with self._lock:  # 使用线程锁保护并发写入
            try:
                # 转换为可JSON序列化的格式
                serializable_data = {}
                for task_id, task_data in self.tasks_log.items():
                    serializable_task = {}
                    for key, value in task_data.items():
                        if isinstance(value, datetime):
                            serializable_task[key] = value.isoformat()
                        elif isinstance(value, TaskStatus):
                            serializable_task[key] = value.value
                        else:
                            serializable_task[key] = value
                    serializable_data[task_id] = serializable_task

                # 确保目录存在
                self.log_file.parent.mkdir(parents=True, exist_ok=True)

                # 使用临时文件+原子移动来确保数据完整性
                temp_file = self.log_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                temp_file.replace(self.log_file)  # 原子操作

            except Exception as e:
                logger.error(f"Failed to save task log: {e}")

    def add_task(self, task: 'Task'):
        """添加任务到日志"""
        task_data = {
            'task_id': task.task_id,
            'status': task.status,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'processing_duration_seconds': task.processing_duration_seconds,
            'timeout_minutes': task.timeout_minutes,
            'downloaded': task.downloaded,
            'result_path': task.result_path,
            'error_message': task.error_message,
            'current_page': task.current_page,
            'total_pages': task.total_pages,
        }
        self.tasks_log[task.task_id] = task_data
        self._save_log()
        logger.info(f"Task {task.task_id} added to persistent log")

    def update_task(self, task: 'Task'):
        """更新任务状态"""
        if task.task_id in self.tasks_log:
            task_data = self.tasks_log[task.task_id]
            task_data['status'] = task.status
            task_data['started_at'] = task.started_at
            task_data['completed_at'] = task.completed_at
            task_data['processing_duration_seconds'] = task.processing_duration_seconds
            task_data['downloaded'] = task.downloaded
            task_data['result_path'] = task.result_path
            task_data['error_message'] = task.error_message
            task_data['current_page'] = task.current_page
            task_data['total_pages'] = task.total_pages
            self._save_log()

    def get_task_history(self, task_id: str = None) -> Dict[str, Any]:
        """获取任务历史"""
        def _make_serializable(data):
            """将数据转换为JSON可序列化格式"""
            if isinstance(data, dict):
                return {k: _make_serializable(v) for k, v in data.items()}
            elif isinstance(data, TaskStatus):
                return data.value
            elif isinstance(data, datetime):
                return data.isoformat()
            else:
                return data

        if task_id:
            task_data = self.tasks_log.get(task_id, {})
            return _make_serializable(task_data)
        else:
            return _make_serializable(self.tasks_log)

    def get_task_stats(self) -> Dict[str, int]:
        """获取任务统计信息"""
        stats = {
            'total_tasks': len(self.tasks_log),
            'pending_tasks': 0,
            'processing_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'timeout_tasks': 0,
            'downloaded_tasks': 0
        }

        for task_data in self.tasks_log.values():
            status = task_data.get('status')
            if status == TaskStatus.PENDING.value:
                stats['pending_tasks'] += 1
            elif status == TaskStatus.PROCESSING.value:
                stats['processing_tasks'] += 1
            elif status == TaskStatus.COMPLETED.value:
                stats['completed_tasks'] += 1
            elif status == TaskStatus.FAILED.value:
                stats['failed_tasks'] += 1
            elif status == TaskStatus.TIMEOUT.value:
                stats['timeout_tasks'] += 1

            if task_data.get('downloaded', False):
                stats['downloaded_tasks'] += 1

        return stats

    def cleanup_old_tasks(self, days: int = 7):
        """清理指定天数之前的任务记录"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        tasks_to_remove = []

        for task_id, task_data in self.tasks_log.items():
            created_at = task_data.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at).timestamp()
                elif isinstance(created_at, datetime):
                    created_at = created_at.timestamp()

                if created_at < cutoff_date:
                    tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks_log[task_id]
            logger.info(f"Cleaned up old task record: {task_id}")

        if tasks_to_remove:
            self._save_log()
            logger.info(f"Cleaned up {len(tasks_to_remove)} old task records")

def _get_inference_args_from_config(cfg: Dict[str, Any], server_args_filtered: Dict[str, Any]) -> Dict[str, Any]:
    """从配置中提取推理采样参数，并基于 max_model_len 计算安全的 max_new_tokens。"""
    model_cfg = (cfg.get("advanced", {}) or {}).get("model", {}) or {}
    infer_cfg = (model_cfg.get("inference", {}) or {})

    recognized_keys = {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "no_repeat_ngram_size",
        # 自定义：并发控制参数
        "max_concurrency_pdf",        # PDF槽位数
        "concurrent_processing_pages", # 页级并发数
        "preprocess_concurrency",      # 预处理并发数
        "prepared_queue_limit",        # 预处理队列上限
    }

    out: Dict[str, Any] = {k: v for k, v in infer_cfg.items() if k in recognized_keys}

    # 从 vLLM 的 max_model_len 获取上下文长度
    context_length = server_args_filtered.get("max_model_len")
    if not isinstance(context_length, int) or context_length <= 0:
        # 合理的兜底（常见大模型上下文）
        context_length = 16384

    # 结合 server_args 中的预填充预算与并发数估算单请求可用的预填充预算
    max_prefill_tokens = server_args_filtered.get("max_num_batched_tokens")
    if not isinstance(max_prefill_tokens, int) or max_prefill_tokens <= 0:
        # 若未配置，采用上下文一半作为保守预算
        max_prefill_tokens = context_length // 2
    # 并发：使用 vLLM 的 max_num_seqs
    max_running_requests = server_args_filtered.get("max_num_seqs")
    if not isinstance(max_running_requests, int) or max_running_requests <= 0:
        max_running_requests = 1

    # 近似按并发请求平分预填充预算，避免多请求时单请求阈值过低
    per_request_prefill_budget = max(256, max_prefill_tokens // max_running_requests)

    requested_max_new = infer_cfg.get("max_new_tokens")

    # 保留至少 64 token 余量，且不超过 context_length
    allowed_max_new = max(32, min(
        (context_length - 64),
        (context_length - per_request_prefill_budget - 64),
    ))

    if requested_max_new is not None:
        try:
            requested_value = int(requested_max_new)
        except Exception:
            requested_value = allowed_max_new
        final_max_new = max(32, min(requested_value, allowed_max_new))
    else:
        final_max_new = allowed_max_new

        if requested_max_new is not None and final_max_new < int(requested_max_new):
            logger.warning(
                f"Clamp max_new_tokens from {requested_max_new} to {final_max_new} "
                f"(max_model_len={context_length}, per_request_prefill_budget={per_request_prefill_budget})"
            )

    out["max_new_tokens"] = final_max_new

    return out




# 根据当前安装的 vllm 版本，过滤与映射 ServerArgs 字段，避免启动时因未知参数报错
def _get_filtered_server_args(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """根据已安装的 vLLM 版本过滤/映射引擎参数，生成可直接传入 vllm.LLM(**kwargs) 的参数集。

    直接使用 vLLM 标准的字段名：max_model_len、max_num_batched_tokens、max_num_seqs、gpu_memory_utilization、tensor_parallel_size。
    """
    import inspect
    allowed_params: set[str] = set()

    # 1) 优先从 vLLM 的 Engine/AsyncEngine 参数对象提取字段
    try:
        try:
            from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs  # type: ignore
        except Exception:
            # vLLM v1/v2 可能包路径有差异
            EngineArgs = None  # type: ignore
            AsyncEngineArgs = None  # type: ignore

        for cls in (EngineArgs, AsyncEngineArgs):
            if cls is None:
                continue
            try:
                sig = inspect.signature(cls)
                allowed_params.update([p for p in sig.parameters.keys() if p not in {"self"}])
            except Exception:
                try:
                    # dataclass 风格
                    from dataclasses import fields
                    allowed_params.update([f.name for f in fields(cls)])
                except Exception:
                    pass
    except Exception:
        pass

    # 2) 回退到 LLM.__init__ 的签名
    try:
        from vllm import LLM  # type: ignore
        sig_llm = inspect.signature(LLM.__init__)
        allowed_params.update([p for p in sig_llm.parameters.keys() if p not in {"self"}])
    except Exception:
        pass

    filtered: Dict[str, Any] = {}
    args = dict(raw_args or {})

    def map_if_supported(target_key: str, value: Any) -> bool:
        if target_key in allowed_params or not allowed_params:
            filtered[target_key] = value
            return True
        return False

    for key, value in args.items():
        mapped = False

        # 直接使用 vLLM 标准键名
        # 上下文长度
        if key in {"max_model_len"}:
            mapped = map_if_supported("max_model_len", value)
        # 总/预填充 token 限制
        elif key in {"max_num_batched_tokens"}:
            mapped = map_if_supported("max_num_batched_tokens", value)
        # 并发会话数
        elif key in {"max_num_seqs"}:
            mapped = map_if_supported("max_num_seqs", value)
        # 显存占比
        elif key in {"gpu_memory_utilization"}:
            mapped = map_if_supported("gpu_memory_utilization", value)
        # 数据/张量并行
        elif key in {"tensor_parallel_size"}:
            mapped = map_if_supported("tensor_parallel_size", value)
        # 显式丢弃：不支持/无需的键
        elif key in {"context_length", "context_len", "max_total_tokens", "max_total_num_tokens", "max_prefill_tokens", "max_running_requests", "max_requests", "mem_fraction_static", "gpu_mem_util", "dp_size", "tp_size", "attn_backend", "attention_backend", "enable_torch_compile", "torch_compile", "use_torch_compile", "enable-torch-compile", "chunked_prefill_size"}:
            # 直接忽略这些旧键名
            mapped = True

        # 直接透传：如果目标参数受支持
        if not mapped and (key in allowed_params or not allowed_params):
            filtered[key] = value
            mapped = True

        if not mapped:
            logger.warning(f"Unknown/unsupported vLLM engine arg ignored: {key}")

    return filtered


# 任务队列管理器
class TaskQueueManager:
    def __init__(self, config: Dict[str, Any], task_persistence_manager=None):
        self.config = config
        self.max_queue_size = config["system"]["queue"]["max_queue_size"]
        self.task_queue = Queue(maxsize=self.max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self._stop_event = threading.Event()
        self._cleanup_thread = None
        self._timeout_thread = None
        self._dispatcher_thread = None
        self._result_thread = None
        self.cleanup_check_interval = config["system"]["timeout"]["cleanup_check_interval"]
        self.cleanup_scan_interval = config["system"]["timeout"]["cleanup_scan_interval"]
        self._cleanup_lock = threading.Lock()  # 添加清理操作锁
        # 新增：任务级并发控制（独立于引擎 max_running_requests）
        try:
            self.max_running_pdf = int((config.get("system", {}).get("queue", {}) or {}).get("max_running_pdf", 3))
            if self.max_running_pdf <= 0:
                self.max_running_pdf = 1
        except Exception:
            self.max_running_pdf = 3
        self._task_concurrency_sema = threading.Semaphore(self.max_running_pdf)
        self._active_task_ids = set()  # 持有并发名额的任务ID
        # 常驻引擎子进程（必须创建并复用）
        self._engine_process: Optional[VLMEngineProcess] = None
        self.task_persistence = task_persistence_manager
        
        # ========== 全局共享参数（所有PDF任务共用，只计算一次）==========
        # 1. 引擎参数（vLLM配置）
        server_args_raw = (config.get("advanced", {}).get("model", {}).get("server_args", {}))
        self.global_server_args = _get_filtered_server_args(server_args_raw)
        
        # 2. 推理参数（采样配置）
        self.global_infer_args = _get_inference_args_from_config(config, self.global_server_args)
        # 注入新的页级并发配置（若提供）
        try:
            _concurrent_pages = int(((config.get("advanced", {}) or {}).get("model", {}) or {}).get("inference", {}).get("concurrent_processing_pages", 0))
            if _concurrent_pages and _concurrent_pages > 0:
                self.global_infer_args["concurrent_processing_pages"] = _concurrent_pages
        except Exception:
            pass
        
        # 3. 任务默认配置（功能开关、输出选项）
        task_defaults = config.get("task_defaults", {})
        self.global_task_config = {
            "output_dir": task_defaults.get("output_dir", "./output"),
            "formula_enable": task_defaults.get("formula_enable", True),
            "table_enable": task_defaults.get("table_enable", True),
            "return_md": task_defaults.get("return_md", True),
            "return_middle_json": task_defaults.get("return_middle_json", True),
            "return_model_output": task_defaults.get("return_model_output", True),
            "return_content_list": task_defaults.get("return_content_list", True),
        }
        
        logger.info(f"Global task config initialized: {self.global_task_config}")
        logger.info(f"Global server args: {self.global_server_args}")
        logger.info(f"Global infer args: {self.global_infer_args}")

    def start_workers(self):
        """启动所有后台线程"""
        # 先启动常驻引擎子进程（避免结果收集线程在引擎未就绪时报错）
        try:
            # 使用预计算的全局参数
            self._engine_process = VLMEngineProcess(self.global_server_args, self.global_infer_args)
            logger.info("VLM engine subprocess started and warmed up")
        except Exception as e:
            logger.error(f"Failed to start engine subprocess: {e}")
            raise

        # 再启动调度器线程（从队列取任务并提交到引擎子进程）
        self._dispatcher_thread = threading.Thread(target=self._dispatcher_loop, name="Dispatcher")
        self._dispatcher_thread.daemon = True
        self._dispatcher_thread.start()

        # 启动结果收集线程（从引擎共享结果队列读取并完成打包/状态更新）
        self._result_thread = threading.Thread(target=self._result_collector_loop, name="ResultCollector")
        self._result_thread.daemon = True
        self._result_thread.start()

        # 启动超时检查线程
        self._timeout_thread = threading.Thread(target=self._timeout_checker_loop, name="TimeoutChecker")
        self._timeout_thread.daemon = True
        self._timeout_thread.start()

        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, name="CleanupWorker")
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()

    def stop_workers(self):
        """停止所有后台线程"""
        self._stop_event.set()

        # 等待调度器与结果收集线程结束
        try:
            if self._dispatcher_thread and self._dispatcher_thread.is_alive():
                self._dispatcher_thread.join(timeout=2.0)
        except Exception as e:
            logger.warning(f"Error joining dispatcher thread: {e}")
        try:
            if self._result_thread and self._result_thread.is_alive():
                self._result_thread.join(timeout=2.0)
        except Exception as e:
            logger.warning(f"Error joining result collector thread: {e}")

        # 停止超时检查线程（会被 _stop_event 立即唤醒）
        if self._timeout_thread and self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=2.0)

        # 停止清理线程（会被 _stop_event 立即唤醒）
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

        # 停止常驻引擎子进程
        try:
            if self._engine_process is not None:
                self._engine_process.stop_engine()
        except Exception as e:
            logger.error(f"Error stopping engine subprocess: {e}")
            raise

    def add_task(self, task: Task) -> bool:
        """添加任务到队列"""
        if self.task_queue.full():
            return False
        self.tasks[task.task_id] = task
        self.task_queue.put(task.task_id)
        return True

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        return self.tasks.get(task_id)

    def _dispatcher_loop(self):
        """调度线程：从任务队列取任务并提交到引擎进程执行"""
        while not self._stop_event.is_set():
            try:
                task_id = self.task_queue.get(timeout=1)
                task = self.tasks.get(task_id)
                if not task:
                    continue

                # 保持为 PENDING，待引擎实际开始处理（收到进度事件）时再置为 PROCESSING

                try:
                    # 任务级并发：阻塞直到有名额再提交给引擎
                    self._task_concurrency_sema.acquire()
                    self._active_task_ids.add(task_id)
                    # 生成 do_parse 的参数并提交到引擎子进程
                    do_parse_kwargs = self._build_do_parse_kwargs(task)
                    if self._engine_process is not None:
                        ok = self._engine_process.submit_task_nonblocking(task.task_id, do_parse_kwargs)
                        if not ok:
                            raise RuntimeError("failed to enqueue job to engine subprocess")
                    else:
                        raise RuntimeError("engine subprocess not available")

                except Exception as e:
                    logger.exception(f"Task {task_id} dispatch failed: {e}")
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now()
                    if self.task_persistence:
                        self.task_persistence.update_task(task)
                    # 释放占用的任务并发名额
                    if task_id in self._active_task_ids:
                        self._active_task_ids.discard(task_id)
                        try:
                            self._task_concurrency_sema.release()
                        except Exception:
                            pass

                finally:
                    self.task_queue.task_done()

            except Exception:
                continue

    def _result_collector_loop(self):
        """结果收集线程：从引擎共享结果队列读取结果并完成打包/状态更新"""
        while not self._stop_event.is_set():
            try:
                if self._engine_process is None:
                    logger.error("Engine process is not available")
                    if self._stop_event.wait(1.0):
                        break
                    continue

                res = self._engine_process.get_task_result(timeout=1.0)
                if not res:
                    continue

                # 处理进度事件
                if isinstance(res, dict) and res.get("type") in ("progress_init", "progress"):
                    task_id = res.get("job_id")
                    task = self.tasks.get(task_id)
                    if task:
                        if res["type"] == "progress_init":
                            task.total_pages = int(res.get("total_pages", 0) or 0)
                            task.current_page = 0
                            logger.info(f"[ResultCollector] Task {task_id} total_pages set to {task.total_pages}")
                        elif res["type"] == "progress":
                            task.current_page = int(res.get("current_page", task.current_page) or task.current_page)
                            # 容错：不超过总页数
                            if task.total_pages and task.current_page > task.total_pages:
                                task.current_page = task.total_pages
                            logger.info(f"[ResultCollector] Task {task_id} progress current_page={task.current_page}/{task.total_pages}")
                        # 首次收到进度事件时，将状态从 PENDING 切换为 PROCESSING，并记录开始时间
                        if task.status != TaskStatus.PROCESSING:
                            task.status = TaskStatus.PROCESSING
                            task.started_at = datetime.now()
                        if self.task_persistence:
                            self.task_persistence.update_task(task)
                    continue

                task_id = res.get("job_id")
                task = self.tasks.get(task_id)
                if not task:
                    continue

                # 若已超时，被标记为 TIMEOUT，则忽略迟到结果
                if task.status == TaskStatus.TIMEOUT:
                    logger.warning(f"Ignore late result for timeout task: {task_id}")
                    continue

                if res.get("ok"):
                    try:
                        task.result_path = self._package_results(task.temp_dir, task.task_id)
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        # 计算处理时间
                        if task.started_at and task.completed_at:
                            task.processing_duration_seconds = max(0.0, (task.completed_at - task.started_at).total_seconds())
                            logger.info(f"Task {task_id} completed successfully in {task.processing_duration_seconds:.2f} seconds")
                        if self.task_persistence:
                            self.task_persistence.update_task(task)
                    except Exception as e:
                        logger.exception(f"Packaging failed for task {task_id}: {e}")
                        task.status = TaskStatus.FAILED
                        task.error_message = str(e)
                        task.completed_at = datetime.now()
                        # 计算处理时间
                        if task.started_at:
                            task.processing_duration_seconds = (task.completed_at - task.started_at).total_seconds()
                            logger.info(f"Task {task_id} failed in {task.processing_duration_seconds:.2f} seconds")
                        if self.task_persistence:
                            self.task_persistence.update_task(task)
                else:
                    task.status = TaskStatus.FAILED
                    task.error_message = res.get("error", "unknown error")
                    task.completed_at = datetime.now()
                    # 计算处理时间
                    if task.started_at:
                        task.processing_duration_seconds = (task.completed_at - task.started_at).total_seconds()
                        logger.info(f"Task {task_id} failed in {task.processing_duration_seconds:.2f} seconds")
                    if self.task_persistence:
                        self.task_persistence.update_task(task)

                # 任务结束：释放任务并发名额
                if task_id in self._active_task_ids:
                    self._active_task_ids.discard(task_id)
                    try:
                        self._task_concurrency_sema.release()
                    except Exception:
                        pass

                # 清理 CUDA 显存缓存（非关键路径，容错）
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Error in result collector: {e}")
                continue

    def _timeout_checker_loop(self):
        """超时检查循环"""
        import time
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                timeout_tasks = []

                # 检查所有进行中的任务是否超时
                for task_id, task in self.tasks.items():
                    if task.status == TaskStatus.PROCESSING and task.started_at:
                        elapsed_minutes = (current_time - task.started_at).total_seconds() / 60
                        if elapsed_minutes > task.timeout_minutes:
                            timeout_tasks.append(task)

                # 处理超时的任务
                for task in timeout_tasks:
                    task.status = TaskStatus.TIMEOUT
                    task.completed_at = current_time
                    task.error_message = f"Task timed out after {task.timeout_minutes} minutes"

                    # 计算处理时间
                    if task.started_at:
                        task.processing_duration_seconds = (task.completed_at - task.started_at).total_seconds()
                        logger.warning(f"Task {task.task_id} timed out after {task.timeout_minutes} minutes (processing time: {task.processing_duration_seconds:.2f} seconds)")

                    # 持久化状态更新
                    if self.task_persistence:
                        self.task_persistence.update_task(task)

                    # 清理任务相关文件
                    self._cleanup_task_files(task)

                    # 超时也释放任务并发名额（不再阻塞新任务）
                    if task.task_id in self._active_task_ids:
                        self._active_task_ids.discard(task.task_id)
                        try:
                            self._task_concurrency_sema.release()
                        except Exception:
                            pass

                # 使用事件等待以便快速响应停止信号
                if self._stop_event.wait(self.cleanup_check_interval):
                    break

            except Exception as e:
                logger.exception(f"Error in timeout checker: {e}")
                if self._stop_event.wait(self.cleanup_check_interval):
                    break

    def _cleanup_loop(self):
        """定期清理循环"""
        import time
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                tasks_to_remove = []

                for task_id, task in list(self.tasks.items()):
                    should_remove = False

                    # 如果任务已下载，立即删除
                    if task.downloaded:
                        should_remove = True
                        logger.info(f"Removing downloaded task: {task_id}")

                    # 如果任务成功完成超过配置时间且未下载，也删除
                    elif (task.status == TaskStatus.COMPLETED and
                          task.completed_at and
                          (current_time - task.completed_at).total_seconds() > self.config["cleanup_policy"]["retention_time"]["completed_undownloaded"]):
                        should_remove = True
                        logger.info(f"Removing old completed task: {task_id}")

                    # 如果任务失败或超时超过配置时间，也删除
                    elif (task.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT] and
                          task.completed_at and
                          (current_time - task.completed_at).total_seconds() > self.config["cleanup_policy"]["retention_time"]["failed_timeout_tasks"]):
                        should_remove = True
                        logger.info(f"Removing old failed/timeout task: {task_id}")

                    if should_remove:
                        tasks_to_remove.append(task_id)

                # 删除任务
                for task_id in tasks_to_remove:
                    task = self.tasks[task_id]
                    self._cleanup_task_files(task)
                    del self.tasks[task_id]

                # 清理持久化日志中的旧记录（7天前）
                try:
                    if self.task_persistence:
                        self.task_persistence.cleanup_old_tasks(7)
                except Exception as e:
                    logger.warning(f"Failed to cleanup persistent log: {e}")

                # 使用事件等待以便快速响应停止信号
                if self._stop_event.wait(self.cleanup_scan_interval):
                    break

            except Exception as e:
                logger.exception(f"Error in cleanup loop: {e}")
                if self._stop_event.wait(self.cleanup_scan_interval):
                    break

    def _cleanup_task_files(self, task: Task):
        """清理任务相关的文件"""
        with self._cleanup_lock:  # 使用锁保护文件清理操作
            try:
                # 清理临时目录
                if task.temp_dir and os.path.exists(task.temp_dir):
                    shutil.rmtree(task.temp_dir)
                    logger.info(f"Cleaned up temp directory: {task.temp_dir}")

                # 清理结果zip文件
                if task.result_path and os.path.exists(task.result_path):
                    os.remove(task.result_path)
                    logger.info(f"Cleaned up result file: {task.result_path}")

            except Exception as e:
                logger.warning(f"Failed to cleanup files for task {task.task_id}: {e}")

    def _build_do_parse_kwargs(self, task: Task) -> Dict[str, Any]:
        """构建并返回 do_parse 所需的参数，同时准备临时输出目录与输入字节
        
        职责分离：
        - 全局共享参数：从 self.global_* 中获取（所有PDF任务共用）
        - 任务特定参数：从 task.params 中获取（每个PDF可能不同）
        """
        # 创建唯一的输出目录
        unique_dir = os.path.join(self.global_task_config["output_dir"], str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        task.temp_dir = unique_dir

        # 处理上传的文件数据 -> 强制仅允许一个 PDF
        if not isinstance(task.file_data_list, list) or len(task.file_data_list) != 1:
            raise Exception("Exactly one PDF must be provided per task")

        file_data = task.file_data_list[0]
        file_path = Path(file_data.filename)
        # 仅允许 .pdf
        if file_path.suffix.lower() != ".pdf":
            raise Exception(f"Unsupported file type: {file_path.suffix}. Only '.pdf' is allowed")

        temp_path = Path(unique_dir) / file_path.name
        with open(temp_path, "wb") as f:
            f.write(file_data.content)
        try:
            pdf_bytes = read_and_convert_to_pdf_bytes(temp_path)
            os.remove(temp_path)
        except Exception as e:
            raise Exception(f"Failed to load file {file_data.filename}: {str(e)}")

        # 构建参数：全局配置 + 任务特定覆盖
        do_parse_kwargs = dict(
            # PDF文件信息
            output_dir=unique_dir,
            pdf_file_name=file_path.stem,
            pdf_bytes=pdf_bytes,
            p_lang="ch",
            
            # 全局共享配置（功能开关）
            formula_enable=self.global_task_config["formula_enable"],
            table_enable=self.global_task_config["table_enable"],
            f_dump_md=self.global_task_config["return_md"],
            f_dump_middle_json=self.global_task_config["return_middle_json"],
            f_dump_model_output=self.global_task_config["return_model_output"],
            f_dump_content_list=self.global_task_config["return_content_list"],
            
            # 任务特定参数（页面范围，允许覆盖）
            start_page_id=task.params.get("start_page_id", 0),
            end_page_id=task.params.get("end_page_id", 99999),
            
            # 全局引擎参数（vLLM配置）
            **self.global_server_args,
            
            # 全局推理参数（采样配置）
            **self.global_infer_args,
        )

        return do_parse_kwargs

    def mark_task_downloaded(self, task_id: str):
        """标记任务已被下载"""
        task = self.tasks.get(task_id)
        if task:
            task.downloaded = True
            logger.info(f"Marked task {task_id} as downloaded")



    def _package_results(self, result_dir: str, task_id: str) -> str:
        """打包结果文件为zip"""
        # 使用系统临时目录创建zip文件，支持跨平台
        temp_dir = tempfile.gettempdir()
        # 压缩包命名调整为 {task_id}.zip（去掉 _results 后缀）
        zip_path = os.path.join(temp_dir, f"{task_id}.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 期望：zip 根目录直接是 vlm 内的文件与子目录，去掉 "pdfname" 和 "vlm" 两层
            # 兼容：当一次任务包含多个 PDF 时，为避免冲突，保留 "pdfname/" 前缀，但仍去掉 "vlm" 层
            pdf_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
            vlm_found = False

            for pdf_dir in pdf_dirs:
                vlm_dir = os.path.join(result_dir, pdf_dir, "vlm")
                if os.path.isdir(vlm_dir):
                    vlm_found = True
                    for root, dirs, files in os.walk(vlm_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path_in_vlm = os.path.relpath(file_path, vlm_dir)

                            # 命名重写规则（仅作用于 vlm 顶层文件，不影响 images 等子目录）
                            # 需求：
                            # - *_content_list.json    -> {task_id}_content_list.json
                            # - *_middle.json          -> layout.json
                            # - *.md                   -> full.md
                            # - *_model_output.txt     -> {task_id}_model_output.txt
                            rel_for_zip = rel_path_in_vlm.replace('\\', '/')
                            if '/' not in rel_for_zip:  # 仅处理顶层文件
                                lower_name = rel_for_zip.lower()
                                if lower_name.endswith('_content_list.json'):
                                    rel_for_zip = f"{task_id}_content_list.json"
                                elif lower_name.endswith('_middle.json'):
                                    rel_for_zip = "layout.json"
                                elif lower_name.endswith('_model_output.txt'):
                                    rel_for_zip = f"{task_id}_model_output.txt"
                                elif lower_name.endswith('.md'):
                                    rel_for_zip = "full.md"

                            # 单文件任务：直接扁平到 zip 根目录；多文件任务：在根目录下加 pdfname 前缀以避免冲突
                            if len(pdf_dirs) == 1:
                                arcname = rel_for_zip
                            else:
                                arcname = os.path.join(pdf_dir, rel_for_zip)
                            zipf.write(file_path, arcname)

            # 兜底：如果未找到期望结构，则保留原先的完整目录打包逻辑
            if not vlm_found:
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, result_dir)
                        zipf.write(file_path, arcname)

        return zip_path
