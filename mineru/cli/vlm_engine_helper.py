# Copyright (c) Opendatalab. All rights reserved.
"""
VLM 引擎子进程管理模块

提供常驻VLM引擎子进程的实现，避免重复初始化模型。
- 主进程：FastAPI服务，负责接收请求和队列管理
- 子进程：VLM引擎进程，负责模型加载和推理计算
- 通信方式：进程间队列（任务队列和结果队列）

这是系统唯一的推理执行方式，确保：
1. 模型只初始化一次，提高效率
2. 推理在独立进程中执行，不阻塞主进程事件循环
3. 支持任务级超时控制
4. 进程隔离保障主服务稳定性
"""

import multiprocessing as mp
from typing import Dict, Any, Optional
from loguru import logger
import threading


class VLMEngineProcess:
    """常驻 VLM 引擎子进程管理器（系统唯一推理模式）

    特性：
    - 单次模型初始化：子进程启动时预热 VLM 模型，后续任务复用
    - 非阻塞：主进程通过队列提交任务，不会阻塞 FastAPI 事件循环
    - 超时控制：支持任务级超时，避免无限等待
    - 进程隔离：推理在独立进程中执行，保障主服务稳定性
    - 强制模式：这是系统唯一的推理执行方式，不支持降级
    """
    
    def __init__(self, server_args: Dict[str, Any], infer_args: Dict[str, Any]):
        """初始化VLM引擎进程管理器
        
        创建进程间通信队列，启动VLM引擎子进程。
        
        Args:
            server_args: vLLM服务器配置参数（如max_model_len, gpu_memory_utilization等）
            infer_args: 推理采样参数（如temperature, top_p, max_new_tokens等）
        """
        self._multiprocessing_context = mp.get_context('spawn')
        self._task_queue: "mp.Queue" = self._multiprocessing_context.Queue()
        
        # 使用Manager管理共享对象，确保可序列化
        self._manager = self._multiprocessing_context.Manager()
        
        # 共享结果队列：子进程将所有任务结果写入此队列
        self._shared_result_queue: "mp.Queue" = self._manager.Queue()
        
        self._server_args = server_args
        self._infer_args = infer_args
        self._engine_process: Optional[mp.Process] = None
        self._start_engine_process()

    def _start_engine_process(self) -> None:
        """启动VLM引擎子进程
        
        创建并启动独立的Python进程来运行VLM模型。
        子进程将在启动时加载模型，然后持续监听任务队列。
        """
        if self._engine_process is not None and self._engine_process.is_alive():
            logger.warning("VLM引擎子进程已在运行，跳过重复启动")
            return
            
        self._engine_process = self._multiprocessing_context.Process(
            target=_vlm_engine_worker_loop,
            args=(self._task_queue, self._shared_result_queue, self._server_args, self._infer_args),
        )
        self._engine_process.daemon = False
        self._engine_process.start()
        logger.info(f"VLM引擎子进程已启动，进程ID: {self._engine_process.pid}")

    def stop_engine(self) -> None:
        """停止VLM引擎子进程
        
        优雅地关闭引擎子进程，释放GPU资源和进程间通信资源。
        """
        try:
            if self._engine_process is not None and self._engine_process.is_alive():
                # 发送停止信号（None表示结束）
                logger.info("发送停止信号到VLM引擎子进程...")
                self._task_queue.put(None)
                
                # 等待子进程优雅退出
                self._engine_process.join(timeout=10)
                
                if self._engine_process.is_alive():
                    logger.warning("VLM引擎子进程未能优雅退出，强制终止...")
                    self._engine_process.terminate()
                    self._engine_process.join(timeout=5)
                    
                logger.info("VLM引擎子进程已停止")
        except Exception as e:
            logger.error(f"停止VLM引擎子进程时出错: {e}")
        finally:
            # 关闭Manager，释放共享内存资源
            try:
                if hasattr(self, "_manager") and self._manager is not None:
                    self._manager.shutdown()
                    logger.debug("进程间通信Manager已关闭")
            except Exception as e:
                logger.warning(f"关闭Manager时出错: {e}")
    

    def submit_task_blocking(self, task_params: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
        """提交任务并阻塞等待结果（同步模式）
        
        将任务提交到VLM引擎子进程，并阻塞等待执行完成。
        适用于需要立即获取结果的场景。
        
        Args:
            task_params: PDF解析任务的参数字典（传递给aio_do_parse）
            timeout_seconds: 任务超时时间（秒）
            
        Returns:
            dict: 执行结果
                - 成功: {"ok": True}
                - 失败: {"ok": False, "error": str, "traceback": str}
                - 超时: {"ok": False, "error": "timeout message"}
        """
        # 为此任务创建专用的返回队列
        task_result_queue = self._manager.Queue(maxsize=1)
        
        task_package = {
            "job_id": task_params.get("output_dir", "unknown"),
            "kwargs": task_params,
            "result_queue": task_result_queue,  # 专用队列，仅此任务使用
        }
        
        try:
            self._task_queue.put(task_package)
            logger.debug(f"任务已提交到VLM引擎: {task_package['job_id']}")
        except Exception as e:
            return {"ok": False, "error": f"提交任务失败: {str(e)}"}

        # 阻塞等待结果
        try:
            result = task_result_queue.get(timeout=max(1.0, timeout_seconds))
            logger.debug(f"任务执行完成: {task_package['job_id']}")
            return result
        except Exception:
            return {
                "ok": False, 
                "error": f"任务超时，等待时间超过 {timeout_seconds} 秒"
            }
    

    def submit_task_nonblocking(self, task_id: str, task_params: Dict[str, Any]) -> bool:
        """提交任务到引擎队列（异步非阻塞模式）
        
        将任务加入队列后立即返回，不等待执行结果。
        结果需要通过 get_task_result() 方法异步获取。

        Args:
            task_id: 任务唯一标识符（如UUID）
            task_params: PDF解析任务的参数字典

        Returns:
            bool: True表示成功加入队列，False表示失败
        """
        task_package = {
            "job_id": task_id,
            "kwargs": task_params,
            # 非阻塞模式：不提供专用result_queue，结果写入共享队列
        }
        
        try:
            self._task_queue.put(task_package)
            logger.debug(f"任务已加入VLM引擎队列（非阻塞模式）: {task_id}")
            return True
        except Exception as e:
            logger.error(f"任务入队失败 {task_id}: {e}")
            return False


    def get_task_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """从共享结果队列中读取一个任务结果（非阻塞模式使用）
        
        用于获取通过 submit_task_nonblocking() 提交的任务的执行结果。
        
        Args:
            timeout: 等待超时时间（秒），None表示永久等待

        Returns:
            dict: 任务结果字典，包含：
                - job_id: 任务ID
                - ok: 是否成功
                - error: 错误信息（如果失败）
                - type: 事件类型（progress_init, progress等）
            None: 超时或队列为空
        """
        try:
            if timeout is not None:
                result = self._shared_result_queue.get(timeout=timeout)
            else:
                result = self._shared_result_queue.get()
            return result
        except Exception:
            return None


    def is_engine_alive(self) -> bool:
        """检查VLM引擎子进程是否存活
        
        Returns:
            bool: True表示子进程正常运行，False表示已停止
        """
        return self._engine_process is not None and self._engine_process.is_alive()
    


def _vlm_engine_worker_loop(
    task_queue: "mp.Queue", 
    shared_result_queue: "mp.Queue", 
    server_args: Dict[str, Any], 
    infer_args: Dict[str, Any]
) -> None:
    """VLM引擎子进程主循环函数
    
    这是VLM引擎子进程的入口函数，负责：
    1. 初始化VLM模型（仅一次，避免重复加载）
    2. 持续监听任务队列并执行PDF解析
    3. 将结果发送回主进程
    
    Args:
        task_queue: 接收来自主进程的任务队列
        shared_result_queue: 发送结果到主进程的共享队列
        server_args: vLLM服务器配置参数
        infer_args: 推理采样参数
    """
    worker_pid = mp.current_process().pid
    logger.info(f"VLM engine worker started (PID: {worker_pid})")
    
    # 额外保障：在子进程内强制应用 GPU 环境，确保内存捕获与显存分配严格遵循 YAML 设置
    try:
        import os
        # 固定设备顺序，避免设备编号漂移
        if not os.environ.get("CUDA_DEVICE_ORDER"):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # 若父进程未设置（或被清空），在子进程内从 YAML 兜底读取并设置
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            from pathlib import Path
            import yaml as _yaml
            _cfg_path = Path(__file__).parent / "fast_api_hyper-parameter.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, "r", encoding="utf-8") as _f:
                    _cfg = _yaml.safe_load(_f) or {}
                _gpu_cfg = ((_cfg.get("system") or {}).get("gpu") or {})
                _cuda_visible = _gpu_cfg.get("cuda_visible_devices")
                if _cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_visible)
                    os.environ["NVIDIA_VISIBLE_DEVICES"] = str(_cuda_visible)
        # 明确将当前 CUDA 设备设置为可见列表中的第 0 个（即物理卡列表中的第一个可见卡）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                try:
                    _cur = torch.cuda.current_device()
                    _name = torch.cuda.get_device_name(0)
                    logger.info(f"Torch CUDA current_device={_cur}, device_name={_name}")
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # 环境兜底失败不阻断启动（仍然依赖父进程环境）
        pass
    
    # 预热模型（仅在子进程中初始化一次）
    try:
        from mineru.backend.vlm.vlm_analyze import ModelSingleton
        logger.info("Warming up VLM model in engine subprocess...")
        logger.info(f"Server args for vLLM engine: {server_args}")
        logger.info(f"Inference args: {infer_args}")
        
        # 打印 GPU 环境变量用于调试
        import os
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'Not set')}")
        
        # 使用异步引擎（过滤掉非 AsyncEngineArgs 支持的键）
        _infer_args_sanitized = dict(infer_args or {})
        # 并发调度相关键仅用于我们自己的调度器，不能传给 AsyncEngineArgs
        for _k in ("concurrent_processing_pages", "max_concurrency_pdf", "preprocess_concurrency", "prepared_queue_limit"):
            _infer_args_sanitized.pop(_k, None)
        # 强制开启 vLLM 统计日志，确保 logger_manager 存在（用于读取 kv_cache_usage）
        _server_args_sanitized = dict(server_args or {})
        try:
            # 显式关闭禁用开关
            _server_args_sanitized["disable_log_stats"] = False
        except Exception:
            pass
        predictor = ModelSingleton().get_model("vllm-async-engine", None, None, **_server_args_sanitized, **_infer_args_sanitized)
        # 关键：在子进程主线程中预先创建 TokenizerManager 的事件循环与信号处理器
        # 避免第一次在工作线程里创建而输出 "Signal handler is not added ..." 的警告
        tm = getattr(getattr(predictor, "engine", None), "tokenizer_manager", None)
        if tm is not None:
            tm.auto_create_handle_loop()
            logger.info("TokenizerManager handle loop initialized in subprocess main thread")
            
        # 打印 vllm 引擎的详细信息用于调试
        engine = getattr(predictor, "engine", None)
        if engine is not None:
            logger.info(f"vLLM async engine initialized successfully")
            logger.info(f"Engine type: {type(engine)}")
            # 尝试获取数据并行相关信息
            try:
                if hasattr(engine, 'server_args'):
                    logger.info(f"Engine server args - dp_size: {getattr(engine.server_args, 'dp_size', 'Not found')}")
                    logger.info(f"Engine server args - tp_size: {getattr(engine.server_args, 'tp_size', 'Not found')}")
            except Exception as e:
                logger.warning(f"Could not access engine server args: {e}")
        
        logger.info("VLM model warmup completed in engine subprocess")
    except Exception as e:
        logger.error(f"Failed to warm up VLM model in subprocess: {e}")
        import traceback
        logger.error(f"Full error traceback: {traceback.format_exc()}")

        # 检查是否是 GPU 相关错误，如果是，提供更友好的错误信息
        error_msg = str(e)
        if "nvidia-smi" in error_msg or "GPU" in error_msg or "cuda" in error_msg.lower():
            logger.error("=" * 60)
            logger.error("GPU 相关错误检测到！")
            logger.error("可能的原因:")
            logger.error("1. 系统没有 NVIDIA GPU")
            logger.error("2. NVIDIA 驱动未正确安装")
            logger.error("3. CUDA_VISIBLE_DEVICES 设置有误")
            logger.error("4. GPU 内存不足")
            logger.error("")

            #  // "prompt": ["请用工具并行完成两件事：1) 统计 'strawberry' 里字母 'r' 的个数；2) 统计 'raspberry' 里字母 'r' 的个数。不要先回答结论，先发起两个工具调用并等待我回传结果后再汇总。","\n\n"],
            
            logger.error("解决方案:")
            logger.error("1. 检查 GPU 是否可用: nvidia-smi")
            logger.error("2. 确认 CUDA_VISIBLE_DEVICES 设置正确")
            logger.error("3. 考虑使用 transformers 后端代替 sglang-engine")
            logger.error("=" * 60)

        # 预热失败时不继续运行，抛出异常让主进程知道
        raise RuntimeError(f"VLM model warmup failed: {e}") from e

    # 旧版按PDF整包处理已移除：不再依赖 aio_do_parse

    # 固定大小线程池：线程数代表同时活跃的 PDF 作业数（pdf_slots）
    try:
        # 总会话并发：优先采用 vLLM 标准键名 max_num_seqs；若未提供则回退到旧键
        max_num_seqs = int(server_args.get("max_num_seqs", server_args.get("max_running_requests", 1)))
        if max_num_seqs <= 0:
            max_num_seqs = 1
    except Exception:
        max_num_seqs = 1

    # PDF 槽位（同时活跃的 PDF 数），缺省为 4
    try:
        pdf_slots = int((infer_args or {}).get("max_concurrency_pdf", 4))
        if pdf_slots <= 0:
            pdf_slots = 1
    except Exception:
        pdf_slots = 4

    # 页级总并发预算（可独立配置），默认回退到 max_num_seqs
    try:
        concurrent_processing_pages = int((infer_args or {}).get("concurrent_processing_pages", max_num_seqs))
        if concurrent_processing_pages <= 0:
            concurrent_processing_pages = max_num_seqs
    except Exception:
        concurrent_processing_pages = max_num_seqs

    # 预处理并发与缓存上限（任务到达即预切页）
    try:
        preprocess_concurrency = int((infer_args or {}).get("preprocess_concurrency", max(2, pdf_slots)))
        if preprocess_concurrency <= 0:
            preprocess_concurrency = max(2, pdf_slots)
    except Exception:
        preprocess_concurrency = max(2, pdf_slots)
    try:
        prepared_queue_limit = int((infer_args or {}).get("prepared_queue_limit", pdf_slots * 2))
        if prepared_queue_limit <= 0:
            prepared_queue_limit = pdf_slots * 2
    except Exception:
        prepared_queue_limit = pdf_slots * 2

    # 打印并发规划（不再输出派生的 per-PDF 值，避免误解为可配置项）
    logger.info(
        f"Engine concurrency plan: max_num_seqs={max_num_seqs}, pdf_slots={pdf_slots}, concurrent_processing_pages={concurrent_processing_pages}"
    )

    # 使用单事件循环并发调度多个 PDF 任务，避免 AsyncLLM 跨线程/多事件循环带来的阻塞或不一致
    import asyncio as _asyncio

    # KV Cache 利用率读取（兼容多版本 vLLM）
    def _read_kv_utilization(_engine) -> dict | None:
        try:
            if _engine is None:
                return None
            # 兼容多版本 vLLM：遍历若干已知路径寻找 block_manager
            candidate_roots = []
            candidate_roots.append(_engine)
            for name in ("engine", "llm_engine", "runner"):
                try:
                    candidate_roots.append(getattr(_engine, name))
                except Exception:
                    pass
            bm = None
            for root in candidate_roots:
                if root is None:
                    continue
                for sched_name in ("scheduler", "cache_engine", "cache_manager", None):
                    try:
                        sched = getattr(root, sched_name) if sched_name else root
                    except Exception:
                        sched = None
                    if sched is None:
                        continue
                    for bm_name in ("block_manager", "gpu_block_manager", "cache_block_manager", "cache_allocator", "block_allocator"):
                        try:
                            candidate = getattr(sched, bm_name)
                        except Exception:
                            candidate = None
                        if candidate is not None:
                            bm = candidate
                            break
                    if bm is not None:
                        break
                if bm is not None:
                    break
            if bm is None:
                # v1 AsyncLLM 多进程场景：直接读取统计日志管理器里的最新值（比例）
                try:
                    lm = getattr(_engine, "logger_manager", None)
                    if lm is not None:
                        per = getattr(lm, "per_engine_logger_dict", None)
                        if isinstance(per, dict) and per:
                            # 取第一个 engine 的第一个 logger（通常包含 LoggingStatLogger）
                            for _, logger_list in per.items():
                                if isinstance(logger_list, (list, tuple)) and logger_list:
                                    for lg in logger_list:
                                        last_stats = getattr(lg, "last_scheduler_stats", None)
                                        if last_stats is not None:
                                            ratio = float(getattr(last_stats, "kv_cache_usage", 0.0) or 0.0)
                                            return {"ratio": ratio}
                            # 找不到则返回 None
                    return None
                except Exception:
                    return None
            # 读取总块数
            total = None
            for t_attr in ("num_gpu_blocks", "total_num_gpu_blocks", "num_blocks", "capacity"):
                try:
                    val = getattr(bm, t_attr)
                    if isinstance(val, (int, float)) or (hasattr(val, "__len__")):
                        total = len(val) if not isinstance(val, (int, float)) else int(val)
                        break
                except Exception:
                    pass
            # 读取空闲块数
            free = None
            for m in ("get_num_free_gpu_blocks", "get_num_free_blocks"):
                fn = getattr(bm, m, None)
                if callable(fn):
                    try:
                        free = int(fn())
                        break
                    except Exception:
                        pass
            if free is None:
                for a in ("free_gpu_blocks", "free_block_ids", "free_blocks"):
                    try:
                        val = getattr(bm, a)
                        if val is not None:
                            free = len(val)
                            break
                    except Exception:
                        pass
            if total is None or free is None or int(total) <= 0:
                return None
            total = int(total)
            free = int(free)
            ratio = float(free) / float(total)
            return {"free": free, "total": total, "ratio": ratio}
        except Exception:
            return None

    # ============== 新版：页级轮询调度（Round-Robin across PDFs） ==============
    # 目标：维持全局页并发 concurrent_processing_pages，公平地在最多 pdf_slots 个桶之间轮询取页
    from mineru.utils.pdf_image_tools import load_images_from_pdf as _load_images_from_pdf
    from mineru.backend.vlm.model_output_to_middle_json import result_to_middle_json as _result_to_middle_json
    from mineru.data.data_reader_writer import FileBasedDataWriter as _FileWriter
    from mineru.cli.common import create_output_directories as _mk_dirs, _write_output_files as _write_out, extract_pdf_page_range as _slice_pdf
    from mineru.utils.enum_class import MakeMode as _MakeMode

    # 旧版 _run_one 已删除

    async def _engine_loop_rr_pages():
        # 活跃 PDF 桶（最多 pdf_slots 个）
        buckets: dict[str, dict] = {}
        bucket_order: list[str] = []  # 维护轮询顺序
        rr_idx: int = 0
        # 全局页级并发信号量
        page_sem = _asyncio.Semaphore(concurrent_processing_pages)
        # 正在运行的页任务集合
        running_pages: set = set()
        # 是否收到关停信号（主队列返回 None）
        shutdown_received: bool = False

        # 预处理：任务到达即预切页，准备好后进入待入场队列
        prepared: dict[str, dict] = {}
        prepared_order: list[str] = []
        prepping_jobs: set[str] = set()
        prep_sem = _asyncio.Semaphore(preprocess_concurrency)

        async def _finalize_bucket(job_id: str, b: dict):
            try:
                # 生成输出目录/写入器
                image_output_dir, markdown_output_dir = _mk_dirs(b["output_dir"], b["pdf_file_name"], "vlm")
                image_writer = _FileWriter(image_output_dir)
                markdown_writer = _FileWriter(markdown_output_dir)
                # 生成中间 JSON（会保存图片）
                layout_json = _result_to_middle_json(b["results"], b["images_list"], b["pdf_doc"], image_writer)
                pdf_info = layout_json.get("pdf_info", {})
                # 写出最终产物
                _write_out(
                    pdf_info,
                    b["pdf_bytes"],
                    b["pdf_file_name"],
                    markdown_output_dir,
                    image_output_dir,
                    markdown_writer,
                    b["f_dump_md"],
                    b["f_dump_content_list"],
                    b["f_dump_middle_json"],
                    b["f_dump_model_output"],
                    b.get("f_make_md_mode", _MakeMode.MM_MD),
                    layout_json,
                    b["results"],
                )
            finally:
                # 通知完成
                try:
                    # 以桶级处理时长为准：从进入桶开始到全部页完成
                    elapsed = None
                    try:
                        if b.get("started_at") is not None:
                            import time as _t
                            elapsed = _t.time() - float(b["started_at"])  # seconds
                    except Exception:
                        pass
                    payload = {"job_id": job_id, "ok": True}
                    if elapsed is not None:
                        payload["processing_duration_seconds"] = max(0.0, float(elapsed))
                    b["result_queue"].put(payload)
                except Exception:
                    pass

        async def _process_one_page(job_id: str, b: dict, page_index: int):
            # 优先使用预处理好的布局输入，减少运行期 CPU 预处理
            try:
                layout_prepped = None
                try:
                    if b.get("layout_prepared"):
                        layout_prepped = b["layout_prepared"][page_index]
                except Exception:
                    layout_prepped = None

                if layout_prepped is not None:
                    # 手动执行 two-step：layout -> extract（不再限流 block）
                    prompt = predictor.prompts.get("[layout]") or predictor.prompts.get("[default]")
                    params = predictor.sampling_params.get("[layout]") or predictor.sampling_params.get("[default]")
                    layout_output = await predictor.client.aio_predict(layout_prepped, prompt, params)
                    blocks = await predictor.helper.aio_parse_layout_output(predictor.executor, layout_output)
                    block_images, prompts, sparams, indices = await predictor.helper.aio_prepare_for_extract(
                        predictor.executor, b["images_pil"][page_index], blocks
                    )
                    outputs = await predictor.client.aio_batch_predict(block_images, prompts, sparams, semaphore=None)
                    for idx, output in zip(indices, outputs):
                        blocks[idx].content = output
                    page_blocks = await predictor.helper.aio_post_process(predictor.executor, blocks)
                    b["results"][page_index] = page_blocks or []
                else:
                    # 回退到标准两步推理
                    imgs = [b["images_pil"][page_index]]
                    page_result_list = await predictor.aio_batch_two_step_extract(images=imgs, semaphore=None)
                    if isinstance(page_result_list, list) and len(page_result_list) > 0:
                        b["results"][page_index] = page_result_list[0]
                    else:
                        b["results"][page_index] = page_result_list or []
            except Exception as _e:
                try:
                    logger.warning(f"page {page_index} two-step failed on job {job_id}: {_e}")
                except Exception:
                    pass
                b["results"][page_index] = []

            # 进度上报
            b["completed_pages"] += 1
            try:
                b["result_queue"].put({
                    "type": "progress",
                    "job_id": job_id,
                    "current_page": b["completed_pages"],
                })
            except Exception:
                pass

            # 若该桶全部完成，做收尾
            if b["completed_pages"] >= b["total_pages"]:
                await _finalize_bucket(job_id, b)
                # 清理桶
                try:
                    del buckets[job_id]
                    bucket_order.remove(job_id)
                except Exception:
                    pass

        async def _prepare_job(job: Dict[str, Any]):
            job_id = job.get("job_id", "unknown")
            do_parse_kwargs = job.get("kwargs", {})
            result_queue = job.get("result_queue") or shared_result_queue
            try:
                # 基础参数校验
                if not isinstance(do_parse_kwargs.get("pdf_file_name"), str) or not isinstance(do_parse_kwargs.get("pdf_bytes"), (bytes, bytearray)):
                    raise ValueError("aio_do_parse requires 'pdf_file_name: str' and 'pdf_bytes: bytes'")

                # 页面裁剪与预切页（在线程中执行以避免阻塞事件循环）
                start_pid = int(do_parse_kwargs.get("start_page_id", 0) or 0)
                end_pid_raw = do_parse_kwargs.get("end_page_id")
                try:
                    end_pid = int(end_pid_raw) if end_pid_raw is not None else None
                except Exception:
                    end_pid = None

                async with prep_sem:
                    pdf_bytes_obj = await _asyncio.to_thread(_slice_pdf, do_parse_kwargs["pdf_bytes"], start_pid, end_pid)
                    pdf_bytes = bytes(pdf_bytes_obj)
                    images_list, pdf_doc = await _asyncio.to_thread(_load_images_from_pdf, pdf_bytes)

                images_pil = [d.get("img_pil") for d in images_list]
                total_pages = len(images_pil)

                # 预先生成布局输入（缩放/转换），减少运行期 CPU 预处理
                layout_prepared: list | None = []
                try:
                    for _im in images_pil:
                        lp = await _asyncio.to_thread(predictor.helper.prepare_for_layout, _im)
                        layout_prepared.append(lp)
                except Exception:
                    layout_prepared = [None] * total_pages

                b = {
                    "result_queue": result_queue,
                    "output_dir": do_parse_kwargs.get("output_dir"),
                    "pdf_file_name": do_parse_kwargs.get("pdf_file_name"),
                    "pdf_bytes": pdf_bytes,
                    "images_list": images_list,
                    "images_pil": images_pil,
                    "pdf_doc": pdf_doc,
                    "layout_prepared": layout_prepared,
                    "results": [None] * total_pages,
                    "total_pages": total_pages,
                    "completed_pages": 0,
                    "next_index": 0,
                    "f_dump_md": bool(do_parse_kwargs.get("f_dump_md", True)),
                    "f_dump_middle_json": bool(do_parse_kwargs.get("f_dump_middle_json", True)),
                    "f_dump_model_output": bool(do_parse_kwargs.get("f_dump_model_output", True)),
                    "f_dump_content_list": bool(do_parse_kwargs.get("f_dump_content_list", True)),
                    "f_make_md_mode": do_parse_kwargs.get("f_make_md_mode", _MakeMode.MM_MD),
                }
                prepared[job_id] = b
                prepared_order.append(job_id)
            except Exception as e:
                try:
                    result_queue.put({"job_id": job_id, "ok": False, "error": str(e)})
                except Exception:
                    pass
            finally:
                prepping_jobs.discard(job_id)

        def _admit_prepared_if_possible():
            # 将已预处理完成的任务按顺序加入活跃桶，直至达到 pdf_slots
            while len(bucket_order) < pdf_slots and prepared_order:
                job_id = prepared_order.pop(0)
                b = prepared.pop(job_id, None)
                if not b:
                    continue
                buckets[job_id] = b
                bucket_order.append(job_id)
                # 进入桶的那一刻开始计时（不包含队列等待时间）
                try:
                    import time as _t
                    b["started_at"] = _t.time()
                except Exception:
                    b["started_at"] = None
                # 上报初始化进度
                try:
                    b["result_queue"].put({"type": "progress_init", "job_id": job_id, "total_pages": b["total_pages"]})
                except Exception:
                    pass

        async def _schedule_if_possible():
            nonlocal rr_idx
            # 填充运行中的页任务至并发上限
            while len(running_pages) < concurrent_processing_pages and bucket_order:
                # 轮询找到下一个仍有剩余页的桶
                found = False
                start_idx = rr_idx
                for _ in range(len(bucket_order)):
                    bid = bucket_order[rr_idx]
                    b = buckets.get(bid)
                    rr_idx = (rr_idx + 1) % len(bucket_order)
                    # 跳过无效/已完成桶
                    if not b or b.get("completed_pages", 0) >= b.get("total_pages", 0):
                        continue
                    # 使用独立指针 next_index 确保每个页仅被调度一次
                    next_page = b.get("next_index", 0)
                    if next_page < b["total_pages"]:
                        b["next_index"] = next_page + 1
                        # 调度日志：观察注入速率与爬坡速度
                        try:
                            logger.info(f"Schedule page task: job={bid}, page={next_page}, running={len(running_pages)+1}/{concurrent_processing_pages}")
                        except Exception:
                            pass
                        t = _asyncio.create_task(_process_one_page(bid, b, next_page))
                        running_pages.add(t)
                        t.add_done_callback(lambda fut, s=running_pages: s.discard(fut))
                        found = True
                        break
                if not found:
                    break

        async def _drain_task_queue():
            # 控制预处理队列长度，避免占用过多内存
            if (len(prepared_order) + len(prepping_jobs)) >= prepared_queue_limit:
                return
            try:
                job = await _asyncio.to_thread(task_queue.get, True, 0.05)
            except Exception:
                return
            if job is None:
                nonlocal shutdown_received
                shutdown_received = True
                return
            job_id = job.get("job_id", "unknown")
            if job_id in prepping_jobs or job_id in prepared or job_id in buckets:
                return
            prepping_jobs.add(job_id)
            _asyncio.create_task(_prepare_job(job))

        # KV 监控后台任务：固定间隔打印 KV 利用率
        async def _kv_monitor_task():
            prev_prompt_tokens_sum: int | float = 0
            while not shutdown_received or bucket_order or running_pages:
                # 优先从 client.vllm_async_llm 获取 AsyncLLM 引擎
                eng = None
                try:
                    eng = getattr(getattr(predictor, "client", None), "vllm_async_llm", None)
                except Exception:
                    eng = None
                if eng is None:
                    try:
                        eng = getattr(predictor, "engine", None)
                    except Exception:
                        eng = None
                info = _read_kv_utilization(eng)
                # running_reqs: 当前运行中的请求数
                # recent_prompt_tokens: 自上次间隔以来处理的预填充 token 量（近似）
                # 读取最近间隔内处理的预填充 tokens（近似值）与运行中请求数
                recent_prompt_tokens = None
                running_reqs = None
                try:
                    lm = getattr(eng, "logger_manager", None)
                    per = getattr(lm, "per_engine_logger_dict", None)
                    if isinstance(per, dict) and per:
                        cur_sum = 0
                        run_sum = 0
                        for _idx, logger_list in per.items():
                            if isinstance(logger_list, (list, tuple)):
                                for lg in logger_list:
                                    # 自增计数：自上次 vLLM 内部 log() 重置以来累计的 prompt tokens
                                    cur_sum += float(getattr(lg, "num_prompt_tokens", 0.0) or 0.0)
                                    last_stats = getattr(lg, "last_scheduler_stats", None)
                                    if last_stats is not None:
                                        run_sum += int(getattr(last_stats, "num_running_reqs", 0) or 0)
                        recent_prompt_tokens = max(0, int(cur_sum - prev_prompt_tokens_sum))
                        prev_prompt_tokens_sum = cur_sum
                        running_reqs = run_sum
                except Exception:
                    pass
                if info:
                    try:
                        extra = ""
                        if recent_prompt_tokens is not None or running_reqs is not None:
                            extra_parts = []
                            if running_reqs is not None:
                                extra_parts.append(f"running_reqs={running_reqs}")
                            if recent_prompt_tokens is not None:
                                extra_parts.append(f"recent_prompt_tokens={recent_prompt_tokens}")
                            if extra_parts:
                                extra = " | " + ", ".join(extra_parts)
                        if isinstance(info, dict) and ("free" in info and "total" in info and info.get("total")):
                            logger.info(f"KVCache utilization: free={info['free']}/{info['total']} ({info['ratio']:.2%}){extra}")
                        elif isinstance(info, dict) and ("ratio" in info):
                            logger.info(f"KVCache utilization: {info['ratio']:.2%}{extra}")
                        else:
                            raise ValueError("invalid info dict")
                    except Exception:
                        try:
                            et = type(eng).__name__ if eng is not None else "none"
                        except Exception:
                            et = "unknown"
                        logger.info(f"KVCache utilization: not available (engine={et})")
                else:
                    try:
                        et = type(eng).__name__ if eng is not None else "none"
                    except Exception:
                        et = "unknown"
                    logger.info(f"KVCache utilization: not available (engine={et})")
                await _asyncio.sleep(1.50)

        kv_task = _asyncio.create_task(_kv_monitor_task())

        # 主循环：泵入任务 -> 调度页 -> 直到结束
        while True:
            # 若已收到关停信号且无桶与页在运行，退出
            if shutdown_received and not bucket_order and not running_pages:
                break
            await _drain_task_queue()
            _admit_prepared_if_possible()
            await _schedule_if_possible()
            # 略作小等待以避免忙等
            await _asyncio.sleep(0.01)
        try:
            kv_task.cancel()
        except Exception:
            pass
        # 旧版按PDF提交的残留逻辑已删除

    # 仅运行新版页级调度循环
    _asyncio.run(_engine_loop_rr_pages())

    logger.info(f"VLM engine worker stopped (PID: {worker_pid})")
