from mineru.utils.profile import nvtx_annotate, tracing_reset, tracing_get_and_reset
from loguru import logger

import copy
import glob
import os
import traceback
import random
import json
import time
import argparse
import gzip
import zipfile
import queue
import threading
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Any, List, Dict, Optional, Tuple
import signal
import psutil

parser = argparse.ArgumentParser(description="MB PDF OCR")
parser.add_argument('--input-dir', type=str, required=True, help='输入数据集的目录')
parser.add_argument('--output-dir', type=str, required=True, help='输出数据集的目录')
parser.add_argument('--vram-size-gb', type=int, default=24, help='使用显存的GB数（默认24）')
parser.add_argument('--cuda-devices', type=str, default='0', help='绑定的GPU编号，多个用逗号分隔，如"0,1,2"')
parser.add_argument('--num-processes', type=int, default=1, help='每个GPU的进程数')
parser.add_argument('--log-dir', type=str, default='/tmp', help='日志目录')
parser.add_argument('--task-timeout', type=int, default=1800, help='任务超时时间（秒）')
parser.add_argument('--max-tasks-per-worker', type=int, default=20, help='每个工作进程最大任务数')
args = parser.parse_args()

def infer_one_pdf(pdf_file_path, lang="en"):
    with nvtx_annotate("py_import"):
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
    with nvtx_annotate("pdf_read"):
        t0 = time.time()
        with open(pdf_file_path, 'rb') as fi:
            pdf_bytes = fi.read()
        t1 = time.time()
    with nvtx_annotate("pdf_convert"):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
        t2 = time.time()
    
    print(f"read pdf file spend :{t1 - t0}, convert pdf spend :{t2 - t1}")
    formula_enable = True
    table_enable = True
    pdf_name = os.path.basename(pdf_file_path)
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            [new_pdf_bytes],
            [lang],
            parse_method="ocr",
            formula_enable=formula_enable, table_enable=table_enable
        )
    )
    t3 = time.time()
    print(f"pipeline_doc_analyze spend:{t3 - t2}")
    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    _ocr_enable = True
    model_json = copy.deepcopy(model_list)

    local_image_dir = f"/ssd/mineru_ocr_local_image_dir/{pdf_name}"
    if not os.path.exists(local_image_dir):
        os.system(f"mkdir -p {local_image_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    with nvtx_annotate("gen_json"):
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            lang, _ocr_enable, formula_enable
        )
    t4 = time.time()
    print(f"pipeline_result_to_middle_json spend:{t4 - t3}")
    print(f"sum time spend:{t4 - t0} doc_ana spend:{t3 - t2} p:{(t3 - t2) / (t4 - t0) * 100}")
    ocr_result = {
        "middle_json": middle_json,
        "model_json": model_json
    }
    return ocr_result

def process_one_pdf_file(pdf_path, save_dir, lang="en"):
    pdf_file_name = os.path.basename(pdf_path).replace(".pdf", "")
    target_file = f"{save_dir}/{pdf_file_name}.json.zip"
    if not os.path.exists(f"{save_dir}/"):
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(target_file):
        raise Exception(f"the pdf result exist...[{target_file}]")
    infer_result = infer_one_pdf(pdf_path, lang=lang)
    infer_result['pdf_path'] = pdf_path
    res_json_str = json.dumps(infer_result, ensure_ascii=False)

    with nvtx_annotate("write_json") as ctx:
        with zipfile.ZipFile(target_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            res_json_bytes = res_json_str.encode("utf-8")
            zf.writestr(f"{pdf_file_name}.json", res_json_bytes)
        
        post_size = os.path.getsize(target_file)
        ctx.add_metadata("json_bytes_uncompressed", len(res_json_bytes))
        ctx.add_metadata("json_bytes_compressed", post_size)

    logger.info(f"Finished processing PDF file: {pdf_path}, and saved to {target_file}")
    return target_file

@dataclass
class GPUProcessPool:
    """单个GPU的进程池"""
    gpu_id: int
    num_processes: int
    task_timeout: int = 1800
    max_tasks_per_worker: int = 15
    
    def __post_init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers: List[mp.Process] = []
        self.worker_task_counts: Dict[int, int] = {}
        self.worker_last_heartbeat: Dict[int, float] = {}
        self.lock = threading.Lock()
        self.is_running = False
        self.monitor_thread = None
        self.worker_counter = 0
        
    def _create_worker(self):
        """创建一个工作进程"""
        worker_id = self.worker_counter
        self.worker_counter += 1
        
        process = mp.Process(
            target=gpu_specific_worker,
            args=(worker_id, self.gpu_id, self.task_queue, self.result_queue),
            daemon=True
        )
        process.start()
        
        with self.lock:
            self.workers.append(process)
            self.worker_task_counts[worker_id] = 0
            self.worker_last_heartbeat[worker_id] = time.time()
        
        print(f"Created worker {worker_id} on GPU {self.gpu_id}")
        return process
    
    def _monitor_pool(self):
        """监控当前GPU池的工作进程"""
        while self.is_running:
            time.sleep(3)  # 每3秒检查一次
            
            current_time = time.time()
            workers_to_remove = []
            
            with self.lock:
                # 检查所有工作进程
                for i, process in enumerate(list(self.workers)):
                    if not process.is_alive():
                        print(f"GPU {self.gpu_id} worker {i} died")
                        workers_to_remove.append(i)
                        continue
                    
                    # 检查心跳超时
                    if current_time - self.worker_last_heartbeat.get(i, 0) > self.task_timeout:
                        print(f"GPU {self.gpu_id} worker {i} timeout")
                        try:
                            process.terminate()
                            process.join(timeout=3)
                        except:
                            pass
                        workers_to_remove.append(i)
                        continue
                    
                    # 检查任务数量限制
                    if self.worker_task_counts.get(i, 0) >= self.max_tasks_per_worker:
                        print(f"GPU {self.gpu_id} worker {i} reached max tasks ({self.worker_task_counts[i]})")
                        try:
                            process.terminate()
                            process.join(timeout=3)
                        except:
                            pass
                        workers_to_remove.append(i)
            
            # 移除死亡的进程并创建新的替代
            for idx in sorted(workers_to_remove, reverse=True):
                with self.lock:
                    if idx < len(self.workers):
                        # 移除旧进程
                        old_process = self.workers.pop(idx)
                        try:
                            old_process.terminate()
                            old_process.join(timeout=2)
                        except:
                            pass
                        
                        # 创建新进程
                        self._create_worker()
            
            # 维持进程数量
            with self.lock:
                current_count = len(self.workers)
                if current_count < self.num_processes:
                    for _ in range(current_count, self.num_processes):
                        self._create_worker()
    
    def submit_task(self, task_data: Any):
        """提交任务到当前GPU池"""
        self.task_queue.put(task_data)
    
    def get_result(self, timeout: float = 1.0) -> Optional[Any]:
        """从当前GPU池获取结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def update_heartbeat(self, worker_id: int):
        """更新工作进程心跳"""
        with self.lock:
            self.worker_last_heartbeat[worker_id] = time.time()
    
    def increment_task_count(self, worker_id: int):
        """增加任务计数"""
        with self.lock:
            self.worker_task_counts[worker_id] = self.worker_task_counts.get(worker_id, 0) + 1
    
    def start(self):
        """启动GPU进程池"""
        self.is_running = True
        
        # 创建初始工作进程
        for _ in range(self.num_processes):
            self._create_worker()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_pool, daemon=True)
        self.monitor_thread.start()
        print(f"GPU {self.gpu_id} process pool started with {self.num_processes} workers")
    
    def stop(self):
        """停止GPU进程池"""
        self.is_running = False
        
        # 发送终止信号给所有工作进程
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # 等待进程结束
        with self.lock:
            for process in self.workers:
                try:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                except:
                    pass
            self.workers.clear()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        print(f"GPU {self.gpu_id} process pool stopped")

class MultiGPUProcessManager:
    """多GPU进程管理器"""
    
    def __init__(self, gpu_devices: List[int], processes_per_gpu: int,
                 task_timeout: int = 1800, max_tasks_per_worker: int = 15):
        self.gpu_pools: Dict[int, GPUProcessPool] = {}
        self.lock = threading.Lock()
        
        # 为每个GPU创建进程池
        for gpu_id in gpu_devices:
            pool = GPUProcessPool(
                gpu_id=gpu_id,
                num_processes=processes_per_gpu,
                task_timeout=task_timeout,
                max_tasks_per_worker=max_tasks_per_worker
            )
            self.gpu_pools[gpu_id] = pool
    
    def start_all(self):
        """启动所有GPU进程池"""
        for gpu_id, pool in self.gpu_pools.items():
            pool.start()
        print(f"Started {len(self.gpu_pools)} GPU process pools")
    
    def stop_all(self):
        """停止所有GPU进程池"""
        for pool in self.gpu_pools.values():
            pool.stop()
        print("All GPU process pools stopped")
    
    def distribute_tasks(self, tasks: List[Any]):
        """分发任务到各个GPU池（轮询分发）"""
        gpu_ids = list(self.gpu_pools.keys())
        for i, task in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            self.gpu_pools[gpu_id].submit_task(task)
            if (i + 1) % 100 == 0:
                print(f"Distributed {i + 1} tasks")
    
    def collect_results(self, total_tasks: int) -> List[Any]:
        """从所有GPU池收集结果"""
        results = []
        completed = 0
        last_print_time = time.time()
        
        while completed < total_tasks:
            for gpu_id, pool in self.gpu_pools.items():
                result_data = pool.get_result(timeout=1.0)
                if result_data:
                    if result_data['type'] == 'result':
                        results.append(result_data['result'])
                        completed += 1
                        # 更新任务计数
                        pool.increment_task_count(result_data['worker_id'])
                    elif result_data['type'] == 'heartbeat':
                        # 更新心跳
                        pool.update_heartbeat(result_data['worker_id'])
                    elif result_data['type'] == 'error':
                        print(f"GPU {gpu_id} worker {result_data['worker_id']} error: {result_data['error']}")
                        results.append({
                            'success': False,
                            'error': result_data['error'],
                            'worker_id': result_data['worker_id'],
                            'gpu_id': gpu_id
                        })
                        completed += 1
            
            # 定期打印进度
            current_time = time.time()
            if current_time - last_print_time > 10:  # 每10秒打印一次进度
                print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                last_print_time = current_time
        
        return results

def gpu_specific_worker(worker_id: int, gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """GPU特定的工作进程函数"""
    # 设置特定的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(args.vram_size_gb)
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    
    print(f"Worker {worker_id} started on GPU {gpu_id}")
    
    # 初始化CUDA
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Worker {worker_id} using GPU {gpu_id}, device name: {torch.cuda.get_device_name(0)}")
    
    # 初始化mineru等资源
    from mineru.utils.profile import tracing_reset, tracing_get_and_reset
    
    while True:
        try:
            # 获取任务
            task_data = task_queue.get()
            if task_data is None:  # 终止信号
                print(f"Worker {worker_id} on GPU {gpu_id} received termination signal")
                break
            
            file_path, save_dir = task_data
            
            # 处理任务
            bt = time.time()
            tracing_reset()
            
            try:
                output_file = process_one_pdf_file(file_path, save_dir)
                et = time.time()
                profile = tracing_get_and_reset()
                
                result = {
                    'success': True,
                    'output': output_file,
                    'start_time': bt,
                    'end_time': et,
                    'profile': profile,
                    'worker_id': worker_id,
                    'gpu_id': gpu_id
                }
            except Exception as e:
                et = time.time()
                result = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'start_time': bt,
                    'end_time': et,
                    'worker_id': worker_id,
                    'gpu_id': gpu_id
                }
                # 清理GPU内存
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # 发送结果
            result_queue.put({
                'type': 'result',
                'result': result,
                'worker_id': worker_id,
                'gpu_id': gpu_id,
                'timestamp': time.time()
            })
            
            # 发送心跳信号
            result_queue.put({
                'type': 'heartbeat',
                'worker_id': worker_id,
                'gpu_id': gpu_id,
                'timestamp': time.time()
            })
            
        except Exception as e:
            print(f"GPU {gpu_id} worker {worker_id} critical error: {e}")
            result_queue.put({
                'type': 'error',
                'error': str(e),
                'worker_id': worker_id,
                'gpu_id': gpu_id,
                'timestamp': time.time()
            })
            break
    
    print(f"Worker {worker_id} on GPU {gpu_id} exiting")

def run_with_multi_gpu_pools():
    """使用多GPU进程池运行任务"""
    devices = [int(d.strip()) for d in args.cuda_devices.split(',')]
    
    # 获取所有PDF文件并检查存在性
    pdf_files = []
    for pdf_path in glob.glob(f"{args.input_dir}/*.pdf"):
        if os.path.exists(pdf_path):
            pdf_files.append(pdf_path)
        else:
            print(f"Warning: PDF file not found, skipping: {pdf_path}")
    
    # 检查输出目录是否存在，提前创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 过滤掉已经处理过的文件
    pdf_files_to_process = []
    for pdf_path in pdf_files:
        pdf_file_name = os.path.basename(pdf_path).replace(".pdf", "")
        target_file = f"{args.output_dir}/{pdf_file_name}.json.zip"
        
        if os.path.exists(target_file):
            pass
            # print(f"File already processed, skipping: {pdf_path} -> {target_file}")
        else:
            pdf_files_to_process.append(pdf_path)
    
    random.seed(20250818)
    random.shuffle(pdf_files_to_process)
    
    print(f"Found {len(pdf_files)} PDF files in total")
    print(f"After filtering, {len(pdf_files_to_process)} PDF files need to be processed")
    print(f"Using GPUs: {devices}, {args.num_processes} processes per GPU")
    
    if len(pdf_files_to_process) == 0:
        print("No files to process, exiting.")
        return []
    
    # 创建多GPU进程管理器
    pool_manager = MultiGPUProcessManager(
        gpu_devices=devices,
        processes_per_gpu=args.num_processes,
        task_timeout=args.task_timeout,
        max_tasks_per_worker=args.max_tasks_per_worker
    )
    
    try:
        pool_manager.start_all()
        time.sleep(2)  # 等待进程池初始化
        
        # 准备任务数据
        tasks = [(pdf_file, args.output_dir) for pdf_file in pdf_files_to_process]
        
        # 分发任务
        print("Distributing tasks to GPU pools...")
        pool_manager.distribute_tasks(tasks)
        
        # 收集结果
        print("Collecting results...")
        results = pool_manager.collect_results(len(tasks))
        
        return results
        
    finally:
        pool_manager.stop_all()
def run_test_task():
    devices = [int(d.strip()) for d in args.cuda_devices.split(',')]
    num_processes_per_gpu = args.num_processes
    
    print(f"input_dir {args.input_dir}")
    print(f"output_dir {args.output_dir}")
    print(f"num_processes_per_gpu {num_processes_per_gpu}")
    print(f"cuda_devices {args.cuda_devices}")
    
    try:
        import mineru.version
        mineru_version = mineru.version.__version__
    except ImportError:
        mineru_version = "unknown"

    print(f"mineru_version: {mineru_version}")

    # 使用多GPU进程池模式
    results = run_with_multi_gpu_pools()
    
    # Filter out None results
    file_logs = [res for res in results if res is not None]
    
    log_data = {
        'version': mineru_version,
        'args': vars(args),
        'files': file_logs,
        'summary': {
            'total_files': len(file_logs),
            'success_count': sum(1 for r in file_logs if r.get('success', False)),
            'failure_count': sum(1 for r in file_logs if not r.get('success', True)),
        }
    }
    
    if args.log_dir is not None:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_path = os.path.join(args.log_dir, f'mineru-pid{os.getpid()}-ts{int(time.time())}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        print(f"Log saved to {log_path}")
    else:
        print("No log_dir specified, skipping log save.")
    
    return file_logs

if __name__ == "__main__":
    # 设置GPU内存分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    start_time = time.time()
    results = run_test_task()
    total_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    
    print(f'Total time: {total_time:.2f} seconds')
    print(f'Processed {total_count} files')
    print(f'Success rate: {success_count/total_count*100:.1f}%' if total_count > 0 else 'No files processed')
    print(f'Average time per file: {total_time/total_count:.2f}s' if total_count > 0 else 'N/A')