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
import fitz  # PyMuPDF

parser = argparse.ArgumentParser(description="MB PDF OCR")
parser.add_argument('--input-dir', type=str, required=True, help='输入数据集的目录')
parser.add_argument('--output-dir', type=str, required=True, help='输出数据集的目录')
parser.add_argument('--vram-size-gb', type=int, default=24, help='使用显存的GB数（默认24）')
parser.add_argument('--cuda-devices', type=str, default='0', help='绑定的GPU编号，多个用逗号分隔，如"0,1,2"')
parser.add_argument('--num-processes', type=int, default=1, help='每个GPU的进程数')
parser.add_argument('--log-dir', type=str, default='/tmp', help='日志目录')
parser.add_argument('--task-timeout', type=int, default=1800, help='任务超时时间（秒）')
parser.add_argument('--max-task-duration', type=int, default=1800, help='每个任务最大执行时间（秒），默认1800秒即30分钟')
parser.add_argument('--monitor-log-path', type=str, default=None, help='进程监控日志路径，用于记录每个进程的处理统计')
args = parser.parse_args()

def infer_one_pdf(pdf_file_path, lang="en"):
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

    t0 = time.time()
    with open(pdf_file_path, 'rb') as fi:
        pdf_bytes = fi.read()
    t1 = time.time()

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

    # Get PDF page count and write to logs
    try:
        pdf_doc = fitz.open(pdf_path)
        page_count = len(pdf_doc)
        pdf_doc.close()

        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        # Write PDF name and page count to count_page.txt (two columns)
        count_file = os.path.join(logs_dir, "count_page.txt")
        pdf_filename = os.path.basename(pdf_path)
        with open(count_file, "a", encoding="utf-8") as f:
            f.write(f"{pdf_filename}\t{page_count}\n")

        logger.info(f"PDF {pdf_path} has {page_count} pages, written to {count_file}")
    except Exception as e:
        logger.error(f"Failed to get page count for {pdf_path}: {e}")

    infer_result = infer_one_pdf(pdf_path, lang=lang)
    infer_result['pdf_path'] = pdf_path
    res_json_str = json.dumps(infer_result, ensure_ascii=False)

    with zipfile.ZipFile(target_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        res_json_bytes = res_json_str.encode("utf-8")
        zf.writestr(f"{pdf_file_name}.json", res_json_bytes)

    post_size = os.path.getsize(target_file)

    logger.info(f"Finished processing PDF file: {pdf_path}, and saved to {target_file}")
    return target_file

@dataclass
class GPUProcessPool:
    """单个GPU的进程池"""
    gpu_id: int
    num_processes: int
    task_timeout: int = 1800
    max_task_duration: int = 1800  # 30分钟 = 1800秒
    monitor_log_path: Optional[str] = None  # 监控日志路径

    def __post_init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers: Dict[int, mp.Process] = {}  # 改为字典，以worker_id为键
        self.worker_task_start_times: Dict[int, float] = {}  # 记录每个worker当前任务开始时间
        self.worker_last_heartbeat: Dict[int, float] = {}
        self.worker_status: Dict[int, str] = {}  # 'idle' or 'busy'
        self.worker_stats: Dict[int, Dict] = {}  # 记录每个worker的统计信息
        self.worker_pids: Dict[int, int] = {}  # 记录worker的真实进程PID
        self.worker_current_tasks: Dict[int, str] = {}  # 记录每个worker当前处理的任务文件
        self.lock = threading.Lock()
        self.is_running = False
        self.monitor_thread = None
        self.stats_thread = None  # 统计输出线程
        self.process_count_thread = None  # 进程数统计线程
        self.worker_counter = 0
        self.restart_count = 0  # 重启计数器
        
    def _create_worker(self):
        """创建一个工作进程"""
        worker_id = self.worker_counter
        self.worker_counter += 1

        process = mp.Process(
            target=gpu_specific_worker,
            args=(worker_id, self.gpu_id, self.task_queue, self.result_queue, self.monitor_log_path),
            daemon=True
        )
        process.start()

        with self.lock:
            self.workers[worker_id] = process  # 使用字典存储
            self.worker_task_start_times[worker_id] = 0
            self.worker_last_heartbeat[worker_id] = time.time()
            self.worker_status[worker_id] = 'idle'
            self.worker_pids[worker_id] = process.pid
            self.worker_stats[worker_id] = {
                'total_pages': 0,
                'total_tasks': 0,
                'start_time': time.time(),
                'last_10min_pages': 0,
                'last_10min_time': time.time()
            }

        print(f"Created worker {worker_id} (PID: {process.pid}) on GPU {self.gpu_id}")
        return worker_id

    def _log_timeout_event(self, worker_id: int, timeout_type: str, duration: float, task_file: str = None):
        """记录超时事件到异常日志文件"""
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        anomaly_log_file = os.path.join(logs_dir, "anomaly_events.txt")
        current_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

        pid = self.worker_pids.get(worker_id, 'unknown')

        timeout_info = {
            'timestamp': timestamp,
            'unix_timestamp': current_time,
            'event_type': 'timeout',
            'gpu_id': self.gpu_id,
            'worker_id': worker_id,
            'pid': pid,
            'timeout_type': timeout_type,  # 'task_timeout' or 'heartbeat_timeout'
            'duration_seconds': duration,
            'task_file': task_file or self.worker_current_tasks.get(worker_id, 'unknown'),
            'action': 'kill_and_restart'
        }

        with open(anomaly_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(timeout_info, ensure_ascii=False) + '\n')

        logger.warning(f"GPU {self.gpu_id} Worker {worker_id} (PID {pid}) {timeout_type}: "
                      f"duration={duration:.1f}s, task={task_file or 'unknown'}, killed and restarted")

    def _log_restart_event(self, worker_id: int, old_pid: int, new_worker_id: int, new_pid: int, reason: str):
        """记录进程重启事件到异常日志文件"""
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        anomaly_log_file = os.path.join(logs_dir, "anomaly_events.txt")
        current_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

        self.restart_count += 1
        restart_info = {
            'timestamp': timestamp,
            'unix_timestamp': current_time,
            'event_type': 'restart',
            'gpu_id': self.gpu_id,
            'old_worker_id': worker_id,
            'old_pid': old_pid,
            'new_worker_id': new_worker_id,
            'new_pid': new_pid,
            'restart_reason': reason,
            'restart_count': self.restart_count,
            'action': 'process_restarted'
        }

        with open(anomaly_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(restart_info, ensure_ascii=False) + '\n')

        logger.info(f"GPU {self.gpu_id} Worker {worker_id} (PID {old_pid}) restarted as Worker {new_worker_id} (PID {new_pid}), reason: {reason}")

    def _monitor_process_count(self):
        """每分钟记录活跃进程数和检测异常情况，分别写入不同日志文件"""
        while self.is_running:
            time.sleep(60)  # 每60秒记录一次

            current_time = time.time()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)

            with self.lock:
                active_count = len([w for w in self.workers.values() if w.is_alive()])
                expected_count = self.num_processes

                # 统计各状态的进程数
                idle_count = sum(1 for status in self.worker_status.values() if status == 'idle')
                busy_count = sum(1 for status in self.worker_status.values() if status == 'busy')

                # 记录定时监控信息到monitoring_status.txt
                monitoring_log_file = os.path.join(logs_dir, "monitoring_status.txt")
                monitoring_info = {
                    'timestamp': timestamp,
                    'unix_timestamp': current_time,
                    'gpu_id': self.gpu_id,
                    'type': 'periodic_monitoring',
                    'active_processes': active_count,
                    'expected_processes': expected_count,
                    'idle_processes': idle_count,
                    'busy_processes': busy_count,
                    'restart_count_total': self.restart_count
                }

                with open(monitoring_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(monitoring_info, ensure_ascii=False) + '\n')

                # 检测异常情况
                anomalies = []

                # 检查死亡进程
                dead_workers = []
                for worker_id, process in self.workers.items():
                    if not process.is_alive():
                        dead_workers.append(worker_id)

                if dead_workers:
                    anomalies.append(f"Dead processes: {dead_workers}")

                # 检查进程数不匹配
                if active_count != expected_count:
                    anomalies.append(f"Process count mismatch: {active_count}/{expected_count}")

                # 检查长时间运行的任务
                long_running_tasks = []
                for worker_id, start_time in self.worker_task_start_times.items():
                    if start_time > 0 and self.worker_status.get(worker_id) == 'busy':
                        duration = current_time - start_time
                        if duration > self.max_task_duration * 0.8:  # 超过80%阈值时预警
                            task_file = self.worker_current_tasks.get(worker_id, 'unknown')
                            long_running_tasks.append(f"Worker {worker_id}: {duration:.1f}s ({task_file})")

                if long_running_tasks:
                    anomalies.append(f"Long running tasks: {long_running_tasks}")

                # 检查心跳超时风险
                stale_heartbeats = []
                for worker_id, last_heartbeat in self.worker_last_heartbeat.items():
                    heartbeat_age = current_time - last_heartbeat
                    if heartbeat_age > self.task_timeout * 0.8:  # 超过80%阈值时预警
                        stale_heartbeats.append(f"Worker {worker_id}: {heartbeat_age:.1f}s")

                if stale_heartbeats:
                    anomalies.append(f"Stale heartbeats: {stale_heartbeats}")

                # 如果发现异常，记录到异常日志文件
                if anomalies:
                    anomaly_log_file = os.path.join(logs_dir, "anomaly_events.txt")
                    anomaly_info = {
                        'timestamp': timestamp,
                        'unix_timestamp': current_time,
                        'event_type': 'anomaly_detection',
                        'gpu_id': self.gpu_id,
                        'active_processes': active_count,
                        'expected_processes': expected_count,
                        'anomalies': anomalies,
                        'action': 'monitoring_alert'
                    }

                    with open(anomaly_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(anomaly_info, ensure_ascii=False) + '\n')

            status_msg = f"GPU {self.gpu_id} Process Status: {active_count}/{expected_count} active, {idle_count} idle, {busy_count} busy, {self.restart_count} restarts"
            if anomalies:
                status_msg += f" [ANOMALIES: {'; '.join(anomalies)}]"
            print(status_msg)
    
    def _monitor_pool(self):
        """监控当前GPU池的工作进程"""
        while self.is_running:
            time.sleep(3)  # 每3秒检查一次

            current_time = time.time()
            workers_to_remove = []

            with self.lock:
                # 检查所有工作进程
                for worker_id, process in list(self.workers.items()):
                    if not process.is_alive():
                        print(f"GPU {self.gpu_id} worker {worker_id} died")
                        workers_to_remove.append((worker_id, "process_died"))
                        continue

                    # 检查心跳超时（空闲时的超时检查）
                    heartbeat_duration = current_time - self.worker_last_heartbeat.get(worker_id, 0)
                    if heartbeat_duration > self.task_timeout:
                        print(f"GPU {self.gpu_id} worker {worker_id} heartbeat timeout")
                        # 记录超时事件到日志
                        self._log_timeout_event(worker_id, "heartbeat_timeout", heartbeat_duration)
                        try:
                            process.terminate()
                            process.join(timeout=3)
                        except:
                            pass
                        workers_to_remove.append((worker_id, "heartbeat_timeout"))
                        continue

                    # 检查任务执行时间限制（只有在忙碌状态时才检查）
                    if (self.worker_status.get(worker_id) == 'busy' and
                        self.worker_task_start_times.get(worker_id, 0) > 0):
                        task_duration = current_time - self.worker_task_start_times[worker_id]
                        if task_duration > self.max_task_duration:
                            print(f"GPU {self.gpu_id} worker {worker_id} task timeout ({task_duration:.1f}s > {self.max_task_duration}s)")
                            # 记录超时事件到日志
                            self._log_timeout_event(worker_id, "task_timeout", task_duration)
                            try:
                                process.terminate()
                                process.join(timeout=3)
                            except:
                                pass
                            workers_to_remove.append((worker_id, "task_timeout"))
                            continue

            # 移除死亡的进程并创建新的替代
            for worker_id, reason in workers_to_remove:
                with self.lock:
                    if worker_id in self.workers:
                        # 记录旧进程信息
                        old_process = self.workers[worker_id]
                        old_pid = self.worker_pids.get(worker_id, 'unknown')

                        # 移除旧进程
                        del self.workers[worker_id]
                        try:
                            old_process.terminate()
                            old_process.join(timeout=2)
                        except:
                            pass

                        # 清理相关数据
                        if worker_id in self.worker_task_start_times:
                            del self.worker_task_start_times[worker_id]
                        if worker_id in self.worker_status:
                            del self.worker_status[worker_id]
                        if worker_id in self.worker_last_heartbeat:
                            del self.worker_last_heartbeat[worker_id]
                        if worker_id in self.worker_current_tasks:
                            del self.worker_current_tasks[worker_id]
                        if worker_id in self.worker_stats:
                            del self.worker_stats[worker_id]
                        if worker_id in self.worker_pids:
                            del self.worker_pids[worker_id]

                        # 创建新进程
                        new_worker_id = self._create_worker()
                        new_pid = self.worker_pids.get(new_worker_id, 'unknown')

                        # 记录重启事件
                        self._log_restart_event(worker_id, old_pid, new_worker_id, new_pid, reason)

            # 维持进程数量
            with self.lock:
                current_count = len(self.workers)
                if current_count < self.num_processes:
                    for _ in range(current_count, self.num_processes):
                        new_worker_id = self._create_worker()
                        new_pid = self.worker_pids.get(new_worker_id, 'unknown')
                        print(f"GPU {self.gpu_id} created additional worker {new_worker_id} (PID {new_pid}) to maintain pool size")
    
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

    def update_task_start(self, worker_id: int, start_time: float, task_file: str = None):
        """更新任务开始时间"""
        with self.lock:
            self.worker_task_start_times[worker_id] = start_time
            self.worker_status[worker_id] = 'busy'
            if task_file:
                self.worker_current_tasks[worker_id] = task_file
            print(f"GPU {self.gpu_id} worker {worker_id} started task at {start_time}, file: {task_file or 'unknown'}")

    def update_task_end(self, worker_id: int):
        """更新任务结束"""
        with self.lock:
            self.worker_task_start_times[worker_id] = 0
            self.worker_status[worker_id] = 'idle'
            if worker_id in self.worker_current_tasks:
                del self.worker_current_tasks[worker_id]
            print(f"GPU {self.gpu_id} worker {worker_id} finished task")

    def update_worker_stats(self, worker_id: int, pages_processed: int):
        """更新worker的统计信息"""
        with self.lock:
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]['total_pages'] += pages_processed
                self.worker_stats[worker_id]['total_tasks'] += 1
                self.worker_stats[worker_id]['last_10min_pages'] += pages_processed

    def _monitor_stats(self):
        """统计监控线程，每10分钟输出一次统计"""
        if not self.monitor_log_path:
            return

        while self.is_running:
            time.sleep(600)  # 10分钟 = 600秒

            current_time = time.time()
            with self.lock:
                for worker_id, stats in self.worker_stats.items():
                    if worker_id in self.worker_pids:
                        pid = self.worker_pids[worker_id]

                        # 计算近10分钟的速率
                        time_elapsed = current_time - stats['last_10min_time']
                        pages_per_10min = stats['last_10min_pages']

                        # 计算总体速率
                        total_time = current_time - stats['start_time']
                        total_pages = stats['total_pages']
                        total_tasks = stats['total_tasks']

                        # 写入日志文件
                        log_file = f"{self.monitor_log_path}/{pid}.log"
                        os.makedirs(self.monitor_log_path, exist_ok=True)

                        with open(log_file, 'a', encoding='utf-8') as f:
                            log_entry = {
                                'timestamp': current_time,
                                'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
                                'worker_id': worker_id,
                                'pid': pid,
                                'gpu_id': self.gpu_id,
                                'pages_last_10min': pages_per_10min,
                                'time_last_10min': time_elapsed,
                                'total_pages': total_pages,
                                'total_tasks': total_tasks,
                                'total_time': total_time,
                                'avg_pages_per_hour': (total_pages / total_time * 3600) if total_time > 0 else 0
                            }
                            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

                        # 重置10分钟统计
                        stats['last_10min_pages'] = 0
                        stats['last_10min_time'] = current_time

                        print(f"GPU {self.gpu_id} Worker {worker_id} (PID {pid}): {pages_per_10min} pages in last 10min, "
                              f"total {total_pages} pages, avg {total_pages/total_time*3600:.1f} pages/hour")
    
    def start(self):
        """启动GPU进程池"""
        self.is_running = True
        
        # 创建初始工作进程
        for _ in range(self.num_processes):
            self._create_worker()

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_pool, daemon=True)
        self.monitor_thread.start()

        # 启动统计监控线程
        if self.monitor_log_path:
            self.stats_thread = threading.Thread(target=self._monitor_stats, daemon=True)
            self.stats_thread.start()
            print(f"GPU {self.gpu_id} stats monitoring started, logs will be saved to {self.monitor_log_path}")

        # 启动进程数监控线程
        self.process_count_thread = threading.Thread(target=self._monitor_process_count, daemon=True)
        self.process_count_thread.start()
        print(f"GPU {self.gpu_id} process count monitoring started")

        print(f"GPU {self.gpu_id} process pool started with {self.num_processes} workers")
    
    def stop(self):
        """停止GPU进程池"""
        print(f"GPU {self.gpu_id} process pool stopping...")
        self.is_running = False

        # 发送终止信号给所有工作进程
        worker_count = len(self.workers)
        for _ in range(worker_count):
            self.task_queue.put(None)

        # 等待进程结束，并强制终止未响应的进程
        with self.lock:
            terminated_count = 0
            for worker_id, process in list(self.workers.items()):
                try:
                    # 先尝试正常结束
                    process.join(timeout=3)
                    if process.is_alive():
                        print(f"GPU {self.gpu_id} worker {worker_id} (PID {process.pid}) not responding, terminating...")
                        process.terminate()
                        process.join(timeout=2)
                        terminated_count += 1

                    # 如果还是没结束，强制杀死
                    if process.is_alive():
                        print(f"GPU {self.gpu_id} worker {worker_id} (PID {process.pid}) forcing kill...")
                        process.kill()
                        terminated_count += 1

                except Exception as e:
                    print(f"GPU {self.gpu_id} error stopping worker {worker_id}: {e}")

            # 清理所有数据结构
            self.workers.clear()
            self.worker_task_start_times.clear()
            self.worker_last_heartbeat.clear()
            self.worker_status.clear()
            self.worker_current_tasks.clear()
            self.worker_stats.clear()
            self.worker_pids.clear()

        # 停止监控线程
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)

        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=3)

        if self.process_count_thread and self.process_count_thread.is_alive():
            self.process_count_thread.join(timeout=3)

        # 记录最终停止状态
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        monitoring_log_file = os.path.join(logs_dir, "monitoring_status.txt")
        shutdown_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'unix_timestamp': time.time(),
            'gpu_id': self.gpu_id,
            'type': 'shutdown_complete',
            'workers_terminated': terminated_count,
            'total_restarts': self.restart_count,
            'action': 'process_pool_stopped'
        }

        with open(monitoring_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(shutdown_info, ensure_ascii=False) + '\n')

        print(f"GPU {self.gpu_id} process pool stopped (terminated {terminated_count} workers, {self.restart_count} total restarts)")

class MultiGPUProcessManager:
    """多GPU进程管理器"""
    
    def __init__(self, gpu_devices: List[int], processes_per_gpu: int,
                 task_timeout: int = 1800, max_task_duration: int = 1800, monitor_log_path: Optional[str] = None):
        self.gpu_pools: Dict[int, GPUProcessPool] = {}
        self.lock = threading.Lock()

        # 为每个GPU创建进程池
        for gpu_id in gpu_devices:
            pool = GPUProcessPool(
                gpu_id=gpu_id,
                num_processes=processes_per_gpu,
                task_timeout=task_timeout,
                max_task_duration=max_task_duration,
                monitor_log_path=monitor_log_path
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
                        # 更新任务结束
                        pool.update_task_end(result_data['worker_id'])
                    elif result_data['type'] == 'heartbeat':
                        # 更新心跳
                        pool.update_heartbeat(result_data['worker_id'])
                    elif result_data['type'] == 'task_start':
                        # 更新任务开始时间和当前任务文件
                        pool.update_task_start(result_data['worker_id'], result_data['start_time'], result_data.get('task_file'))
                    elif result_data['type'] == 'task_stats':
                        # 更新任务统计信息
                        pool.update_worker_stats(result_data['worker_id'], result_data['pages_processed'])
                    elif result_data['type'] == 'error':
                        print(f"GPU {gpu_id} worker {result_data['worker_id']} error: {result_data['error']}")
                        results.append({
                            'success': False,
                            'error': result_data['error'],
                            'worker_id': result_data['worker_id'],
                            'gpu_id': gpu_id
                        })
                        completed += 1
                        # 错误时也要更新任务结束
                        pool.update_task_end(result_data['worker_id'])
            
            # 定期打印进度
            current_time = time.time()
            if current_time - last_print_time > 10:  # 每10秒打印一次进度
                print(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                last_print_time = current_time
        
        return results

def gpu_specific_worker(worker_id: int, gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, monitor_log_path: Optional[str] = None):
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
    
    while True:
        try:
            # 获取任务
            task_data = task_queue.get()
            if task_data is None:  # 终止信号
                print(f"Worker {worker_id} on GPU {gpu_id} received termination signal")
                break
            
            file_path, save_dir = task_data

            # 发送任务开始信号
            bt = time.time()
            result_queue.put({
                'type': 'task_start',
                'worker_id': worker_id,
                'gpu_id': gpu_id,
                'start_time': bt,
                'timestamp': bt,
                'task_file': file_path
            })

            try:
                # 获取PDF页数
                pages_processed = 0
                try:
                    import fitz
                    pdf_doc = fitz.open(file_path)
                    pages_processed = len(pdf_doc)
                    pdf_doc.close()
                except Exception as e:
                    print(f"Warning: Could not get page count for {file_path}: {e}")
                    pages_processed = 1  # 默认至少有1页

                output_file = process_one_pdf_file(file_path, save_dir)
                et = time.time()

                # 发送统计信息
                if monitor_log_path:
                    result_queue.put({
                        'type': 'task_stats',
                        'worker_id': worker_id,
                        'gpu_id': gpu_id,
                        'pages_processed': pages_processed,
                        'timestamp': et
                    })

                result = {
                    'success': True,
                    'output': output_file,
                    'start_time': bt,
                    'end_time': et,
                    'worker_id': worker_id,
                    'gpu_id': gpu_id,
                    'pages_processed': pages_processed
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

            # 发送心跳信号（任务结束后变为空闲状态）
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
        max_task_duration=args.max_task_duration,
        monitor_log_path=args.monitor_log_path
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