#!/usr/bin/env python3
"""
MinerU OCR API Server
基于FastAPI的PDF处理服务，支持持续队列监听
"""

import os
import uuid
import asyncio
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import zipfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 导入现有的OCR处理模块
from ocr_pdf import process_one_pdf_file
from process_pool import SimpleProcessPool


def process_task_wrapper_global(task_id: str, file_path: str, output_dir: str, max_pages: int, gpu_id: Optional[int] = None):
    """全局任务包装器，用于处理单个PDF文件 - 避免pickle问题"""
    print(f"[DEBUG] process_task_wrapper_global called: task_id={task_id}, file_path={file_path}")
    try:
        # 执行OCR处理
        # 从环境变量获取参数
        gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.5))
        mm_processor_cache_gb = int(os.environ.get("MM_PROCESSOR_CACHE_GB", 0))
        split_pdf_chunk_size = int(os.environ.get("SPLIT_PDF_CHUNK_SIZE", 0))
        backend = os.environ.get("BACKEND","vllm-engine")
        result = process_one_pdf_file(
            pdf_path=file_path,
            save_dir=output_dir,
            max_pages_per_pdf=max_pages,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            mm_processor_cache_gb=mm_processor_cache_gb,
            split_pdf_chunk_size=split_pdf_chunk_size
        )

        # 处理不同的结果状态
        if result.get('success', False):
            if result.get('skipped', False):
                # 文件被跳过（页数超限）
                return {
                    "task_id": task_id,
                    "status": "skipped",
                    "message": f"PDF skipped: {result.get('page_count', 0)} pages exceeds limit of {max_pages}",
                    "result": result
                }
            elif result.get('output_path'):
                # 处理成功，返回zip文件信息
                zip_path = result['output_path']
                if os.path.exists(zip_path):
                    file_size = os.path.getsize(zip_path)
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "message": "PDF processing completed successfully",
                        "zip_file": {
                            "path": zip_path,
                            "size": file_size,
                            "filename": os.path.basename(zip_path)
                        },
                        "result": result
                    }
                else:
                    return {
                        "task_id": task_id,
                        "status": "error",
                        "message": "Output file not found",
                        "result": result
                    }
            else:
                return {
                    "task_id": task_id,
                    "status": "error",
                    "message": "Processing completed but no output file generated",
                    "result": result
                }
        else:
            # 处理失败
            return {
                "task_id": task_id,
                "status": "failed",
                "message": "PDF processing failed",
                "error": result.get('error', 'Unknown error'),
                "traceback": result.get('traceback'),
                "result": result
            }

    except Exception as e:
        import traceback
        return {
            "task_id": task_id,
            "status": "failed",
            "message": "Task wrapper exception",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


class TaskStatus(BaseModel):
    task_id: str
    chunk_id: Optional[str] = None  # 批次分组ID
    pdf_name: Optional[str] = None  # 原始PDF文件名
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    updated_at: datetime
    message: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    message: str
    status: str


class BatchSubmitRequest(BaseModel):
    input_dir: str  # 输入目录路径
    chunk_id: Optional[str] = None  # 批次分组ID，如果不提供则自动生成


class BatchSubmitResponse(BaseModel):
    chunk_id: str
    task_ids: List[str]
    message: str
    total_files: int
    successful_submissions: int


class MinerUAPIServer:
    def __init__(self,
                 gpu_ids: str = "0",
                 workers_per_gpu: int = 2,
                 vram_size_gb: int = 24,
                 max_pages: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 output_dir: str = "api_results"):

        self.gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        self.workers_per_gpu = workers_per_gpu
        self.vram_size_gb = vram_size_gb
        self.max_pages = max_pages
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 任务存储
        self.tasks: Dict[str, TaskStatus] = {}
        self.task_file_mapping: Dict[str, str] = {}  # task_id -> file_path

        # 初始化进程池（持续模式）
        self.process_pool = SimpleProcessPool(
            gpu_ids=self.gpu_ids,
            workers_per_gpu=self.workers_per_gpu,
            continuous_mode=True
        )

        # 启动持续监听
        self.process_pool.start_continuous_monitoring()

        # 创建FastAPI应用
        self.app = FastAPI(
            title="MinerU OCR API",
            description="PDF处理和OCR服务API",
            version="1.0.0"
        )

        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""

        @self.app.post("/submit_task", response_model=TaskSubmitResponse)
        async def submit_task(file: UploadFile = File(...), chunk_id: Optional[str] = Form(None)):
            """提交PDF处理任务"""
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="只支持PDF文件")

            # 生成任务ID
            task_id = str(uuid.uuid4())

            # 保存上传的文件
            temp_dir = self.output_dir / "temp" / task_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            file_path = temp_dir / file.filename

            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                # 创建任务状态
                task_status = TaskStatus(
                    task_id=task_id,
                    chunk_id=chunk_id,
                    pdf_name=file.filename,
                    status="pending",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    message="任务已提交，等待处理"
                )
                self.tasks[task_id] = task_status
                self.task_file_mapping[task_id] = str(file_path)

                # 提交任务到进程池
                self._submit_ocr_task(task_id, str(file_path))

                return TaskSubmitResponse(
                    task_id=task_id,
                    message="任务提交成功",
                    status="pending"
                )

            except Exception as e:
                # 清理临时文件
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

        @self.app.get("/get_status/{task_id}", response_model=TaskStatus)
        async def get_task_status(task_id: str):
            """获取任务状态"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="任务不存在")

            # 检查进程池中的任务状态
            task = self.tasks[task_id]
            if task.status == "processing":
                # 这里可以添加更详细的进度检查
                pass

            return task

        @self.app.get("/download_result/{task_id}")
        async def download_result(task_id: str):
            """下载处理结果"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="任务不存在")

            task = self.tasks[task_id]
            if task.status != "completed":
                raise HTTPException(status_code=400, detail="任务尚未完成")

            if not task.result_path or not os.path.exists(task.result_path):
                raise HTTPException(status_code=404, detail="结果文件不存在")

            return FileResponse(
                path=task.result_path,
                filename=f"{task_id}_result.zip",
                media_type="application/zip"
            )

        @self.app.get("/list_tasks")
        async def list_tasks():
            """列出所有任务"""
            tasks = list(self.tasks.values())
            # 创建简化的任务列表，包含关键信息
            simplified_tasks = []
            for task in tasks:
                simplified_tasks.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "pdf_name": task.pdf_name or "未知文件",
                    "chunk_id": task.chunk_id or "无分组",
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                })

            return {
                "tasks": simplified_tasks,
                "total_count": len(tasks),
                "status_breakdown": {
                    "pending": len([t for t in tasks if t.status == "pending"]),
                    "processing": len([t for t in tasks if t.status == "processing"]),
                    "completed": len([t for t in tasks if t.status == "completed"]),
                    "failed": len([t for t in tasks if t.status == "failed"])
                }
            }

        @self.app.delete("/delete_task/{task_id}")
        async def delete_task(task_id: str):
            """删除任务及相关文件"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="任务不存在")

            # 删除相关文件
            temp_dir = self.output_dir / "temp" / task_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            task = self.tasks[task_id]
            if task.result_path and os.path.exists(task.result_path):
                os.remove(task.result_path)

            # 删除任务记录
            del self.tasks[task_id]
            if task_id in self.task_file_mapping:
                del self.task_file_mapping[task_id]

            return {"message": "任务已删除"}

        @self.app.post("/batch_submit", response_model=BatchSubmitResponse)
        async def batch_submit_tasks(request: BatchSubmitRequest):
            """批次提交PDF处理任务"""
            input_path = Path(request.input_dir)

            # 验证输入目录
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"输入目录不存在: {request.input_dir}")

            if not input_path.is_dir():
                raise HTTPException(status_code=400, detail=f"输入路径不是目录: {request.input_dir}")

            # 查找所有PDF文件
            pdf_files = list(input_path.glob("*.pdf"))
            if not pdf_files:
                raise HTTPException(status_code=400, detail=f"目录中没有找到PDF文件: {request.input_dir}")

            # 生成或使用提供的chunk_id
            chunk_id = request.chunk_id or f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(pdf_files)}"

            submitted_tasks = []

            # 批量提交任务
            for pdf_file in pdf_files:
                try:
                    # 生成任务ID
                    task_id = str(uuid.uuid4())

                    # 创建任务状态
                    task_status = TaskStatus(
                        task_id=task_id,
                        chunk_id=chunk_id,
                        pdf_name=pdf_file.name,
                        status="pending",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        message=f"批次任务已提交，等待处理"
                    )
                    self.tasks[task_id] = task_status
                    self.task_file_mapping[task_id] = str(pdf_file)

                    # 提交任务到进程池
                    self._submit_ocr_task(task_id, str(pdf_file))

                    submitted_tasks.append(task_id)

                except Exception as e:
                    # 记录失败但继续处理其他文件
                    print(f"提交任务失败 {pdf_file}: {e}")
                    continue

            if not submitted_tasks:
                raise HTTPException(status_code=500, detail="所有任务提交失败")

            return BatchSubmitResponse(
                chunk_id=chunk_id,
                task_ids=submitted_tasks,
                message=f"成功提交 {len(submitted_tasks)}/{len(pdf_files)} 个任务",
                total_files=len(pdf_files),
                successful_submissions=len(submitted_tasks)
            )

        @self.app.get("/list_tasks_by_chunk/{chunk_id}")
        async def list_tasks_by_chunk(chunk_id: str):
            """按chunk_id列出任务"""
            chunk_tasks = [task for task in self.tasks.values() if task.chunk_id == chunk_id]

            if not chunk_tasks:
                raise HTTPException(status_code=404, detail=f"没有找到chunk_id为 {chunk_id} 的任务")

            # 创建简化的任务列表，包含关键信息
            simplified_tasks = []
            for task in chunk_tasks:
                simplified_tasks.append({
                    "task_id": task.task_id,
                    "status": task.status,
                    "pdf_name": task.pdf_name or "未知文件",
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                })

            return {
                "chunk_id": chunk_id,
                "tasks": simplified_tasks,
                "total_tasks": len(chunk_tasks),
                "status_breakdown": {
                    "pending": len([t for t in chunk_tasks if t.status == "pending"]),
                    "processing": len([t for t in chunk_tasks if t.status == "processing"]),
                    "completed": len([t for t in chunk_tasks if t.status == "completed"]),
                    "failed": len([t for t in chunk_tasks if t.status == "failed"])
                }
            }

        @self.app.get("/download_chunk_results/{chunk_id}")
        async def download_chunk_results(chunk_id: str):
            """下载整个chunk的所有结果"""
            chunk_tasks = [task for task in self.tasks.values() if task.chunk_id == chunk_id]
            completed_tasks = [task for task in chunk_tasks if task.status == "completed" and task.result_path]

            if not completed_tasks:
                raise HTTPException(status_code=404, detail=f"chunk_id {chunk_id} 没有已完成的任务")

            # 创建临时zip文件
            temp_zip_path = self.output_dir / f"chunk_{chunk_id}_results.zip"

            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for task in completed_tasks:
                    if task.result_path and os.path.exists(task.result_path):
                        # 将每个结果文件添加到zip中，使用task_id作为文件名前缀
                        zipf.write(task.result_path, f"{task.task_id}_{os.path.basename(task.result_path)}")

            return FileResponse(
                path=temp_zip_path,
                filename=f"chunk_{chunk_id}_results.zip",
                media_type="application/zip"
            )

        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "workers": len(self.process_pool.workers),
                "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
                "processing_tasks": len([t for t in self.tasks.values() if t.status == "processing"]),
                "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"])
            }

    def _submit_ocr_task(self, task_id: str, file_path: str):
        """提交OCR任务到进程池"""
        print(f"[DEBUG] _submit_ocr_task called: task_id={task_id}, file_path={file_path}")
        try:
            # 更新任务状态
            self.tasks[task_id].status = "processing"
            self.tasks[task_id].updated_at = datetime.now()
            self.tasks[task_id].message = "正在处理中..."

            # 创建输出目录
            task_output_dir = self.output_dir / "results" / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)

            # 提交任务 - 使用全局函数而不是实例方法避免pickle问题
            process_task_id = self.process_pool.submit_task(
                process_task_wrapper_global,
                task_id,
                file_path,
                str(task_output_dir),
                self.max_pages
            )

            # 启动后台任务监听完成状态
            asyncio.create_task(self._monitor_task_completion(task_id, process_task_id))

        except Exception as e:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].updated_at = datetime.now()
            self.tasks[task_id].error = str(e)
            self.tasks[task_id].message = "任务提交失败"

  
    async def _monitor_task_completion(self, task_id: str, process_task_id: int):
        """监听任务完成状态"""
        print(f"[DEBUG] _monitor_task_completion started: task_id={task_id}, process_task_id={process_task_id}")
        while True:
            await asyncio.sleep(2)  # 每2秒检查一次

            if self.process_pool.is_task_completed(process_task_id):
                result = self.process_pool.get_task_result(process_task_id)
                if result:
                    _, status, data = result

                    if status == "success":
                        # 处理新的返回值结构
                        task_status = data.get("status", "unknown")

                        if task_status == "completed":
                            # 任务成功完成，有zip文件
                            zip_info = data.get("zip_file", {})
                            zip_path = zip_info.get("path")

                            if zip_path and os.path.exists(zip_path):
                                self.tasks[task_id].status = "completed"
                                self.tasks[task_id].updated_at = datetime.now()
                                self.tasks[task_id].message = data.get("message", "处理完成")
                                self.tasks[task_id].result_path = zip_path
                                self.tasks[task_id].progress = data.get("result", {})
                            else:
                                self.tasks[task_id].status = "failed"
                                self.tasks[task_id].updated_at = datetime.now()
                                self.tasks[task_id].error = "结果文件不存在或无法访问"
                                self.tasks[task_id].message = "处理完成但结果文件丢失"

                        elif task_status == "skipped":
                            # 文件被跳过（页数超限等）
                            self.tasks[task_id].status = "failed"
                            self.tasks[task_id].updated_at = datetime.now()
                            self.tasks[task_id].error = data.get("message", "文件被跳过")
                            self.tasks[task_id].message = "文件被跳过"
                            self.tasks[task_id].progress = data.get("result", {})

                        elif task_status in ["error", "failed"]:
                            # 处理失败
                            self.tasks[task_id].status = "failed"
                            self.tasks[task_id].updated_at = datetime.now()
                            self.tasks[task_id].error = data.get("error", data.get("message", "未知错误"))
                            self.tasks[task_id].message = data.get("message", "处理失败")
                            self.tasks[task_id].progress = data.get("result", {})

                        else:
                            # 未知状态
                            self.tasks[task_id].status = "failed"
                            self.tasks[task_id].updated_at = datetime.now()
                            self.tasks[task_id].error = f"未知任务状态: {task_status}"
                            self.tasks[task_id].message = "任务状态异常"
                    else:
                        # 进程池级别的失败
                        self.tasks[task_id].status = "failed"
                        self.tasks[task_id].updated_at = datetime.now()
                        self.tasks[task_id].error = str(data)
                        self.tasks[task_id].message = "进程池执行失败"

                break

            # 检查任务是否超时（比如30分钟）
            if datetime.now() - self.tasks[task_id].created_at > timedelta(minutes=30):
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].updated_at = datetime.now()
                self.tasks[task_id].error = "任务超时"
                self.tasks[task_id].message = "处理超时"
                break


# 全局服务器实例
server_instance = None


def create_app(**kwargs) -> FastAPI:
    """创建FastAPI应用"""
    global server_instance
    server_instance = MinerUAPIServer(**kwargs)
    return server_instance.app


if __name__ == "__main__":
    # 从环境变量读取配置
    gpu_ids = os.getenv("GPU_IDS", "0")
    workers_per_gpu = int(os.getenv("WORKERS_PER_GPU", "2"))
    vram_size_gb = int(os.getenv("VRAM_SIZE_GB", "24"))
    max_pages = int(os.getenv("MAX_PAGES", "1000")) if os.getenv("MAX_PAGES") else None
    batch_size = int(os.getenv("BATCH_SIZE", "384")) if os.getenv("BATCH_SIZE") else None
    output_dir = os.getenv("OUTPUT_DIR", "api_results")

    # 创建并运行服务器
    app = create_app(
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        vram_size_gb=vram_size_gb,
        max_pages=max_pages,
        batch_size=batch_size,
        output_dir=output_dir
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # 使用8001端口避免冲突
        workers=1,  # 单进程模式，因为内部已经有进程池
        log_level="info"
    )