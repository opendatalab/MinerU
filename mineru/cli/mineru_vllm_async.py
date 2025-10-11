import os
import uuid
import asyncio
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
import sys
# ===================== 1. 配置与设置 =====================
app = FastAPI(
    title="MinerU vLLM 异步服务 (生产完善版)",
    description="""
    提供同步和异步两种文件解析接口，并内置并发控制以保护后端服务。
    - **同步接口 (`/parse_file`)**: 上传文件并等待解析完成，直接返回结果。
    - **异步接口 (`/submit_task`)**: 提交文件后立即返回任务ID，客户端可稍后查询结果。
    """
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 配置 Loguru 日志记录 ★★★
# 1. 移除默认的控制台输出处理器
logger.remove()
# 2. 添加一个新的处理器，用于输出到控制台
logger.add(sys.stderr, level="INFO")
# 3. 添加一个文件处理器，用于将日志写入文件
# rotation="100 MB": 当日志文件达到 100MB 时，会自动创建一个新的。
# retention="7 days": 只保留最近 7 天的日志文件。
# level="INFO": 只记录 INFO 级别及以上的日志。
logger.add(
    "logs/server_{time}.log", 
    rotation="100 MB", 
    retention="7 days", 
    level="INFO",
    encoding="utf-8"
)
# --- 目录与环境变量配置 ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
for d in [INPUT_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:30000")
TASK_TTL_MINUTES = int(os.getenv("TASK_TTL_MINUTES", "60"))

# 限制同时运行的 mineru 子进程数量，保护后端vLLM服务。
# 这个值根据vLLM服务承受能力进行调整。
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "12"))
SEM = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


# --- 内存任务存储 ---
TASKS: Dict[str, Dict[str, Any]] = {}


# ===================== 2. 核心逻辑与工具函数 =====================
# ... (cleanup_path 和 run_mineru_command 函数保持不变) ...
def cleanup_path(path: Path):
    """安全地清理指定的文件或目录。"""
    try:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info(f"🧹 已清理资源: {path}")
    except Exception as e:
        logger.warning(f"清理资源失败: {path}，原因: {e}")


async def run_mineru_command(**kwargs: Any) -> tuple[int, str, str]:
    """异步执行 mineru 命令行工具。"""
    cmd = [
        "mineru", "-p", str(kwargs["input_path"]), "-o", str(kwargs["output_dir"]),
        "-b", "vlm-http-client", "-u", VLLM_URL, "-m", kwargs["method"], "-l", kwargs["lang"],
        "-f", str(kwargs["formula"]).lower(), "-t", str(kwargs["table"]).lower()
    ]
    if kwargs.get("start") is not None:
        cmd += ["-s", str(kwargs["start"])]
    if kwargs.get("end") is not None:
        cmd += ["-e", str(kwargs["end"])]

    task_id = kwargs.get('task_id', 'sync_task')
    logger.info(f"🚀 开始为任务 {task_id} 执行命令: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"❌ 任务 {task_id} 的 MinerU 命令执行失败。错误信息: {stderr.decode()}")
    else:
        logger.success(f"✅ 任务 {task_id} 的 MinerU 命令执行成功。")
        
    return proc.returncode, stdout.decode(), stderr.decode()

async def process_background_task(task_id: str, params: Dict[str, Any], file_info: Dict[str, Any]):
    """处理异步任务的核心后台协程，现在受信号量控制。"""

    # 当并发任务达到上限时，新的任务会在这里异步等待，直到有其他任务完成并释放信号量。
    async with SEM:
        logger.info(f"🚦 任务 {task_id} 获得执行许可，当前并发数: {MAX_CONCURRENT_TASKS - SEM._value}/{MAX_CONCURRENT_TASKS}")
        
        task = TASKS[task_id]
        input_path = file_info["input_path"]
        output_dir = file_info["output_dir"]

        try:
            task["status"] = "processing"
            task["start_time"] = datetime.utcnow()
            
            code, _, err = await run_mineru_command(task_id=task_id, **params, **file_info)

            if code != 0:
                raise RuntimeError(f"MinerU 进程执行失败: {err}")

            if params["return_zip"]:
                zip_path = OUTPUT_DIR / f"{task_id}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in output_dir.rglob('*'):
                        zf.write(p, p.relative_to(output_dir))
                task["result_path"] = str(zip_path)
            else:
                md_files = list(output_dir.rglob("*.md"))
                if not md_files:
                    raise FileNotFoundError("解析成功，但未在输出目录中找到 Markdown 文件。")
                task["result_content"] = md_files[0].read_text(encoding="utf-8")

            task["status"] = "completed"

        except Exception as e:
            logger.error(f"任务 {task_id} 因异常而失败: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
        finally:
            task["end_time"] = datetime.utcnow()
            cleanup_path(input_path)
            cleanup_path(output_dir)
            logger.info(f"🚦 任务 {task_id} 执行完毕，释放信号量。")


# ===================== 3. API 端点 =====================
@app.post("/parse_file", tags=["同步接口"])
async def parse_file_sync(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="需要解析的文档文件。"),
    method: str = Form("auto", description="解析方法: auto, txt, ocr。"),
    lang: str = Form("ch", description="文档语言。"),
    start: Optional[int] = Form(None, description="起始页码 (从0开始)。"),
    end: Optional[int] = Form(None, description="结束页码。"),
    formula: bool = Form(True, description="是否启用公式解析。"),
    table: bool = Form(True, description="是否启用表格解析。"),
    return_zip: bool = Form(False, description="是否将所有输出打包为ZIP文件返回。")
):
    """上传文件，等待解析完成，然后直接返回结果。"""
    req_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{req_id}_{file.filename}"
    output_dir = OUTPUT_DIR / req_id
    output_dir.mkdir()

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 将清理任务添加到后台，确保响应发送后再执行
    background_tasks.add_task(cleanup_path, input_path)
    background_tasks.add_task(cleanup_path, output_dir)

    code, _, err = await run_mineru_command(
        input_path=input_path, output_dir=output_dir, method=method, lang=lang,
        start=start, end=end, formula=formula, table=table
    )

    if code != 0:
        raise HTTPException(status_code=500, detail={"message": "MinerU 命令执行失败", "stderr": err})

    if return_zip:
        zip_path = OUTPUT_DIR / f"{req_id}.zip"
        background_tasks.add_task(cleanup_path, zip_path) # 下载后清理zip
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in output_dir.rglob('*'):
                zf.write(p, p.relative_to(output_dir))
        return FileResponse(zip_path, filename=f"{Path(file.filename).stem}_result.zip", media_type="application/zip")

    md_files = list(output_dir.rglob("*.md"))
    if not md_files:
        raise HTTPException(status_code=404, detail="解析已完成，但未找到任何 Markdown 输出文件。")
    
    return JSONResponse({"markdown_content": md_files[0].read_text(encoding="utf-8")})


@app.post("/submit_task", tags=["异步接口"])
async def submit_task(
    file: UploadFile = File(..., description="需要解析的文档文件。"),
    method: str = Form("auto"), lang: str = Form("ch"), start: Optional[int] = Form(None),
    end: Optional[int] = Form(None), formula: bool = Form(True), table: bool = Form(True),
    return_zip: bool = Form(False, description="是否将结果准备成一个可下载的ZIP文件。")
):
    """提交一个解析任务，并立即返回任务ID，不阻塞等待。"""
    task_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{task_id}_{file.filename}"
    output_dir = OUTPUT_DIR / task_id
    output_dir.mkdir()

    with open(input_path, "wb") as f:
        f.write(await file.read())

    TASKS[task_id] = {
        "status": "queued",
        "submitted_time": datetime.utcnow(),
        "filename": file.filename,
        "result_path": None,
        "result_content": None,
        "error": None
    }
    
    # 将所有请求参数打包
    params = locals()
    params.pop("file", None) # 不需要传递UploadFile对象

    # 创建一个后台任务来执行真正的处理逻辑
    asyncio.create_task(process_background_task(
        task_id, params, {"input_path": input_path, "output_dir": output_dir}
    ))

    return {"task_id": task_id, "status": "queued", "message": "任务已提交，请稍后使用任务ID查询结果。"}


@app.get("/task/{task_id}", tags=["异步接口"])
async def get_task_result(task_id: str, background_tasks: BackgroundTasks):
    """根据任务ID查询任务状态和结果。"""
    prune_old_tasks()  # 每次查询时顺便清理一下旧任务
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务ID不存在或已过期被清理。")

    if task["status"] == "completed":
        if task.get("result_path"):  # 结果是一个ZIP文件
            zip_path = Path(task["result_path"])
            # 客户端下载文件后，通过后台任务清理ZIP文件，并从内存中移除任务记录
            background_tasks.add_task(cleanup_path, zip_path)
            TASKS.pop(task_id, None)
            return FileResponse(zip_path, filename=f"{Path(task['filename']).stem}_result.zip")
        else:  # 结果是Markdown文本
            # 客户端获取内容后，直接从内存中移除任务记录
            result_content = task["result_content"]
            TASKS.pop(task_id, None)
            return {"task_id": task_id, "status": "completed", "markdown_content": result_content}
    
    return {"task_id": task_id, "status": task["status"], "error": task.get("error")}

# ===================== 4. 系统维护与启动 =====================
def prune_old_tasks():
    """从内存中清理超过设定时间（TTL）的旧任务，防止内存泄漏。"""
    cutoff_time = datetime.utcnow() - timedelta(minutes=TASK_TTL_MINUTES)
    tasks_to_prune = [
        tid for tid, task in TASKS.items()
        if task.get("end_time") and task.get("end_time") < cutoff_time
    ]
    for tid in tasks_to_prune:
        TASKS.pop(tid, None)
        logger.info(f"🗑️ 已清理过期任务: {tid}")

@app.on_event("startup")
async def startup_event():
    """服务器启动时执行的事件。"""
    logger.info("🚀 MinerU vLLM 异步服务正在启动...")
    logger.info(f"📂 输入文件目录: {INPUT_DIR}")
    logger.info(f"📂 输出文件目录: {OUTPUT_DIR}")
    logger.info(f"🔗 目标 vLLM 服务地址: {VLLM_URL}")
    logger.info(f"⏰ 任务记录保留时间: {TASK_TTL_MINUTES} 分钟")
    logger.info(f"🚦 最大并发处理任务数: {MAX_CONCURRENT_TASKS}")


if __name__ == "__main__":
    import uvicorn
    # 文件名是 mineru_vllm_async.py, 所以使用 "mineru_vllm_async:app"
    print("✅ MinerU vLLM 异步服务已启动: http://127.0.0.1:8000")
    print("📘 API 文档地址: http://127.0.0.1:8000/docs")
    uvicorn.run("mineru_vllm_async:app", host="127.0.0.1", port=8000, reload=True)