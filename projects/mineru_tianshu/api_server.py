"""
MinerU Tianshu - API Server
天枢API服务器

提供RESTful API接口用于任务提交、查询和管理
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
from loguru import logger
import uvicorn
from typing import Optional
from datetime import datetime
import os
import re
import uuid
import json
from minio import Minio

from task_db import TaskDB

# 初始化 FastAPI 应用
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级多GPU文档解析服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
db = TaskDB()

# 配置输出目录
OUTPUT_DIR = Path('/tmp/mineru_tianshu_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MinIO 配置
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', ''),
    'access_key': os.getenv('MINIO_ACCESS_KEY', ''),
    'secret_key': os.getenv('MINIO_SECRET_KEY', ''),
    'secure': True,
    'bucket_name': os.getenv('MINIO_BUCKET', '')
}


def get_minio_client():
    """获取MinIO客户端实例"""
    return Minio(
        endpoint=MINIO_CONFIG['endpoint'],
        access_key=MINIO_CONFIG['access_key'],
        secret_key=MINIO_CONFIG['secret_key'],
        secure=MINIO_CONFIG['secure']
    )


def process_markdown_images(md_content: str, image_dir: Path, upload_images: bool = False):
    """
    处理 Markdown 中的图片引用
    
    Args:
        md_content: Markdown 内容
        image_dir: 图片所在目录
        upload_images: 是否上传图片到 MinIO 并替换链接
        
    Returns:
        处理后的 Markdown 内容
    """
    if not upload_images:
        return md_content
    
    try:
        minio_client = get_minio_client()
        bucket_name = MINIO_CONFIG['bucket_name']
        minio_endpoint = MINIO_CONFIG['endpoint']
        
        # 查找所有 markdown 格式的图片
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # 构建完整的本地图片路径
            full_image_path = image_dir / Path(image_path).name
            
            if full_image_path.exists():
                # 获取文件后缀
                file_extension = full_image_path.suffix
                # 生成 UUID 作为新文件名
                new_filename = f"{uuid.uuid4()}{file_extension}"
                
                try:
                    # 上传到 MinIO
                    object_name = f"images/{new_filename}"
                    minio_client.fput_object(bucket_name=bucket_name, object_name=object_name, file_path=str(full_image_path))
                    
                    # 生成 MinIO 访问 URL
                    scheme = 'https' if MINIO_CONFIG['secure'] else 'http'
                    minio_url = f"{scheme}://{minio_endpoint}/{bucket_name}/{object_name}"
                    
                    # 返回 HTML 格式的 img 标签
                    return f'<img src="{minio_url}" alt="{alt_text}">'
                except Exception as e:
                    logger.error(f"Failed to upload image to MinIO: {e}")
                    return match.group(0)  # 上传失败，保持原样
            
            return match.group(0)
        
        # 替换所有图片引用
        new_content = re.sub(img_pattern, replace_image, md_content)
        return new_content
        
    except Exception as e:
        logger.error(f"Error processing markdown images: {e}")
        return md_content  # 出错时返回原内容


def read_json_file(file_path: Path):
    """
    读取 JSON 文件

    Args:
        file_path: JSON 文件路径

    Returns:
        解析后的 JSON 数据，失败返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        return None


def get_file_metadata(file_path: Path):
    """
    获取文件元数据

    Args:
        file_path: 文件路径

    Returns:
        包含文件元数据的字典
    """
    if not file_path.exists():
        return None

    stat = file_path.stat()
    return {
        'size': stat.st_size,
        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
    }


def get_images_info(image_dir: Path, upload_to_minio: bool = False):
    """
    获取图片目录信息

    Args:
        image_dir: 图片目录路径
        upload_to_minio: 是否上传到 MinIO

    Returns:
        图片信息字典
    """
    if not image_dir.exists() or not image_dir.is_dir():
        return {
            'count': 0,
            'list': [],
            'uploaded_to_minio': False
        }

    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
    image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    images_list = []

    for img_file in sorted(image_files):
        img_info = {
            'name': img_file.name,
            'size': img_file.stat().st_size,
            'path': str(img_file.relative_to(image_dir.parent))
        }

        # 如果需要上传到 MinIO
        if upload_to_minio:
            try:
                minio_client = get_minio_client()
                bucket_name = MINIO_CONFIG['bucket_name']
                minio_endpoint = MINIO_CONFIG['endpoint']

                # 生成 UUID 作为新文件名
                file_extension = img_file.suffix
                new_filename = f"{uuid.uuid4()}{file_extension}"
                object_name = f"images/{new_filename}"

                # 上传到 MinIO
                minio_client.fput_object(bucket_name=bucket_name, object_name=object_name, file_path=str(img_file))

                # 生成访问 URL
                scheme = 'https' if MINIO_CONFIG['secure'] else 'http'
                img_info['url'] = f"{scheme}://{minio_endpoint}/{bucket_name}/{object_name}"

            except Exception as e:
                logger.error(f"Failed to upload image {img_file.name} to MinIO: {e}")
                img_info['url'] = None

        images_list.append(img_info)

    return {
        'count': len(images_list),
        'list': images_list,
        'uploaded_to_minio': upload_to_minio
    }


@app.get("/")
async def root():
    """API根路径"""
    return {
        "service": "MinerU Tianshu",
        "version": "1.0.0",
        "description": "天枢 - 企业级多GPU文档解析服务",
        "docs": "/docs"
    }


@app.post("/api/v1/tasks/submit")
async def submit_task(
    file: UploadFile = File(..., description="文档文件: PDF/图片(MinerU解析) 或 Office/HTML/文本等(MarkItDown解析)"),
    backend: str = Form('pipeline', description="处理后端: pipeline/vlm-transformers/vlm-vllm-engine"),
    lang: str = Form('ch', description="语言: ch/en/korean/japan等"),
    method: str = Form('auto', description="解析方法: auto/txt/ocr"),
    formula_enable: bool = Form(True, description="是否启用公式识别"),
    table_enable: bool = Form(True, description="是否启用表格识别"),
    priority: int = Form(0, description="优先级，数字越大越优先"),
):
    """
    提交文档解析任务
    
    立即返回 task_id，任务在后台异步处理
    """
    try:
        # 保存上传的文件到临时目录
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
        
        # 流式写入文件到磁盘，避免高内存使用
        while True:
            chunk = await file.read(1 << 23)  # 8MB chunks
            if not chunk:
                break
            temp_file.write(chunk)
        
        temp_file.close()
        
        # 创建任务
        task_id = db.create_task(
            file_name=file.filename,
            file_path=temp_file.name,
            backend=backend,
            options={
                'lang': lang,
                'method': method,
                'formula_enable': formula_enable,
                'table_enable': table_enable,
            },
            priority=priority
        )
        
        logger.info(f"✅ Task submitted: {task_id} - {file.filename} (priority: {priority})")
        
        return {
            'success': True,
            'task_id': task_id,
            'status': 'pending',
            'message': 'Task submitted successfully',
            'file_name': file.filename,
            'created_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}/data")
async def get_task_data(
    task_id: str,
    include_fields: str = Query(
        "md,content_list,middle_json,model_output,images",
        description="需要返回的字段，逗号分隔：md,content_list,middle_json,model_output,images,layout_pdf,span_pdf,origin_pdf"
    ),
    upload_images: bool = Query(False, description="是否上传图片到MinIO并返回URL"),
    include_metadata: bool = Query(True, description="是否包含文件元数据")
):
    """
    按需获取任务的解析数据

    支持灵活获取 MinerU 解析后的数据，包括：
    - Markdown 内容
    - Content List JSON（结构化内容列表）
    - Middle JSON（中间处理结果）
    - Model Output JSON（模型原始输出）
    - 图片列表
    - 其他辅助文件（layout PDF、span PDF、origin PDF）

    通过 include_fields 参数按需选择需要返回的字段
    """
    # 获取任务信息
    task = db.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 构建基础响应
    response = {
        'success': True,
        'task_id': task_id,
        'status': task['status'],
        'file_name': task['file_name'],
        'backend': task['backend'],
        'created_at': task['created_at'],
        'completed_at': task['completed_at']
    }

    # 如果任务未完成，直接返回状态
    if task['status'] != 'completed':
        response['message'] = f"Task is in {task['status']} status, data not available yet"
        return response

    # 检查结果路径
    if not task['result_path']:
        response['message'] = 'Task completed but result files have been cleaned up (older than retention period)'
        return response

    result_dir = Path(task['result_path'])
    if not result_dir.exists():
        response['message'] = 'Result directory does not exist'
        return response

    # 解析需要返回的字段
    fields = [f.strip() for f in include_fields.split(',')]

    # 初始化 data 字段
    response['data'] = {}  # type: ignore

    logger.info(f"📦 Getting complete data for task {task_id}, fields: {fields}")

    # 查找文件（递归搜索，MinerU 输出结构：task_id/filename/auto/*.md）
    try:
        # 1. 处理 Markdown 文件
        if 'md' in fields:
            md_files = list(result_dir.rglob('*.md'))
            # 排除带特殊后缀的 md 文件
            md_files = [f for f in md_files if not any(f.stem.endswith(suffix) for suffix in ['_layout', '_span', '_origin'])] 

            if md_files:
                md_file = md_files[0]
                logger.info(f"📄 Reading markdown file: {md_file}")

                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()

                # 处理图片（如果需要上传）
                image_dir = md_file.parent / 'images'
                if upload_images and image_dir.exists():
                    md_content = process_markdown_images(md_content, image_dir, upload_images)

                response['data']['markdown'] = {
                    'content': md_content,
                    'file_name': md_file.name
                }

                if include_metadata:
                    metadata = get_file_metadata(md_file)
                    if metadata:
                        response['data']['markdown']['metadata'] = metadata

        # 2. 处理 Content List JSON
        if 'content_list' in fields:
            content_list_files = list(result_dir.rglob('*_content_list.json'))
            if content_list_files:
                content_list_file = content_list_files[0]
                logger.info(f"📄 Reading content list file: {content_list_file}")

                content_data = read_json_file(content_list_file)
                if content_data is not None:
                    response['data']['content_list'] = {
                        'content': content_data,
                        'file_name': content_list_file.name
                    }

                    if include_metadata:
                        metadata = get_file_metadata(content_list_file)
                        if metadata:
                            response['data']['content_list']['metadata'] = metadata

        # 3. 处理 Middle JSON
        if 'middle_json' in fields:
            middle_json_files = list(result_dir.rglob('*_middle.json'))
            if middle_json_files:
                middle_json_file = middle_json_files[0]
                logger.info(f"📄 Reading middle json file: {middle_json_file}")

                middle_data = read_json_file(middle_json_file)
                if middle_data is not None:
                    response['data']['middle_json'] = {
                        'content': middle_data,
                        'file_name': middle_json_file.name
                    }

                    if include_metadata:
                        metadata = get_file_metadata(middle_json_file)
                        if metadata:
                            response['data']['middle_json']['metadata'] = metadata

        # 4. 处理 Model Output JSON
        if 'model_output' in fields:
            model_output_files = list(result_dir.rglob('*_model.json'))
            if model_output_files:
                model_output_file = model_output_files[0]
                logger.info(f"📄 Reading model output file: {model_output_file}")

                model_data = read_json_file(model_output_file)
                if model_data is not None:
                    response['data']['model_output'] = {
                        'content': model_data,
                        'file_name': model_output_file.name
                    }

                    if include_metadata:
                        metadata = get_file_metadata(model_output_file)
                        if metadata:
                            response['data']['model_output']['metadata'] = metadata

        # 5. 处理图片
        if 'images' in fields:
            image_dirs = list(result_dir.rglob('images'))
            if image_dirs:
                image_dir = image_dirs[0]
                logger.info(f"🖼️  Getting images info from: {image_dir}")

                images_info = get_images_info(image_dir, upload_images)
                response['data']['images'] = images_info

        # 6. 处理 Layout PDF
        if 'layout_pdf' in fields:
            layout_pdf_files = list(result_dir.rglob('*_layout.pdf'))
            if layout_pdf_files:
                layout_pdf_file = layout_pdf_files[0]
                response['data']['layout_pdf'] = {
                    'file_name': layout_pdf_file.name,
                    'path': str(layout_pdf_file.relative_to(result_dir))
                }

                if include_metadata:
                    metadata = get_file_metadata(layout_pdf_file)
                    if metadata:
                        response['data']['layout_pdf']['metadata'] = metadata

        # 7. 处理 Span PDF
        if 'span_pdf' in fields:
            span_pdf_files = list(result_dir.rglob('*_span.pdf'))
            if span_pdf_files:
                span_pdf_file = span_pdf_files[0]
                response['data']['span_pdf'] = {
                    'file_name': span_pdf_file.name,
                    'path': str(span_pdf_file.relative_to(result_dir))
                }

                if include_metadata:
                    metadata = get_file_metadata(span_pdf_file)
                    if metadata:
                        response['data']['span_pdf']['metadata'] = metadata

        # 8. 处理 Origin PDF
        if 'origin_pdf' in fields:
            origin_pdf_files = list(result_dir.rglob('*_origin.pdf'))
            if origin_pdf_files:
                origin_pdf_file = origin_pdf_files[0]
                response['data']['origin_pdf'] = {
                    'file_name': origin_pdf_file.name,
                    'path': str(origin_pdf_file.relative_to(result_dir))
                }

                if include_metadata:
                    metadata = get_file_metadata(origin_pdf_file)
                    if metadata:
                        response['data']['origin_pdf']['metadata'] = metadata

        logger.info(f"✅ Complete data retrieved successfully for task {task_id}")

    except Exception as e:
        logger.error(f"❌ Failed to get complete data for task {task_id}: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")  

    return response


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    upload_images: bool = Query(False, description="是否上传图片到MinIO并替换链接（仅当任务完成时有效）")
):
    """
    查询任务状态和详情
    
    当任务完成时，会自动返回解析后的 Markdown 内容（data 字段）
    可选择是否上传图片到 MinIO 并替换为 URL
    """
    task = db.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    response = {
        'success': True,
        'task_id': task_id,
        'status': task['status'],
        'file_name': task['file_name'],
        'backend': task['backend'],
        'priority': task['priority'],
        'error_message': task['error_message'],
        'created_at': task['created_at'],
        'started_at': task['started_at'],
        'completed_at': task['completed_at'],
        'worker_id': task['worker_id'],
        'retry_count': task['retry_count']
    }
    logger.info(f"✅ Task status: {task['status']} - (result_path: {task['result_path']})")
    
    # 如果任务已完成，尝试返回解析内容
    if task['status'] == 'completed':
        if not task['result_path']:
            # 结果文件已被清理
            response['data'] = None
            response['message'] = 'Task completed but result files have been cleaned up (older than retention period)'
            return response
        
        result_dir = Path(task['result_path'])
        logger.info(f"📂 Checking result directory: {result_dir}")
        
        if result_dir.exists():
            logger.info(f"✅ Result directory exists")
            # 递归查找 Markdown 文件（MinerU 输出结构：task_id/filename/auto/*.md）
            md_files = list(result_dir.rglob('*.md'))
            logger.info(f"📄 Found {len(md_files)} markdown files: {[f.relative_to(result_dir) for f in md_files]}")
            
            if md_files:
                try:
                    # 读取 Markdown 内容
                    md_file = md_files[0]
                    logger.info(f"📖 Reading markdown file: {md_file}")
                    with open(md_file, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    
                    logger.info(f"✅ Markdown content loaded, length: {len(md_content)} characters")
                    
                    # 查找图片目录（在 markdown 文件的同级目录下）
                    image_dir = md_file.parent / 'images'
                    
                    # 处理图片（如果需要）
                    if upload_images and image_dir.exists():
                        logger.info(f"🖼️  Processing images for task {task_id}, upload_images={upload_images}")
                        md_content = process_markdown_images(md_content, image_dir, upload_images)
                    
                    # 添加 data 字段
                    response['data'] = {
                        'markdown_file': md_file.name,
                        'content': md_content,
                        'images_uploaded': upload_images,
                        'has_images': image_dir.exists() if not upload_images else None
                    }
                    logger.info(f"✅ Response data field added successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to read markdown content: {e}")
                    logger.exception(e)
                    # 读取失败不影响状态查询，只是不返回 data
                    response['data'] = None
            else:
                logger.warning(f"⚠️  No markdown files found in {result_dir}")
        else:
            logger.error(f"❌ Result directory does not exist: {result_dir}")
    elif task['status'] == 'completed':
        logger.warning(f"⚠️  Task completed but result_path is empty")
    else:
        logger.info(f"ℹ️  Task status is {task['status']}, skipping content loading")
    
    return response


@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    取消任务（仅限 pending 状态）
    """
    task = db.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task['status'] == 'pending':
        db.update_task_status(task_id, 'cancelled')
        
        # 删除临时文件
        file_path = Path(task['file_path'])
        if file_path.exists():
            file_path.unlink()
        
        logger.info(f"⏹️  Task cancelled: {task_id}")
        return {
            'success': True,
            'message': 'Task cancelled successfully'
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel task in {task['status']} status"
        )


@app.get("/api/v1/queue/stats")
async def get_queue_stats():
    """
    获取队列统计信息
    """
    stats = db.get_queue_stats()
    
    return {
        'success': True,
        'stats': stats,
        'total': sum(stats.values()),
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/v1/queue/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="筛选状态: pending/processing/completed/failed"),
    limit: int = Query(100, description="返回数量限制", le=1000)
):
    """
    获取任务列表
    """
    if status:
        tasks = db.get_tasks_by_status(status, limit)
    else:
        # 返回所有任务（需要修改 TaskDB 添加这个方法）
        with db.get_cursor() as cursor:
            cursor.execute('''
                SELECT * FROM tasks 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            tasks = [dict(row) for row in cursor.fetchall()]
    
    return {
        'success': True,
        'count': len(tasks),
        'tasks': tasks
    }


@app.post("/api/v1/admin/cleanup")
async def cleanup_old_tasks(days: int = Query(7, description="清理N天前的任务")):
    """
    清理旧任务记录（管理接口）
    """
    deleted_count = db.cleanup_old_tasks(days)
    
    logger.info(f"🧹 Cleaned up {deleted_count} old tasks")
    
    return {
        'success': True,
        'deleted_count': deleted_count,
        'message': f'Cleaned up tasks older than {days} days'
    }


@app.post("/api/v1/admin/reset-stale")
async def reset_stale_tasks(timeout_minutes: int = Query(60, description="超时时间（分钟）")):
    """
    重置超时的 processing 任务（管理接口）
    """
    reset_count = db.reset_stale_tasks(timeout_minutes)
    
    logger.info(f"🔄 Reset {reset_count} stale tasks")
    
    return {
        'success': True,
        'reset_count': reset_count,
        'message': f'Reset tasks processing for more than {timeout_minutes} minutes'
    }


@app.get("/api/v1/health")
async def health_check():
    """
    健康检查接口
    """
    try:
        # 检查数据库连接
        stats = db.get_queue_stats()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'queue_stats': stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e)
            }
        )


if __name__ == '__main__':
    # 从环境变量读取端口，默认为8000
    api_port = int(os.getenv('API_PORT', '8000'))
    
    display_port = os.getenv('HOST_API_PORT', api_port)
    logger.info("🚀 Starting MinerU Tianshu API Server...")
    logger.info(f"📖 API Documentation: http://localhost:{display_port}/docs")
    
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=api_port,
        log_level='info'
    )

