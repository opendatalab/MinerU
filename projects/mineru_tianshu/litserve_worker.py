"""
MinerU Tianshu - LitServe Worker
天枢 LitServe Worker

使用 LitServe 实现 GPU 资源的自动负载均衡
从 SQLite 队列拉取任务并处理
"""
import os
import json
import sys
from pathlib import Path
import litserve as ls
from loguru import logger
from typing import Optional

# 添加父目录到路径以导入 MinerU
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from task_db import TaskDB
from mineru.cli.common import do_parse, read_fn
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram

# 尝试导入 markitdown
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logger.warning("⚠️  markitdown not available, Office format parsing will be disabled")


class MinerUWorkerAPI(ls.LitAPI):
    """
    LitServe API Worker
    
    从 SQLite 队列拉取任务，利用 LitServe 的自动 GPU 负载均衡
    支持两种解析方式：
    - PDF/图片 -> MinerU 解析（GPU 加速）
    - 其他所有格式 -> MarkItDown 解析（快速处理）
    """
    
    # 支持的文件格式定义
    # MinerU 专用格式：PDF 和图片
    PDF_IMAGE_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    # 其他所有格式都使用 MarkItDown 解析
    
    def __init__(self, output_dir='/tmp/mineru_tianshu_output', worker_id_prefix='tianshu'):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id_prefix = worker_id_prefix
        self.db = TaskDB()
        self.worker_id = None
        self.markitdown = None
    
    def setup(self, device):
        """
        初始化环境（每个 worker 进程调用一次）
        
        Args:
            device: LitServe 分配的设备 (cuda:0, cuda:1, etc.)
        """
        # 生成唯一的 worker_id
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"{self.worker_id_prefix}-{hostname}-{device}-{pid}"
        
        logger.info(f"⚙️  Worker {self.worker_id} setting up on device: {device}")
        
        # 配置 MinerU 环境
        if os.getenv('MINERU_DEVICE_MODE', None) is None:
            os.environ['MINERU_DEVICE_MODE'] = device if device != 'auto' else get_device()
        
        device_mode = os.environ['MINERU_DEVICE_MODE']
        
        # 配置显存
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) is None:
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                try:
                    vram = round(get_vram(device_mode))
                    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
                except:
                    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '8'  # 默认值
            else:
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '1'
        
        # 初始化 MarkItDown（如果可用）
        if MARKITDOWN_AVAILABLE:
            self.markitdown = MarkItDown()
            logger.info(f"✅ MarkItDown initialized for Office format parsing")
        
        logger.info(f"✅ Worker {self.worker_id} ready")
        logger.info(f"   Device: {device_mode}")
        logger.info(f"   VRAM: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}GB")
    
    def decode_request(self, request):
        """
        解码请求
        
        接收一个 'poll' 信号来触发从数据库拉取任务
        """
        return request.get('action', 'poll')
    
    def _get_file_type(self, file_path: str) -> str:
        """
        判断文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            'pdf_image': PDF 或图片格式，使用 MinerU 解析
            'markitdown': 其他所有格式，使用 markitdown 解析
        """
        suffix = Path(file_path).suffix.lower()
        
        if suffix in self.PDF_IMAGE_FORMATS:
            return 'pdf_image'
        else:
            # 所有非 PDF/图片格式都使用 markitdown
            return 'markitdown'
    
    def _parse_with_mineru(self, file_path: Path, file_name: str, task_id: str, 
                           backend: str, options: dict, output_path: Path):
        """
        使用 MinerU 解析 PDF 和图片格式
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            task_id: 任务ID
            backend: 后端类型
            options: 解析选项
            output_path: 输出路径
        """
        logger.info(f"📄 Using MinerU to parse: {file_name}")
        
        # 读取文件
        pdf_bytes = read_fn(file_path)
        
        # 执行解析
        do_parse(
            output_dir=str(output_path),
            pdf_file_names=[Path(file_name).stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[options.get('lang', 'ch')],
            backend=backend,
            parse_method=options.get('method', 'auto'),
            formula_enable=options.get('formula_enable', True),
            table_enable=options.get('table_enable', True),
        )
    
    def _parse_with_markitdown(self, file_path: Path, file_name: str, 
                               output_path: Path):
        """
        使用 markitdown 解析文档（支持 Office、HTML、文本等多种格式）
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            output_path: 输出路径
        """
        if not MARKITDOWN_AVAILABLE or self.markitdown is None:
            raise RuntimeError("markitdown is not available. Please install it: pip install markitdown")
        
        logger.info(f"📊 Using MarkItDown to parse: {file_name}")
        
        # 使用 markitdown 转换文档
        result = self.markitdown.convert(str(file_path))
        
        # 保存为 markdown 文件
        output_file = output_path / f"{Path(file_name).stem}.md"
        output_file.write_text(result.text_content, encoding='utf-8')
        
        logger.info(f"📝 Markdown saved to: {output_file}")
    
    def predict(self, action):
        """
        从数据库拉取任务并处理
        
        这里是实际的任务处理逻辑，LitServe 会自动管理 GPU 负载均衡
        支持根据文件类型选择不同的解析器：
        - PDF/图片 -> MinerU（GPU 加速）
        - 其他所有格式 -> MarkItDown（快速处理）
        """
        if action != 'poll':
            return {
                'status': 'error', 
                'message': 'Invalid action. Use {"action": "poll"} to trigger task processing.'
            }
        
        # 从数据库获取任务
        task = self.db.get_next_task(self.worker_id)
        
        if not task:
            # 没有任务时返回空闲状态
            return {
                'status': 'idle', 
                'message': 'No pending tasks in queue',
                'worker_id': self.worker_id
            }
        
        # 提取任务信息
        task_id = task['task_id']
        file_path = task['file_path']
        file_name = task['file_name']
        backend = task['backend']
        options = json.loads(task['options'])
        
        logger.info(f"🔄 Worker {self.worker_id} processing task {task_id}: {file_name}")
        
        try:
            # 准备输出目录
            output_path = self.output_dir / task_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 判断文件类型并选择解析方式
            file_type = self._get_file_type(file_path)
            
            if file_type == 'pdf_image':
                # 使用 MinerU 解析 PDF 和图片
                self._parse_with_mineru(
                    file_path=Path(file_path),
                    file_name=file_name,
                    task_id=task_id,
                    backend=backend,
                    options=options,
                    output_path=output_path
                )
                parse_method = 'MinerU'
                
            else:  # file_type == 'markitdown'
                # 使用 markitdown 解析所有其他格式
                self._parse_with_markitdown(
                    file_path=Path(file_path),
                    file_name=file_name,
                    output_path=output_path
                )
                parse_method = 'MarkItDown'
            
            # 更新状态为成功
            self.db.update_task_status(task_id, 'completed', str(output_path))
            
            logger.info(f"✅ Task {task_id} completed by {self.worker_id}")
            logger.info(f"   Parser: {parse_method}")
            logger.info(f"   Output: {output_path}")
            
            return {
                'status': 'completed',
                'task_id': task_id,
                'file_name': file_name,
                'parse_method': parse_method,
                'file_type': file_type,
                'output_path': str(output_path),
                'worker_id': self.worker_id
            }
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")
            self.db.update_task_status(task_id, 'failed', error_message=str(e))
            
            return {
                'status': 'failed',
                'task_id': task_id,
                'error': str(e),
                'worker_id': self.worker_id
            }
        
        finally:
            # 清理临时文件
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")
    
    def encode_response(self, response):
        """编码响应"""
        return response


def start_litserve_workers(
    output_dir='/tmp/mineru_tianshu_output',
    accelerator='auto',
    devices='auto',
    workers_per_device=1,
    port=9000
):
    """
    启动 LitServe Worker Pool
    
    Args:
        output_dir: 输出目录
        accelerator: 加速器类型 (auto/cuda/cpu/mps)
        devices: 使用的设备 (auto/[0,1,2])
        workers_per_device: 每个 GPU 的 worker 数量
        port: 服务端口
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting MinerU Tianshu LitServe Worker Pool")
    logger.info("=" * 60)
    logger.info(f"📂 Output Directory: {output_dir}")
    logger.info(f"🎮 Accelerator: {accelerator}")
    logger.info(f"💾 Devices: {devices}")
    logger.info(f"👷 Workers per Device: {workers_per_device}")
    logger.info(f"🔌 Port: {port}")
    logger.info("=" * 60)
    
    # 创建 LitServe 服务器
    api = MinerUWorkerAPI(output_dir=output_dir)
    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=False,  # 不设置超时
    )
    
    logger.info(f"✅ LitServe worker pool initialized")
    logger.info(f"📡 Listening on: http://0.0.0.0:{port}/predict")
    logger.info(f"🔄 Workers will poll SQLite queue for tasks")
    logger.info("=" * 60)
    
    # 启动服务器
    server.run(port=port, generate_client_file=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MinerU Tianshu LitServe Worker Pool')
    parser.add_argument('--output-dir', type=str, default='/tmp/mineru_tianshu_output',
                       help='Output directory for processed files')
    parser.add_argument('--accelerator', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Accelerator type')
    parser.add_argument('--devices', type=str, default='auto',
                       help='Devices to use (auto or comma-separated list like 0,1,2)')
    parser.add_argument('--workers-per-device', type=int, default=1,
                       help='Number of workers per device')
    parser.add_argument('--port', type=int, default=9000,
                       help='Server port')
    
    args = parser.parse_args()
    
    # 处理 devices 参数
    devices = args.devices
    if devices != 'auto':
        try:
            devices = [int(d) for d in devices.split(',')]
        except:
            logger.warning(f"Invalid devices format: {devices}, using 'auto'")
            devices = 'auto'
    
    start_litserve_workers(
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=devices,
        workers_per_device=args.workers_per_device,
        port=args.port
    )

