"""
MinerU Tianshu - Unified Startup Script
天枢统一启动脚本

一键启动所有服务：API Server + LitServe Workers + Task Scheduler
"""
import subprocess
import signal
import sys
import time
import os
from loguru import logger
from pathlib import Path
import argparse


class TianshuLauncher:
    """天枢服务启动器"""
    
    def __init__(
        self,
        output_dir='/tmp/mineru_tianshu_output',
        api_port=8000,
        worker_port=9000,
        workers_per_device=1,
        devices='auto',
        accelerator='auto'
    ):
        self.output_dir = output_dir
        self.api_port = api_port
        self.worker_port = worker_port
        self.workers_per_device = workers_per_device
        self.devices = devices
        self.accelerator = accelerator
        self.processes = []
    
    def start_services(self):
        """启动所有服务"""
        logger.info("=" * 70)
        logger.info("🚀 MinerU Tianshu - Starting All Services")
        logger.info("=" * 70)
        logger.info("天枢 - 企业级多GPU文档解析服务")
        logger.info("")
        
        try:
            # 1. 启动 API Server
            logger.info("📡 [1/3] Starting API Server...")
            env = os.environ.copy()
            env['API_PORT'] = str(self.api_port)
            api_proc = subprocess.Popen(
                [sys.executable, 'api_server.py'],
                cwd=Path(__file__).parent,
                env=env
            )
            self.processes.append(('API Server', api_proc))
            time.sleep(3)
            
            if api_proc.poll() is not None:
                logger.error("❌ API Server failed to start!")
                return False
            
            logger.info(f"   ✅ API Server started (PID: {api_proc.pid})")
            logger.info(f"   📖 API Docs: http://localhost:{self.api_port}/docs")
            logger.info("")
            
            # 2. 启动 LitServe Worker Pool
            logger.info("⚙️  [2/3] Starting LitServe Worker Pool...")
            worker_cmd = [
                sys.executable, 'litserve_worker.py',
                '--output-dir', self.output_dir,
                '--accelerator', self.accelerator,
                '--workers-per-device', str(self.workers_per_device),
                '--port', str(self.worker_port),
                '--devices', str(self.devices) if isinstance(self.devices, str) else ','.join(map(str, self.devices))
            ]
            
            worker_proc = subprocess.Popen(
                worker_cmd,
                cwd=Path(__file__).parent
            )
            self.processes.append(('LitServe Workers', worker_proc))
            time.sleep(5)
            
            if worker_proc.poll() is not None:
                logger.error("❌ LitServe Workers failed to start!")
                return False
            
            logger.info(f"   ✅ LitServe Workers started (PID: {worker_proc.pid})")
            logger.info(f"   🔌 Worker Port: {self.worker_port}")
            logger.info(f"   👷 Workers per Device: {self.workers_per_device}")
            logger.info("")
            
            # 3. 启动 Task Scheduler
            logger.info("🔄 [3/3] Starting Task Scheduler...")
            scheduler_cmd = [
                sys.executable, 'task_scheduler.py',
                '--litserve-url', f'http://localhost:{self.worker_port}/predict',
                '--wait-for-workers'
            ]
            
            scheduler_proc = subprocess.Popen(
                scheduler_cmd,
                cwd=Path(__file__).parent
            )
            self.processes.append(('Task Scheduler', scheduler_proc))
            time.sleep(3)
            
            if scheduler_proc.poll() is not None:
                logger.error("❌ Task Scheduler failed to start!")
                return False
            
            logger.info(f"   ✅ Task Scheduler started (PID: {scheduler_proc.pid})")
            logger.info("")
            
            # 启动成功
            logger.info("=" * 70)
            logger.info("✅ All Services Started Successfully!")
            logger.info("=" * 70)
            logger.info("")
            logger.info("📚 Quick Start:")
            logger.info(f"   • API Documentation: http://localhost:{self.api_port}/docs")
            logger.info(f"   • Submit Task:       POST http://localhost:{self.api_port}/api/v1/tasks/submit")
            logger.info(f"   • Query Status:      GET  http://localhost:{self.api_port}/api/v1/tasks/{{task_id}}")
            logger.info(f"   • Queue Stats:       GET  http://localhost:{self.api_port}/api/v1/queue/stats")
            logger.info("")
            logger.info("🔧 Service Details:")
            for name, proc in self.processes:
                logger.info(f"   • {name:20s} PID: {proc.pid}")
            logger.info("")
            logger.info("⚠️  Press Ctrl+C to stop all services")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start services: {e}")
            self.stop_services()
            return False
    
    def stop_services(self, signum=None, frame=None):
        """停止所有服务"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("⏹️  Stopping All Services...")
        logger.info("=" * 70)
        
        for name, proc in self.processes:
            if proc.poll() is None:  # 进程仍在运行
                logger.info(f"   Stopping {name} (PID: {proc.pid})...")
                proc.terminate()
        
        # 等待所有进程结束
        for name, proc in self.processes:
            try:
                proc.wait(timeout=10)
                logger.info(f"   ✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"   ⚠️  {name} did not stop gracefully, forcing...")
                proc.kill()
                proc.wait()
        
        logger.info("=" * 70)
        logger.info("✅ All Services Stopped")
        logger.info("=" * 70)
        sys.exit(0)
    
    def wait(self):
        """等待所有服务"""
        try:
            while True:
                time.sleep(1)
                
                # 检查进程状态
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        logger.error(f"❌ {name} unexpectedly stopped!")
                        self.stop_services()
                        return
                        
        except KeyboardInterrupt:
            self.stop_services()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MinerU Tianshu - 统一启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置启动（自动检测GPU）
  python start_all.py
  
  # 使用CPU模式
  python start_all.py --accelerator cpu
  
  # 指定输出目录和端口
  python start_all.py --output-dir /data/output --api-port 8080
  
  # 每个GPU启动2个worker
  python start_all.py --accelerator cuda --workers-per-device 2
  
  # 只使用指定的GPU
  python start_all.py --accelerator cuda --devices 0,1
        """
    )
    
    parser.add_argument('--output-dir', type=str, default='/tmp/mineru_tianshu_output',
                       help='输出目录 (默认: /tmp/mineru_tianshu_output)')
    parser.add_argument('--api-port', type=int, default=8000,
                       help='API服务器端口 (默认: 8000)')
    parser.add_argument('--worker-port', type=int, default=9000,
                       help='Worker服务器端口 (默认: 9000)')
    parser.add_argument('--accelerator', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='加速器类型 (默认: auto，自动检测)')
    parser.add_argument('--workers-per-device', type=int, default=1,
                       help='每个GPU的worker数量 (默认: 1)')
    parser.add_argument('--devices', type=str, default='auto',
                       help='使用的GPU设备，逗号分隔 (默认: auto，使用所有GPU)')
    
    args = parser.parse_args()
    
    # 处理 devices 参数
    devices = args.devices
    if devices != 'auto':
        try:
            devices = [int(d) for d in devices.split(',')]
        except:
            logger.warning(f"Invalid devices format: {devices}, using 'auto'")
            devices = 'auto'
    
    # 创建启动器
    launcher = TianshuLauncher(
        output_dir=args.output_dir,
        api_port=args.api_port,
        worker_port=args.worker_port,
        workers_per_device=args.workers_per_device,
        devices=devices,
        accelerator=args.accelerator
    )
    
    # 设置信号处理
    signal.signal(signal.SIGINT, launcher.stop_services)
    signal.signal(signal.SIGTERM, launcher.stop_services)
    
    # 启动服务
    if launcher.start_services():
        launcher.wait()
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

