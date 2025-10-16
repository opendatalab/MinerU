"""
MinerU Tianshu - Task Scheduler
天枢任务调度器

定期检查任务队列，触发 LitServe Workers 拉取和处理任务
"""
import asyncio
import aiohttp
from loguru import logger
from task_db import TaskDB
import signal
import sys


class TaskScheduler:
    """
    任务调度器
    
    职责：
    1. 监控 SQLite 任务队列
    2. 当有待处理任务时，触发 LitServe Workers
    3. 管理调度策略（轮询间隔、并发控制等）
    """
    
    def __init__(
        self, 
        litserve_url='http://localhost:9000/predict', 
        poll_interval=2,
        max_concurrent_polls=10
    ):
        """
        初始化调度器
        
        Args:
            litserve_url: LitServe Worker 的 URL
            poll_interval: 轮询间隔（秒）
            max_concurrent_polls: 最大并发轮询数
        """
        self.litserve_url = litserve_url
        self.poll_interval = poll_interval
        self.max_concurrent_polls = max_concurrent_polls
        self.db = TaskDB()
        self.running = True
        self.active_polls = 0
    
    async def trigger_worker_poll(self, session: aiohttp.ClientSession):
        """
        触发一个 worker 拉取任务
        """
        self.active_polls += 1
        try:
            async with session.post(
                self.litserve_url,
                json={'action': 'poll'},
                timeout=aiohttp.ClientTimeout(total=600)  # 10分钟超时
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    if result.get('status') == 'completed':
                        logger.info(f"✅ Task completed: {result.get('task_id')} by {result.get('worker_id')}")
                    elif result.get('status') == 'failed':
                        logger.error(f"❌ Task failed: {result.get('task_id')} - {result.get('error')}")
                    elif result.get('status') == 'idle':
                        # Worker 空闲，没有任务
                        pass
                    
                    return result
                else:
                    logger.error(f"Worker poll failed with status {resp.status}")
                    
        except asyncio.TimeoutError:
            logger.warning("Worker poll timeout")
        except Exception as e:
            logger.error(f"Worker poll error: {e}")
        finally:
            self.active_polls -= 1
    
    async def schedule_loop(self):
        """
        主调度循环
        """
        logger.info("🔄 Task scheduler started")
        logger.info(f"   LitServe URL: {self.litserve_url}")
        logger.info(f"   Poll Interval: {self.poll_interval}s")
        logger.info(f"   Max Concurrent Polls: {self.max_concurrent_polls}")
        
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # 获取队列统计
                    stats = self.db.get_queue_stats()
                    pending_count = stats.get('pending', 0)
                    processing_count = stats.get('processing', 0)
                    
                    if pending_count > 0:
                        logger.info(f"📋 Queue status: {pending_count} pending, {processing_count} processing")
                        
                        # 计算需要触发的 worker 数量
                        # 考虑：待处理任务数、当前处理中的任务数、活跃的轮询数
                        needed_workers = min(
                            pending_count,  # 待处理任务数
                            self.max_concurrent_polls - self.active_polls  # 剩余并发数
                        )
                        
                        if needed_workers > 0:
                            # 并发触发多个 worker
                            tasks = [
                                self.trigger_worker_poll(session) 
                                for _ in range(needed_workers)
                            ]
                            await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 等待下一次轮询
                    await asyncio.sleep(self.poll_interval)
                    
                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")
                    await asyncio.sleep(self.poll_interval)
        
        logger.info("⏹️  Task scheduler stopped")
    
    def start(self):
        """启动调度器"""
        logger.info("🚀 Starting MinerU Tianshu Task Scheduler...")
        
        # 设置信号处理
        def signal_handler(sig, frame):
            logger.info("\n🛑 Received stop signal, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 运行调度循环
        asyncio.run(self.schedule_loop())
    
    def stop(self):
        """停止调度器"""
        self.running = False


async def health_check(litserve_url: str) -> bool:
    """
    健康检查：验证 LitServe Worker 是否可用
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                litserve_url.replace('/predict', '/health'),
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
    except:
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MinerU Tianshu Task Scheduler')
    parser.add_argument('--litserve-url', type=str, default='http://localhost:9000/predict',
                       help='LitServe worker URL')
    parser.add_argument('--poll-interval', type=int, default=2,
                       help='Poll interval in seconds')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent worker polls')
    parser.add_argument('--wait-for-workers', action='store_true',
                       help='Wait for workers to be ready before starting')
    
    args = parser.parse_args()
    
    # 等待 workers 就绪（可选）
    if args.wait_for_workers:
        logger.info("⏳ Waiting for LitServe workers to be ready...")
        import time
        max_retries = 30
        for i in range(max_retries):
            if asyncio.run(health_check(args.litserve_url)):
                logger.info("✅ LitServe workers are ready!")
                break
            time.sleep(2)
            if i == max_retries - 1:
                logger.error("❌ LitServe workers not responding, starting anyway...")
    
    # 创建并启动调度器
    scheduler = TaskScheduler(
        litserve_url=args.litserve_url,
        poll_interval=args.poll_interval,
        max_concurrent_polls=args.max_concurrent
    )
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("👋 Scheduler interrupted by user")

