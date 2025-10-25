"""
MinerU Tianshu - Client Example
天枢客户端示例

演示如何使用 Python 客户端提交任务和查询状态
"""
import asyncio
import aiohttp
from pathlib import Path
from loguru import logger
import time
from typing import Dict


class TianshuClient:
    """天枢客户端"""
    
    def __init__(self, api_url='http://localhost:8000'):
        self.api_url = api_url
        self.base_url = f"{api_url}/api/v1"
    
    async def submit_task(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        backend: str = 'pipeline',
        lang: str = 'ch',
        method: str = 'auto',
        formula_enable: bool = True,
        table_enable: bool = True,
        priority: int = 0
    ) -> Dict:
        """
        提交任务
        
        Args:
            session: aiohttp session
            file_path: 文件路径
            backend: 处理后端
            lang: 语言
            method: 解析方法
            formula_enable: 是否启用公式识别
            table_enable: 是否启用表格识别
            priority: 优先级
            
        Returns:
            响应字典，包含 task_id
        """
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=Path(file_path).name)
            data.add_field('backend', backend)
            data.add_field('lang', lang)
            data.add_field('method', method)
            data.add_field('formula_enable', str(formula_enable).lower())
            data.add_field('table_enable', str(table_enable).lower())
            data.add_field('priority', str(priority))
            
            async with session.post(f'{self.base_url}/tasks/submit', data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Submitted: {file_path} -> Task ID: {result['task_id']}")
                    return result
                else:
                    error = await resp.text()
                    logger.error(f"❌ Failed to submit {file_path}: {error}")
                    return {'success': False, 'error': error}
    
    async def get_task_status(self, session: aiohttp.ClientSession, task_id: str) -> Dict:
        """
        查询任务状态
        
        Args:
            session: aiohttp session
            task_id: 任务ID
            
        Returns:
            任务状态字典
        """
        async with session.get(f'{self.base_url}/tasks/{task_id}') as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return {'success': False, 'error': 'Task not found'}
    
    async def wait_for_task(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        timeout: int = 600,
        poll_interval: int = 2
    ) -> Dict:
        """
        等待任务完成
        
        Args:
            session: aiohttp session
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            最终任务状态
        """
        start_time = time.time()
        
        while True:
            status = await self.get_task_status(session, task_id)
            
            if not status.get('success'):
                logger.error(f"❌ Failed to get status for task {task_id}")
                return status
            
            task_status = status.get('status')
            
            if task_status == 'completed':
                logger.info(f"✅ Task {task_id} completed!")
                logger.info(f"   Output: {status.get('result_path')}")
                return status
            
            elif task_status == 'failed':
                logger.error(f"❌ Task {task_id} failed!")
                logger.error(f"   Error: {status.get('error_message')}")
                return status
            
            elif task_status == 'cancelled':
                logger.warning(f"⚠️  Task {task_id} was cancelled")
                return status
            
            # 检查超时
            if time.time() - start_time > timeout:
                logger.error(f"⏱️  Task {task_id} timeout after {timeout}s")
                return {'success': False, 'error': 'timeout'}
            
            # 等待后继续轮询
            await asyncio.sleep(poll_interval)
    
    async def get_queue_stats(self, session: aiohttp.ClientSession) -> Dict:
        """获取队列统计"""
        async with session.get(f'{self.base_url}/queue/stats') as resp:
            return await resp.json()
    
    async def cancel_task(self, session: aiohttp.ClientSession, task_id: str) -> Dict:
        """取消任务"""
        async with session.delete(f'{self.base_url}/tasks/{task_id}') as resp:
            return await resp.json()


async def example_single_task():
    """示例1：提交单个任务并等待完成"""
    logger.info("=" * 60)
    logger.info("示例1：提交单个任务")
    logger.info("=" * 60)
    
    client = TianshuClient()
    
    async with aiohttp.ClientSession() as session:
        # 提交任务
        result = await client.submit_task(
            session,
            file_path='../../demo/pdfs/demo1.pdf',
            backend='pipeline',
            lang='ch',
            formula_enable=True,
            table_enable=True
        )
        
        if result.get('success'):
            task_id = result['task_id']
            
            # 等待完成
            logger.info(f"⏳ Waiting for task {task_id} to complete...")
            final_status = await client.wait_for_task(session, task_id)
            
            return final_status


async def example_batch_tasks():
    """示例2：批量提交多个任务并并发等待"""
    logger.info("=" * 60)
    logger.info("示例2：批量提交多个任务")
    logger.info("=" * 60)
    
    client = TianshuClient()
    
    # 准备任务列表
    files = [
        '../../demo/pdfs/demo1.pdf',
        '../../demo/pdfs/demo2.pdf',
        '../../demo/pdfs/demo3.pdf',
    ]
    
    async with aiohttp.ClientSession() as session:
        # 并发提交所有任务
        logger.info(f"📤 Submitting {len(files)} tasks...")
        submit_tasks = [
            client.submit_task(session, file) 
            for file in files
        ]
        results = await asyncio.gather(*submit_tasks)
        
        # 提取 task_ids
        task_ids = [r['task_id'] for r in results if r.get('success')]
        logger.info(f"✅ Submitted {len(task_ids)} tasks successfully")
        
        # 并发等待所有任务完成
        logger.info(f"⏳ Waiting for all tasks to complete...")
        wait_tasks = [
            client.wait_for_task(session, task_id) 
            for task_id in task_ids
        ]
        final_results = await asyncio.gather(*wait_tasks)
        
        # 统计结果
        completed = sum(1 for r in final_results if r.get('status') == 'completed')
        failed = sum(1 for r in final_results if r.get('status') == 'failed')
        
        logger.info("=" * 60)
        logger.info(f"📊 Results: {completed} completed, {failed} failed")
        logger.info("=" * 60)
        
        return final_results


async def example_priority_tasks():
    """示例3：使用优先级队列"""
    logger.info("=" * 60)
    logger.info("示例3：优先级队列")
    logger.info("=" * 60)
    
    client = TianshuClient()
    
    async with aiohttp.ClientSession() as session:
        # 提交低优先级任务
        low_priority = await client.submit_task(
            session,
            file_path='../../demo/pdfs/demo1.pdf',
            priority=0
        )
        logger.info(f"📝 Low priority task: {low_priority['task_id']}")
        
        # 提交高优先级任务
        high_priority = await client.submit_task(
            session,
            file_path='../../demo/pdfs/demo2.pdf',
            priority=10
        )
        logger.info(f"🔥 High priority task: {high_priority['task_id']}")
        
        # 高优先级任务会先被处理
        logger.info("⏳ 高优先级任务将优先处理...")


async def example_queue_monitoring():
    """示例4：监控队列状态"""
    logger.info("=" * 60)
    logger.info("示例4：监控队列状态")
    logger.info("=" * 60)
    
    client = TianshuClient()
    
    async with aiohttp.ClientSession() as session:
        # 获取队列统计
        stats = await client.get_queue_stats(session)
        
        logger.info("📊 Queue Statistics:")
        logger.info(f"   Total: {stats.get('total', 0)}")
        for status, count in stats.get('stats', {}).items():
            logger.info(f"   {status:12s}: {count}")


async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        example = 'all'
    
    try:
        if example == 'single' or example == 'all':
            await example_single_task()
            print()
        
        if example == 'batch' or example == 'all':
            await example_batch_tasks()
            print()
        
        if example == 'priority' or example == 'all':
            await example_priority_tasks()
            print()
        
        if example == 'monitor' or example == 'all':
            await example_queue_monitoring()
            print()
            
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """
    使用方法:
    
    # 运行所有示例
    python client_example.py
    
    # 运行特定示例
    python client_example.py single
    python client_example.py batch
    python client_example.py priority
    python client_example.py monitor
    """
    asyncio.run(main())

