"""
MinerU Tianshu - SQLite Task Database Manager
天枢任务数据库管理器

负责任务的持久化存储、状态管理和原子性操作
"""
import sqlite3
import json
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict
from pathlib import Path


class TaskDB:
    """任务数据库管理类"""
    
    def __init__(self, db_path='mineru_tianshu.db'):
        self.db_path = db_path
        self._init_db()
    
    def _get_conn(self):
        """获取数据库连接（每次创建新连接，避免 pickle 问题）
        
        并发安全说明：
            - 使用 check_same_thread=False 是安全的，因为：
              1. 每次调用都创建新连接，不跨线程共享
              2. 连接使用完立即关闭（在 get_cursor 上下文管理器中）
              3. 不使用连接池，避免线程间共享同一连接
            - timeout=30.0 防止死锁，如果锁等待超过30秒会抛出异常
        """
        conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def get_cursor(self):
        """上下文管理器，自动提交和错误处理"""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()  # 关闭连接
    
    def _init_db(self):
        """初始化数据库表"""
        with self.get_cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    backend TEXT DEFAULT 'pipeline',
                    options TEXT,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            ''')
            
            # 创建索引加速查询
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_worker_id ON tasks(worker_id)')
    
    def create_task(self, file_name: str, file_path: str, 
                   backend: str = 'pipeline', options: dict = None,
                   priority: int = 0) -> str:
        """
        创建新任务
        
        Args:
            file_name: 文件名
            file_path: 文件路径
            backend: 处理后端 (pipeline/vlm-transformers/vlm-vllm-engine)
            options: 处理选项 (dict)
            priority: 优先级，数字越大越优先
            
        Returns:
            task_id: 任务ID
        """
        task_id = str(uuid.uuid4())
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO tasks (task_id, file_name, file_path, backend, options, priority)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (task_id, file_name, file_path, backend, json.dumps(options or {}), priority))
        return task_id
    
    def get_next_task(self, worker_id: str, max_retries: int = 3) -> Optional[Dict]:
        """
        获取下一个待处理任务（原子操作，防止并发冲突）
        
        Args:
            worker_id: Worker ID
            max_retries: 当任务被其他 worker 抢走时的最大重试次数（默认3次）
            
        Returns:
            task: 任务字典，如果没有任务返回 None
            
        并发安全说明：
            1. 使用 BEGIN IMMEDIATE 立即获取写锁
            2. UPDATE 时检查 status = 'pending' 防止重复拉取
            3. 检查 rowcount 确保更新成功
            4. 如果任务被抢走，立即重试而不是返回 None（避免不必要的等待）
        """
        for attempt in range(max_retries):
            with self.get_cursor() as cursor:
                # 使用事务确保原子性
                cursor.execute('BEGIN IMMEDIATE')
                
                # 按优先级和创建时间获取任务
                cursor.execute('''
                    SELECT * FROM tasks 
                    WHERE status = 'pending' 
                    ORDER BY priority DESC, created_at ASC 
                    LIMIT 1
                ''')
                
                task = cursor.fetchone()
                if task:
                    # 立即标记为 processing，并确保状态仍是 pending
                    cursor.execute('''
                        UPDATE tasks 
                        SET status = 'processing', 
                            started_at = CURRENT_TIMESTAMP, 
                            worker_id = ?
                        WHERE task_id = ? AND status = 'pending'
                    ''', (worker_id, task['task_id']))
                    
                    # 检查是否更新成功（防止被其他 worker 抢走）
                    if cursor.rowcount == 0:
                        # 任务被其他进程抢走了，立即重试
                        # 因为队列中可能还有其他待处理任务
                        continue
                    
                    return dict(task)
                else:
                    # 队列中没有待处理任务，返回 None
                    return None
            
        # 重试次数用尽，仍未获取到任务（高并发场景）
        return None
    
    def _build_update_clauses(self, status: str, result_path: str = None, 
                             error_message: str = None, worker_id: str = None, 
                             task_id: str = None):
        """
        构建 UPDATE 和 WHERE 子句的辅助方法
        
        Args:
            status: 新状态
            result_path: 结果路径（可选）
            error_message: 错误信息（可选）
            worker_id: Worker ID（可选）
            task_id: 任务ID（可选）
            
        Returns:
            tuple: (update_clauses, update_params, where_clauses, where_params)
        """
        update_clauses = ['status = ?']
        update_params = [status]
        where_clauses = []
        where_params = []
        
        # 添加 task_id 条件（如果提供）
        if task_id:
            where_clauses.append('task_id = ?')
            where_params.append(task_id)
        
        # 处理 completed 状态
        if status == 'completed':
            update_clauses.append('completed_at = CURRENT_TIMESTAMP')
            if result_path:
                update_clauses.append('result_path = ?')
                update_params.append(result_path)
            # 只更新正在处理的任务
            where_clauses.append("status = 'processing'")
            if worker_id:
                where_clauses.append('worker_id = ?')
                where_params.append(worker_id)
        
        # 处理 failed 状态
        elif status == 'failed':
            update_clauses.append('completed_at = CURRENT_TIMESTAMP')
            if error_message:
                update_clauses.append('error_message = ?')
                update_params.append(error_message)
            # 只更新正在处理的任务
            where_clauses.append("status = 'processing'")
            if worker_id:
                where_clauses.append('worker_id = ?')
                where_params.append(worker_id)
        
        return update_clauses, update_params, where_clauses, where_params
    
    def update_task_status(self, task_id: str, status: str, 
                          result_path: str = None, error_message: str = None,
                          worker_id: str = None):
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态 (pending/processing/completed/failed/cancelled)
            result_path: 结果路径（可选）
            error_message: 错误信息（可选）
            worker_id: Worker ID（可选，用于并发检查）
            
        Returns:
            bool: 更新是否成功
            
        并发安全说明：
            1. 更新为 completed/failed 时会检查状态是 processing
            2. 如果提供 worker_id，会检查任务是否属于该 worker
            3. 返回 False 表示任务被其他进程修改了
        """
        with self.get_cursor() as cursor:
            # 使用辅助方法构建 UPDATE 和 WHERE 子句
            update_clauses, update_params, where_clauses, where_params = \
                self._build_update_clauses(status, result_path, error_message, worker_id, task_id)
            
            # 合并参数：先 UPDATE 部分，再 WHERE 部分
            all_params = update_params + where_params
            
            sql = f'''
                UPDATE tasks 
                SET {', '.join(update_clauses)}
                WHERE {' AND '.join(where_clauses)}
            '''
            
            cursor.execute(sql, all_params)
            
            # 检查更新是否成功
            success = cursor.rowcount > 0
            
            # 调试日志（仅在失败时）
            if not success and status in ['completed', 'failed']:
                from loguru import logger
                logger.debug(
                    f"Status update failed: task_id={task_id}, status={status}, "
                    f"worker_id={worker_id}, SQL: {sql}, params: {all_params}"
                )
            
            return success
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        查询任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            task: 任务字典，如果不存在返回 None
        """
        with self.get_cursor() as cursor:
            cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
            task = cursor.fetchone()
            return dict(task) if task else None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        获取队列统计信息
        
        Returns:
            stats: 各状态的任务数量
        """
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM tasks 
                GROUP BY status
            ''')
            stats = {row['status']: row['count'] for row in cursor.fetchall()}
            return stats
    
    def get_tasks_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """
        根据状态获取任务列表
        
        Args:
            status: 任务状态
            limit: 返回数量限制
            
        Returns:
            tasks: 任务列表
        """
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT * FROM tasks 
                WHERE status = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (status, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_task_files(self, days: int = 7):
        """
        清理旧任务的结果文件（保留数据库记录）
        
        Args:
            days: 清理多少天前的任务文件
            
        Returns:
            int: 删除的文件目录数
            
        注意：
            - 只删除结果文件，保留数据库记录
            - 数据库中的 result_path 字段会被清空
            - 用户仍可查询任务状态和历史记录
        """
        from pathlib import Path
        import shutil
        
        with self.get_cursor() as cursor:
            # 查询要清理文件的任务
            cursor.execute('''
                SELECT task_id, result_path FROM tasks 
                WHERE completed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('completed', 'failed')
                AND result_path IS NOT NULL
            ''', (days,))
            
            old_tasks = cursor.fetchall()
            file_count = 0
            
            # 删除结果文件
            for task in old_tasks:
                if task['result_path']:
                    result_path = Path(task['result_path'])
                    if result_path.exists() and result_path.is_dir():
                        try:
                            shutil.rmtree(result_path)
                            file_count += 1
                            
                            # 清空数据库中的 result_path，表示文件已被清理
                            cursor.execute('''
                                UPDATE tasks 
                                SET result_path = NULL
                                WHERE task_id = ?
                            ''', (task['task_id'],))
                            
                        except Exception as e:
                            from loguru import logger
                            logger.warning(f"Failed to delete result files for task {task['task_id']}: {e}")
            
            return file_count
    
    def cleanup_old_task_records(self, days: int = 30):
        """
        清理极旧的任务记录（可选功能）
        
        Args:
            days: 删除多少天前的任务记录
            
        Returns:
            int: 删除的记录数
            
        注意：
            - 这个方法会永久删除数据库记录
            - 建议设置较长的保留期（如30-90天）
            - 一般情况下不需要调用此方法
        """
        with self.get_cursor() as cursor:
            cursor.execute('''
                DELETE FROM tasks 
                WHERE completed_at < datetime('now', '-' || ? || ' days')
                AND status IN ('completed', 'failed')
            ''', (days,))
            
            deleted_count = cursor.rowcount
            return deleted_count
    
    def reset_stale_tasks(self, timeout_minutes: int = 60):
        """
        重置超时的 processing 任务为 pending
        
        Args:
            timeout_minutes: 超时时间（分钟）
        """
        with self.get_cursor() as cursor:
            cursor.execute('''
                UPDATE tasks 
                SET status = 'pending',
                    worker_id = NULL,
                    retry_count = retry_count + 1
                WHERE status = 'processing' 
                AND started_at < datetime('now', '-' || ? || ' minutes')
            ''', (timeout_minutes,))
            reset_count = cursor.rowcount
            return reset_count


if __name__ == '__main__':
    # 测试代码
    db = TaskDB('test_tianshu.db')
    
    # 创建测试任务
    task_id = db.create_task(
        file_name='test.pdf',
        file_path='/tmp/test.pdf',
        backend='pipeline',
        options={'lang': 'ch', 'formula_enable': True},
        priority=1
    )
    print(f"Created task: {task_id}")
    
    # 查询任务
    task = db.get_task(task_id)
    print(f"Task details: {task}")
    
    # 获取统计
    stats = db.get_queue_stats()
    print(f"Queue stats: {stats}")
    
    # 清理测试数据库
    Path('test_tianshu.db').unlink(missing_ok=True)
    print("Test completed!")

