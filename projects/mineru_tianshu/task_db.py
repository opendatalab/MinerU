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
        """获取数据库连接（每次创建新连接，避免 pickle 问题）"""
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
    
    def get_next_task(self, worker_id: str) -> Optional[Dict]:
        """
        获取下一个待处理任务（原子操作，防止并发冲突）
        
        Args:
            worker_id: Worker ID
            
        Returns:
            task: 任务字典，如果没有任务返回 None
        """
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
                # 立即标记为 processing
                cursor.execute('''
                    UPDATE tasks 
                    SET status = 'processing', 
                        started_at = CURRENT_TIMESTAMP, 
                        worker_id = ?
                    WHERE task_id = ?
                ''', (worker_id, task['task_id']))
                
                return dict(task)
            
            return None
    
    def update_task_status(self, task_id: str, status: str, 
                          result_path: str = None, error_message: str = None):
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态 (pending/processing/completed/failed/cancelled)
            result_path: 结果路径（可选）
            error_message: 错误信息（可选）
        """
        with self.get_cursor() as cursor:
            updates = ['status = ?']
            params = [status]
            
            if status == 'completed':
                updates.append('completed_at = CURRENT_TIMESTAMP')
                if result_path:
                    updates.append('result_path = ?')
                    params.append(result_path)
            
            if status == 'failed' and error_message:
                updates.append('error_message = ?')
                params.append(error_message)
                updates.append('completed_at = CURRENT_TIMESTAMP')
            
            params.append(task_id)
            cursor.execute(f'''
                UPDATE tasks SET {', '.join(updates)}
                WHERE task_id = ?
            ''', params)
    
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
    
    def cleanup_old_tasks(self, days: int = 7):
        """
        清理旧任务记录
        
        Args:
            days: 保留最近N天的任务
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

