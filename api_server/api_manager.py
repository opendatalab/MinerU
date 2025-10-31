#!/usr/bin/env python3
"""
MinerU API 高级管理工具
提供更复杂的API操作和批量处理功能
"""

import argparse
import json
import time
import os
import sys
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class MinerUAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()

    def submit_task(self, pdf_path: str, chunk_id: str = None) -> Dict:
        """提交PDF处理任务"""
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if chunk_id:
                data['chunk_id'] = chunk_id
            response = self.session.post(f"{self.base_url}/submit_task", files=files, data=data)
        response.raise_for_status()
        return response.json()

    def batch_submit(self, input_dir: str, chunk_id: str = None) -> Dict:
        """批次提交PDF处理任务"""
        data = {
            "input_dir": input_dir
        }
        if chunk_id:
            data["chunk_id"] = chunk_id

        response = self.session.post(f"{self.base_url}/batch_submit", json=data)
        response.raise_for_status()
        return response.json()

    def list_tasks_by_chunk(self, chunk_id: str) -> Dict:
        """按chunk_id列出任务"""
        response = self.session.get(f"{self.base_url}/list_tasks_by_chunk/{chunk_id}")
        response.raise_for_status()
        return response.json()

    def download_chunk_results(self, chunk_id: str, save_path: str) -> bool:
        """下载整个chunk的结果"""
        response = self.session.get(f"{self.base_url}/download_chunk_results/{chunk_id}")
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False

    def get_status(self, task_id: str) -> Dict:
        """获取任务状态"""
        response = self.session.get(f"{self.base_url}/get_status/{task_id}")
        response.raise_for_status()
        return response.json()

    def download_result(self, task_id: str, save_path: str) -> bool:
        """下载处理结果"""
        response = self.session.get(f"{self.base_url}/download_result/{task_id}")
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False

    def list_tasks(self) -> Dict:
        """列出所有任务"""
        response = self.session.get(f"{self.base_url}/list_tasks")
        response.raise_for_status()
        return response.json()

    def delete_task(self, task_id: str) -> Dict:
        """删除任务"""
        response = self.session.delete(f"{self.base_url}/delete_task/{task_id}")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

def submit_batch(client: MinerUAPIClient, pdf_files: List[str], delay: float = 1.0, chunk_id: str = None) -> List[Dict]:
    """批量提交任务"""
    results = []
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"文件不存在: {pdf_file}")
            continue

        print(f"提交任务: {pdf_file}")
        try:
            result = client.submit_task(pdf_file, chunk_id)
            results.append(result)
            print(f"  ✓ 任务ID: {result.get('task_id')}")
            if delay > 0:
                time.sleep(delay)
        except Exception as e:
            print(f"  ✗ 提交失败: {e}")

    return results

def monitor_tasks(client: MinerUAPIClient, task_ids: List[str], interval: int = 5, timeout: int = 1800) -> Dict:
    """监控任务进度"""
    start_time = time.time()
    completed = []
    failed = []

    print(f"监控 {len(task_ids)} 个任务...")

    while task_ids and (time.time() - start_time) < timeout:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 检查任务状态...")

        remaining_tasks = []
        for task_id in task_ids:
            try:
                status = client.get_status(task_id)
                task_status = status.get('status')

                if task_status == 'completed':
                    completed.append(task_id)
                    print(f"  ✓ {task_id}: 已完成")
                elif task_status == 'failed':
                    failed.append(task_id)
                    print(f"  ✗ {task_id}: 失败 - {status.get('error', 'Unknown error')}")
                else:
                    remaining_tasks.append(task_id)
                    print(f"  ⟳ {task_id}: {task_status}")
            except Exception as e:
                print(f"  ✗ {task_id}: 状态检查失败 - {e}")
                failed.append(task_id)

        task_ids = remaining_tasks

        if task_ids:
            print(f"等待 {interval} 秒...")
            time.sleep(interval)

    print(f"\n监控完成!")
    print(f"已完成: {len(completed)}")
    print(f"失败: {len(failed)}")
    print(f"超时/未完成: {len(task_ids)}")

    return {
        'completed': completed,
        'failed': failed,
        'timeout': task_ids
    }

def download_results(client: MinerUAPIClient, task_ids: List[str], output_dir: str = "downloads") -> Dict:
    """批量下载结果"""
    results = {'success': [], 'failed': []}

    os.makedirs(output_dir, exist_ok=True)

    for task_id in task_ids:
        output_path = os.path.join(output_dir, f"result_{task_id}.zip")
        print(f"下载 {task_id} 到 {output_path}")

        try:
            if client.download_result(task_id, output_path):
                file_size = os.path.getsize(output_path)
                results['success'].append({
                    'task_id': task_id,
                    'path': output_path,
                    'size': file_size
                })
                print(f"  ✓ 成功 ({file_size} bytes)")
            else:
                results['failed'].append({'task_id': task_id, 'reason': 'Download failed'})
                print(f"  ✗ 下载失败")
        except Exception as e:
            results['failed'].append({'task_id': task_id, 'reason': str(e)})
            print(f"  ✗ 下载出错: {e}")

    return results

def cleanup_tasks(client: MinerUAPIClient, older_than_hours: int = 24) -> Dict:
    """清理旧任务"""
    tasks = client.list_tasks().get('tasks', [])
    current_time = datetime.now()

    deleted = []
    failed = []

    for task in tasks:
        created_time = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
        age_hours = (current_time - created_time).total_seconds() / 3600

        if age_hours > older_than_hours:
            task_id = task['task_id']
            try:
                client.delete_task(task_id)
                deleted.append(task_id)
                print(f"删除任务: {task_id} (年龄: {age_hours:.1f}小时)")
            except Exception as e:
                failed.append({'task_id': task_id, 'reason': str(e)})
                print(f"删除失败: {task_id} - {e}")

    return {'deleted': deleted, 'failed': failed}

def generate_report(client: MinerUAPIClient, output_file: str = "api_report.json") -> Dict:
    """生成API使用报告"""
    tasks = client.list_tasks().get('tasks', [])
    health = client.health_check()

    # 统计数据
    total_tasks = len(tasks)
    status_counts = {}
    for task in tasks:
        status = task.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

    # 时间分析
    if tasks:
        earliest = min(task['created_at'] for task in tasks)
        latest = max(task['created_at'] for task in tasks)
    else:
        earliest = latest = None

    # 文件大小统计
    completed_tasks = [t for t in tasks if t.get('status') == 'completed' and t.get('progress', {}).get('file_size')]
    total_size = sum(t['progress']['file_size'] for t in completed_tasks)

    report = {
        'generated_at': datetime.now().isoformat(),
        'api_health': health,
        'statistics': {
            'total_tasks': total_tasks,
            'status_breakdown': status_counts,
            'completed_with_size': len(completed_tasks),
            'total_output_size_bytes': total_size,
            'average_size_mb': total_size / len(completed_tasks) / (1024*1024) if completed_tasks else 0
        },
        'time_period': {
            'earliest_task': earliest,
            'latest_task': latest
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def main():
    parser = argparse.ArgumentParser(description='MinerU API 高级管理工具')
    parser.add_argument('--url', default='http://localhost:8001', help='API服务器地址')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 服务器状态
    subparsers.add_parser('health', help='检查服务器健康状态')
    subparsers.add_parser('list', help='列出所有任务')

    # 批量提交
    batch_parser = subparsers.add_parser('batch', help='批量提交任务')
    batch_parser.add_argument('pdf_files', nargs='+', help='PDF文件路径')
    batch_parser.add_argument('--delay', type=float, default=1.0, help='提交间隔(秒)')
    batch_parser.add_argument('--chunk-id', help='chunk_id标识')

    # 批次提交目录
    batch_dir_parser = subparsers.add_parser('batch-dir', help='批量提交目录中的PDF')
    batch_dir_parser.add_argument('input_dir', help='输入目录路径')
    batch_dir_parser.add_argument('--chunk-id', help='chunk_id标识')

    # 按chunk查询
    chunk_list_parser = subparsers.add_parser('chunk-list', help='按chunk_id查询任务')
    chunk_list_parser.add_argument('chunk_id', help='chunk_id标识')

    # 下载chunk结果
    chunk_download_parser = subparsers.add_parser('chunk-download', help='下载chunk结果')
    chunk_download_parser.add_argument('chunk_id', help='chunk_id标识')
    chunk_download_parser.add_argument('--output', default='chunk_results.zip', help='输出文件路径')

    # 监控任务
    monitor_parser = subparsers.add_parser('monitor', help='监控任务进度')
    monitor_parser.add_argument('task_ids', nargs='*', help='任务ID列表(留空监控所有)')
    monitor_parser.add_argument('--interval', type=int, default=5, help='检查间隔(秒)')
    monitor_parser.add_argument('--timeout', type=int, default=1800, help='超时时间(秒)')

    # 下载结果
    download_parser = subparsers.add_parser('download', help='下载任务结果')
    download_parser.add_argument('task_ids', nargs='*', help='任务ID列表(留空下载所有完成的)')
    download_parser.add_argument('--output-dir', default='downloads', help='输出目录')

    # 清理任务
    cleanup_parser = subparsers.add_parser('cleanup', help='清理旧任务')
    cleanup_parser.add_argument('--older-than', type=int, default=24, help='清理多少小时前的任务')

    # 生成报告
    report_parser = subparsers.add_parser('report', help='生成使用报告')
    report_parser.add_argument('--output', default='api_report.json', help='报告文件路径')

    args = parser.parse_args()
    client = MinerUAPIClient(args.url)

    try:
        if args.command == 'health':
            health = client.health_check()
            print("服务器健康状态:")
            print(json.dumps(health, indent=2, ensure_ascii=False))

        elif args.command == 'list':
            tasks = client.list_tasks()
            print(f"任务列表 (共 {len(tasks['tasks'])} 个):")
            for task in tasks['tasks']:
                print(f"  {task['task_id']}: {task['status']}")

        elif args.command == 'batch':
            results = submit_batch(client, args.pdf_files, args.delay, args.chunk_id)
            print(f"\n提交完成: {len(results)}/{len(args.pdf_files)}")

        elif args.command == 'batch-dir':
            print(f"批量提交目录: {args.input_dir}")
            try:
                result = client.batch_submit(args.input_dir, args.chunk_id)
                print(f"✓ 批次提交成功")
                print(f"  chunk_id: {result.get('chunk_id')}")
                print(f"  任务数量: {result.get('successful_submissions', 0)}/{result.get('total_files', 0)}")
                print(f"  任务IDs: {', '.join(result.get('task_ids', [])[:3])}{'...' if len(result.get('task_ids', [])) > 3 else ''}")
            except Exception as e:
                print(f"✗ 批次提交失败: {e}")

        elif args.command == 'chunk-list':
            print(f"查询chunk: {args.chunk_id}")
            try:
                result = client.list_tasks_by_chunk(args.chunk_id)
                print(f"✓ 查询成功")
                print(f"  总任务数: {result.get('total_tasks', 0)}")
                breakdown = result.get('status_breakdown', {})
                print(f"  状态分布: 待处理({breakdown.get('pending', 0)}) | 处理中({breakdown.get('processing', 0)}) | 已完成({breakdown.get('completed', 0)}) | 失败({breakdown.get('failed', 0)})")
            except Exception as e:
                print(f"✗ 查询失败: {e}")

        elif args.command == 'chunk-download':
            print(f"下载chunk结果: {args.chunk_id}")
            try:
                if client.download_chunk_results(args.chunk_id, args.output):
                    file_size = os.path.getsize(args.output)
                    print(f"✓ 下载成功: {args.output}")
                    print(f"  文件大小: {file_size} bytes")
                else:
                    print("✗ 下载失败")
            except Exception as e:
                print(f"✗ 下载失败: {e}")

        elif args.command == 'monitor':
            if not args.task_ids:
                # 获取所有任务ID
                tasks = client.list_tasks().get('tasks', [])
                args.task_ids = [t['task_id'] for t in tasks]

            if not args.task_ids:
                print("没有找到任务")
                return

            results = monitor_tasks(client, args.task_ids, args.interval, args.timeout)

        elif args.command == 'download':
            if not args.task_ids:
                # 获取所有已完成任务的ID
                tasks = client.list_tasks().get('tasks', [])
                args.task_ids = [t['task_id'] for t in tasks if t.get('status') == 'completed']

            if not args.task_ids:
                print("没有找到可下载的任务")
                return

            results = download_results(client, args.task_ids, args.output_dir)
            print(f"\n下载完成: {len(results['success'])}/{len(args.task_ids)}")

        elif args.command == 'cleanup':
            results = cleanup_tasks(client, args.older_than)
            print(f"\n清理完成: 删除 {len(results['deleted'])} 个任务")
            if results['failed']:
                print(f"失败 {len(results['failed'])} 个任务")

        elif args.command == 'report':
            report = generate_report(client, args.output)
            print(f"报告已生成: {args.output}")
            print(json.dumps(report, indent=2, ensure_ascii=False))

    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到API服务器 {args.url}")
        print("请确保服务器正在运行")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()