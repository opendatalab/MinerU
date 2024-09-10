import json
from flask import url_for, request
from flask_restful import Resource
from sqlalchemy import func
from ..extentions import db
from .models import AnalysisTask, AnalysisPdf
from .ext import task_state_map
from common.custom_response import generate_response


class TaskView(Resource):
    def get(self):
        """
        查询正在进行的任务
        :return:
        """
        analysis_task_running = AnalysisTask.query.filter(AnalysisTask.status == 0).first()
        analysis_task_pending = AnalysisTask.query.filter(AnalysisTask.status == 2).order_by(
            AnalysisTask.create_date.asc()).all()
        pending_total = db.session.query(func.count(AnalysisTask.id)).filter(AnalysisTask.status == 2).scalar()
        if analysis_task_running:
            task_nums = pending_total + 1
            file_name_split = analysis_task_running.file_name.split("_")
            new_file_name = file_name_split[-1] if file_name_split else analysis_task_running.file_name
            data = [
                {
                    "queues": task_nums,  # 正在排队的任务总数
                    "rank": 1,
                    "id": analysis_task_running.id,
                    "url": url_for('analysis.uploadpdfview', filename=analysis_task_running.file_name, as_attachment=False),
                    "fileName": new_file_name,
                    "type": analysis_task_running.task_type,
                    "state": task_state_map.get(analysis_task_running.status),
                }
            ]
        else:
            task_nums = pending_total
            data = []
        for n, task in enumerate(analysis_task_pending):
            file_name_split = task.file_name.split("_")
            new_file_name = file_name_split[-1] if file_name_split else task.file_name
            data.append({
                "queues": task_nums,  # 正在排队的任务总数
                "rank": n + 2,
                "id": task.id,
                "url": url_for('analysis.uploadpdfview', filename=task.file_name, as_attachment=False),
                "fileName": new_file_name,
                "type": task.task_type,
                "state": task_state_map.get(task.status),
            })
        data.reverse()
        return generate_response(data=data, total=task_nums)


class HistoricalTasksView(Resource):
    def get(self):
        """
        获取任务历史记录
        :return:
        """
        params = request.args
        page_no = params.get('pageNo', 1)
        page_size = params.get('pageSize', 10)
        total = db.session.query(func.count(AnalysisTask.id)).scalar()
        analysis_task = AnalysisTask.query.order_by(AnalysisTask.create_date.desc()).paginate(page=int(page_no),
                                                                                              per_page=int(page_size),
                                                                                              error_out=False)
        data = []
        for n, task in enumerate(analysis_task):
            file_name_split = task.file_name.split("_")
            new_file_name = file_name_split[-1] if file_name_split else task.file_name
            data.append({
                "fileName": new_file_name,
                "id": task.id,
                "type": task.task_type,
                "state": task_state_map.get(task.status),
            })
        data = {
            "list": data,
            "total": total,
            "pageNo": page_no,
            "pageSize": page_size,
        }
        return generate_response(data=data)


class DeleteTaskView(Resource):
    def delete(self, id):
        """
        删除任务历史记录
        :return:
        """
        analysis_task = AnalysisTask.query.filter(AnalysisTask.id == id, AnalysisTask.status != 0).first()
        if analysis_task:
            analysis_pdf = AnalysisPdf.query.filter(AnalysisPdf.id == AnalysisTask.analysis_pdf_id).first()
            with db.auto_commit():
                db.session.delete(analysis_pdf)
                db.session.delete(analysis_task)
        else:
            return generate_response(code=400, msg="The ID is incorrect", msgZH="id不正确")

        return generate_response(data={"id": id})
