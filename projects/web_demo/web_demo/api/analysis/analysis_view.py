import json
import threading
from multiprocessing import Process
from pathlib import Path
from flask import request, current_app, url_for
from flask_restful import Resource
from .ext import find_file, task_state_map
# from .formula_ext import formula_detection, formula_recognition
from .serialization import AnalysisViewSchema
from marshmallow import ValidationError
from ..extentions import db
from .models import AnalysisTask, AnalysisPdf
from .pdf_ext import analysis_pdf_task
from common.custom_response import generate_response


class AnalysisTaskProgressView(Resource):

    def get(self):
        """
        获取任务进度
        :return:
        """
        params = request.args
        id = params.get('id')
        analysis_task = AnalysisTask.query.filter(AnalysisTask.id == id).first()
        if not analysis_task:
            return generate_response(code=400, msg="Invalid ID", msgZH="无效id")
        match analysis_task.task_type:
            case 'pdf':
                analysis_pdf = AnalysisPdf.query.filter(AnalysisPdf.id == analysis_task.analysis_pdf_id).first()
                file_url = url_for('analysis.uploadpdfview', filename=analysis_task.file_name, as_attachment=False)
                file_name_split = analysis_task.file_name.split("_")
                file_name = file_name_split[-1] if file_name_split else analysis_task.file_name
                if analysis_task.status == 0:
                    data = {
                        "state": task_state_map.get(analysis_task.status),
                        "status": analysis_pdf.status,
                        "url": file_url,
                        "fileName": file_name,
                        "file_key": analysis_task.file_key,
                        "content": [],
                        "markdownUrl": [],
                        "fullMdLink": "",
                        "type": analysis_task.task_type,
                    }
                    return generate_response(data=data)
                elif analysis_task.status == 1:
                    if analysis_pdf.status == 1:  # 任务正常完成
                        bbox_info = json.loads(analysis_pdf.bbox_info)
                        md_link_list = json.loads(analysis_pdf.md_link_list)
                        full_md_link = analysis_pdf.full_md_link
                        data = {
                            "state": task_state_map.get(analysis_task.status),
                            "status": analysis_pdf.status,
                            "url": file_url,
                            "fileName": file_name,
                            "file_key": analysis_task.file_key,
                            "content": bbox_info,
                            "markdownUrl": md_link_list,
                            "fullMdLink": full_md_link,
                            "type": analysis_task.task_type,
                        }
                        return generate_response(data=data)
                    else:  # 任务异常结束
                        data = {
                            "state": "failed",
                            "status": analysis_pdf.status,
                            "url": file_url,
                            "fileName": file_name,
                            "file_key": analysis_task.file_key,
                            "content": [],
                            "markdownUrl": [],
                            "fullMdLink": "",
                            "type": analysis_task.task_type,
                        }
                        return generate_response(code=-60004, data=data, msg="Failed to retrieve PDF parsing progress",
                                                 msgZh="无法获取PDF解析进度")
                else:
                    data = {
                        "state": task_state_map.get(analysis_task.status),
                        "status": analysis_pdf.status,
                        "url": file_url,
                        "fileName": file_name,
                        "file_key": analysis_task.file_key,
                        "content": [],
                        "markdownUrl": [],
                        "fullMdLink": "",
                        "type": analysis_task.task_type,
                    }
                    return generate_response(data=data)
            case 'formula-detect':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'formula-extract':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'table-recogn':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case _:
                return generate_response(code=400, msg="Not yet supported", msgZH="参数不支持")


class AnalysisTaskView(Resource):

    def post(self):
        """
        提交任务
        :return:
        """
        analysis_view_schema = AnalysisViewSchema()
        try:
            params = analysis_view_schema.load(request.get_json())
        except ValidationError as err:
            return generate_response(code=400, msg=err.messages)
        file_key = params.get("fileKey")
        file_name = params.get("fileName")
        task_type = params.get("taskType")
        is_ocr = params.get("isOcr", False)

        pdf_upload_folder = current_app.config['PDF_UPLOAD_FOLDER']
        upload_dir = f"{current_app.static_folder}/{pdf_upload_folder}"
        file_path = find_file(file_key, upload_dir)
        match task_type:
            case 'pdf':
                if not file_path:
                    return generate_response(code=400, msg="FileKey is invalid, no PDF file found",
                                             msgZH="fileKey无效，未找到pdf文件")
                analysis_task = AnalysisTask.query.filter(AnalysisTask.status.in_([0, 2])).first()
                file_name = Path(file_path).name
                with db.auto_commit():
                    analysis_pdf_object = AnalysisPdf(
                        file_name=file_name,
                        file_path=file_path,
                        status=3 if analysis_task else 0,
                    )
                    db.session.add(analysis_pdf_object)
                    db.session.flush()
                    analysis_pdf_id = analysis_pdf_object.id
                with db.auto_commit():
                    analysis_task_object = AnalysisTask(
                        file_key=file_key,
                        file_name=file_name,
                        task_type=task_type,
                        is_ocr=is_ocr,
                        status=2 if analysis_task else 0,
                        analysis_pdf_id=analysis_pdf_id
                    )
                    db.session.add(analysis_task_object)
                    db.session.flush()
                    analysis_task_id = analysis_task_object.id
                if not analysis_task:  # 已有同类型任务在执行，请等待执行完成
                    file_stem = Path(file_path).stem
                    pdf_analysis_folder = current_app.config['PDF_ANALYSIS_FOLDER']
                    pdf_dir = f"{current_app.static_folder}/{pdf_analysis_folder}/{file_stem}"
                    image_dir = f"{pdf_dir}/images"
                    t = threading.Thread(target=analysis_pdf_task,
                                         args=(pdf_dir, image_dir, file_path, is_ocr, analysis_pdf_id))
                    t.start()
                # 生成文件的URL路径
                file_url = url_for('analysis.uploadpdfview', filename=file_name, as_attachment=False)
                data = {
                    "url": file_url,
                    "fileName": file_name,
                    "id": analysis_task_id
                }
                return generate_response(data=data)
            case 'formula-detect':
                # if not file_path:
                #     return generate_response(code=400, msg="FileKey is invalid, no image file found",
                #                              msgZH="fileKey无效，未找到图片")
                # return formula_detection(file_path, upload_dir)
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'formula-extract':
                # if not file_path:
                #     return generate_response(code=400, msg="FileKey is invalid, no image file found",
                #                              msgZH="fileKey无效，未找到图片")
                # return formula_recognition(file_path, upload_dir)
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'table-recogn':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case _:
                return generate_response(code=400, msg="Not yet supported", msgZH="参数不支持")

    def put(self):
        """
        重新发起任务
        :return:
        """
        params = json.loads(request.data)
        id = params.get('id')
        analysis_task = AnalysisTask.query.filter(AnalysisTask.id == id).first()
        if not analysis_task:
            return generate_response(code=400, msg="Invalid ID", msgZH="无效id")
        match analysis_task.task_type:
            case 'pdf':
                task_r_p = AnalysisTask.query.filter(AnalysisTask.status.in_([0, 2])).first()
                if task_r_p:
                    with db.auto_commit():
                        analysis_pdf_object = AnalysisPdf.query.filter_by(id=analysis_task.analysis_pdf_id).first()
                        analysis_pdf_object.status = 3
                        db.session.add(analysis_pdf_object)
                    with db.auto_commit():
                        analysis_task.status = 2
                        db.session.add(analysis_task)
                else:
                    with db.auto_commit():
                        analysis_pdf_object = AnalysisPdf.query.filter_by(id=analysis_task.analysis_pdf_id).first()
                        analysis_pdf_object.status = 0
                        db.session.add(analysis_pdf_object)
                    with db.auto_commit():
                        analysis_task.status = 0
                        db.session.add(analysis_task)

                    pdf_upload_folder = current_app.config['PDF_UPLOAD_FOLDER']
                    upload_dir = f"{current_app.static_folder}/{pdf_upload_folder}"
                    file_path = find_file(analysis_task.file_key, upload_dir)
                    file_stem = Path(file_path).stem
                    pdf_analysis_folder = current_app.config['PDF_ANALYSIS_FOLDER']
                    pdf_dir = f"{current_app.static_folder}/{pdf_analysis_folder}/{file_stem}"
                    image_dir = f"{pdf_dir}/images"
                    process = Process(target=analysis_pdf_task,
                                      args=(pdf_dir, image_dir, file_path, analysis_task.is_ocr,
                                            analysis_task.analysis_pdf_id))
                    process.start()

                # 生成文件的URL路径
                file_url = url_for('analysis.uploadpdfview', filename=analysis_task.file_name, as_attachment=False)
                file_name_split = analysis_task.file_name.split("_")
                new_file_name = file_name_split[-1] if file_name_split else analysis_task.file_name
                data = {
                    "url": file_url,
                    "fileName": new_file_name,
                    "id": analysis_task.id
                }
                return generate_response(data=data)
            case 'formula-detect':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'formula-extract':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case 'table-recogn':
                return generate_response(code=400, msg="Not yet supported", msgZH="功能待开发")
            case _:
                return generate_response(code=400, msg="Not yet supported", msgZH="参数不支持")
