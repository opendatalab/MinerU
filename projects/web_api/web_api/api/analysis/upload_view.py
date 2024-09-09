import json
import time
import traceback
import requests
from flask import request, current_app, url_for, send_from_directory
from flask_restful import Resource
from werkzeug.utils import secure_filename
from pathlib import Path
from common.ext import is_pdf, calculate_file_hash, url_is_pdf
from io import BytesIO
from werkzeug.datastructures import FileStorage
from common.custom_response import generate_response
from loguru import logger


class UploadPdfView(Resource):

    def get(self):
        """
        获取pdf
        :return:
        """
        params = request.args
        filename = params.get('filename')
        as_attachment = params.get('as_attachment')
        if str(as_attachment).lower() == "true":
            as_attachment = True
        else:
            as_attachment = False
        pdf_upload_folder = current_app.config['PDF_UPLOAD_FOLDER']
        response = send_from_directory(f"{current_app.static_folder}/{pdf_upload_folder}", filename,
                                       as_attachment=as_attachment)
        return response

    def post(self):
        """
        上传pdf
        :return:
        """
        file_list = request.files.getlist("file")
        if file_list:
            file = file_list[0]
            filename = secure_filename(file.filename)
            if not file or file and not is_pdf(filename, file):
                return generate_response(code=400, msg="Invalid PDF file", msgZH="PDF文件参数无效")
        else:
            params = json.loads(request.data)
            pdf_url = params.get('pdfUrl')
            try:
                response = requests.get(pdf_url, stream=True)
            except ConnectionError as e:
                logger.error(traceback.format_exc())
                return generate_response(code=400, msg="params is not valid", msgZh="参数错误，pdf链接无法访问")
            if response.status_code != 200:
                return generate_response(code=400, msg="params is not valid", msgZh="参数错误，pdf链接响应状态异常")
            # 创建一个模拟的 FileStorage 对象
            file_content = BytesIO(response.content)
            filename = Path(pdf_url).name if ".pdf" in pdf_url else f"{Path(pdf_url).name}.pdf"
            file = FileStorage(
                stream=file_content,
                filename=filename,
                content_type=response.headers.get('Content-Type', 'application/octet-stream')
            )
            if not file or file and not url_is_pdf(file):
                return generate_response(code=400, msg="Invalid PDF file", msgZH="PDF文件参数无效")

        pdf_upload_folder = current_app.config['PDF_UPLOAD_FOLDER']
        upload_dir = f"{current_app.static_folder}/{pdf_upload_folder}"
        if not Path(upload_dir).exists():
            Path(upload_dir).mkdir(parents=True, exist_ok=True)
        file_key = f"{calculate_file_hash(file)}{int(time.time())}"
        new_filename = f"{file_key}_{filename}"
        file_path = f"{upload_dir}/{new_filename}"
        # file.save(file_path)
        chunk_size = 8192
        with open(file_path, 'wb') as f:
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        # 生成文件的URL路径
        file_url = url_for('analysis.uploadpdfview', filename=new_filename, as_attachment=False)
        data = {
            "url": file_url,
            "file_key": file_key
        }
        return generate_response(data=data)
