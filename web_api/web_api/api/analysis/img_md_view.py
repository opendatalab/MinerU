from pathlib import Path
from flask import request, current_app, send_from_directory
from flask_restful import Resource


class ImgView(Resource):
    def get(self):
        """
        获取pdf解析的图片
        :return:
        """
        params = request.args
        pdf = params.get('pdf')
        filename = params.get('filename')
        as_attachment = params.get('as_attachment')
        if str(as_attachment).lower() == "true":
            as_attachment = True
        else:
            as_attachment = False
        file_stem = Path(pdf).stem
        pdf_analysis_folder = current_app.config['PDF_ANALYSIS_FOLDER']
        pdf_dir = f"{current_app.static_folder}/{pdf_analysis_folder}/{file_stem}"
        image_dir = f"{pdf_dir}/images"
        response = send_from_directory(image_dir, filename, as_attachment=as_attachment)
        return response


class MdView(Resource):
    def get(self):
        """
        获取pdf解析的markdown
        :return:
        """
        params = request.args
        pdf = params.get('pdf')
        filename = params.get('filename')
        as_attachment = params.get('as_attachment')
        if str(as_attachment).lower() == "true":
            as_attachment = True
        else:
            as_attachment = False
        file_stem = Path(pdf).stem
        pdf_analysis_folder = current_app.config['PDF_ANALYSIS_FOLDER']
        pdf_dir = f"{current_app.static_folder}/{pdf_analysis_folder}/{file_stem}"
        response = send_from_directory(pdf_dir, filename, as_attachment=as_attachment)
        return response
