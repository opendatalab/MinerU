from flask import render_template, Response
from flask_restful import Resource


class ReactAppView(Resource):
    def get(self):
        # 创建自定义的响应对象
        rendered_template = render_template('index.html')
        response = Response(rendered_template, mimetype='text/html')

        return response
