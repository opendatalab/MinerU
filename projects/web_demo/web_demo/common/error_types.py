import json
from flask import request
from werkzeug.exceptions import HTTPException


class ApiException(HTTPException):
    """API错误基类"""
    code = 500
    msg = 'Sorry, we made a mistake Σ(っ °Д °;)っ'
    msgZH = ""
    error_code = 999

    def __init__(self, msg=None, msgZH=None, code=None, error_code=None, headers=None):
        if code:
            self.code = code
        if msg:
            self.msg = msg
        if msgZH:
            self.msgZH = msgZH
        if error_code:
            self.error_code = error_code
        super(ApiException, self).__init__(msg, None)

    @staticmethod
    def get_error_url():
        """获取出错路由和请求方式"""
        method = request.method
        full_path = str(request.full_path)
        main_path = full_path.split('?')[0]
        res = method + ' ' + main_path
        return res

    def get_body(self, environ=None, scope=None):
        """异常返回信息"""
        body = dict(
            msg=self.msg,
            error_code=self.error_code,
            request=self.get_error_url()
        )
        text = json.dumps(body)
        return text

    def get_headers(self, environ=None, scope=None):
        """异常返回格式"""
        return [("Content-Type", "application/json")]