from flask import jsonify


class ResponseCode:
    SUCCESS = 200
    PARAM_WARING = 400
    MESSAGE = "success"


def generate_response(data=None, code=ResponseCode.SUCCESS, msg=ResponseCode.MESSAGE, **kwargs):
    """
    自定义响应
    :param code:状态码
    :param data:返回数据
    :param msg:返回消息
    :param kwargs:
    :return:
    """
    msg = msg or 'success' if code == 200 else msg or 'fail'
    success = True if code == 200 else False
    res = jsonify(dict(code=code, success=success, data=data, msg=msg, **kwargs))
    res.status_code = 200
    return res
