from contextlib import contextmanager

from common.error_types import ApiException
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
from flask_restful import Api as _Api
from flask_sqlalchemy import SQLAlchemy as _SQLAlchemy
from loguru import logger
from werkzeug.exceptions import HTTPException


class Api(_Api):
    def handle_error(self, e):
        if isinstance(e, ApiException):
            code = e.code
            msg = e.msg
            msgZH = e.msgZH
            error_code = e.error_code
        elif isinstance(e, HTTPException):
            code = e.code
            msg = e.description
            msgZH = '服务异常，详细信息请查看日志'
            error_code = e.code
        else:
            code = 500
            msg = str(e)
            error_code = 500
            msgZH = '服务异常，详细信息请查看日志'

        # 使用 loguru 记录异常信息
        logger.opt(exception=e).error(f'An error occurred: {msg}')

        return jsonify({
            'error': 'Internal Server Error' if code == 500 else e.name,
            'msg': msg,
            'msgZH': msgZH,
            'code': code,
            'error_code': error_code
        }), code


class SQLAlchemy(_SQLAlchemy):
    @contextmanager
    def auto_commit(self):
        try:
            yield
            db.session.commit()
            db.session.flush()
        except Exception as e:
            db.session.rollback()
            raise e


app = Flask(__name__)
CORS(app, supports_credentials=True)
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
ma = Marshmallow()
folder = app.config.get('REACT_APP_DIST')
