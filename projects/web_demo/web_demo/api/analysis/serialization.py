from marshmallow import Schema, fields, validates_schema, validates
from common.error_types import ApiException
from .models import AnalysisTask


class BooleanField(fields.Boolean):
    def _deserialize(self, value, attr, data, **kwargs):
        # 进行自定义验证
        if not isinstance(value, bool):
            raise ApiException(code=400, msg="isOcr not a valid boolean", msgZH="isOcr不是有效的布尔值")

        return value


class AnalysisViewSchema(Schema):
    fileKey = fields.Str(required=True)
    fileName = fields.Str()
    taskType = fields.Str(required=True)
    isOcr = BooleanField()

    @validates_schema(pass_many=True)
    def validate_passwords(self, data, **kwargs):
        task_type = data['taskType']
        file_key = data['fileKey']
        if not file_key:
            raise ApiException(code=400, msg="fileKey cannot be empty", msgZH="fileKey不能为空")
        if not task_type:
            raise ApiException(code=400, msg="taskType cannot be empty", msgZH="taskType不能为空")
