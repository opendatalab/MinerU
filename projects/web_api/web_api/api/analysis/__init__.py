from flask import Blueprint
from ..extentions import Api
from .upload_view import UploadPdfView
from .analysis_view import AnalysisTaskView, AnalysisTaskProgressView
from .img_md_view import ImgView, MdView
from .task_view import TaskView, HistoricalTasksView, DeleteTaskView
from .markdown_view import MarkdownView

analysis_blue = Blueprint('analysis', __name__)

api_v2 = Api(analysis_blue, prefix='/api/v2')
api_v2.add_resource(UploadPdfView, '/analysis/upload_pdf')
api_v2.add_resource(AnalysisTaskView, '/extract/task/submit')
api_v2.add_resource(AnalysisTaskProgressView, '/extract/task/progress')
api_v2.add_resource(ImgView, '/analysis/pdf_img')
api_v2.add_resource(MdView, '/analysis/pdf_md')
api_v2.add_resource(TaskView, '/extract/taskQueue')
api_v2.add_resource(HistoricalTasksView, '/extract/list')
api_v2.add_resource(DeleteTaskView, '/extract/task/<int:id>')
api_v2.add_resource(MarkdownView, '/extract/markdown')