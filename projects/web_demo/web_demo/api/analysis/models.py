from datetime import datetime
from ..extentions import db


class AnalysisTask(db.Model):
    __tablename__ = 'analysis_task'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_key = db.Column(db.Text, comment="文件唯一哈希")
    file_name = db.Column(db.Text, comment="文件名称")
    task_type = db.Column(db.String(128), comment="任务类型")
    is_ocr = db.Column(db.Boolean, default=False, comment="是否ocr")
    status = db.Column(db.Integer, default=0, comment="状态")  # 0 running  1 done  2 pending
    analysis_pdf_id = db.Column(db.Integer, comment="analysis_pdf的id")
    create_date = db.Column(db.DateTime(), nullable=False, default=datetime.now)
    update_date = db.Column(db.DateTime(), nullable=False, default=datetime.now, onupdate=datetime.now)


class AnalysisPdf(db.Model):
    __tablename__ = 'analysis_pdf'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_name = db.Column(db.Text, comment="文件名称")
    file_url = db.Column(db.Text, comment="文件原路径")
    file_path = db.Column(db.Text, comment="文件路径")
    status = db.Column(db.Integer, default=3, comment="状态")  # 0 转换中  1 已完成  2 转换失败 3 init
    bbox_info = db.Column(db.Text, comment="坐标数据")
    md_link_list = db.Column(db.Text, comment="markdown分页链接")
    full_md_link = db.Column(db.Text, comment="markdown全文链接")
    create_date = db.Column(db.DateTime(), nullable=False, default=datetime.now)
    update_date = db.Column(db.DateTime(), nullable=False, default=datetime.now, onupdate=datetime.now)