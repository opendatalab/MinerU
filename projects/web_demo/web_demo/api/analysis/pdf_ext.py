import json
import os
import shutil
import traceback
from pathlib import Path

from common.error_types import ApiException
from common.mk_markdown.mk_markdown import \
    ocr_mk_mm_markdown_with_para_and_pagination
from flask import current_app, url_for
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult

from ..extentions import app, db
from .ext import find_file
from .models import AnalysisPdf, AnalysisTask

model_config.__use_inside_model__ = True


def analysis_pdf(image_url_prefix, image_dir, pdf_bytes, is_ocr=False):
    try:
        model_json = []  # model_json传空list使用内置模型解析
        image_writer = FileBasedDataWriter(image_dir)
        logger.info(f'is_ocr: {is_ocr}')
        parse_method = 'ocr'
        ds = PymuDocDataset(pdf_bytes)
        # Choose parsing method
        if not is_ocr:
            if ds.classify() == SupportedPdfParseMethod.OCR:
                parse_method = 'ocr'
            else:
                parse_method = 'txt'

        if parse_method == 'ocr':
            infer_result = ds.apply(doc_analyze, ocr=True)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

        if parse_method == 'ocr':
            pipe_res = infer_result.pipe_ocr_mode(image_writer)
        else:
            pipe_res = infer_result.pipe_txt_mode(image_writer)

        pdf_mid_data = pipe_res._pipe_res
        pdf_info_list = pdf_mid_data['pdf_info']
        md_content = json.dumps(ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_list, image_url_prefix),
                                ensure_ascii=False)
        bbox_info = get_bbox_info(pdf_info_list)
        return md_content, bbox_info
    except Exception as e:  # noqa: F841
        logger.error(traceback.format_exc())

def get_bbox_info(data):
    bbox_info = []
    for page in data:
        preproc_blocks = page.get('preproc_blocks', [])
        discarded_blocks = page.get('discarded_blocks', [])
        bbox_info.append({
            'preproc_blocks': preproc_blocks,
            'page_idx': page.get('page_idx'),
            'page_size': page.get('page_size'),
            'discarded_blocks': discarded_blocks,
        })
    return bbox_info


def analysis_pdf_task(pdf_dir, image_dir, pdf_path, is_ocr, analysis_pdf_id):
    """解析pdf.

    :param pdf_dir:  pdf解析目录
    :param image_dir:  图片目录
    :param pdf_path:  pdf路径
    :param is_ocr:  是否启用ocr
    :param analysis_pdf_id:  pdf解析表id
    :return:
    """
    try:
        logger.info(f'start task: {pdf_path}')
        logger.info(f'image_dir: {image_dir}')
        if not Path(image_dir).exists():
            Path(image_dir).mkdir(parents=True, exist_ok=True)
        else:
            # 清空image_dir，避免同文件多次解析图片积累
            shutil.rmtree(image_dir, ignore_errors=True)
            os.makedirs(image_dir, exist_ok=True)

        # 获取文件内容
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()
        # 生成图片链接
        with app.app_context():
            image_url_prefix = f"http://{current_app.config['SERVER_NAME']}{current_app.config['FILE_API']}&pdf={Path(pdf_path).name}&filename="
        # 解析文件
        md_content, bbox_info = analysis_pdf(image_url_prefix, image_dir, pdf_bytes, is_ocr)

        # ############ markdown #############
        pdf_name = Path(pdf_path).name

        full_md_content = ''
        for item in json.loads(md_content):
            full_md_content += item['md_content'] + '\n'

        full_md_name = 'full.md'
        with open(f'{pdf_dir}/{full_md_name}', 'w', encoding='utf-8') as file:
            file.write(full_md_content)
        with app.app_context():
            full_md_link = url_for('analysis.mdview', filename=full_md_name, as_attachment=False)
            full_md_link = f'{full_md_link}&pdf={pdf_name}'

        md_link_list = []
        with app.app_context():
            for n, md in enumerate(json.loads(md_content)):
                md_content = md['md_content']
                md_name = f"{md.get('page_no', n)}.md"
                with open(f'{pdf_dir}/{md_name}', 'w', encoding='utf-8') as file:
                    file.write(md_content)
                md_url = url_for('analysis.mdview', filename=md_name, as_attachment=False)
                md_link_list.append(f'{md_url}&pdf={pdf_name}')

        with app.app_context():
            with db.auto_commit():
                analysis_pdf_object = AnalysisPdf.query.filter_by(id=analysis_pdf_id).first()
                analysis_pdf_object.status = 1
                analysis_pdf_object.bbox_info = json.dumps(bbox_info, ensure_ascii=False)
                analysis_pdf_object.md_link_list = json.dumps(md_link_list, ensure_ascii=False)
                analysis_pdf_object.full_md_link = full_md_link
                db.session.add(analysis_pdf_object)
            with db.auto_commit():
                analysis_task_object = AnalysisTask.query.filter_by(analysis_pdf_id=analysis_pdf_id).first()
                analysis_task_object.status = 1
                db.session.add(analysis_task_object)
        logger.info('finished!')
    except Exception as e:  # noqa: F841
        logger.error(traceback.format_exc())
        with app.app_context():
            with db.auto_commit():
                analysis_pdf_object = AnalysisPdf.query.filter_by(id=analysis_pdf_id).first()
                analysis_pdf_object.status = 2
                db.session.add(analysis_pdf_object)
            with db.auto_commit():
                analysis_task_object = AnalysisTask.query.filter_by(analysis_pdf_id=analysis_pdf_id).first()
                analysis_task_object.status = 1
                db.session.add(analysis_task_object)
        raise ApiException(code=500, msg='PDF parsing failed', msgZH='pdf解析失败')
    finally:
        # 执行pending
        with app.app_context():
            analysis_task_object = AnalysisTask.query.filter_by(status=2).order_by(
                AnalysisTask.update_date.asc()).first()
            if analysis_task_object:
                pdf_upload_folder = current_app.config['PDF_UPLOAD_FOLDER']
                upload_dir = f'{current_app.static_folder}/{pdf_upload_folder}'
                file_path = find_file(analysis_task_object.file_key, upload_dir)
                file_stem = Path(file_path).stem
                pdf_analysis_folder = current_app.config['PDF_ANALYSIS_FOLDER']
                pdf_dir = f'{current_app.static_folder}/{pdf_analysis_folder}/{file_stem}'
                image_dir = f'{pdf_dir}/images'
                with db.auto_commit():
                    analysis_pdf_object = AnalysisPdf.query.filter_by(id=analysis_task_object.analysis_pdf_id).first()
                    analysis_pdf_object.status = 0
                    db.session.add(analysis_pdf_object)
                with db.auto_commit():
                    analysis_task_object.status = 0
                    db.session.add(analysis_task_object)
                analysis_pdf_task(pdf_dir, image_dir, file_path, analysis_task_object.is_ocr, analysis_task_object.analysis_pdf_id)
            else:
                logger.info('all task finished!')
