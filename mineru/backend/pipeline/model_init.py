import os

import torch
from loguru import logger

from .model_list import AtomicModel
from ...model.layout.doclayout_yolo import DocLayoutYOLOModel
from ...model.mfd.yolo_v8 import YOLOv8MFDModel
from ...model.mfr.unimernet.Unimernet import UnimernetModel
from ...model.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
from ...model.table.rapid_table import RapidTableModel
from ...utils.enum_class import ModelPath
from ...utils.models_download_utils import auto_download_and_get_model_root_path


def table_model_init(lang=None):
    atom_model_manager = AtomModelSingleton()
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name='ocr',
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang
    )
    table_model = RapidTableModel(ocr_engine)
    return table_model


def mfd_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    mfd_model = YOLOv8MFDModel(weight, device)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    mfr_model = UnimernetModel(weight_dir, device)
    return mfr_model


def doclayout_yolo_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    model = DocLayoutYOLOModel(weight, device)
    return model

def ocr_model_init(det_db_box_thresh=0.3,
                   lang=None,
                   use_dilation=True,
                   det_db_unclip_ratio=1.8,
                   ):
    if lang is not None and lang != '':
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    else:
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    return model


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):

        lang = kwargs.get('lang', None)
        table_model_name = kwargs.get('table_model_name', None)

        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
        elif atom_model_name in [AtomicModel.Table]:
            key = (atom_model_name, table_model_name, lang)
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = doclayout_yolo_model_init(
            kwargs.get('doclayout_yolo_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get('lang'),
        )
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model


class MineruPipelineModel:
    def __init__(self, **kwargs):
        self.formula_config = kwargs.get('formula_config')
        self.apply_formula = self.formula_config.get('enable', True)
        self.table_config = kwargs.get('table_config')
        self.apply_table = self.table_config.get('enable', True)
        self.lang = kwargs.get('lang', None)
        self.device = kwargs.get('device', 'cpu')
        logger.info(
            'DocAnalysis init, this may take some times......'
        )
        atom_model_manager = AtomModelSingleton()

        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd), ModelPath.yolo_v8_mfd)
                ),
                device=self.device,
            )

            # 初始化公式解析模型
            mfr_weight_dir = os.path.join(auto_download_and_get_model_root_path(ModelPath.unimernet_small), ModelPath.unimernet_small)

            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=mfr_weight_dir,
                device=self.device,
            )

        # 初始化layout模型
        self.layout_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            doclayout_yolo_weights=str(
                os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
            ),
            device=self.device,
        )
        # 初始化ocr
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=self.lang
        )
        # init table model
        if self.apply_table:
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
                lang=self.lang,
            )

        logger.info('DocAnalysis init done!')