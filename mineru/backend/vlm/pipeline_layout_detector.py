# Copyright (c) Opendatalab. All rights reserved.
"""基于pipeline小模型（PP-DocLayoutV2）的layout检测器实现。

作为"第一阶段可替换"的内置示例：不依赖VLM即可产出layout，CPU可运行。
所有重依赖（torch/cv2等）均在实例化/调用时惰性导入，保证vlm-only安装下
`mineru.backend.vlm.stages` 仍可导入。
"""
import os
from types import SimpleNamespace

from .stages import LayoutDetector


class PipelineLayoutDetector(LayoutDetector):
    """用PP-DocLayoutV2出layout，复用hybrid medium档的标签映射与bbox归一化。"""

    emits_formula_number = True
    name = "pp_doclayout_v2"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 8,
        enable_table_orientation: bool = True,
        lang: str | None = None,
    ):
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.backend.pipeline.model_list import AtomicModel
        from mineru.utils.config_reader import get_device
        from mineru.utils.enum_class import ModelPath
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

        self.device = device if device is not None else get_device()
        self.batch_size = batch_size
        self.enable_table_orientation = enable_table_orientation
        self.lang = lang
        self._atom_model_manager = AtomModelSingleton()
        self.layout_model = self._atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            pp_doclayout_v2_weights=str(
                os.path.join(
                    auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2),
                    ModelPath.pp_doclayout_v2,
                )
            ),
            device=self.device,
        )

    def batch_detect(self, images, start_page_idx: int = 0):
        # 惰性导入hybrid辅助函数：标签映射/归一化/表格方向分类均以hybrid为单一事实来源
        from mineru.backend.hybrid.hybrid_analyze import (
            _apply_medium_table_orientation_labels,
            _build_medium_vlm_layout_blocks,
        )
        from mineru.backend.pipeline.model_init import run_layout_inference

        images = list(images)
        images_layout_res = run_layout_inference(
            self.layout_model.batch_predict,
            images,
            batch_size=self.batch_size,
        )
        if self.enable_table_orientation:
            # _apply_medium_table_orientation_labels 只用到 .atom_model_manager 和 .lang
            model_shim = SimpleNamespace(
                atom_model_manager=self._atom_model_manager,
                lang=self.lang,
            )
            _apply_medium_table_orientation_labels(
                images,
                images_layout_res,
                model_shim,
                batch_ratio=1,
            )
        return [
            _build_medium_vlm_layout_blocks(page_layout_res, pil_img.width, pil_img.height)
            for page_layout_res, pil_img in zip(images_layout_res, images)
        ]
