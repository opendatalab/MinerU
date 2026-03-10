import math
import os
import torch
import yaml
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from mineru.model.utils.tools.infer import pytorchocr_utility
from mineru.model.utils.pytorchocr.base_ocr_v20 import BaseOCRV20
from .processors import (
    UniMERNetImgDecode,
    UniMERNetTestTransform,
    LatexImageFormat,
    ToBatch,
    UniMERNetDecode,
)


class FormulaRecognizer(BaseOCRV20):
    def __init__(
        self,
        weight_dir,
        device="cpu",
    ):
        self.weights_path = os.path.join(
            weight_dir,
            "PP-FormulaNet_plus-M.pth",
        )
        self.yaml_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "utils",
            "pytorchocr",
            "utils",
            "resources",
            "pp_formulanet_arch_config.yaml"
        )
        self.infer_yaml_path = os.path.join(
            weight_dir,
            "PP-FormulaNet_plus-M_inference.yml",
        )

        network_config = pytorchocr_utility.AnalysisConfig(
            self.weights_path, self.yaml_path
        )
        weights = self.read_pytorch_weights(self.weights_path)

        super(FormulaRecognizer, self).__init__(network_config)

        self.load_state_dict(weights)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.net.to(self.device)
        self.net.eval()

        with open(self.infer_yaml_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.pre_tfs = {
            "UniMERNetImgDecode": UniMERNetImgDecode(input_size=(384, 384)),
            "UniMERNetTestTransform": UniMERNetTestTransform(),
            "LatexImageFormat": LatexImageFormat(),
            "ToBatch": ToBatch(),
        }

        self.post_op = UniMERNetDecode(
            character_list=data["PostProcess"]["character_dict"]
        )

    @staticmethod
    def _normalize_bbox(bbox, image):
        if bbox is None:
            return None

        xmin, ymin, xmax, ymax = [float(v) for v in bbox]
        xmin = math.floor(xmin)
        ymin = math.floor(ymin)
        xmax = math.ceil(xmax)
        ymax = math.ceil(ymax)
        height, width = image.shape[:2]
        xmin = max(0, min(width, xmin))
        xmax = max(0, min(width, xmax))
        ymin = max(0, min(height, ymin))
        ymax = max(0, min(height, ymax))
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _item_to_bbox(item, image):
        return FormulaRecognizer._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(self, mfd_res, image, interline_enable=True):
        formula_list = []
        crop_targets = []

        for item in mfd_res or []:
            if not isinstance(item, dict):
                continue
            if item.get("label") != "formula":
                continue

            new_item = dict(item)
            new_item.setdefault("latex", "")
            formula_list.append(new_item)

            bbox = self._item_to_bbox(new_item, image)
            if bbox is not None:
                crop_targets.append((new_item, bbox))

        return formula_list, crop_targets

    def predict(self, img_list, batch_size: int = 64):
        # Reduce batch size by 50% to avoid potential memory issues during inference.
        batch_size = max(1, int(0.5 * batch_size))
        batch_imgs = self.pre_tfs["UniMERNetImgDecode"](imgs=img_list)
        batch_imgs = self.pre_tfs["UniMERNetTestTransform"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["LatexImageFormat"](imgs=batch_imgs)
        inp = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        inp = torch.from_numpy(inp[0])
        inp = inp.to(self.device)
        rec_formula = []
        with torch.no_grad():
            with tqdm(total=len(inp), desc="MFR Predict") as pbar:
                for index in range(0, len(inp), batch_size):
                    batch_data = inp[index: index + batch_size]
                    # with torch.amp.autocast(device_type=self.device.type):
                    #     batch_preds = [self.net(batch_data)]
                    batch_preds = [self.net(batch_data)]
                    batch_preds = [p.reshape([-1]) for p in batch_preds[0]]
                    batch_preds = [bp.cpu().numpy() for bp in batch_preds]
                    rec_formula += self.post_op(batch_preds)
                    pbar.update(len(batch_preds))
        return rec_formula

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []  # Store (area, original_index, image) tuples

        # Collect images with their original indices
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            image = images[image_index]
            formula_list, crop_targets = self._build_formula_items(
                mfd_res, image, interline_enable=interline_enable
            )

            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)
                backfill_list.append(formula_item)

            images_formula_list.append(formula_list)

        # Stable sort by area
        image_info.sort(key=lambda x: x[0])  # sort by area
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]

        # Create mapping for results
        index_mapping = {
            new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)
        }

        if len(sorted_images) > 0:
            # 进行预测
            batch_size = min(batch_size, max(1, 2 ** (len(sorted_images).bit_length() - 1))) if sorted_images else 1
            rec_formula = self.predict(sorted_images, batch_size)
        else:
            rec_formula = []

        # Restore original order
        unsorted_results = [""] * len(rec_formula)
        for new_idx, latex in enumerate(rec_formula):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
