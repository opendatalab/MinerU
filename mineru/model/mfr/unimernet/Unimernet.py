import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils import build_mfr_batch_groups


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image


class UnimernetModel(object):
    def __init__(self, weight_dir, _device_="cpu"):
        from .unimernet_hf import UnimernetModel

        if _device_.startswith("mps") or _device_.startswith("npu") or _device_.startswith("musa"):
            self.model = UnimernetModel.from_pretrained(
                weight_dir,
                attn_implementation="eager",
            )
        else:
            self.model = UnimernetModel.from_pretrained(weight_dir)
        self.device = torch.device(_device_)
        self.model.to(self.device)
        if not _device_.startswith("cpu"):
            self.model = self.model.to(dtype=torch.float16)
        self.model.eval()

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
        return UnimernetModel._normalize_bbox(item.get("bbox"), image)

    def _build_formula_items(self, mfd_res, image, interline_enable=True):
        formula_list = []
        crop_targets = []

        for item in mfd_res or []:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label not in ["inline_formula", "display_formula"]:
                continue
            if not interline_enable and label == "display_formula":
                continue

            new_item = dict(item)
            new_item.setdefault("latex", "")
            formula_list.append(new_item)

            bbox = self._item_to_bbox(new_item, image)
            if bbox is not None:
                crop_targets.append((new_item, bbox))

        return formula_list, crop_targets

    def predict(
        self,
        mfd_res,
        image,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        return self.batch_predict(
            [mfd_res],
            [image],
            batch_size=batch_size,
            interline_enable=interline_enable,
        )[0]

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        if not images_mfd_res:
            return []

        if len(images_mfd_res) != len(images):
            raise ValueError("images_mfd_res and images must have the same length.")

        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []

        for mfd_res, image in zip(images_mfd_res, images):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res,
                image,
                interline_enable=interline_enable,
            )

            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)
                backfill_list.append(formula_item)

            images_formula_list.append(formula_list)

        if not image_info:
            return images_formula_list

        image_info.sort(key=lambda x: x[0])
        sorted_areas = [x[0] for x in image_info]
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]
        index_mapping = {
            new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)
        }

        batch_groups = build_mfr_batch_groups(sorted_areas, batch_size)
        dataset = MathDataset(sorted_images, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_sampler=batch_groups, num_workers=0)

        mfr_res = []
        with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar:
            for batch_group, mf_img in zip(batch_groups, dataloader):
                current_batch_size = len(batch_group)
                mf_img = mf_img.to(dtype=self.model.dtype)
                mf_img = mf_img.to(self.device)
                with torch.no_grad():
                    output = self.model.generate(
                        {"image": mf_img},
                        batch_size=current_batch_size,
                    )
                mfr_res.extend(output["fixed_str"])
                pbar.update(current_batch_size)

        unsorted_results = [""] * len(mfr_res)
        for new_idx, latex in enumerate(mfr_res):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
