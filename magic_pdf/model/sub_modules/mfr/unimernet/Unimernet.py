import argparse
import os
import re

import torch
import unimernet.tasks as tasks
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from unimernet.common.config import Config
from unimernet.processors import load_processor


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = "[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == s:
            break
    return s


class UnimernetModel(object):
    def __init__(self, weight_dir, cfg_path, _device_="cpu"):
        args = argparse.Namespace(cfg_path=cfg_path, options=None)
        cfg = Config(args)
        cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.pth")
        cfg.config.model.model_config.model_name = weight_dir
        cfg.config.model.tokenizer_config.path = weight_dir
        task = tasks.setup_task(cfg)
        self.model = task.build_model(cfg)
        self.device = _device_
        self.model.to(_device_)
        self.model.eval()
        vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        self.mfr_transform = transforms.Compose(
            [
                vis_processor,
            ]
        )

    def predict(self, mfd_res, image):
        formula_list = []
        mf_image_list = []
        for xyxy, conf, cla in zip(
            mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
                "latex": "",
            }
            formula_list.append(new_item)
            pil_img = Image.fromarray(image)
            bbox_img = pil_img.crop((xmin, ymin, xmax, ymax))
            mf_image_list.append(bbox_img)

        dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.model.generate({"image": mf_img})
            mfr_res.extend(output["pred_str"])
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex_rm_whitespace(latex)
        return formula_list

    def batch_predict(
        self, images_mfd_res: list, images: list, batch_size: int = 64
    ) -> list:
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            pil_img = Image.fromarray(images[image_index])
            formula_list = []

            for xyxy, conf, cla in zip(
                mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            ):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": 13 + int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 2),
                    "latex": "",
                }
                formula_list.append(new_item)
                bbox_img = pil_img.crop((xmin, ymin, xmax, ymax))
                mf_image_list.append(bbox_img)

            images_formula_list.append(formula_list)
            backfill_list += formula_list

        dataset = MathDataset(mf_image_list, transform=self.mfr_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.model.generate({"image": mf_img})
            mfr_res.extend(output["pred_str"])
        for res, latex in zip(backfill_list, mfr_res):
            res["latex"] = latex_rm_whitespace(latex)
        return images_formula_list
