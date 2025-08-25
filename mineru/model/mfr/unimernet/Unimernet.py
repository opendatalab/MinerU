import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        if _device_.startswith("mps") or _device_.startswith("npu"):
            self.model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
        else:
            self.model = UnimernetModel.from_pretrained(weight_dir)
        self.device = _device_
        self.model.to(_device_)
        if not _device_.startswith("cpu"):
            self.model = self.model.to(dtype=torch.float16)
        self.model.eval()

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
            bbox_img = image[ymin:ymax, xmin:xmax]
            mf_image_list.append(bbox_img)

        dataset = MathDataset(mf_image_list, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(dtype=self.model.dtype)
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.model.generate({"image": mf_img})
            mfr_res.extend(output["fixed_str"])
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex
        return formula_list

    def batch_predict(self, images_mfd_res: list, images: list, batch_size: int = 64) -> list:
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []  # Store (area, original_index, image) tuples

        # Collect images with their original indices
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            image = images[image_index]
            formula_list = []

            for idx, (xyxy, conf, cla) in enumerate(zip(
                    mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            )):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": 13 + int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 2),
                    "latex": "",
                }
                formula_list.append(new_item)
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)

            images_formula_list.append(formula_list)
            backfill_list += formula_list

        # Stable sort by area
        image_info.sort(key=lambda x: x[0])  # sort by area
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]

        # Create mapping for results
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # Create dataset with sorted images
        dataset = MathDataset(sorted_images, transform=self.model.transform)

        # 如果batch_size > len(sorted_images)，则设置为不超过len(sorted_images)的2的幂
        batch_size = min(batch_size, max(1, 2 ** (len(sorted_images).bit_length() - 1))) if sorted_images else 1

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        # Process batches and store results
        mfr_res = []
        # for mf_img in dataloader:

        with tqdm(total=len(sorted_images), desc="MFR Predict") as pbar:
            for index, mf_img in enumerate(dataloader):
                mf_img = mf_img.to(dtype=self.model.dtype)
                mf_img = mf_img.to(self.device)
                with torch.no_grad():
                    output = self.model.generate({"image": mf_img}, batch_size=batch_size)
                mfr_res.extend(output["fixed_str"])

                # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                pbar.update(current_batch_size)

        # Restore original order
        unsorted_results = [""] * len(mfr_res)
        for new_idx, latex in enumerate(mfr_res):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        # Fill results back
        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list
