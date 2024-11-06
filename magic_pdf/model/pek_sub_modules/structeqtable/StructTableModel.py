import re

import torch
from struct_eqtable import build_model


class StructTableModel:
    def __init__(self, model_path, max_new_tokens=1024, max_time=60):
        # init
        assert torch.cuda.is_available(), "CUDA must be available for StructEqTable model."
        self.model = build_model(
            model_ckpt=model_path,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            lmdeploy=False,
            flash_attn=False,
            batch_size=1,
        ).cuda()
        self.default_format = "html"

    def predict(self, images, output_format=None, **kwargs):

        if output_format is None:
            output_format = self.default_format
        else:
            if output_format not in ['latex', 'markdown', 'html']:
                raise ValueError(f"Output format {output_format} is not supported.")

        results = self.model(
            images, output_format=output_format
        )

        if output_format == "html":
            results = [self.minify_html(html) for html in results]

        return results

    def minify_html(self, html):
        # 移除多余的空白字符
        html = re.sub(r'\s+', ' ', html)
        # 移除行尾的空白字符
        html = re.sub(r'\s*>\s*', '>', html)
        # 移除标签前的空白字符
        html = re.sub(r'\s*<\s*', '<', html)
        return html.strip()