import torch
from struct_eqtable import build_model

from magic_pdf.model.sub_modules.table.table_utils import minify_html


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
            results = [minify_html(html) for html in results]

        return results

