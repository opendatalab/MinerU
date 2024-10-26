import os
import fitz
import torch
import base64
import litserve as ls
from uuid import uuid4
from fastapi import HTTPException
from filetype import guess_extension
from magic_pdf.tools.common import do_parse
from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        self.output_dir = output_dir

    def setup(self, device):
        if device.startswith('cuda'):
            os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
            if torch.cuda.device_count() > 1:
                raise RuntimeError("Remove any CUDA actions before setting 'CUDA_VISIBLE_DEVICES'.")

        model_manager = ModelSingleton()
        model_manager.get_model(True, False)
        model_manager.get_model(False, False)
        print(f'Model initialization complete on {device}!')

    def decode_request(self, request):
        file = request['file']
        file = self.to_pdf(file)
        opts = request.get('kwargs', {})
        opts.setdefault('debug_able', False)
        opts.setdefault('parse_method', 'auto')
        return file, opts

    def predict(self, inputs):
        try:
            do_parse(self.output_dir, pdf_name := str(uuid4()), inputs[0], [], **inputs[1])
            return pdf_name
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.clean_memory()

    def encode_response(self, response):
        return {'output_dir': response}

    def clean_memory(self):
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def to_pdf(self, file_base64):
        try:
            file_bytes = base64.b64decode(file_base64)
            file_ext = guess_extension(file_bytes)
            with fitz.open(stream=file_bytes, filetype=file_ext) as f:
                if f.is_pdf: return f.tobytes()
                return f.convert_to_pdf()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    server = ls.LitServer(
        MinerUAPI(output_dir='/tmp'),
        accelerator='cuda',
        devices='auto',
        workers_per_device=1,
        timeout=False
    )
    server.run(port=8000)
