import os
import uuid
import shutil
import tempfile
import gc
import fitz
import torch
import base64
import filetype
import litserve as ls
from pathlib import Path
from fastapi import HTTPException


class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        self.output_dir = Path(output_dir)

    def setup(self, device):
        if device.startswith('cuda'):
            os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
            if torch.cuda.device_count() > 1:
                raise RuntimeError("Remove any CUDA actions before setting 'CUDA_VISIBLE_DEVICES'.")

        from magic_pdf.tools.cli import do_parse, convert_file_to_pdf
        from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton

        self.do_parse = do_parse
        self.convert_file_to_pdf = convert_file_to_pdf

        model_manager = ModelSingleton()
        model_manager.get_model(True, False)
        model_manager.get_model(False, False)
        print(f'Model initialization complete on {device}!')

    def decode_request(self, request):
        file = request['file']
        file = self.cvt2pdf(file)
        opts = request.get('kwargs', {})
        opts.setdefault('debug_able', False)
        opts.setdefault('parse_method', 'auto')
        return file, opts

    def predict(self, inputs):
        try:
            pdf_name = str(uuid.uuid4())
            output_dir = self.output_dir.joinpath(pdf_name)
            self.do_parse(self.output_dir, pdf_name, inputs[0], [], **inputs[1])
            return output_dir
        except Exception as e:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.clean_memory()

    def encode_response(self, response):
        return {'output_dir': response}

    def clean_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def cvt2pdf(self, file_base64):
        try:
            temp_dir = Path(tempfile.mkdtemp())
            temp_file = temp_dir.joinpath('tmpfile')
            file_bytes = base64.b64decode(file_base64)
            file_ext = filetype.guess_extension(file_bytes)

            if file_ext in ['pdf', 'jpg', 'png', 'doc', 'docx', 'ppt', 'pptx']:
                if file_ext == 'pdf':
                    return file_bytes
                elif file_ext in ['jpg', 'png']:
                    with fitz.open(stream=file_bytes, filetype=file_ext) as f:
                        return f.convert_to_pdf()
                else:
                    temp_file.write_bytes(file_bytes)
                    self.convert_file_to_pdf(temp_file, temp_dir)
                    return temp_file.with_suffix('.pdf').read_bytes()
            else:
                raise Exception('Unsupported file format')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    server = ls.LitServer(
        MinerUAPI(output_dir='/tmp'),
        accelerator='cuda',
        devices='auto',
        workers_per_device=1,
        timeout=False
    )
    server.run(port=8000)
