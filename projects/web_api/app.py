import json
import os
from io import StringIO
from typing import Tuple, Union
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.dataset import PymuDocDataset
import magic_pdf.model as model_config
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.model.operators import InferenceResult
from magic_pdf.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from magic_pdf.pipe.operators import PipeResult
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.config_reader import get_s3_config, get_bucket_name
from magic_pdf.data.data_reader_writer import FileBasedDataWriter

model_config.__use_inside_model__ = True

app = FastAPI()



class MemoryDataWriter(DataWriter):
    def __init__(self):
        self.buffer = StringIO()
        
    def write(self, path: str, data: bytes) -> None:
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            self.buffer.write(data.decode('utf-8'))
            
    def write_string(self, path: str, data: str) -> None:
        self.buffer.write(data)
        
    def get_value(self) -> str:
        return self.buffer.getvalue()
        
    def close(self):
        self.buffer.close()

def init_writers(
    pdf_path: str = None,
    pdf_file: UploadFile = None,
    output_path: str = None,
    output_image_path: str = None,
) -> Tuple[Union[S3DataWriter, FileBasedDataWriter], Union[S3DataWriter, FileBasedDataWriter], bytes]:
    """
    Initialize writers based on path type
    
    Args:
        pdf_path: PDF file path (local path or S3 path)
        pdf_file: Uploaded PDF file object
        output_path: Output directory path
        output_image_path: Image output directory path
        
    Returns:
        Tuple[writer, image_writer, pdf_bytes]: Returns initialized writer tuple and PDF file content
    """
    if pdf_path:
        is_s3_path = pdf_path.startswith('s3://')
        if is_s3_path:
            bucket = get_bucket_name(pdf_path)
            ak, sk, endpoint = get_s3_config(bucket)
            
            writer = S3DataWriter(output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            image_writer = S3DataWriter(output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            # 临时创建reader读取文件内容
            temp_reader = S3DataReader("", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            pdf_bytes = temp_reader.read(pdf_path)
        else:
            writer = FileBasedDataWriter(output_path)
            image_writer = FileBasedDataWriter(output_image_path)
            os.makedirs(output_image_path, exist_ok=True)
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
    else:
        # 处理上传的文件
        pdf_bytes = pdf_file.file.read()
        writer = FileBasedDataWriter(output_path)
        image_writer = FileBasedDataWriter(output_image_path)
        os.makedirs(output_image_path, exist_ok=True)
        
    return writer, image_writer, pdf_bytes

def process_pdf(
    pdf_bytes: bytes,
    parse_method: str,
    image_writer: Union[S3DataWriter, FileBasedDataWriter]
) -> Tuple[InferenceResult, PipeResult]:
    """
    Process PDF file content
    
    Args:
        pdf_bytes: Binary content of PDF file
        parse_method: Parse method ('ocr', 'txt', 'auto')
        image_writer: Image writer
        
    Returns:
        Tuple[InferenceResult, PipeResult]: Returns inference result and pipeline result
    """
    ds = PymuDocDataset(pdf_bytes)
    infer_result : InferenceResult = None
    pipe_result : PipeResult = None

    if parse_method == 'ocr':
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    elif parse_method == 'txt':
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    else:  # auto
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
            
    return infer_result, pipe_result

@app.post('/pdf_parse', tags=['projects'], summary='Parse PDF files (supports local files and S3)')
async def pdf_parse(
    pdf_file: UploadFile = None,
    pdf_path: str = None,
    parse_method: str = 'auto',
    is_json_md_dump: bool = True,
    output_dir: str = 'output',
    return_layout: bool = False,
    return_info: bool = False,
    return_content_list: bool = False,
):
    try:
        if pdf_file is None and pdf_path is None:
            raise HTTPException(status_code=400, detail="Must provide either pdf_file or pdf_path")

        # Get PDF filename
        pdf_name = os.path.basename(pdf_path if pdf_path else pdf_file.filename).split('.')[0]
        output_path = f"{output_dir}/{pdf_name}"
        output_image_path = f"{output_path}/images"

        # Initialize readers/writers and get PDF content
        writer, image_writer, pdf_bytes = init_writers(
            pdf_path=pdf_path,
            pdf_file=pdf_file,
            output_path=output_path,
            output_image_path=output_image_path
        )
        
        # Process PDF
        infer_result, pipe_result = process_pdf(pdf_bytes, parse_method, image_writer)
        
        # Use MemoryDataWriter to get results
        content_list_writer = MemoryDataWriter()
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()
        
        # Use PipeResult's dump method to get data
        pipe_result.dump_content_list(content_list_writer, "", "images")
        pipe_result.dump_md(md_content_writer, "", "images")
        pipe_result.dump_middle_json(middle_json_writer, "")
        
        # Get content
        content_list = json.loads(content_list_writer.get_value())
        md_content = md_content_writer.get_value()
        middle_json = json.loads(middle_json_writer.get_value())
        model_json = infer_result.get_infer_res()

        # If results need to be saved
        if is_json_md_dump:
            writer.write_string(f"{pdf_name}_content_list.json", content_list_writer.get_value())
            writer.write_string(f"{pdf_name}.md", md_content)
            writer.write_string(f"{pdf_name}_middle.json", middle_json_writer.get_value())
            writer.write_string(f"{pdf_name}_model.json", json.dumps(model_json, indent=4, ensure_ascii=False))
            # Save visualization results
            pipe_result.draw_layout(os.path.join(output_path, f'{pdf_name}_layout.pdf'))
            pipe_result.draw_span(os.path.join(output_path, f'{pdf_name}_spans.pdf'))
            pipe_result.draw_line_sort(os.path.join(output_path, f'{pdf_name}_line_sort.pdf'))
            infer_result.draw_model(os.path.join(output_path, f'{pdf_name}_model.pdf'))
        
        # Build return data
        data = {}
        if return_layout:
            data['layout'] = model_json
        if return_info:
            data['info'] = middle_json
        if return_content_list:
            data['content_list'] = content_list
        data['md_content'] = md_content  # md_content is always returned
        
        # Clean up memory writers
        content_list_writer.close()
        md_content_writer.close()
        middle_json_writer.close()
        
        return JSONResponse(data, status_code=200)
        
    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)
