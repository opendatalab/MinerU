import os
from fastapi import FastAPI, Request, UploadFile, File
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import aiofiles
app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    try:
        # 使用 aiofiles 异步打开文件并写入数据
        async with aiofiles.open(file.filename, 'wb') as f:
            pdf_bytes = await file.read()
            # read bytes
            # proc
            ## Create Dataset Instance
            ds = PymuDocDataset(pdf_bytes)

            ## inference
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                ## pipeline
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                ## pipeline
                pipe_result = infer_result.pipe_txt_mode(image_writer)


            ### get markdown content
            md_content = pipe_result.get_markdown(image_dir)


            return {"message": "File processed successfully", "md_content": md_content}
        return {"filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
