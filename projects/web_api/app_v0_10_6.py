from typing import Literal

from fastapi import FastAPI, UploadFile, HTTPException

from pdf_parse_main import pdf_parse_main

app = FastAPI()

parse_allowed_methods = Literal["auto", "txt", "ocr"]

# Here to set the default output path of the parsing result file
PDF_OUTPUT_PATH = "/tmp/output"


@app.post("/pdf-parse")
async def pdf_parse(
    file: UploadFile,
    parse_method: parse_allowed_methods = "auto",
    is_output: bool = False,
    save_path: str = PDF_OUTPUT_PATH,
):
    """
    is_output: Whether to keep the parsing result file
    save_path: Parse result file save path
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="File type error")

    pdf_bytes = await file.read()
    pdf_file_name = file.filename.split(".")[0]

    try:
        md_content, list_content, txt_content = await pdf_parse_main(
            pdf_bytes, pdf_file_name, parse_method, is_output, save_path
        )

        return {"md_data": md_content, "content_list_data": list_content, "txt_data": txt_content}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
