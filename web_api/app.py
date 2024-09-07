import copy
import json
import os
from tempfile import NamedTemporaryFile

import magic_pdf.model as model_config
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

model_config.__use_inside_model__ = True

app = FastAPI()

def json_md_dump(
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content,
):
    # Write model results to model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # Write intermediate results to middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # Write text content results to content_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    # Write results to .md file
    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )

@app.post("/pdf_parse", tags=["projects"], summary="Parse PDF file")
async def pdf_parse_main(
        pdf_file: UploadFile = File(...),
        parse_method: str = 'auto',
        model_json_path: str = None,
        is_json_md_dump: bool = True,
        output_dir: str = "output"
):
    """
    Execute the process of converting PDF to JSON and MD, outputting MD and JSON files to the specified directory
    :param pdf_file: The PDF file to be parsed
    :param parse_method: Parsing method, can be auto, ocr, or txt. Default is auto. If results are not satisfactory, try ocr
    :param model_json_path: Path to existing model data file. If empty, use built-in model. PDF and model_json must correspond
    :param is_json_md_dump: Whether to write parsed data to .json and .md files. Default is True. Different stages of data will be written to different .json files (3 in total), md content will be saved to .md file
    :param output_dir: Output directory for results. A folder named after the PDF file will be created to store all results
    """
    try:
        # Create a temporary file to store the uploaded PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await pdf_file.read())
            temp_pdf_path = temp_pdf.name

        pdf_name = os.path.basename(pdf_file.filename).split(".")[0]

        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            output_path = os.path.join(os.path.dirname(temp_pdf_path), pdf_name)

        output_image_path = os.path.join(output_path, 'images')

        # Get parent path of images for relative path in .md and content_list.json
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(temp_pdf_path, "rb").read()  # Read binary data of PDF file

        if model_json_path:
            # Read original JSON data of PDF file parsed by model, list type
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # Execute parsing steps
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

        # Choose parsing method
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("Unknown parse method, only auto, ocr, txt allowed")
            return JSONResponse(content={"error": "Invalid parse method"}, status_code=400)

        # Execute classification
        pipe.pipe_classify()

        # If no model data is provided, use built-in model for parsing
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # Parse
            else:
                logger.error("Need model list input")
                return JSONResponse(content={"error": "Model list input required"}, status_code=400)

        # Execute parsing
        pipe.pipe_parse()

        # Save results in text and md format
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)
        data = {"layout": copy.deepcopy(pipe.model_list), "info": pipe.pdf_mid_data, "content_list": content_list,'md_content':md_content}
        return JSONResponse(data, status_code=200)

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up the temporary file
        if 'temp_pdf_path' in locals():
            os.unlink(temp_pdf_path)

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8888)