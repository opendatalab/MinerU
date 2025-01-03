import os
from shutil import rmtree
from datetime import datetime

from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset


async def pdf_parse_main(
        pdf_file,
        pdf_file_name: str = "noname",
        parse_method: str = "auto",
        is_save_output: bool = False,
        save_path: str = None,
):
    """

    Args:
        pdf_file: file path or file bytes
        pdf_file_name: Use the file name as the folder name for saving the result file.
        parse_method: auto/txt/ocr
        is_save_output: Whether to save the output file.
        save_path: Directory to save the output file. By default, the output file will be saved in the current workspace directory.

    Returns:
        md_content: markdown result without images
        list_content: list result
        txt_content: just text from list_content

    """
    local_md_dir = None

    try:
        # In the case that the pdf_file is the file path, read its byte data.
        if isinstance(pdf_file, str) and os.path.exists(pdf_file):
            file_reader = FileBasedDataReader()
            pdf_bytes = file_reader.read(pdf_file)
            pdf_file_name = os.path.splitext(os.path.basename(pdf_file))[0]
        elif isinstance(pdf_file, bytes):
            pdf_bytes = pdf_file
            pdf_file_name = pdf_file_name
        else:
            raise ValueError(
                "pdf_file must be a file path or byte data. \
                Please ensure the path is correct or provide the correct byte data."
            )

        # Create the output directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M%f")[:-4]
        if save_path:
            local_md_dir = os.path.join(save_path, f"{pdf_file_name}_{timestamp}")
        else:
            local_md_dir = os.path.join(os.getcwd(), f"{pdf_file_name}_{timestamp}")
        local_image_dir = os.path.join(local_md_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)

        md_writer = FileBasedDataWriter(local_md_dir)
        image_writer = FileBasedDataWriter(local_image_dir)

        ds = PymuDocDataset(pdf_bytes)

        if parse_method == "auto":
            parse_method = "ocr" if ds.classify() == SupportedPdfParseMethod.OCR else "txt"

        if parse_method == "txt":
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer, debug_mode=True)
        elif parse_method == "ocr":
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer, debug_mode=True)
        else:
            raise ValueError(f"Unsupported parsing method: {parse_method}, please choose [auto, txt, ocr].")

        # result
        md_content = pipe_result.dump_md(md_writer, f"{pdf_file_name}.md", local_image_dir)
        list_content = pipe_result.dump_content_list(md_writer, f"{pdf_file_name}_content_list.json", local_image_dir)
        # middle_content = pipe_result._pipe_res

        # get text
        txt_content = "\n".join(i.get("text", "") for i in list_content)

        return md_content, list_content, txt_content

    except Exception as e:
        raise Exception(f"An error occurred when processing the file: {e}")

    finally:
        if not is_save_output and local_md_dir and os.path.exists(local_md_dir):
            # delete the output directory
            rmtree(local_md_dir)


# test
if __name__ == "__main__":
    import asyncio

    pdf_path = "/home/yzz/pdf_file_test/Quality.pdf"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # asyncio.run(pdf_parse_main(pdf_path, parse_method="auto", is_save_output=True))
    asyncio.run(pdf_parse_main(pdf_bytes, parse_method="auto", is_save_output=True))
