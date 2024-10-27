import os

from loguru import logger
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter


try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_name = "demo1"
    pdf_path = os.path.join(current_script_dir, f"{demo_name}.pdf")
    pdf_bytes = open(pdf_path, "rb").read()
    jso_useful_key = {"_pdf_type": "", "model_list": []}
    local_image_dir = os.path.join(current_script_dir, 'images')
    image_dir = str(os.path.basename(local_image_dir))
    image_writer = DiskReaderWriter(local_image_dir)
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
    with open(f"{demo_name}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
except Exception as e:
    logger.exception(e)