import os
import click
from loguru import logger
from pathlib import Path

from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
import magic_pdf.model as model_config
from magic_pdf.tools.common import parse_pdf_methods, do_parse
from magic_pdf.libs.version import __version__


@click.command()
@click.version_option(__version__, "--version", "-v", help="display the version and exit")
@click.option(
    "-p",
    "--path",
    "path",
    type=click.Path(exists=True),
    required=True,
    help="local pdf filepath or directory",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=str,
    help="output local directory",
    default="",
)
@click.option(
    "-m",
    "--method",
    "method",
    type=parse_pdf_methods,
    help="""the method for parsing pdf. 
ocr: using ocr technique to extract information from pdf.
txt: suitable for the text-based pdf only and outperform ocr.
auto: automatically choose the best method for parsing pdf from ocr and txt.
without method specified, auto will be used by default.""",
    default="auto",
)
def cli(path, output_dir, method):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = "full"
    if output_dir == "":
        if os.path.isdir(path):
            output_dir = os.path.join(path, "output")
        else:
            output_dir = os.path.join(os.path.dirname(path), "output")

    def read_fn(path):
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    def parse_doc(doc_path: str):
        try:
            file_name = str(Path(doc_path).stem)
            pdf_data = read_fn(doc_path)
            do_parse(
                output_dir,
                file_name,
                pdf_data,
                [],
                method,
            )

        except Exception as e:
            logger.exception(e)

    if os.path.isdir(path):
        for doc_path in Path(path).glob("*.pdf"):
            parse_doc(doc_path)
    else:
        parse_doc(path)


if __name__ == "__main__":
    cli()
