import os
from pathlib import Path

import click
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.libs.version import __version__
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, parse_pdf_methods


@click.command()
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
@click.option(
    '-p',
    '--path',
    'path',
    type=click.Path(exists=True),
    required=True,
    help='local pdf filepath or directory',
)
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(),
    required=True,
    help='output local directory',
)
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help="""the method for parsing pdf.
ocr: using ocr technique to extract information from pdf.
txt: suitable for the text-based pdf only and outperform ocr.
auto: automatically choose the best method for parsing pdf from ocr and txt.
without method specified, auto will be used by default.""",
    default='auto',
)
@click.option(
    '-d',
    '--debug',
    'debug_able',
    type=bool,
    help='Enables detailed debugging information during the execution of the CLI commands.',
    default=False,
)
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='The starting page for PDF parsing, beginning from 0.',
    default=0,
)
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='The ending page for PDF parsing, beginning from 0.',
    default=None,
)
def cli(path, output_dir, method, debug_able, start_page_id, end_page_id):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = 'full'
    os.makedirs(output_dir, exist_ok=True)

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
                debug_able,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

        except Exception as e:
            logger.exception(e)

    if os.path.isdir(path):
        for doc_path in Path(path).glob('*.pdf'):
            parse_doc(doc_path)
    else:
        parse_doc(path)


if __name__ == '__main__':
    cli()
