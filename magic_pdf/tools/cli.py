import os
import shutil
import tempfile
from pathlib import Path

import click
import fitz
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.data.batch_build_dataset import batch_build_dataset
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.version import __version__
from magic_pdf.tools.common import batch_do_parse, do_parse, parse_pdf_methods
from magic_pdf.utils.office_to_pdf import convert_file_to_pdf

pdf_suffixes = ['.pdf']
ms_office_suffixes = ['.ppt', '.pptx', '.doc', '.docx']
image_suffixes = ['.png', '.jpeg', '.jpg']


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
    help='local filepath or directory. support PDF, PPT, PPTX, DOC, DOCX, PNG, JPG files',
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
    '-l',
    '--lang',
    'lang',
    type=str,
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
    You should input "Abbreviation" with language form url:
    https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/blog/multi_languages.html#5-support-languages-and-abbreviations
    """,
    default=None,
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
def cli(path, output_dir, method, lang, debug_able, start_page_id, end_page_id):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    def read_fn(path: Path):
        if path.suffix in ms_office_suffixes:
            convert_file_to_pdf(str(path), temp_dir)
            fn = os.path.join(temp_dir, f'{path.stem}.pdf')
        elif path.suffix in image_suffixes:
            with open(str(path), 'rb') as f:
                bits = f.read()
            pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
            fn = os.path.join(temp_dir, f'{path.stem}.pdf')
            with open(fn, 'wb') as f:
                f.write(pdf_bytes)
        elif path.suffix in pdf_suffixes:
            fn = str(path)
        else:
            raise Exception(f'Unknown file suffix: {path.suffix}')

        disk_rw = FileBasedDataReader(os.path.dirname(fn))
        return disk_rw.read(os.path.basename(fn))

    def parse_doc(doc_path: Path, dataset: Dataset | None = None):
        try:
            file_name = str(Path(doc_path).stem)
            if dataset is None:
                pdf_data_or_dataset = read_fn(doc_path)
            else:
                pdf_data_or_dataset = dataset
            do_parse(
                output_dir,
                file_name,
                pdf_data_or_dataset,
                [],
                method,
                debug_able,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                lang=lang
            )

        except Exception as e:
            logger.exception(e)

    if os.path.isdir(path):
        doc_paths = []
        for doc_path in Path(path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes + ms_office_suffixes:
                if doc_path.suffix in ms_office_suffixes:
                    convert_file_to_pdf(str(doc_path), temp_dir)
                    doc_path = Path(os.path.join(temp_dir, f'{doc_path.stem}.pdf'))
                elif doc_path.suffix in image_suffixes:
                    with open(str(doc_path), 'rb') as f:
                        bits = f.read()
                        pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
                    fn = os.path.join(temp_dir, f'{doc_path.stem}.pdf')
                    with open(fn, 'wb') as f:
                        f.write(pdf_bytes)
                    doc_path = Path(fn)
                doc_paths.append(doc_path)
        datasets = batch_build_dataset(doc_paths, 4, lang)
        batch_do_parse(output_dir, [str(doc_path.stem) for doc_path in doc_paths], datasets, method, debug_able, lang=lang)
    else:
        parse_doc(Path(path))

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    cli()
