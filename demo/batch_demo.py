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


def batch(pdf_dir, output_dir, method, lang):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = 'full'
    os.makedirs(output_dir, exist_ok=True)

    doc_paths = []
    for doc_path in Path(pdf_dir).glob('*'):
        if doc_path.suffix == '.pdf':
            doc_paths.append(doc_path)

    # build dataset with 2 workers
    datasets = batch_build_dataset(doc_paths, 2, lang)

    os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = "10" # every 10 pages will be parsed in one batch
    batch_do_parse(output_dir, [str(doc_path.stem) for doc_path in doc_paths], datasets, method, True)


if __name__ == '__main__':
    batch("batch_data", "output", "ocr", "en")

