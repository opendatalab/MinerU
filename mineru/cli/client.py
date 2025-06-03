# Copyright (c) Opendatalab. All rights reserved.
import os
import click
from pathlib import Path
from loguru import logger
from ..version import __version__
from .common import do_parse, read_fn, pdf_suffixes, image_suffixes


@click.command()
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
@click.option(
    '-p',
    '--path',
    'input_path',
    type=click.Path(exists=True),
    required=True,
    help='local filepath or directory. support pdf, png, jpg, jpeg files',
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
    '-b',
    '--backend',
    'backend',
    type=click.Choice(['pipeline', 'vlm-huggingface', 'vlm-sglang-engine', 'vlm-sglang-client']),
    help="""the backend for parsing pdf:
    pipeline: More general.
    vlm-huggingface: More general.
    vlm-sglang-engine: Faster(engine).
    vlm-sglang-client: Faster(client).
    without method specified, huggingface will be used by default.""",
    default='pipeline',
)
@click.option(
    '-l',
    '--lang',
    'lang',
    type=click.Choice(['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']),
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
    Without languages specified, 'ch' will be used by default.
    """,
    default='ch',
)
@click.option(
    '-u',
    '--url',
    'server_url',
    type=str,
    help="""
    When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """,
    default=None,
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

def main(input_path, output_dir, backend, lang, server_url, start_page_id, end_page_id):
    os.makedirs(output_dir, exist_ok=True)

    def parse_doc(path_list: list[Path]):
        try:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
            do_parse(output_dir, file_name_list, pdf_bytes_list, lang_list, backend, server_url,
                         start_page_id=start_page_id, end_page_id=end_page_id)
        except Exception as e:
            logger.exception(e)

    if os.path.isdir(input_path):
        doc_path_list = []
        for doc_path in Path(input_path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)
        parse_doc(doc_path_list)
    else:
        parse_doc([Path(input_path)])

if __name__ == '__main__':
    main()
