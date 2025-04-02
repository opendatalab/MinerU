import os
from pathlib import Path
from magic_pdf.data.batch_build_dataset import batch_build_dataset
from magic_pdf.tools.common import batch_do_parse


def batch(pdf_dir, output_dir, method, lang):
    os.makedirs(output_dir, exist_ok=True)
    doc_paths = []
    for doc_path in Path(pdf_dir).glob('*'):
        if doc_path.suffix == '.pdf':
            doc_paths.append(doc_path)

    # build dataset with 2 workers
    datasets = batch_build_dataset(doc_paths, 4, lang)

    # os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = "200"  # every 200 pages will be parsed in one batch
    batch_do_parse(output_dir, [str(doc_path.stem) for doc_path in doc_paths], datasets, method)


if __name__ == '__main__':
    batch("pdfs", "output", "auto", "")

