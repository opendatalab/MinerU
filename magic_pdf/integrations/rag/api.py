import os
from pathlib import Path

from loguru import logger

from magic_pdf.integrations.rag.type import (ElementRelation, LayoutElements,
                                             Node)
from magic_pdf.integrations.rag.utils import inference


class RagPageReader:

    def __init__(self, pagedata: LayoutElements):
        self.o = [
            Node(
                category_type=v.category_type,
                text=v.text,
                image_path=v.image_path,
                anno_id=v.anno_id,
                latex=v.latex,
                html=v.html,
            ) for v in pagedata.layout_dets
        ]

        self.pagedata = pagedata

    def __iter__(self):
        return iter(self.o)

    def get_rel_map(self) -> list[ElementRelation]:
        return self.pagedata.extra.element_relation


class RagDocumentReader:

    def __init__(self, ragdata: list[LayoutElements]):
        self.o = [RagPageReader(v) for v in ragdata]

    def __iter__(self):
        return iter(self.o)


class DataReader:

    def __init__(self, path_or_directory: str, method: str, output_dir: str):
        self.path_or_directory = path_or_directory
        self.method = method
        self.output_dir = output_dir
        self.pdfs = []
        if os.path.isdir(path_or_directory):
            for doc_path in Path(path_or_directory).glob('*.pdf'):
                self.pdfs.append(doc_path)
        else:
            assert path_or_directory.endswith('.pdf')
            self.pdfs.append(Path(path_or_directory))

    def get_documents_count(self) -> int:
        """Returns the number of documents in the directory."""
        return len(self.pdfs)

    def get_document_result(self, idx: int) -> RagDocumentReader | None:
        """
        Args:
            idx (int): the index of documents under the
                directory path_or_directory

        Returns:
            RagDocumentReader | None: RagDocumentReader is an iterable object,
            more details @RagDocumentReader
        """
        if idx >= self.get_documents_count() or idx < 0:
            logger.error(f'invalid idx: {idx}')
            return None
        res = inference(str(self.pdfs[idx]), self.output_dir, self.method)
        if res is None:
            logger.warning(f'failed to inference pdf {self.pdfs[idx]}')
            return None
        return RagDocumentReader(res)

    def get_document_filename(self, idx: int) -> Path:
        """get the filename of the document."""
        return self.pdfs[idx]
