import json
import os
import shutil
import tempfile

from magic_pdf.integrations.rag.api import DataReader, RagDocumentReader
from magic_pdf.integrations.rag.type import CategoryType
from magic_pdf.integrations.rag.utils import \
    convert_middle_json_to_layout_elements


def test_rag_document_reader():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    # test
    with open('tests/unittest/test_integrations/test_rag/assets/middle.json') as f:
        json_data = json.load(f)
    res = convert_middle_json_to_layout_elements(json_data, temp_output_dir)

    doc = RagDocumentReader(res)
    assert len(list(iter(doc))) == 1

    page = list(iter(doc))[0]
    assert len(list(iter(page))) >= 10
    assert len(page.get_rel_map()) >= 3

    item = list(iter(page))[0]
    assert item.category_type == CategoryType.text

    # teardown
    shutil.rmtree(temp_output_dir)


def test_data_reader():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    # test
    data_reader = DataReader('tests/unittest/test_integrations/test_rag/assets', 'ocr',
                             temp_output_dir)

    assert data_reader.get_documents_count() == 2
    for idx in range(data_reader.get_documents_count()):
        document = data_reader.get_document_result(idx)
        assert document is not None

    # teardown
    shutil.rmtree(temp_output_dir)
