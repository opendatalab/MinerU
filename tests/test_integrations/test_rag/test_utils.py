import json
import os
import shutil
import tempfile

from magic_pdf.integrations.rag.type import CategoryType
from magic_pdf.integrations.rag.utils import (
    convert_middle_json_to_layout_elements, inference)


def test_convert_middle_json_to_layout_elements():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    # test
    with open('tests/test_integrations/test_rag/assets/middle.json') as f:
        json_data = json.load(f)
    res = convert_middle_json_to_layout_elements(json_data, temp_output_dir)

    assert len(res) == 1
    assert len(res[0].layout_dets) == 10
    assert res[0].layout_dets[0].anno_id == 0
    assert res[0].layout_dets[0].category_type == CategoryType.text
    assert len(res[0].extra.element_relation) == 3

    # teardown
    shutil.rmtree(temp_output_dir)


def test_inference():

    asset_dir = 'tests/test_integrations/test_rag/assets'
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'
    os.makedirs(unitest_dir, exist_ok=True)
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    # test
    res = inference(
        asset_dir + '/one_page_with_table_image.pdf',
        temp_output_dir,
        'ocr',
    )

    assert res is not None
    assert len(res) == 1
    assert len(res[0].layout_dets) == 10
    assert res[0].layout_dets[0].anno_id == 0
    assert res[0].layout_dets[0].category_type == CategoryType.text
    assert len(res[0].extra.element_relation) == 3

    # teardown
    shutil.rmtree(temp_output_dir)
