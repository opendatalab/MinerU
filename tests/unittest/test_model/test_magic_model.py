import json

from magic_pdf.data.read_api import read_local_pdfs
from magic_pdf.model.magic_model import MagicModel


def test_magic_model_image_v2():
    datasets = read_local_pdfs('tests/unittest/test_model/assets/test_01.pdf')
    with open('tests/unittest/test_model/assets/test_01.model.json') as f:
        model_json = json.load(f)

    magic_model = MagicModel(model_json, datasets[0])

    imgs = magic_model.get_imgs_v2(0)
    print(imgs)

    tables = magic_model.get_tables_v2(0)
    print(tables)


def test_magic_model_table_v2():
    datasets = read_local_pdfs('tests/unittest/test_model/assets/test_02.pdf')
    with open('tests/unittest/test_model/assets/test_02.model.json') as f:
        model_json = json.load(f)

    magic_model = MagicModel(model_json, datasets[0])
    tables = magic_model.get_tables_v2(5)
    print(tables)

    tables = magic_model.get_tables_v2(8)
    print(tables)
