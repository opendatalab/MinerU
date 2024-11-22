
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset


def test_pymudataset():
    with open('tests/unittest/test_data/assets/pdfs/test_01.pdf', 'rb') as f:
        bits = f.read()
    datasets = PymuDocDataset(bits)
    assert len(datasets) > 0
    assert datasets.get_page(0).get_page_info().h > 100


def test_imagedataset():
    with open('tests/unittest/test_data/assets/pngs/test_01.png', 'rb') as f:
        bits = f.read()
    datasets = ImageDataset(bits)
    assert len(datasets) == 1
    assert datasets.get_page(0).get_page_info().w > 100
