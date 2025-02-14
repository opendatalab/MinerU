import os
import shutil

from magic_pdf.data.data_reader_writer import (FileBasedDataReader,
                                               FileBasedDataWriter)


def test_filebased_reader_writer():

    unitest_dir = '/tmp/magic_pdf/unittest/data/filebased_reader_writer'
    sub_dir = os.path.join(unitest_dir, 'sub')
    abs_fn = os.path.join(unitest_dir, 'abspath.txt')

    os.makedirs(sub_dir, exist_ok=True)

    writer = FileBasedDataWriter(sub_dir)
    reader = FileBasedDataReader(sub_dir)

    writer.write('test.txt', b'hello world')
    assert reader.read('test.txt') == b'hello world'

    writer.write(abs_fn, b'hello world')
    assert reader.read(abs_fn) == b'hello world'
    shutil.rmtree(unitest_dir)
