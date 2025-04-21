import json
import os
import tempfile
import shutil
from pathlib import Path

from magic_pdf.config.exceptions import EmptyData, InvalidParams
from magic_pdf.data.data_reader_writer import (FileBasedDataReader,
                                               MultiBucketS3DataReader)
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.utils.office_to_pdf import convert_file_to_pdf, ConvertToPdfError

def read_jsonl(
    s3_path_or_local: str, s3_client: MultiBucketS3DataReader | None = None
) -> list[PymuDocDataset]:
    """Read the jsonl file and return the list of PymuDocDataset.

    Args:
        s3_path_or_local (str): local file or s3 path
        s3_client (MultiBucketS3DataReader | None, optional): s3 client that support multiple bucket. Defaults to None.

    Raises:
        InvalidParams: if s3_path_or_local is s3 path but s3_client is not provided.
        EmptyData: if no pdf file location is provided in some line of jsonl file.
        InvalidParams: if the file location is s3 path but s3_client is not provided

    Returns:
        list[PymuDocDataset]: each line in the jsonl file will be converted to a PymuDocDataset
    """
    bits_arr = []
    if s3_path_or_local.startswith('s3://'):
        if s3_client is None:
            raise InvalidParams('s3_client is required when s3_path is provided')
        jsonl_bits = s3_client.read(s3_path_or_local)
    else:
        jsonl_bits = FileBasedDataReader('').read(s3_path_or_local)
    jsonl_d = [
        json.loads(line) for line in jsonl_bits.decode().split('\n') if line.strip()
    ]
    for d in jsonl_d:
        pdf_path = d.get('file_location', '') or d.get('path', '')
        if len(pdf_path) == 0:
            raise EmptyData('pdf file location is empty')
        if pdf_path.startswith('s3://'):
            if s3_client is None:
                raise InvalidParams('s3_client is required when s3_path is provided')
            bits_arr.append(s3_client.read(pdf_path))
        else:
            bits_arr.append(FileBasedDataReader('').read(pdf_path))
    return [PymuDocDataset(bits) for bits in bits_arr]


def read_local_pdfs(path: str) -> list[PymuDocDataset]:
    """Read pdf from path or directory.

    Args:
        path (str): pdf file path or directory that contains pdf files

    Returns:
        list[PymuDocDataset]: each pdf file will converted to a PymuDocDataset
    """
    if os.path.isdir(path):
        reader = FileBasedDataReader()
        ret = []
        for root, _, files in os.walk(path):
            for file in files:
                suffix = file.split('.')
                if suffix[-1] == 'pdf':
                    ret.append( PymuDocDataset(reader.read(os.path.join(root, file))))
        return ret
    else:
        reader = FileBasedDataReader()
        bits = reader.read(path)
        return [PymuDocDataset(bits)]

def read_local_office(path: str) -> list[PymuDocDataset]:
    """Read ms-office file (ppt, pptx, doc, docx) from path or directory.

    Args:
        path (str): ms-office file or directory that contains ms-office files

    Returns:
        list[PymuDocDataset]: each ms-office file will converted to a PymuDocDataset
        
    Raises:
        ConvertToPdfError: Failed to convert ms-office file to pdf via libreoffice
        FileNotFoundError: File not Found
        Exception: Unknown Exception raised
    """
    suffixes = ['.ppt', '.pptx', '.doc', '.docx']
    fns = []
    ret = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                suffix = Path(file).suffix
                if suffix in suffixes:
                    fns.append((os.path.join(root, file)))
    else:
        fns.append(path)
        
    reader = FileBasedDataReader()
    temp_dir = tempfile.mkdtemp()
    for fn in fns:
        try:
            convert_file_to_pdf(fn, temp_dir)
        except ConvertToPdfError as e:
            raise e
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise e
        fn_path = Path(fn)
        pdf_fn = f"{temp_dir}/{fn_path.stem}.pdf"
        ret.append(PymuDocDataset(reader.read(pdf_fn)))
    shutil.rmtree(temp_dir)
    return ret

def read_local_images(path: str, suffixes: list[str]=['.png', '.jpg', '.jpeg']) -> list[ImageDataset]:
    """Read images from path or directory.

    Args:
        path (str): image file path or directory that contains image files
        suffixes (list[str]): the suffixes of the image files used to filter the files. Example: ['.jpg', '.png']

    Returns:
        list[ImageDataset]: each image file will converted to a ImageDataset
    """
    if os.path.isdir(path):
        imgs_bits = []
        s_suffixes = set(suffixes)
        reader = FileBasedDataReader()
        for root, _, files in os.walk(path):
            for file in files:
                suffix = Path(file).suffix
                if suffix in s_suffixes:
                    imgs_bits.append(reader.read(os.path.join(root, file)))
        return [ImageDataset(bits) for bits in imgs_bits]
    else:
        reader = FileBasedDataReader()
        bits = reader.read(path)
        return [ImageDataset(bits)]
