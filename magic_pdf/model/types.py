
import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import Dataset


class InferenceResult:
    def __init__(self, inference_results: list, dataset: Dataset):
        self._infer_res = inference_results
        self._dataset = dataset

    def draw_model(self, writer: FileBasedDataWriter, dump_file_path: str):
        dir_name = os.path.dirname(dump_file_path)
        if dir_name not in ('', '.', '..'):
            os.makedirs(dir_name, exist_ok=True)

    def get_infer_res(self):
        return self._infer_res
