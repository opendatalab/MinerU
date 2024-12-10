import os
import subprocess
from pathlib import Path


class ConvertToPdfError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


def convert_file_to_pdf(input_path, output_dir):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'soffice',
        '--headless',
        '--convert-to', 'pdf',
        '--outdir', str(output_dir),
        str(input_path)
    ]
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if process.returncode != 0:
        raise ConvertToPdfError(process.stderr.decode())
