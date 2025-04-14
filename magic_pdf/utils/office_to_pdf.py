import os
import subprocess
import platform
from pathlib import Path


class ConvertToPdfError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


# Chinese font list
REQUIRED_CHS_FONTS = ['SimSun', 'Microsoft YaHei', 'Noto Sans CJK SC']


def check_fonts_installed():
    """Check if required Chinese fonts are installed."""
    system_type = platform.system()

    if system_type == 'Windows':
        # Windows: check fonts via registry or system font folder
        font_dir = Path("C:/Windows/Fonts")
        installed_fonts = [f.name for f in font_dir.glob("*.ttf")]
        if any(font for font in REQUIRED_CHS_FONTS if any(font in f for f in installed_fonts)):
            return True
        raise EnvironmentError(
            f"Missing Chinese font. Please install at least one of: {', '.join(REQUIRED_CHS_FONTS)}"
        )
    else:
        # Linux/macOS: use fc-list
        try:
            output = subprocess.check_output(['fc-list', ':lang=zh'], encoding='utf-8')
            for font in REQUIRED_CHS_FONTS:
                if font in output:
                    return True
            raise EnvironmentError(
                f"Missing Chinese font. Please install at least one of: {', '.join(REQUIRED_CHS_FONTS)}"
            )
        except Exception as e:
            raise EnvironmentError(f"Font detection failed. Please install 'fontconfig' and fonts: {str(e)}")


def get_soffice_command():
    """Return the path to LibreOffice's soffice executable depending on the platform."""
    if platform.system() == 'Windows':
        possible_paths = [
            Path("C:/Program Files/LibreOffice/program/soffice.exe"),
            Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe")
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise ConvertToPdfError(
            "LibreOffice not found. Please install LibreOffice and ensure soffice.exe is located in a standard path."
        )
    else:
        return 'soffice'  # Assume it's in PATH on Linux/macOS


def convert_file_to_pdf(input_path, output_dir):
    """Convert a single document (ppt, doc, etc.) to PDF."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    check_fonts_installed()

    soffice_cmd = get_soffice_command()

    cmd = [
        soffice_cmd,
        '--headless',
        '--norestore',
        '--invisible',
        '--convert-to', 'pdf',
        '--outdir', str(output_dir),
        str(input_path)
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        raise ConvertToPdfError(f"LibreOffice convert failed: {process.stderr.decode()}")
