from pathlib import Path
from setuptools import setup, find_packages
from magic_pdf.libs.version import __version__


def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    requires = []

    for line in lines:
        if "http" in line:
            pkg_name_without_url = line.split('@')[0].strip()
            requires.append(pkg_name_without_url)
        else:
            requires.append(line)

    return requires


if __name__ == '__main__':
    with Path(Path(__file__).parent,
              'README.md').open(encoding='utf-8') as file:
        long_description = file.read()
    setup(
        name="magic_pdf",  # 项目名
        version=__version__,  # 自动从tag中获取版本号
        packages=find_packages() + ["magic_pdf.resources"] + ["magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorchocr.utils.resources"],  # 包含所有的包
        package_data={
            "magic_pdf.resources": ["**"],  # 包含magic_pdf.resources目录下的所有文件
            "magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorchocr.utils.resources": ["**"],  # pytorchocr.resources目录下的所有文件
        },
        install_requires=parse_requirements('requirements.txt'),  # 项目依赖的第三方库
        extras_require={
            "lite": [
                    "paddleocr==2.7.3",
                    "paddlepaddle==3.0.0b1;platform_system=='Linux'",
                    "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
            ],
            "full": [
                     "matplotlib<=3.9.0;platform_system=='Windows'",  # 3.9.1及之后不提供windows的预编译包，避免一些没有编译环境的windows设备安装失败
                     "matplotlib>=3.10;platform_system=='Linux' or platform_system=='Darwin'",  # linux 和 macos 不应限制matplotlib的最高版本，以避免无法更新导致的一些bug
                     "ultralytics>=8.3.48",  # yolov8,公式检测
                     "doclayout_yolo==0.0.2b1",  # doclayout_yolo
                     "dill>=0.3.9,<1",  # doclayout_yolo
                     "rapid_table>=1.0.5,<2.0.0",  # rapid_table
                     "PyYAML>=6.0.2,<7",  # yaml
                     "ftfy>=6.3.1,<7", # unimernet_hf
                     "openai>=1.70.0,<2",  # openai SDK
                     "shapely>=2.0.7,<3",  # imgaug-paddleocr2pytorch
                     "pyclipper>=1.3.0,<2",  # paddleocr2pytorch
                     "omegaconf>=2.3.0,<3",  # paddleocr2pytorch
            ],
            "full_old_linux":[
                    "matplotlib>=3.10",
                    "ultralytics>=8.3.48",  # yolov8,公式检测
                    "doclayout_yolo==0.0.2b1",  # doclayout_yolo
                    "dill>=0.3.9,<1",  # doclayout_yolo
                    "PyYAML>=6.0.2,<7",  # yaml
                    "ftfy>=6.3.1,<7",  # unimernet_hf
                    "openai>=1.70.0,<2",  # openai SDK
                    "shapely>=2.0.7,<3",  # imgaug-paddleocr2pytorch
                    "pyclipper>=1.3.0,<2",  # paddleocr2pytorch
                    "omegaconf>=2.3.0,<3",  # paddleocr2pytorch
                    "albumentations<=1.4.20", # 1.4.21引入的simsimd不支持2019年及更早的linux系统
                    "rapid_table==1.0.3",  # rapid_table新版本依赖的onnxruntime不支持2019年及更早的linux系统
            ],
        },
        description="A practical tool for converting PDF to Markdown",  # 简短描述
        long_description=long_description,  # 详细描述
        long_description_content_type="text/markdown",  # 如果README是Markdown格式
        url="https://github.com/opendatalab/MinerU",
        python_requires=">=3.9",  # 项目依赖的 Python 版本
        entry_points={
            "console_scripts": [
                "magic-pdf = magic_pdf.tools.cli:cli",
                "magic-pdf-dev = magic_pdf.tools.cli_dev:cli" 
            ],
        },  # 项目提供的可执行命令
        include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
        zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
    )
