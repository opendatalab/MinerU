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
        packages=find_packages() + ["magic_pdf.resources"],  # 包含所有的包
        package_data={
            "magic_pdf.resources": ["**"],  # 包含magic_pdf.resources目录下的所有文件
        },
        install_requires=parse_requirements('requirements.txt'),  # 项目依赖的第三方库
        extras_require={
            "gpu": ["paddleocr", "paddlepaddle-gpu"],
            "cpu": ["paddleocr", "paddlepaddle"],
            "full-cpu": ["unimernet", "matplotlib", "ultralytics", "paddleocr", "paddlepaddle"],
        },
        description="A practical tool for converting PDF to Markdown",  # 简短描述
        long_description=long_description,  # 详细描述
        long_description_content_type="text/markdown",  # 如果README是Markdown格式
        url="https://github.com/opendatalab/MinerU",
        python_requires=">=3.9",  # 项目依赖的 Python 版本
        entry_points={
            "console_scripts": [
                "magic-pdf = magic_pdf.cli.magicpdf:cli"
            ],
        },  # 项目提供的可执行命令
        include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
        zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
    )
