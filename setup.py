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
            "lite": ["paddleocr==2.7.3",
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",
                     ],
            "full": ["unimernet==0.1.6",  # 0.1.6版本大幅裁剪依赖包范围，推荐使用此版本
                     "matplotlib<=3.9.0;platform_system=='Windows'",  # 3.9.1及之后不提供windows的预编译包，避免一些没有编译环境的windows设备安装失败
                     "matplotlib;platform_system=='Linux' or platform_system=='Darwin'",  # linux 和 macos 不应限制matplotlib的最高版本，以避免无法更新导致的一些bug
                     "ultralytics",  # yolov8,公式检测
                     "paddleocr==2.7.3",  # 2.8.0及2.8.1版本与detectron2有冲突，需锁定2.7.3
                     "paddlepaddle==3.0.0b1;platform_system=='Linux'",  # 解决linux的段异常问题
                     "paddlepaddle==2.6.1;platform_system=='Windows' or platform_system=='Darwin'",  # windows版本3.0.0b1效率下降，需锁定2.6.1
                     "pypandoc",  # 表格解析latex转html
                     "struct-eqtable==0.1.0",  # 表格解析
                     "detectron2"
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
