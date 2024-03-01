from setuptools import setup, find_packages

setup(
    name="Magic-PDF",  # 项目名
    version="0.1.0",  # 版本号
    packages=find_packages(), # 包含所有的包
    install_requires=['PyMuPDF>=1.23.25',
                      'boto3>=1.34.52',
                      'botocore>=1.34.52',
                      'Brotli>=1.1.0',
                      'click>=8.1.7',
                      'Distance>=0.1.3',
                      'loguru>=0.7.2',
                      'matplotlib>=3.8.3',
                      'numpy>=1.26.4',
                      'pandas>=2.2.1',
                      'pycld2>=0.41',
                      'regex>=2023.12.25',
                      'spacy>=3.7.4',
                      'termcolor>=2.4.0',
                      'en_core_web_sm>=3.7.1',
                      'zh_core_web_sm>=3.7.0',
                      ],  # 项目依赖的第三方库
    python_requires=">=3.9",  # 项目依赖的 Python 版本
    # entry_points={"console_scripts": ["my_command=my_project.main:run"]}, # 项目提供的可执行命令
    include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
    zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
)
