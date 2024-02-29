from setuptools import setup, find_packages

setup(
    name="Magic-PDF", # 项目名
    version="0.1.0", # 版本号
    packages=find_packages(), # 包含所有的包
    install_requires=['PyMuPDF>=1.23.25',
                      ], # 项目依赖的第三方库
    python_requires=">=3.9", # 项目依赖的 Python 版本
    # entry_points={"console_scripts": ["my_command=my_project.main:run"]}, # 项目提供的可执行命令
    include_package_data=True, # 是否包含非代码文件，如数据文件、配置文件等
    zip_safe=False, # 是否使用 zip 文件格式打包，一般设为 False
)
