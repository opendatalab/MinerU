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
    setup(
        name="magic_pdf",  # 项目名
        version=__version__,  # 自动从tag中获取版本号
        packages=find_packages(),  # 包含所有的包
        install_requires=parse_requirements('requirements.txt'),  # 项目依赖的第三方库
        python_requires=">=3.9",  # 项目依赖的 Python 版本
        # entry_points={"console_scripts": ["my_command=my_project.main:run"]}, # 项目提供的可执行命令
        include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
        zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
    )
