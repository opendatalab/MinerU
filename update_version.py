import os
import subprocess


def get_version():
    command = ["git", "describe", "--tags"]
    try:
        version = subprocess.check_output(command).decode().strip()
        version_parts = version.split("-")
        if len(version_parts) > 1 and version_parts[0].startswith("magic_pdf"):
            return version_parts[1]
        else:
            raise ValueError(f"Invalid version tag {version}. Expected format is magic_pdf-<version>-released.")
    except Exception as e:
        print(e)
        return "0.0.0"


def write_version_to_commons(version):
    commons_path = os.path.join(os.path.dirname(__file__), 'magic_pdf', 'libs', 'version.py')
    with open(commons_path, 'w') as f:
        f.write(f'__version__ = "{version}"\n')


if __name__ == '__main__':
    version_name = get_version()
    write_version_to_commons(version_name)
