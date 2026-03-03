from typing import BinaryIO

from mineru.model.docx.docx_converter import DocxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO):
    converter = DocxConverter()
    converter.convert(file_binary)
    return converter.pages


if __name__ == "__main__":
    # provide a more robust command-line interface and resolve the demo
    # document path relative to the project root instead of depending on
    # the current working directory.
    from pathlib import Path
    import argparse

    # climb up until we find pyproject.toml or reach a reasonable depth
    def find_project_root(start: Path) -> Path:
        current = start
        for _ in range(6):  # avoid infinite loops
            if (current / "pyproject.toml").exists() or (current / "README.md").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return start

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path.parent)
    default_docx = project_root / "demo" / "docx" / "demo1.docx"

    parser = argparse.ArgumentParser(
        description="Convert a DOCX file to internal JSON representation"
    )
    parser.add_argument(
        "docx",
        nargs="?",
        default=str(default_docx),
        help="path to the .docx file to convert (defaults to demo/docx/demo1.docx)"
    )
    args = parser.parse_args()

    print(convert_path(args.docx))
