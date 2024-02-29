import os


def sanitize_filename(filename, replacement="_"):
    if os.name == 'nt':
        invalid_chars = '<>:"|?*'

        for char in invalid_chars:
            filename = filename.replace(char, replacement)

    return filename
