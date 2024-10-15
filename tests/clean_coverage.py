"""
clean coverage
"""
import os
import shutil

def delete_file(path):
    """delete file."""
    if not os.path.exists(path):
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"File '{path}' deleted.")
            except TypeError as e:
                print(f"Error deleting file '{path}': {e}")
    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' and its contents deleted.")
        except TypeError as e:
            print(f"Error deleting directory '{path}': {e}")

if __name__ == "__main__":
    delete_file("htmlcov/")
    #delete_file(".coverage")
