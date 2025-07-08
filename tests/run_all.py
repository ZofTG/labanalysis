"""run all tests as once."""

#! IMPORTS

import subprocess
from os import listdir
from os.path import dirname, join, sep

#! MAIN

if __name__ == "__main__":
    folder_path = join(dirname(__file__), "single_tests")
    for filename in listdir(folder_path):
        if filename.endswith(".py"):
            file_path = join(folder_path, filename)
            print(f"Running {file_path.rsplit(sep, 1)[-1]}...")
            subprocess.run(["python", file_path], check=True)
