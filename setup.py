import os
from setuptools import find_packages, setup

print("WHEN IMPORTS DON'T WORK, EXECUTE THE FOLLOWING COMMAND FROM THE PROJECT FOLDER:")
print("python -m pip install -e .")
print("python3 -m pip install -e .")
print(f'export PYTHONPATH="${{PYTHONPATH}}:{os.path.dirname(__file__)}"')


# https://stackoverflow.com/a/72124527
setup(name="bsc2022-usage-info", packages=find_packages())


