from setuptools import find_packages
from setuptools import setup

setup(
    name="skillmodels",
    version="0.0.56",
    packages=find_packages(),
    package_data={"skillmodels": ["visualization/preamble.tex"]},
    include_package_data=True,
)
