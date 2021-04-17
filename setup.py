from setuptools import find_packages
from setuptools import setup

setup(
    name="skillmodels",
    version="0.2.0",
    packages=find_packages(),
    package_data={"skillmodels": ["tests/model2.yaml"]},
    include_package_data=True,
)
