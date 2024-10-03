from setuptools import setup, find_packages

setup(
    name="tvo",
    version="0.2.0",
    packages=find_packages(exclude=("test",)),
    zip_safe=False,
)
