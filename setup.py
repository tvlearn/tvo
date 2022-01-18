from setuptools import setup, find_packages, Extension

setup(
    name="tvo",
    packages=find_packages(exclude=("test",)),
    zip_safe=False,
    ext_modules=[
        Extension(
            name="tvo.variational._set_redundant_lpj_to_low_CPU",
            sources=["tvo/variational/_set_redundant_lpj_to_low_CPU.pyx"],
        )
    ],
)
