from setuptools import setup, find_packages, Extension

setup(
    name="tvem",
    packages=find_packages(exclude=("test",)),
    zip_safe=False,
    ext_modules=[
        Extension(
            name="tvem.variational._set_redundant_lpj_to_low_CPU",
            sources=["tvem/variational/_set_redundant_lpj_to_low_CPU.pyx"],
        )
    ],
)
