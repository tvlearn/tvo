# TVO - Truncated Variational Optimization <br>
[![build status](https://github.com/tvlearn/tvo/actions/workflows/test.yml/badge.svg)](https://github.com/tvlearn/tvo/actions/workflows/test.yml?query=branch%3Amaster)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tvlearn.github.io/tvo)
[![coverage](https://raw.githubusercontent.com/tvlearn/tvo/gh-pages/docs/cov_badge.svg)](https://tvlearn.github.io/tvo/htmlcov/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![trello board](https://img.shields.io/badge/trello%20board-private-blue.svg)](https://trello.com/b/EuWTcm4w/tvem-repo)

This repository provides a PyTorch package for Truncated Variational Optimization. For related publications, see [1-3]. Check out the [examples](/examples) to get started.


## Installation

### with CUDA, without MPI

To install from sources:

```bash
git@github.com:tvlearn/tvo.git # clone this repo
python -m pip install .
```

### without CUDA, with MPI (requires installing PyTorch from sources)

```bash
# load modules hosting MPI and Anaconda installations
$ module load gcc/9.3.0 openmpi/gcc.9/3.1.5 anaconda2/2019.10

# create and active conda environment to host MPI-based installation
$ conda create -n tvo python==3.8
$ source activate tvo

# install MPI for Python
$ MPICC=$(which mpicc) pip install mpi4py

# install PyTorch from source
$ conda install -c conda-forge typing_extensions numpy ninja pyyaml mkl mkl-include setuptools cmake cffi future six requests dataclasses
$ git clone --recursive https://github.com/pytorch/pytorch
$ cd pytorch
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
$ python setup.py install

# install TVO including required packages
$ cd ..
$ git clone https://github.com/tvlearn/tvo.git
$ cd tvo
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
$ python setup.py develop  # change develop->install (optionally)

# install tvutil (optionally, required to run the examples)
$ cd ..
$ git clone https://github.com/tvlearn/tvutil.git
$ cd tvutil
$ python setup.py develop  # change develop->install (optionally)
```


## Running tests

To run the tests, first make sure `cython`, `pytest` and `setuptools` are installed in your local environment.
Then run:

```bash
python setup.py build_ext --inplace # compile extension modules
pytest test # run tests
```


## References

[1] Evolutionary Variational Optimization of Generative Models. Jakob Drefs, Enrico Guiraud, Jörg Lücke. _Journal of Machine Learning Research_ 23(21):1-51, 2022. [(online access)](https://www.jmlr.org/papers/v23/20-233.html)

[2] Truncated Variational Sampling for ‘Black Box’ Optimization of Generative Models. Jörg Lücke, Zhenwen Dai, Georgios Exarchakis. _International Conference on Latent Variable Analysis and Signal Separation_, Springer, Cham, 2018. [(online access)](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_43)

[3] Truncated Variational Expectation Maximization. Jörg Lücke. _arXiv preprint_ arXiv:1610.03113, 2019. [(online access)](https://arxiv.org/abs/1610.03113)
