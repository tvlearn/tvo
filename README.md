# TVO - Truncated Variational Optimization <br>
[![build status](https://github.com/tvlearn/tvo/actions/workflows/test.yml/badge.svg)](https://github.com/tvlearn/tvo/actions/workflows/test.yml?query=branch%3Amaster)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tvlearn.github.io/tvo)
[![coverage](https://raw.githubusercontent.com/tvlearn/tvo/gh-pages/docs/cov_badge.svg)](https://tvlearn.github.io/tvo/htmlcov/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![trello board](https://img.shields.io/badge/trello%20board-private-blue.svg)](https://trello.com/b/EuWTcm4w/tvem-repo)

This repository provides a PyTorch package for Truncated Variational Optimization. See [related publications](#related-publications) below. To get started, check out the [examples](/examples).


## Setup
Clone this repository:

```bash
git clone git@github.com:tvlearn/tvo.git
```

We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting installed packages:

```bash
$ conda create -n tvo python==3.8
$ conda activate tvo
```

For large problem sizes, we strongly recommend distributed execution of the algorithms using CPU and/or GPU parallelization. To enable MPI-based CPU parallelization, make sure to complete the steps described under [Installing PyTorch with MPI support](#installing-pytorch-with-mpi-support-(optional\)) below. Execution on GPU requires the CUDA Toolkit to be installed, e.g. via `conda install -c anaconda cudatoolkit`.

After completing the optional installations for distributed execution, run

```bash
$ pip install -r requirements.txt  # use requirements-mpi.txt in case of MPI-usage
```

Finally, TVO can be set up:

```bash
$ cd tvo
$ python setup.py build_ext
$ python setup.py install  # optionally replace install by develop to facilitate development
```

Running the examples additionally requires an installation of [tvutil](https://github.com/tvlearn/tvutil), e.g. via:

```bash
$ cd ..
$ git clone git@github.com:tvlearn/tvutil.git
$ cd tvutil
$ python setup.py install
```


## Running tests

Unit tests are implemented in [test](/test) and can be executed via:

```bash
python setup.py build_ext --inplace  # compile extension modules
pytest test  # run tests
```


## Installing PyTorch with MPI support (optional)
This requires a system level installation of MPI run (please consult the official documentation of [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) and [MPICH](https://www.mpich.org/documentation/guides/) if you need help):
```
$ sudo apt install mpich
$ MPICC=$(which mpicc) pip install mpi4py
```

Once MPICH is installed, you are ready to install PyTorch from source:
```bash
$ conda install -c conda-forge typing_extensions numpy ninja pyyaml mkl mkl-include setuptools cmake cffi future six requests dataclasses
$ git clone --recursive https://github.com/pytorch/pytorch
$ cd pytorch
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
$ python setup.py install
```


## Related publications

J. Drefs\*, E. Guiraud\*, F. Panagiotou, J. Lücke. Direct Evolutionary Optimization of Variational Autoencoders With Binary Latents. _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_, Springer, Cham, 2022, accepted. \*Joint first authorship.

Jakob Drefs, Enrico Guiraud, Jörg Lücke. Evolutionary Variational Optimization of Generative Models. _Journal of Machine Learning Research_ 23(21):1-51, 2022. [(online access)](https://www.jmlr.org/papers/v23/20-233.html)

Jörg Lücke, Zhenwen Dai, Georgios Exarchakis. Truncated Variational Sampling for ‘Black Box’ Optimization of Generative Models. _International Conference on Latent Variable Analysis and Signal Separation_, Springer, Cham, 2018. [(online access)](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_43)

Jörg Lücke. Truncated Variational Expectation Maximization. _arXiv preprint_ arXiv:1610.03113, 2019. [(online access)](https://arxiv.org/abs/1610.03113)
