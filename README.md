# TVO - Truncated Variational Optimization <br>
<!-- [![build status](https://img.shields.io/gitlab/pipeline/mloldenburg/tvem.svg?style=flat-square)](https://gitlab.com/mloldenburg/tvem/pipelines)-->
[![pipeline status](https://gitlab.com/mloldenburg/tvem/badges/master/pipeline.svg)](https://gitlab.com/mloldenburg/tvem/commits/master)
[![trello board](https://img.shields.io/badge/trello%20board-private-blue.svg?style=flat-square)](https://trello.com/b/EuWTcm4w/tvem-repo)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square)](https://mloldenburg.gitlab.io/tvem)
[![coverage](https://mloldenburg.gitlab.io/tvem/cov_badge.svg)](https://mloldenburg.gitlab.io/tvem/htmlcov)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
<!--[![conda](https://anaconda.org/mloldenburg/tvem/badges/installer/conda.svg)](https://anaconda.org/mloldenburg/tvem)-->

## Installing

TVO is available on [conda](https://anaconda.org/mloldenburg/tvem):

```bash
conda install -c conda-forge -c pytorch -c mloldenburg tvem
```

Alternatively, to install from sources:

```bash
git clone git@gitlab.com:mloldenburg/tvem.git# clone this repo
python -m pip install .
```

## Get started
Examples how to setup and run the algorithms can be found [here](/examples).


## Running tests

To run the tests, first make sure `cython`, `pytest` and `setuptools` are installed in your local environment.
Then run:

```bash
python setup.py build_ext --inplace # compile extension modules
pytest test # run tests
```
