# TVO - Truncated Variational Optimization <br>
[![build status](https://github.com/tvlearn/tvo/actions/workflows/test.yml/badge.svg)](https://github.com/tvlearn/tvo/actions/workflows/test.yml?query=branch%3Amaster)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tvlearn.github.io/tvo)
[![coverage](https://raw.githubusercontent.com/tvlearn/tvo/gh-pages/docs/cov_badge.svg)](https://tvlearn.github.io/tvo/htmlcov/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![trello board](https://img.shields.io/badge/trello%20board-private-blue.svg)](https://trello.com/b/EuWTcm4w/tvem-repo)

This repository provides a PyTorch package for Truncated Variational Optimization. For related publications, see [1-3]. Check out the [examples](/examples) to get started.


## Installing

To install from sources:

```bash
git@github.com:tvlearn/tvo.git # clone this repo
python -m pip install .
```


## Running tests

To run the tests, first make sure `cython`, `pytest` and `setuptools` are installed in your local environment.
Then run:

```bash
python setup.py build_ext --inplace # compile extension modules
pytest test # run tests
```


## References

[1] "Evolutionary Variational Optimization of Generative Models." Jakob Drefs, Enrico Guiraud, Jörg Lücke. Journal of Machine Learning Research, in press, 2022. We will link the paper very soon, see [here](https://arxiv.org/abs/2012.12294) for a preliminary arXiv version.

[2] "Truncated Variational Sampling for ‘Black Box’ Optimization of Generative Models." Jörg Lücke, Zhenwen Dai, Georgios Exarchakis. [International Conference on Latent Variable Analysis and Signal Separation, Springer, Cham, 2018](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_43).

[3] "Truncated Variational Expectation Maximization." Jörg Lücke. [arXiv preprint arXiv:1610.03113 (2019)](https://arxiv.org/abs/1610.03113).



