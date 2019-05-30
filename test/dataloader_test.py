# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import pytest
from munch import Munch

from tvem.utils.data import TVEMDataLoader
import tvem


@pytest.fixture(
    scope="module", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    N, D = 10, 4
    return Munch(N=N, D=D, batch_size=2, set1=to.rand(N, D), set2=to.rand(N, D))


def test_one_dataset(setup):
    DataLoader = TVEMDataLoader(setup.set1, batch_size=setup.batch_size)
    assert DataLoader.dataset.tensors[0].equal(to.arange(setup.N))
    assert to.allclose(DataLoader.dataset.tensors[1], setup.set1)


def test_two_datasets(setup):
    DataLoader = TVEMDataLoader(setup.set1, setup.set2, batch_size=setup.batch_size)
    assert DataLoader.dataset.tensors[0].equal(to.arange(setup.N))
    assert to.allclose(DataLoader.dataset.tensors[1], setup.set1)
    assert to.allclose(DataLoader.dataset.tensors[2], setup.set2)
