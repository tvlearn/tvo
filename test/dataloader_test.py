# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import torch.distributed as dist
import pytest
from munch import Munch

from tvo.utils.data import TVODataLoader, ShufflingSampler
from tvo.utils.parallel import init_processes
import tvo


@pytest.fixture(scope="module", params=[pytest.param(tvo.get_device().type, marks=pytest.mark.gpu)])
def setup(request):
    N, D = 10, 4
    return Munch(N=N, D=D, batch_size=2, set1=to.rand(N, D), set2=to.rand(N, D))


def test_one_dataset(setup):
    DataLoader = TVODataLoader(setup.set1, batch_size=setup.batch_size)
    assert DataLoader.dataset.tensors[0].equal(to.arange(setup.N))
    assert to.allclose(DataLoader.dataset.tensors[1], setup.set1)


def test_two_datasets(setup):
    DataLoader = TVODataLoader(setup.set1, setup.set2, batch_size=setup.batch_size)
    assert DataLoader.dataset.tensors[0].equal(to.arange(setup.N))
    assert to.allclose(DataLoader.dataset.tensors[1], setup.set1)
    assert to.allclose(DataLoader.dataset.tensors[2], setup.set2)


@pytest.mark.mpi
def test_shuffling_sampler(setup):
    if tvo.get_run_policy() == "mpi":
        init_processes()
    n_procs = dist.get_world_size() if tvo.get_run_policy() == "mpi" else 1

    data = setup.set1
    n_samples = (setup.N + n_procs - 1) // n_procs
    sampler = ShufflingSampler(data, n_samples)
    dl = TVODataLoader(data, batch_size=setup.batch_size, sampler=sampler)
    n_entries_per_proc = 0
    for idx, batch in dl:
        n_entries_per_proc += idx.numel()
    # check all processes looped over n_samples datapoints, independently of the dataset size
    assert n_entries_per_proc == n_samples
