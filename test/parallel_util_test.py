# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.util.parallel import (
    init_processes,
    scatter_to_processes,
    gather_from_processes,
    all_reduce,
)
import tvem

import os
import torch as to
import pytest
from collections import namedtuple


@pytest.fixture(scope="module")
def setup(request):
    if tvem.get_run_policy() == "seq":
        rank, n_procs = 0, 1
    else:
        assert tvem.get_run_policy() == "mpi"
        init_processes()
        rank = to.distributed.get_rank()
        n_procs = to.distributed.get_world_size()
    Setup = namedtuple("Setup", "rank, n_procs")
    return Setup(rank, n_procs)


@pytest.mark.mpi
def test_scatter_to_processes(setup):
    t = to.arange(setup.n_procs * 2).reshape(setup.n_procs, 2)
    my_t = scatter_to_processes(t)
    assert my_t.shape == (1, 2)
    assert to.allclose(my_t, to.arange(2) + setup.rank * 2)


@pytest.mark.mpi
def test_gather_from_processes(setup):
    t = gather_from_processes((to.arange(2) + setup.rank * 2)[None, :])
    if setup.rank == 0:
        assert t.shape == (setup.n_procs, 2)
        assert to.allclose(t, to.arange(setup.n_procs * 2).reshape(setup.n_procs, 2))


@pytest.mark.mpi
def test_gather_from_processes_uneven_chunks(setup):
    if setup.rank == setup.n_procs - 1:
        my_t = (to.arange(4) + setup.rank * 2).view(2, 2)
    else:
        my_t = (to.arange(2) + setup.rank * 2)[None, :]
    t = gather_from_processes(my_t)
    if setup.rank == 0:
        assert t.shape == (setup.n_procs + 1, 2)
        assert to.allclose(t, to.arange((setup.n_procs + 1) * 2).view(setup.n_procs + 1, 2))


@pytest.mark.mpi
def test_all_reduce(setup):
    t = to.ones(1)
    all_reduce(t)
    assert t == setup.n_procs


@pytest.mark.gpu
@pytest.mark.mpi
def test_device(setup):
    if "TVEM_GPU" in os.environ:
        assert tvem.get_device().type == "cuda"
    else:
        assert tvem.get_device() == to.device("cpu")
