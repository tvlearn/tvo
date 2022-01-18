# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvo.utils.parallel import (
    init_processes,
    scatter_to_processes,
    gather_from_processes,
    all_reduce,
)
import tvo

import os
import torch as to
import pytest
from munch import Munch


@pytest.fixture(scope="module")
def setup(request):
    if tvo.get_run_policy() == "seq":
        rank, n_procs = 0, 1
    else:
        assert tvo.get_run_policy() == "mpi"
        init_processes()
        rank = to.distributed.get_rank()
        n_procs = to.distributed.get_world_size()
    return Munch(rank=rank, n_procs=n_procs)


@pytest.mark.mpi
def test_scatter_to_processes(setup):
    t = to.arange(setup.n_procs * 2).view(setup.n_procs, 2)
    my_t = scatter_to_processes(t)
    assert my_t.shape == (1, 2)
    assert to.allclose(my_t, to.arange(2) + setup.rank * 2)


@pytest.mark.mpi
def test_gather_from_processes(setup):
    t = gather_from_processes((to.arange(2) + setup.rank * 2)[None, :])
    if setup.rank == 0:
        assert t.shape == (setup.n_procs, 2)
        assert to.allclose(t, to.arange(setup.n_procs * 2).view(setup.n_procs, 2))


@pytest.mark.mpi
def test_scatter_from_processes_uneven_chunks(setup):
    if setup.n_procs != 4:
        pytest.skip("test unreliable for n_procs!=4")
    t = to.arange((setup.n_procs + 1) * 2).view(setup.n_procs + 1, 2)
    my_t = scatter_to_processes(t)
    assert my_t.shape[0] > 0  # make sure that every MPI process is assigned a non-empty chunk
    assert my_t.shape[1] == 2
    assert not (my_t.sum(dim=1) == 0).any()  # scatter_to_processes appends dummy zeros
    # intermediately if the number of data points to be
    # scattered is not evenly divisible by n_procs. Here,
    # make sure that the scattered tensors do not contain
    # any of these dummy zeros anymore.


@pytest.mark.mpi
def test_gather_from_processes_uneven_chunks(setup):
    if setup.n_procs != 4:
        pytest.skip("test unreliable for n_procs!=4")
    my_t = (
        (to.arange(2) + (setup.rank * 2))[None, :]
        if setup.rank < setup.n_procs - 1
        else (to.arange(4) + (setup.rank * 2)).view(2, 2)
    )
    t = gather_from_processes(my_t)
    if setup.rank == 0:
        assert t.shape == (setup.n_procs + 1, 2)
        assert to.allclose(t, to.arange(setup.n_procs * 2 + 2).view(setup.n_procs + 1, 2))


@pytest.mark.mpi
def test_scatter_and_gather(setup):
    if setup.rank == 0:
        t = to.arange(5).unsqueeze(1)
    else:
        t = []

    my_t = scatter_to_processes(t)
    mpi_t = gather_from_processes(my_t)

    if setup.rank == 0:
        assert (mpi_t == t).all()
    else:
        assert mpi_t == []


@pytest.mark.mpi
def test_all_reduce(setup):
    t = to.ones(1)
    all_reduce(t)
    assert t == setup.n_procs


@pytest.mark.gpu
@pytest.mark.mpi
def test_device(setup):
    if "TVO_GPU" in os.environ:
        assert tvo.get_device().type == "cuda"
    else:
        assert tvo.get_device() == to.device("cpu")
