# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.exp import Training, Testing as _Testing  # otherwise Testing is picked up as a test class
from tvem.models import NoisyOR
from tvem.util.parallel import init_processes
import tvem
import os
import numpy as np
import h5py
import pytest
import torch.distributed as dist

device_name = tvem.get_device().type
gpu_and_mpi_marks = (pytest.mark.gpu, pytest.mark.mpi)


@pytest.fixture(scope='module', params=[pytest.param(device_name, marks=gpu_and_mpi_marks)])
def setup(request):
    class Setup:
        N, D = 10, 8
        S, H = 4, 10
        data_fname = 'experiment_test_data.h5'
        model = NoisyOR(H, D)

    if tvem.get_run_policy() == 'mpi':
        init_processes()
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        f = h5py.File(Setup.data_fname, mode='w')
        data = f.create_dataset('data', (Setup.N, Setup.D), dtype=np.uint8)
        data[:] = np.random.randint(2, size=(Setup.N, Setup.D), dtype=np.uint8)
        f.close()
    if tvem.get_run_policy() == 'mpi':
        dist.barrier()

    yield Setup

    if rank == 0:
        os.remove(Setup.data_fname)


def test_training(setup):
    exp = Training({'n_states': setup.S}, model=setup.model, train_data_file=setup.data_fname)
    exp.run(10)


def test_training_and_validation(setup):
    exp = Training({'n_states': setup.S}, model=setup.model, train_data_file=setup.data_fname,
                   val_data_file=setup.data_fname)
    exp.run(10)


def test_testing(setup):
    exp = _Testing({'n_states': setup.S}, model=setup.model, data_file=setup.data_fname)
    exp.run(10)
