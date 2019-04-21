# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvem.exp import ExpConfig, EEMConfig, Training, Testing as _Testing
from tvem.models import NoisyOR, BSC
from tvem.util.parallel import init_processes
from tvem.util import get
import tvem
import os
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist
from collections import namedtuple


gpu_and_mpi_marks = pytest.param(tvem.get_device().type,
                                 marks=(pytest.mark.gpu, pytest.mark.mpi))


@pytest.fixture(scope='module', params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope='module')
def hyperparams():
    """Return an object containing hyperparametrs N,D,S,H as data members."""
    class HyperParams:
        N = 10
        D = 8
        S = 4
        H = 10
    return HyperParams


@pytest.fixture(scope='module')
def input_files(hyperparams):
    """Create hd5 input files for tests, remove them before exiting the module."""
    if tvem.get_run_policy() == 'mpi':
        init_processes()
    rank = dist.get_rank() if dist.is_initialized() else 0

    binary_fname = 'experiment_test_data_binary.h5'
    continuous_fname = 'experiment_test_data_continous.h5'

    if rank == 0:
        N, D = hyperparams.N, hyperparams.D

        f = h5py.File(binary_fname, mode='w')
        data = f.create_dataset('data', (N, D), dtype=np.uint8)
        data[:] = np.random.randint(2, size=(N, D), dtype=np.uint8)
        f.close()

        f = h5py.File(continuous_fname, mode='w')
        data = f.create_dataset('data', (N, D), dtype=np.float32)
        data[:] = np.random.rand(N, D)
        f.close()

    if tvem.get_run_policy() == 'mpi':
        dist.barrier()

    FileNames = namedtuple('FileNames', 'binary_data, continuous_data')
    yield FileNames(binary_data=binary_fname, continuous_data=continuous_fname)

    if rank == 0:
        os.remove(binary_fname)
        os.remove(continuous_fname)


@pytest.fixture(scope='function', params=('NoisyOR', 'BSC'))
def model_and_data(request, hyperparams, input_files):
    """Return a tuple of a TVEMModel and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    N, S, D, H = get(hyperparams.__dict__, 'N', 'S', 'D', 'H')
    if request.param == 'NoisyOR':
        return NoisyOR(H=H, D=D, precision=to.float32), input_files.binary_data
    elif request.param == 'BSC':
        conf = {'N': N, 'D': D, 'H': H, 'S': S, 'Snew': 6,
                'batch_size': 1, 'dtype': to.float32}
        return BSC(conf), input_files.continuous_data


@pytest.fixture(scope='module')
def estep_conf(hyperparams):
    return EEMConfig(n_states=hyperparams.S, n_parents=3, n_children=2, n_generations=1,
                     crossover=False)


@pytest.fixture(scope='module')
def exp_conf():
    return ExpConfig(precision=to.float32)


def test_training(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file)
    exp.run(10)


def test_training_and_validation(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file,
                   val_data_file=input_file)
    exp.run(10)


def test_testing(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = _Testing(exp_conf, estep_conf, model, data_file=input_file)
    exp.run(10)
