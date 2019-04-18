# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvem.exp import Training, Testing as _Testing
from tvem.models import BSC
from tvem.util.parallel import init_processes
from tvem.util import get
import tvem
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist

gpu_and_mpi_marks = pytest.param(tvem.get_device().type,
                                 marks=(pytest.mark.gpu, pytest.mark.mpi))


def generate_bars(H, amp=1., neg_amp=False, background=None, dtype=to.float64):
    """ Generate a ground-truth dictionary W suitable for a std. bars test

    Creates H bases vectors with horizontal and vertival bars on a R*R pixel grid,
    (wth R = H // 2).  The function thus returns a matrix storing H dictionaries of
    size D=R*R.

    :param H: Number of latent variables
    :type  H: int
    :param neg_amp: Set probability of amplitudes taking negative values to 50 percent.
    :type  neg_amp: bool
    :rtype: ndarray (D x H)
    """
    R = H // 2
    D = R ** 2
    W = to.zeros((R, R, H), dtype=dtype, device=tvem.get_device())
    for i in range(R):
        W[i, :, i] = amp
        W[:, i, R + i] = amp

    if neg_amp:
        sign = 1 - 2 * to.randint(high=2, size=(H), device=tvem.get_device())
        W = sign[None, None, :] * W
    return W.view((D, H))

    D = W.shape[0]
    if background:
        background_field = background * to.ones((D, 1), device=tvem.get_device())
        W = to.cat((W, background_field), dim=1)
        H += 1

    return W


@pytest.fixture(scope='module', params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope='module')
def hyperparams():
    """Return an object containing hyperparametrs N,D,S,H as data members."""
    class HyperParams:
        dtype = to.float32
        N = 500
        H = 10
        D = int((H / 2)**2)
        S = 60
        W_gt = generate_bars(H, amp=10., dtype=dtype)
        sigma_gt = to.ones((1,), dtype=dtype, device=tvem.get_device())
        pies_gt = to.full((H,), 2./H, dtype=dtype, device=tvem.get_device())
        batch_size = 1
    return HyperParams


@pytest.fixture(scope='function', params=('BSC',))
def model_and_data(request, hyperparams):
    """Return a tuple of a TVEMModel and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    if tvem.get_run_policy() == 'mpi':
        init_processes()
    rank = dist.get_rank() if dist.is_initialized() else 0

    dtype, N, S, D, H, batch_size = get(hyperparams.__dict__, 'dtype', 'N', 'S', 'D',
                                        'H', 'batch_size')
    if request.param == 'BSC':

        conf = {'N': N, 'D': D, 'H': H, 'S': S, 'Snew': 6,
                'batch_size': batch_size, 'dtype': dtype}
        model = BSC(conf, hyperparams.W_gt, hyperparams.sigma_gt, hyperparams.pies_gt)

        fname = 'bars_test_data_bsc.h5'

        if rank == 0:
            f = h5py.File(fname, mode='w')
            data = f.create_dataset('data', (N, D), dtype=np.float32)
            data[:] = model.generate_data(N)['data']
            f.close()

    if tvem.get_run_policy() == 'mpi':
        dist.barrier()

    return model, fname


def test_training(model_and_data, hyperparams, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training({'n_states': hyperparams.S, 'dtype': hyperparams.dtype}, model=model,
                   train_data_file=input_file)
    exp.run(5)


def test_training_and_validation(model_and_data, hyperparams, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training({'n_states': hyperparams.S, 'dtype': hyperparams.dtype}, model=model,
                   train_data_file=input_file, val_data_file=input_file)
    exp.run(5)


def test_testing(model_and_data, hyperparams, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = _Testing({'n_states': hyperparams.S, 'dtype': hyperparams.dtype}, model=model,
                   data_file=input_file)
    exp.run(5)
