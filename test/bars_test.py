# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvem.exp import ExpConfig, EEMConfig, Training
from tvem.models import NoisyOR, BSC
from tvem.util.parallel import init_processes, broadcast
from tvem.util import get
import tvem
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist

gpu_and_mpi_marks = pytest.param(tvem.get_device().type, marks=(pytest.mark.gpu, pytest.mark.mpi))


def generate_bars(
    H: int,
    bar_amp: float = 1.0,
    neg_amp: bool = False,
    bg_amp: float = 0.0,
    add_unit: float = None,
    dtype: to.dtype = to.float64,
):
    """ Generate a ground-truth dictionary W suitable for a std. bars test

    Creates H bases vectors with horizontal and vertival bars on a R*R pixel grid,
    (wth R = H // 2).

    :param H: Number of latent variables
    :param bar_amp: Amplitude of each bar
    :param neg_amp: Set probability of amplitudes taking negative values to 50 percent
    :param bg_amp: Background amplitude
    :param add_unit: If not None an additional unit with amplitude add_unit will be inserted
    :param dtype: torch.dtype of the returned tensor
    :returns: tensor containing the bars dictionary
    """
    R = H // 2
    D = R ** 2

    W = bg_amp * to.ones((R, R, H), dtype=dtype, device=tvem.get_device())
    for i in range(R):
        W[i, :, i] = bar_amp
        W[:, i, R + i] = bar_amp

    if neg_amp:
        sign = 1 - 2 * to.randint(high=2, size=(H), device=tvem.get_device())
        W = sign[None, None, :] * W

    if add_unit is not None:
        add_unit = add_unit * to.ones((D, 1), device=tvem.get_device())
        W = to.cat((W, add_unit), dim=1)
        H += 1

    return W.view((D, H))


@pytest.fixture(scope="module", params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope="module")
def hyperparams():
    """Return an object containing hyperparametrs N,D,S,H as data members."""

    class BarsParams:
        N = 500
        H = 10
        D = (H // 2) ** 2
        S = 60
        batch_size = 10
        precision = to.float32

    return BarsParams


@pytest.fixture(scope="module")
def estep_conf(request, hyperparams):
    return EEMConfig(
        n_states=hyperparams.S, n_parents=3, n_children=2, n_generations=2, crossover=False
    )


def get_eem_new_states(c: EEMConfig):
    if c.crossover:
        return c.n_parents * (c.n_parents - 1) * c.n_children * c.n_generations
    else:
        return c.n_parents * c.n_children * c.n_generations


@pytest.fixture(scope="function", params=("BSC", "NoisyOR"))
def model_and_data(request, hyperparams, estep_conf):
    """Return a tuple of a TVEMModel and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    if tvem.get_run_policy() == "mpi":
        init_processes()
    rank = dist.get_rank() if dist.is_initialized() else 0

    precision, N, S, D, H, batch_size = get(
        hyperparams.__dict__, "precision", "N", "S", "D", "H", "batch_size"
    )

    if request.param == "BSC":

        W_gt = generate_bars(H, bar_amp=10.0, dtype=precision)
        sigma_gt = to.ones((1,), dtype=precision, device=tvem.get_device())
        pies_gt = to.full((H,), 2.0 / H, dtype=precision, device=tvem.get_device())

        W_init = to.rand((D, H), dtype=precision, device=tvem.get_device())
        broadcast(W_init)

        sigma_init = to.tensor([1.0], dtype=precision, device=tvem.get_device())
        pies_init = to.full((H,), 1.0 / H, dtype=precision, device=tvem.get_device())

        conf = {
            "N": N,
            "D": D,
            "H": H,
            "S": S,
            "Snew": get_eem_new_states(estep_conf),
            "batch_size": batch_size,
            "dtype": precision,
        }
        model = BSC(conf, W_gt, sigma_gt, pies_gt)

        fname = "bars_test_data_bsc.h5"

        if rank == 0:
            f = h5py.File(fname, mode="w")
            data = f.create_dataset("data", (N, D), dtype=np.float32)
            data[:] = model.generate_data(N)["data"].cpu()
            f.close()

        model.theta["W"] = W_init
        model.theta["sigma"] = sigma_init
        model.theta["pies"] = pies_init

    elif request.param == "NoisyOR":

        W_gt = generate_bars(H, bar_amp=0.8, bg_amp=0.1, dtype=precision)
        pies_gt = to.full((H,), 2.0 / H, dtype=precision, device=tvem.get_device())

        W_init = to.rand((D, H), dtype=precision, device=tvem.get_device())
        broadcast(W_init)
        pies_init = to.full((H,), 1.0 / H, dtype=precision, device=tvem.get_device())

        model = NoisyOR(H=H, D=D, W_init=W_gt, pi_init=pies_gt, precision=precision)

        fname = "bars_test_data_nor.h5"

        if rank == 0:
            f = h5py.File(fname, mode="w")
            data = f.create_dataset("data", (N, D), dtype=np.uint8)
            data[:] = model.generate_data(N)["data"].cpu()
            f.close()

        model.theta["W"] = W_init
        model.theta["pies"] = pies_init

    if tvem.get_run_policy() == "mpi":
        dist.barrier()

    return model, fname


@pytest.fixture(scope="module")
def exp_conf(hyperparams):
    return ExpConfig(batch_size=hyperparams.batch_size, precision=hyperparams.precision)


def test_training(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file)
    exp.run(10)
