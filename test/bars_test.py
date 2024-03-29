# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvo.exp import ExpConfig, FullEMConfig, Training
from tvo.models import NoisyOR, BSC
from tvo.utils.parallel import init_processes, broadcast
from tvo.utils import get, generate_bars
import tvo
import os
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist
from munch import Munch

gpu_and_mpi_marks = pytest.param(tvo.get_device().type, marks=(pytest.mark.gpu, pytest.mark.mpi))


@pytest.fixture(scope="module", params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope="module")
def hyperparams():
    """Return an object containing hyperparametrs N,D,H as data members."""
    H = 6
    return Munch(N=500, H=H, D=(H // 2) ** 2, batch_size=10, precision=to.float32)


@pytest.fixture(scope="module")
def estep_conf(request, hyperparams):
    return FullEMConfig(hyperparams.H)


def write_dataset(fname, N, D, dtype, model):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        if not os.path.exists(fname):
            f = h5py.File(fname, mode="w")
            data = f.create_dataset("data", (N, D), dtype=dtype)
            data[:] = model.generate_data(N)[0].cpu()
            f.close()


@pytest.fixture(scope="function", params=("BSC", "NoisyOR"))
def model_and_data(request, hyperparams, estep_conf):
    """Return a tuple of a model and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    if tvo.get_run_policy() == "mpi":
        init_processes()

    precision, N, D, H, batch_size = get(hyperparams, "precision", "N", "D", "H", "batch_size")

    if request.param == "BSC":
        W_gt = generate_bars(H, bar_amp=10.0, precision=precision)
        sigma2_gt = to.ones((1,), dtype=precision, device=tvo.get_device())
        pies_gt = to.full((H,), 2.0 / H, dtype=precision, device=tvo.get_device())

        to.manual_seed(999)
        W_init = to.rand((D, H), dtype=precision)
        W_init = W_init.to(device=tvo.get_device())
        broadcast(W_init)

        sigma2_init = to.tensor([1.0], dtype=precision, device=tvo.get_device())
        pies_init = to.full((H,), 1.0 / H, dtype=precision, device=tvo.get_device())

        model = BSC(
            H=H, D=D, W_init=W_gt, sigma2_init=sigma2_gt, pies_init=pies_gt, precision=precision
        )

        fname = "bars_test_data_bsc.h5"

        write_dataset(fname, N, D, np.float32, model)

        model.theta["W"] = W_init
        model.theta["sigma2"] = sigma2_init
        model.theta["pies"] = pies_init

    elif request.param == "NoisyOR":
        W_gt = generate_bars(H, bar_amp=0.8, bg_amp=0.1, precision=precision)
        pies_gt = to.full((H,), 2.0 / H, dtype=precision, device=tvo.get_device())

        to.manual_seed(999)
        W_init = to.rand((D, H), dtype=precision)
        W_init = W_init.to(device=tvo.get_device())
        broadcast(W_init)
        pies_init = to.full((H,), 1.0 / H, dtype=precision, device=tvo.get_device())

        model = NoisyOR(H=H, D=D, W_init=W_gt, pi_init=pies_gt, precision=precision)

        fname = "bars_test_data_nor.h5"

        write_dataset(fname, N, D, np.uint8, model)

        model.theta["W"] = W_init
        model.theta["pies"] = pies_init

    if tvo.get_run_policy() == "mpi":
        dist.barrier()

    return model, fname


@pytest.fixture(scope="module")
def exp_conf(hyperparams):
    return ExpConfig(batch_size=hyperparams.batch_size)


def check_file(input_file):
    rank = dist.get_rank() if tvo.get_run_policy() == "mpi" else 0
    if rank != 0:
        return

    ofname = input_file.replace("data", "exp")
    output_file_mpi = ofname.replace(".h5", "_mpi.h5")
    output_file_seq_cpu = ofname.replace(".h5", "_seq_cpu.h5")
    output_file_seq_cuda = ofname.replace(".h5", "_seq_cuda.h5")
    # to tolerate a bit of noise in floating point calculations
    eps = 1e-4

    if (
        os.path.exists(output_file_mpi)
        and os.path.exists(output_file_seq_cpu)
        and os.path.exists(output_file_seq_cuda)
    ):
        f = h5py.File(output_file_mpi, "r")
        F_mpi = to.tensor(f["train_F"])
        f.close()
        f = h5py.File(output_file_seq_cpu, "r")
        F_seq_cpu = to.tensor(f["train_F"])
        f.close()
        f = h5py.File(output_file_seq_cuda, "r")
        F_seq_cuda = to.tensor(f["train_F"])
        f.close()
        assert np.all(np.diff(F_mpi) >= -eps)
        assert np.all(np.diff(F_seq_cpu) >= -eps)
        assert np.all(np.diff(F_seq_cuda) >= -eps)
        assert to.allclose(F_mpi, F_seq_cpu)
        assert to.allclose(F_seq_cpu, F_seq_cuda)

        import glob

        for p in glob.glob("*.h5"):
            os.remove(p)
    else:
        return


def test_training(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data

    if tvo.get_run_policy() == "mpi":
        suffix = "mpi"
    elif tvo.get_run_policy() == "seq":
        suffix = "seq_%s" % tvo.get_device().type
    exp_conf.output = input_file.replace("data", "exp").replace(".h5", "_%s.h5" % suffix)

    model, input_file = model_and_data
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file)
    for log in exp.run(10):
        log.print()

    check_file(input_file)
