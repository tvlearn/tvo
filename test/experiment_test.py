# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvem.exp import ExpConfig, EEMConfig, Training, Testing as _Testing
from tvem.models import NoisyOR, BSC, TVAE, TVEMModel
from tvem.utils.parallel import init_processes
from tvem.utils import get
import tvem
import os
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist
from munch import Munch


class LogJointOnly(TVEMModel):
    """A dummy TVEMModel that only implements log_joint."""

    def __init__(self, H, D, precision):
        self._H = H
        self._D = D
        self.precision = precision
        self.theta = dict(pies=to.zeros(H), W=to.zeros(D, H))

    def log_joint(self, data, states):
        N, S = data.shape[0], states.shape[1]
        return to.ones(N, S, dtype=self.precision, device=tvem.get_device())

    def update_param_batch(self, *args, **kwargs):
        pass

    @property
    def shape(self):
        return (self._H, self._D)

    def generate_from_hidden(self, hidden_state):
        N, D = hidden_state.shape[0], self._D
        return to.zeros(N, D, dtype=self.precision, device=tvem.device())


gpu_and_mpi_marks = pytest.param(tvem.get_device().type, marks=(pytest.mark.gpu, pytest.mark.mpi))


@pytest.fixture(scope="module", params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope="module")
def hyperparams():
    """Return an object containing hyperparametrs N,D,S,H as data members."""
    return Munch(N=10, D=8, S=4, H=10)


@pytest.fixture(scope="module", params=[to.float32, to.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture(scope="module")
def input_files(hyperparams):
    """Create hd5 input files for tests, remove them before exiting the module."""
    if tvem.get_run_policy() == "mpi":
        init_processes()
    rank = dist.get_rank() if dist.is_initialized() else 0

    binary_fname = "experiment_test_data_binary.h5"
    continuous_fname = "experiment_test_data_continous.h5"

    if rank == 0:
        N, D = hyperparams.N, hyperparams.D

        f = h5py.File(binary_fname, mode="w")
        data = f.create_dataset("data", (N, D), dtype=np.uint8)
        data[:] = np.random.randint(2, size=(N, D), dtype=np.uint8)
        f.close()

        f = h5py.File(continuous_fname, mode="w")
        data = f.create_dataset("data", (N, D), dtype=np.float32)
        data[:] = np.random.rand(N, D)
        f.close()

    if tvem.get_run_policy() == "mpi":
        dist.barrier()

    yield Munch(binary_data=binary_fname, continuous_data=continuous_fname)

    if rank == 0:
        os.remove(binary_fname)
        os.remove(continuous_fname)
        os.remove("tvem_exp.h5")  # default experiment output file


@pytest.fixture(scope="module", params=(True, False), ids=("cross", "nocross"))
def estep_conf(request, hyperparams):
    # randomly select mutation algorithm (testing both for every model and experiment is a bit much)
    mutation = ["sparsity", "uniform"][np.random.randint(2)]
    return EEMConfig(
        n_states=hyperparams.S,
        n_parents=3,
        n_children=2,
        n_generations=2,
        mutation=mutation,
        crossover=request.param,
        bitflip_frequency=1 / hyperparams.H if mutation == "sparsity" else None,
    )


def get_eem_new_states(c: EEMConfig):
    if c.crossover:
        return c.n_parents * (c.n_parents - 1) * c.n_children * c.n_generations
    else:
        return c.n_parents * c.n_children * c.n_generations


@pytest.fixture(scope="module", params=(1, 2, 3), ids=("batch1", "batch2", "batch3"))
def batch_size(request):
    return request.param


@pytest.fixture(scope="function", params=("NoisyOR", "BSC", "TVAE", "LogJointOnly"))
def model_and_data(request, hyperparams, input_files, precision, estep_conf, batch_size):
    """Return a tuple of a TVEMModel and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    N, S, D, H = get(hyperparams, "N", "S", "D", "H")
    if request.param == "NoisyOR":
        return NoisyOR(H=H, D=D, precision=precision), input_files.binary_data
    elif request.param == "BSC":
        conf = {
            "D": D,
            "H": H,
            "S": S,
            "Snew": get_eem_new_states(estep_conf),
            "batch_size": batch_size,
            "precision": precision,
        }
        return BSC(conf), input_files.continuous_data
    elif request.param == "TVAE":
        return TVAE(shape=(D, H * 2, H), precision=precision), input_files.continuous_data
    elif request.param == "LogJointOnly":
        return LogJointOnly(H, D, precision), input_files.continuous_data


@pytest.fixture(scope="module", params=(0, 3), ids=("nowarmup", "warmup"))
def warmup_Esteps(request):
    return request.param


@pytest.fixture(scope="module")
def exp_conf(precision, batch_size, warmup_Esteps):
    return ExpConfig(batch_size=batch_size, precision=precision, warmup_Esteps=warmup_Esteps)


def check_file(fname, *prefixes: str):
    rank = dist.get_rank() if tvem.get_run_policy() == "mpi" else 0
    if rank != 0:
        return

    f = h5py.File(fname, "r")
    eps = 1e-4  # to tolerate a bit of noise in floating point calculations

    for prefix in prefixes:
        F = to.tensor(f[prefix + "_F"])
        subs = to.tensor(f[prefix + "_subs"])

        assert subs.shape[0] == F.shape[0]
        if prefix == "test":
            assert np.all(np.diff(F) >= -eps)
        else:
            warmup_Esteps: int = f["exp_config"]["warmup_Esteps"][...]
            assert np.all(np.diff(F[:warmup_Esteps]) >= -eps)
            # TODO for models other than NoisyOR without rollback, we can check F always increases

        K = f[prefix + "_states"]
        lpj = f[prefix + "_lpj"]
        assert K.shape[:2] == lpj.shape

    assert "theta" in f and len(f["theta"].keys()) > 0
    f.close()


def test_training(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file)
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "train")


def test_training_and_validation(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = Training(
        exp_conf, estep_conf, model, train_data_file=input_file, val_data_file=input_file
    )
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "train", "valid")


def test_testing(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    exp = _Testing(exp_conf, estep_conf, model, data_file=input_file)
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "test")
