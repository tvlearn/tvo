# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# otherwise Testing is picked up as a test class
from tvem.exp import ExpConfig, EEMConfig, FullEMConfig, Training, Testing as _Testing
from tvem.models import NoisyOR, BSC, TVAE
from tvem.utils.model_protocols import Trainable, Reconstructor
from tvem.utils.parallel import init_processes, broadcast
from tvem.utils import get
import tvem
import os
import numpy as np
import h5py
import pytest
import torch as to
import torch.distributed as dist
from munch import Munch
from contextlib import suppress
from tvem.trainer import Trainer


class LogJointOnly(Trainable):
    """A dummy model that only implements log_joint."""

    def __init__(self, H, D, precision):
        self._H = H
        self._D = D
        self._precision = precision
        self._theta = dict(pies=to.zeros(H), W=to.zeros(D, H))
        self._shape = (D, H)
        self._config = {}

    def log_joint(self, data, states):
        N, S = data.shape[0], states.shape[1]
        return to.ones(N, S, dtype=self.precision, device=tvem.get_device())

    def update_param_batch(self, *args, **kwargs):
        pass


gpu_and_mpi_marks = pytest.param(tvem.get_device().type, marks=(pytest.mark.gpu, pytest.mark.mpi))


@pytest.fixture(scope="module", params=(gpu_and_mpi_marks,))
def add_gpu_and_mpi_marks():
    """No-op fixture, use it to add the 'gpu' and 'mpi' marks to a test or fixture."""
    pass


@pytest.fixture(scope="module")
def hyperparams():
    """Return an object containing hyperparametrs N,D,H as data members."""
    return Munch(N=16, D=8, S=4, H=10)


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
    continuous_fname_distributed_reco = "experiment_test_data_continous_distributed_reco.h5"

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

        f = h5py.File(continuous_fname_distributed_reco, mode="w")
        N_ = 17  # cause data points to be unevenly distributed across MPI processes
        # (assume 4 MPI processes)
        data = f.create_dataset("data", (N_, D), dtype=np.float32)
        data_ = np.random.rand(N_, D)
        data_[data_ < 0.5] = float("nan")
        data[:] = data_
        f.close()

    if tvem.get_run_policy() == "mpi":
        dist.barrier()

    yield Munch(
        binary_data=binary_fname,
        continuous_data=continuous_fname,
        continuous_data_distributed_reco=continuous_fname_distributed_reco,
    )

    if rank == 0:
        with suppress(FileNotFoundError):
            os.remove(binary_fname)
            os.remove(continuous_fname)
            os.remove(continuous_fname_distributed_reco)
            os.remove("tvem_exp.h5")  # default experiment output file


@pytest.fixture(scope="function", params=(True, False), ids=("cross", "nocross"))
def estep_conf(request, hyperparams):
    # randomly select parent selection and mutation algorithm
    # (testing all combinations for every model and experiment takes too much)
    parent_selection = ["fitness", "uniform"][np.random.randint(2)]
    mutation = ["sparsity", "uniform"][np.random.randint(2)]
    crossover = request.param
    return EEMConfig(
        n_states=hyperparams.S,
        n_parents=3,
        n_children=2 if not crossover else None,
        n_generations=2,
        parent_selection=parent_selection,
        mutation=mutation,
        crossover=crossover,
        bitflip_frequency=1 / hyperparams.H if mutation == "sparsity" else None,
    )


@pytest.fixture(scope="module", params=(1, 2, 3), ids=("batch1", "batch2", "batch3"))
def batch_size(request):
    return request.param


@pytest.fixture(scope="function", params=("NoisyOR", "BSC", "TVAE", "TVAE_reco", "LogJointOnly"))
def model_and_data(request, hyperparams, input_files, precision):
    """Return a tuple of a model and a filename (dataset for the model).

    Parametrized fixture, use it to test on several models.
    """
    N, D, H = get(hyperparams, "N", "D", "H")
    if request.param == "NoisyOR":
        return NoisyOR(H=H, D=D, precision=precision), input_files.binary_data
    elif request.param == "BSC":
        return BSC(H=H, D=D, precision=precision), input_files.continuous_data
    elif request.param == "TVAE":
        return TVAE(shape=(D, H * 2, H), precision=precision), input_files.continuous_data
    elif request.param == "TVAE_reco":
        return (
            TVAE(shape=(D, H * 2, H), precision=precision),
            input_files.continuous_data_distributed_reco,
        )
    elif request.param == "LogJointOnly":
        return LogJointOnly(H, D, precision), input_files.continuous_data


@pytest.fixture(scope="module", params=(0, 3), ids=("nowarmup", "warmup"))
def warmup_Esteps(request):
    return request.param


@pytest.fixture(scope="function")
def exp_conf(precision, batch_size, warmup_Esteps):
    # randomly activate some toggle options
    keep_best_states = to.rand((1,))
    broadcast(keep_best_states)
    keep_best_states = (keep_best_states < 0.5).item()

    eval_F_at_epoch_end = to.rand((1,))
    broadcast(eval_F_at_epoch_end)
    eval_F_at_epoch_end = (eval_F_at_epoch_end < 0.5).item()

    return ExpConfig(
        batch_size=batch_size,
        warmup_Esteps=warmup_Esteps,
        keep_best_states=keep_best_states,
        eval_F_at_epoch_end=eval_F_at_epoch_end,
    )


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
    if input_file == "experiment_test_data_continous_distributed_reco.h5":
        return
    exp = Training(exp_conf, estep_conf, model, train_data_file=input_file)
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "train")


def test_training_and_validation(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    if input_file == "experiment_test_data_continous_distributed_reco.h5":
        return
    exp = Training(
        exp_conf, estep_conf, model, train_data_file=input_file, val_data_file=input_file
    )
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "train", "valid")


def test_testing(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks):
    model, input_file = model_and_data
    if input_file == "experiment_test_data_continous_distributed_reco.h5":
        return
    exp = _Testing(exp_conf, estep_conf, model, data_file=input_file)
    for log in exp.run(10):
        log.print()
    check_file(exp_conf.output, "test")


def test_reconstruction(model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks, warmup_Esteps):
    model, input_file = model_and_data
    if input_file == "experiment_test_data_continous_distributed_reco.h5":
        return
    if not isinstance(model, Reconstructor):
        return

    exp_conf.reco_epochs = range(10)
    exp_conf.warmup_Esteps = 0
    exp = Training(
        exp_conf, estep_conf, model, train_data_file=input_file, val_data_file=input_file
    )
    for log in exp.run(10):
        log.print()

    rank = dist.get_rank() if tvem.get_run_policy() == "mpi" else 0
    if rank == 0:
        f = h5py.File(exp_conf.output, "r")
        assert "train_reconstruction" in f.keys()
        train_reconstruction = to.tensor(f["train_reconstruction"], dtype=model.precision)
        f.close()

        f = h5py.File(input_file, "r")
        train_data = to.tensor(f["data"], dtype=model.precision)
        f.close()
        assert train_reconstruction.shape == train_data.shape
        assert (train_reconstruction != train_data).any()


@pytest.mark.mpi
def test_reconstruction_with_missing_distributed(
    model_and_data, exp_conf, estep_conf, add_gpu_and_mpi_marks, warmup_Esteps
):
    model, input_file = model_and_data
    if input_file != "experiment_test_data_continous_distributed_reco.h5":
        return
    if not isinstance(model, Reconstructor):
        return
    comm_rank = dist.get_rank() if tvem.get_run_policy() == "mpi" else 0
    comm_size = dist.get_world_size() if dist.is_initialized() else 1
    if comm_size != 4:
        pytest.skip(f"test obsolete for n_procs=={comm_size}")

    exp_conf.reco_epochs = range(10)
    exp_conf.warmup_Esteps = 0
    exp = Training(
        exp_conf, estep_conf, model, train_data_file=input_file, val_data_file=input_file
    )
    for log in exp.run(10):
        log.print()

    if comm_rank == 0:
        f = h5py.File(exp_conf.output, "r")
        assert "train_reconstruction" in f.keys()
        train_reconstruction = to.tensor(f["train_reconstruction"], dtype=model.precision)
        f.close()

        f = h5py.File(input_file, "r")
        train_data = to.tensor(f["data"], dtype=model.precision)
        f.close()
        assert train_reconstruction.shape == train_data.shape
        assert (to.logical_not(to.isnan(train_reconstruction))).any()
        inds_not_is_nan = to.logical_not(to.isnan(train_data))
        assert to.allclose(train_data[inds_not_is_nan], train_reconstruction[inds_not_is_nan])


def test_data_transform(model_and_data):
    model, input_file = model_and_data
    if input_file == "experiment_test_data_continous_distributed_reco.h5":
        return
    exp_conf = ExpConfig(data_transform=lambda x: to.zeros_like(x))
    estep_conf = FullEMConfig(n_latents=model.shape[1])
    exp = _Testing(exp_conf, estep_conf, model, input_file)
    with h5py.File(input_file, "r") as f:
        data = to.from_numpy(f["data"][...])
    zeros = to.zeros_like(data).to(model.precision if data.dtype is not to.uint8 else to.uint8)
    t = Trainer(model, test_data=zeros, test_states=exp.test_states)
    model_F_for_zeros = t.eval_free_energies()["test_F"]
    print(model_F_for_zeros)
    for log in exp.run(1):
        log.print()
        assert log._results["test_F"] == model_F_for_zeros
