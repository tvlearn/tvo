# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import tvem
from tvem.trainer import Trainer
from tvem.models import NoisyOR
from tvem.variational import RandomSampledVarStates, FullEM
from tvem.utils.data import TVEMDataLoader

import pytest
import torch as to
import numpy as np


@pytest.fixture(
    scope="function", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    class Setup:
        device = tvem.get_device()
        precision = to.float32
        N, D, S, H, S_new = 10, 16, 8, 8, 10
        model = NoisyOR(H, D, precision=precision)
        _td = to.randint(2, size=(N, D), dtype=to.uint8, device=device)
        data = TVEMDataLoader(_td, batch_size=N)
        _td = to.randint(2, size=(N, D), dtype=to.uint8, device=device)
        test_data = TVEMDataLoader(_td, batch_size=N)
        var_states = RandomSampledVarStates(N, H, S, precision, S_new)
        test_states = RandomSampledVarStates(N, H, S, precision, S_new)

    return Setup


def test_training(setup):
    trainer = Trainer(setup.model, setup.data, setup.var_states)
    d1 = trainer.em_step()
    d2 = trainer.em_step()
    assert "train_F" in d1 and "train_subs" in d1
    assert "test_F" not in d1 and "test_subs" not in d1
    assert d1["train_F"] < d2["train_F"]


def test_training_with_tensor(setup):
    data = to.randint(2, size=(setup.N, setup.D)).byte()
    trainer = Trainer(setup.model, data, setup.var_states)
    d1 = trainer.em_step()
    d2 = trainer.em_step()
    assert "train_F" in d1 and "train_subs" in d1
    assert "test_F" not in d1 and "test_subs" not in d1
    assert d1["train_F"] < d2["train_F"]


def test_training_with_valid(setup):
    trainer = Trainer(
        setup.model,
        train_data=setup.data,
        train_states=setup.var_states,
        test_data=setup.test_data,
        test_states=setup.test_states,
    )
    d1 = trainer.em_step()
    d2 = trainer.em_step()
    assert "train_F" in d1 and "train_subs" in d1
    assert "test_F" in d1 and "test_subs" in d1
    assert d1["train_F"] < d2["train_F"]


def test_testing(setup):
    trainer = Trainer(setup.model, test_data=setup.test_data, test_states=setup.test_states)
    d1 = trainer.em_step()
    d2 = trainer.em_step()
    assert "train_F" not in d1 and "train_subs" not in d1
    assert "test_F" in d1 and "test_subs" in d1
    assert d1["test_F"] < d2["test_F"]


def test_estep(setup):
    trainer = Trainer(setup.model, test_data=setup.test_data, test_states=setup.test_states)
    d1 = trainer.e_step()
    d2 = trainer.e_step()
    assert d1["test_F"] < d2["test_F"]


@pytest.mark.gpu
def test_rollback():
    # A seed for which we know the unconstrained M-step will decrease free energies.
    to.manual_seed(123)

    eps = 1e-6
    N, H, D = 10, 4, 32
    data = (to.rand(N, D, device=tvem.get_device()) < 0.8).byte()  # noisy data
    dataloader = TVEMDataLoader(data, batch_size=N)

    # make two identical copies of the model: we'll train twice with same initial conditions
    # parameters are initialized stochastically.
    m1 = NoisyOR(H, D, precision=to.float64)
    m2 = NoisyOR(H, D, precision=m1.precision, W_init=m1.theta["W"], pi_init=m1.theta["pies"])

    # use FullEM so training is deterministic
    states = FullEM(N, H, m1.precision)

    # without rollback, we get decreasing free energies
    trainer = Trainer(m1, dataloader, states)
    F = [trainer.em_step()["train_F"] for _ in range(50)]
    assert np.any(np.diff(F) < -eps)

    # with rollback, F always increases
    trainer = Trainer(m2, dataloader, states, rollback_if_F_decreases=["W"])
    F = [trainer.em_step()["train_F"] for _ in range(50)]
    assert np.any(np.diff(F) >= -eps)
