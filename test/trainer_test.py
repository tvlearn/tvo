# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import tvem
from tvem.trainer import Trainer
from tvem.models import NoisyOR
from tvem.variational import RandomSampledVarStates
from tvem.utils.data import TVEMDataLoader

import pytest
import torch as to


@pytest.fixture(
    scope="function", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    class Setup:
        N, D, S, H = 10, 16, 8, 8
        model = NoisyOR(N, H, D, precision=to.float32)
        _td = to.randint(2, size=(N, D), dtype=to.uint8, device=tvem.get_device())
        data = TVEMDataLoader(_td, batch_size=N)
        _td = to.randint(2, size=(N, D), dtype=to.uint8, device=tvem.get_device())
        test_data = TVEMDataLoader(_td, batch_size=N)
        _varstates_conf = {
            "N": N,
            "H": H,
            "S": S,
            "precision": to.float32,
            "device": tvem.get_device(),
        }
        var_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)
        test_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)

    return Setup


def test_training(setup):
    trainer = Trainer(setup.model, setup.data, setup.var_states)
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
