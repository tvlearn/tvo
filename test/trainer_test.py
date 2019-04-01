# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import tvem
from tvem.trainer import Trainer
from tvem.models import NoisyOR
from tvem.variational import RandomSampledVarStates
from tvem.util.data import TVEMDataLoader

import pytest
import torch as to
from torch.utils.data.dataset import TensorDataset


@pytest.fixture(scope='function',
                params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)])
def setup(request):
    class Setup:
        N, D, S, H = 10, 16, 8, 8
        model = NoisyOR(H, D)
        _td = TensorDataset(to.randint(2, size=(N, D), dtype=to.uint8, device=tvem.get_device()))
        data = TVEMDataLoader(_td, batch_size=N)
        _td = TensorDataset(to.randint(2, size=(N, D), dtype=to.uint8, device=tvem.get_device()))
        test_data = TVEMDataLoader(_td, batch_size=N)
        _varstates_conf = {'N': N, 'H': H, 'S': S, 'dtype': to.float32, 'device': tvem.get_device()}
        var_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)
        test_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)
    return Setup


def test_training(setup):
    trainer = Trainer(setup.model, setup.data, setup.var_states)
    trainer.em_step()


def test_training_with_valid(setup):
    trainer = Trainer(setup.model, train_data=setup.data, train_states=setup.var_states,
                      test_data=setup.test_data, test_states=setup.test_states)
    trainer.em_step()


def test_testing(setup):
    trainer = Trainer(setup.model, test_data=setup.test_data, test_states=setup.test_states)
    trainer.em_step()
