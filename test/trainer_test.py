# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.trainer import Trainer
from tvem.models import NoisyOR
from tvem.variational import RandomSampledVarStates

import os
import pytest
import torch as to

test_devices = ['cpu']
if 'TVEM_USE_GPU' in os.environ:
    test_devices.append('cuda:0')


@pytest.fixture(scope='function', params=test_devices)
def setup(request):
    class Setup:
        N, D, S, H = 10, 16, 8, 8
        device = request.param
        trainer = Trainer(NoisyOR(H, D, device=device))
        data = to.randint(2, size=(N, D), dtype=to.uint8, device=device)
        val_data = to.randint(2, size=(N, D), dtype=to.uint8, device=device)
        _varstates_conf = {'N': N, 'H': H, 'S': S, 'dtype': to.float32, 'device': device}
        var_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)
        val_var_states = RandomSampledVarStates(n_new_states=10, conf=_varstates_conf)
    return Setup


def test_training(setup):
    setup.trainer.train(epochs=10, batch_size=setup.N,
                        train_data=setup.data, train_states=setup.var_states)


def test_training_with_valid(setup):
    setup.trainer.train(epochs=10, batch_size=setup.N,
                        train_data=setup.data, train_states=setup.var_states,
                        val_data=setup.val_data, val_states=setup.val_var_states)


def test_testing(setup):
    setup.trainer.test(epochs=10, test_data=setup.data, test_states=setup.var_states)
