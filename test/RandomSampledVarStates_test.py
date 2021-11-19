# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import pytest
import torch as to
from tvem.variational import RandomSampledVarStates
import tvem
from tvem.utils.model_protocols import Trainable


class DummyModel(Trainable):
    def log_joint(self, data, states):
        return states.sum(dim=2, dtype=to.float32)

    def update_param_batch(self):
        pass


@pytest.mark.gpu
def test_update():
    device = tvem.get_device()
    precision = to.float32
    N, H, S, S_new = 10, 8, 4, 10
    var_states = RandomSampledVarStates(N, H, S, precision, S_new)
    data = to.rand(N, 1, device=device)
    idx = to.arange(data.shape[0], device=device)

    # lpj simply counts active units in each latent state:
    # check that E-step does not decrease total number of active units
    n_active_units = var_states.K.sum()
    var_states.update(idx, data, DummyModel())
    new_n_active_units = var_states.K.sum()
    assert new_n_active_units >= n_active_units

    # check that E-step does not perform any substitution
    # if K contains a single state with all units on (max lpj possible)
    var_states.K = to.ones(10, 1, 50, dtype=to.uint8, device=device)
    var_states.lpj = DummyModel().log_joint(data=None, states=var_states.K)
    n_subs = var_states.update(idx, data, DummyModel())
    assert (var_states.K == 1).all()
    # n_subs _could_ be greater than zero if one or more of the states
    # sampled by RandomSampledVarStates are all ones.
    # The probability of this happening, however, is sparsity**H == (1/2)**50
    assert n_subs == 0
