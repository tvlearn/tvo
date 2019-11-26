# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import pytest
import torch as to
from tvem.variational import RandomSampledVarStates
from tvem.models import TVEMModel
import tvem


class DummyModel(TVEMModel):
    def __init__(self):
        super().__init__({})

    def log_joint(self, data, states):
        return states.sum(dim=2, dtype=to.float32)

    def generate_from_hidden(self):
        pass

    def shape(self):
        pass

    def update_param_batch(self):
        pass


@pytest.mark.gpu
def test_update():
    device = tvem.get_device()
    conf = {"N": 10, "H": 8, "S": 4, "S_new": 10, "precision": to.float32, "device": device}
    var_states = RandomSampledVarStates(conf)
    data = to.rand(conf["N"], 1, device=device)
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
