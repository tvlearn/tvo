# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import pytest
import torch as to
from tvem.variational import FullEM, FullEMSingleCauseModels, state_matrix  # type: ignore
import tvem
from tvem.utils.model_protocols import Trainable
from munch import Munch
import sys

print(sys.path)


class DummyModel(Trainable):
    def log_joint(self, data, states):
        return states.sum(dim=2, dtype=to.float32)

    def update_param_batch(self):
        pass


@pytest.fixture(
    scope="function", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    s = Munch(N=10, H=8, precision=to.float32)
    s.update(var_states=FullEM(s.N, s.H, s.precision))
    s.update(var_states_SCM=FullEMSingleCauseModels(s.N, s.H, s.precision))
    s.update(K_multiple_causes=state_matrix(s.H)[None, :, :].expand(s.N, -1, -1))
    s.update(
        K_single_cause=to.eye(s.H, dtype=s.precision, device=tvem.get_device())[None, :, :].expand(
            s.N, -1, -1
        )
    )
    return s


def test_init(setup):
    var_states = setup.var_states
    var_states_SCM = setup.var_states_SCM
    assert var_states.K.shape == (setup.N, 2 ** setup.H, setup.H)
    assert to.unique(var_states.K[0], dim=0).shape[0] == 2 ** setup.H
    assert var_states_SCM.K.shape == (setup.N, setup.H, setup.H)
    assert to.unique(var_states_SCM.K[0], dim=0).shape[0] == setup.H


def test_update(setup):
    var_states = setup.var_states
    var_states_SCM = setup.var_states_SCM
    data = to.rand(setup.N, 1, device=tvem.get_device())
    idx = to.arange(data.shape[0], device=tvem.get_device())
    lpj = DummyModel(setup.N, setup.H, setup.precision).log_joint(data=None, states=var_states.K)
    lpj_SCM = DummyModel(setup.N, setup.H, setup.precision).log_joint(
        data=None, states=var_states_SCM.K
    )
    n_subs = var_states.update(idx, data, model=DummyModel(setup.N, setup.H, setup.precision))
    n_subs_SCM = var_states_SCM.update(
        idx, data, model=DummyModel(setup.N, setup.H, setup.precision)
    )
    assert n_subs == 0
    assert n_subs_SCM == 0
    assert (var_states.lpj == lpj).all()
    assert (var_states_SCM.lpj == lpj_SCM).all()

    try:
        var_states_SCM.K = var_states.K
        var_states_SCM.update(idx, setup.N, model=DummyModel(setup.N, setup.H, setup.precision))
        raise Exception("FullEMSingleCauseModels accepted multiple cause K's")
    except AssertionError:
        pass
