# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import pytest
import torch as to
from tvem.variational import FullEM
import tvem
from munch import Munch


def count_active_units(data, states):
    return states.sum(dim=2, dtype=to.float32)


@pytest.fixture(
    scope="function", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    s = Munch(N=10, H=8, precision=to.float32)
    s.update(var_states=FullEM(s.N, s.H, s.precision))
    return s


def test_init(setup):
    var_states = setup.var_states
    assert var_states.K.shape == (setup.N, 2 ** setup.H, setup.H)
    assert to.unique(var_states.K[0], dim=0).shape[0] == 2 ** setup.H


def test_update(setup):
    var_states = setup.var_states
    data = to.rand(setup.N, 1, device=tvem.get_device())
    idx = to.arange(data.shape[0], device=tvem.get_device())
    lpj = count_active_units(data=None, states=var_states.K)
    n_subs = var_states.update(idx, data, lpj_fn=count_active_units)
    assert n_subs == 0
    assert (var_states.lpj == lpj).all()
