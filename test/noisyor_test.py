# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to
import pytest
from tvem.models import NoisyOR
from tvem.variational import FullEM
import math
import tvem


@pytest.fixture(
    scope="function", params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)]
)
def setup(request):
    class Setup:
        _device = tvem.get_device()
        N, D, H = 2, 1, 2
        pi_init = to.full((H,), 0.5)
        W_init = to.full((D, H), 0.5)
        m = NoisyOR(H, D, W_init, pi_init, precision=to.float32)
        all_s = FullEM(N, H, m.precision)
        data = to.tensor([[0], [1]], dtype=to.uint8, device=_device)
        # p(s) = 1/4 p(y=1|0,0) = 0, p(y=1|0,1) = p(y=1|1,0) = 1/2, p(y=1|1,1) = 3/4
        # free_energy = np.log((1/4)*(0 + 1/2 + 1/2 + 3/4)) + np.log((1/4)*(1 + 1/2 + 1/2 + 1/4))
        true_free_energy = -1.4020427180880297
        true_lpj = to.tensor(
            [
                [0, np.log(1 / 2), np.log(1 / 2), np.log(1 / 4)],
                [-1e30, np.log(1 / 2), np.log(1 / 2), np.log(3 / 4)],
            ],
            device=_device,
        )

    return Setup


def test_lpj(setup):
    lpj = setup.m.log_pseudo_joint(setup.data, setup.all_s.K)
    assert lpj.device == setup.all_s.K.device
    assert to.allclose(lpj, setup.true_lpj)


def test_free_energy(setup):
    setup.all_s.lpj[:] = setup.m.log_pseudo_joint(setup.data, setup.all_s.K)
    f = setup.m.free_energy(idx=to.arange(setup.N), batch=setup.data, states=setup.all_s)
    assert math.isclose(f, setup.true_free_energy, rel_tol=1e-6)


def test_train(setup):
    m = setup.m
    N = setup.N
    data, states = setup.data, setup.all_s

    states.lpj[:] = m.log_pseudo_joint(data, states.K)
    first_F = m.free_energy(idx=to.arange(N), batch=data, states=states)

    m.init_epoch()
    m.update_param_batch(idx=to.arange(N), batch=data, states=states)
    m.update_param_epoch()
    states.lpj[:] = m.log_pseudo_joint(data, states.K)
    new_F = m.free_energy(idx=to.arange(N), batch=data, states=states)

    assert new_F > first_F


def test_generate_from_hidden(setup):
    N = 1
    S = to.zeros(N, setup.H, dtype=to.uint8, device=tvem.get_device())
    data = setup.m.generate_data(N, S)
    assert data.shape == (N, setup.D)
    assert (data == to.zeros(N, setup.D).to(S)).all()


def test_generate_data(setup):
    N = 3
    data, hidden_state = setup.m.generate_data(N)
    assert data.shape == (N, setup.D)
    assert hidden_state.shape == (N, setup.H)
