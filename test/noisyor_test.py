# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to
import pytest
from tvem.models import NoisyOR
from tvem.variational import TVEMVariationalStates
import math
import tvem


class AllStatesExceptZero(TVEMVariationalStates):
    """All possible latent states except the all-zero one, which NoisyOR deals with separately."""

    def __init__(self, N, H):
        conf = {'N': N, 'H': H, 'S': 2**H - 1, 'dtype': to.float32}
        super().__init__(conf, self._generate_all_states(N, H))

    def update(self, idx, batch, lpj_fn, sort_by_lpj):
        self.lpj = lpj_fn(batch, self.K[idx])
        return 0

    def _generate_all_states(self, N, H):
        all_states = []
        for i in range(1, 2**H):
            i_as_binary_string = f'{i:0{H}b}'
            s = tuple(map(int, i_as_binary_string))
            all_states.append(s)
        return to.tensor(all_states, dtype=to.uint8, device=tvem.get_device())\
                 .unsqueeze(0).expand(N, -1, -1)


@pytest.fixture(scope="module",
                params=[pytest.param(tvem.get_device().type, marks=pytest.mark.gpu)])
def setup(request):
    class Setup:
        _device = tvem.get_device()
        N, D, H = 2, 1, 2
        pi_init = to.full((H,), .5)
        W_init = to.full((D, H), .5)
        m = NoisyOR(H, D, W_init, pi_init)
        all_s = AllStatesExceptZero(N, H)
        data = to.tensor([[0], [1]], dtype=to.uint8, device=_device)
        # p(s) = 1/4 p(y=1|0,0) = 0, p(y=1|0,1) = p(y=1|1,0) = 1/2, p(y=1|1,1) = 3/4
        # free_energy = np.log((1/4)*(0 + 1/2 + 1/2 + 3/4)) + np.log((1/4)*(1 + 1/2 + 1/2 + 1/4))
        true_free_energy = -1.4020427180880297
        true_lpj = to.tensor([[np.log(1/2), np.log(1/2), np.log(1/4)],
                             [np.log(1/2), np.log(1/2), np.log(3/4)]],
                             device=_device)
    return Setup


def test_lpj(setup):
    lpj = setup.m.log_pseudo_joint(setup.data, setup.all_s.K)
    assert to.allclose(lpj, setup.true_lpj)


def test_free_energy(setup):
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
    new_F = m.free_energy(idx=to.arange(N), batch=data, states=states)

    assert new_F > first_F


def test_generate_from_hidden(setup):
    zeros = to.zeros(1, setup.H, dtype=to.uint8, device=tvem.get_device())
    assert (setup.m.generate_from_hidden(zeros) == zeros).all()


def test_generate_data(setup):
    N = 3
    d = setup.m.generate_data(N)
    assert d['data'].shape == (N, setup.D)
    assert d['hidden_state'].shape == (N, setup.H)
