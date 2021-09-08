# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to
import pytest
import math

import tvem
from tvem.models import BSC
from tvem.variational import FullEM


@pytest.fixture(
    scope="module", params=[pytest.param(tvem.get_device().type, marks=[pytest.mark.gpu])]
)
def add_gpu_mark():
    """No-op fixture, use it to add the 'gpu' mark to a test or fixture."""
    pass


@pytest.fixture(scope="function", params=["single-prior", "individual-priors"])
def setup(request, add_gpu_mark):
    class Setup:
        _device = tvem.get_device()
        individual = request.param == "individual-priors"
        N, D, H = 2, 1, 2
        precision = to.float32
        pies_init = (
            to.full((H,), 0.5, dtype=precision, device=_device)
            if individual
            else to.tensor([0.5], dtype=precision, device=_device)
        )
        W_init = to.full((D, H), 1.0, dtype=precision, device=_device)
        sigma2_init = to.tensor([1.0], dtype=precision, device=_device)

        m = BSC(
            H=H,
            D=D,
            W_init=W_init,
            sigma2_init=sigma2_init,
            pies_init=pies_init,
            precision=precision,
            individual_priors=individual,
        )
        all_s = FullEM(N, H, precision)
        all_s.lpj = to.zeros_like(all_s.lpj)
        data = to.tensor([[0], [1]], dtype=precision, device=_device)
        # lpj = \sum_h s_h \log( \pi_h/(1-\pi_h) )
        #        - 1/(2\sigma^2) ( \vec{y}-W\vec{s})^T (\vec{y}-W\vec{s}) )
        # const = \sum_h \log(1-\pi_h) - (D/2) \log(2\pi\sigma^2)
        # free_energy_all_datapoints = to.log(to.exp(lpj + const).sum(dim=1)).sum()
        true_lpj = to.tensor(
            [
                [
                    0.0,
                    np.log(1.0) - (1.0 / 2),
                    np.log(1.0) - (1.0 / 2),
                    2.0 * np.log(1.0) - (1.0 / 2) * 2.0 ** 2,
                ],
                [-(1.0 / 2), np.log(1.0), np.log(1.0), 2.0 * np.log(1.0) - (1.0 / 2)],
            ],
            device=_device,
            dtype=precision,
        )
        true_const = 2 * np.log(0.5) - 0.5 * np.log(2 * math.pi)
        # per datap.
        true_free_energy = to.log(to.exp(true_lpj + true_const).sum(dim=1)).sum() / N

    return Setup


def test_lpj(setup):
    lpj = setup.m.log_pseudo_joint(setup.data, setup.all_s.K)
    assert lpj.device == setup.all_s.K.device
    assert to.allclose(lpj, setup.true_lpj)


def test_free_energy(setup):
    batch, states, m = setup.data, setup.all_s, setup.m
    states.lpj[:] = m.log_pseudo_joint(batch, states.K)
    f = m.free_energy(idx=to.arange(setup.N), batch=batch, states=states) / setup.N
    assert math.isclose(f, setup.true_free_energy, rel_tol=1e-6)


def test_train(setup):
    m = setup.m
    N = setup.N
    batch, states = setup.data, setup.all_s

    states.lpj[:] = m.log_pseudo_joint(batch, states.K)
    first_F = m.free_energy(idx=to.arange(N), batch=batch, states=states)

    m.update_param_batch(idx=to.arange(N), batch=batch, states=states)
    m.update_param_epoch()
    states.lpj[:] = m.log_pseudo_joint(batch, states.K)
    new_F = m.free_energy(idx=to.arange(N), batch=batch, states=states)
    assert new_F > first_F


def test_generate_from_hidden(setup):
    zeros = to.zeros(1, setup.H, dtype=to.uint8, device=tvem.get_device())
    assert setup.m.generate_data(zeros.shape[0], zeros).shape == (1, setup.D)


def test_generate_data(setup):
    N = 3
    data, hidden_states = setup.m.generate_data(N)
    assert data.shape == (N, setup.D)
    assert hidden_states.shape == (N, setup.H)
