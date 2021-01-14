# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from tvem.models import PSC
from tvem.variational import FullEM
import tvem
import math
import pytest


@pytest.fixture(
    scope="module", params=[pytest.param(tvem.get_device().type, marks=[pytest.mark.gpu])]
)
def add_gpu_mark():
    """No-op fixture, use it to add the 'gpu' mark to a test or fixture."""
    pass


def fullem_for(model, N):
    _, H = model.shape
    return FullEM(N, H, model.precision)


@pytest.fixture(scope="function")
def simple_psc(add_gpu_mark):
    H, D = 2, 1
    W_init = 3.0 * to.ones((H, D))
    pies_init = to.full((H,), 0.2)
    return PSC(H, D, W_init, pies_init, precision=W_init.dtype)


def true_lpj(model, data, states):
    pies, W = model.theta["pies"], model.theta["W"]
    pies_ = pies.clamp(1e-2, 1.0 - 1e-2)
    precision, device = pies.dtype, tvem.get_device()
    Kfloat = states.K.type_as(pies)
    N, S = Kfloat.shape[:2]
    tiny = to.finfo(precision).tiny

    pies_term = (pies_ / (1 - pies_)).log()
    true_lpj = to.zeros((N, S), dtype=precision, device=device)

    for ind_n in range(N):
        for ind_s in range(S):
            prior_term = (Kfloat[ind_n, ind_s] * pies_term).sum()  # is (1,)
            Wbar = (Kfloat[ind_n, ind_s].unsqueeze(1) * W).sum(dim=0)  # is (D,)
            true_lpj[ind_n, ind_s] = (data[ind_n] * (Wbar + tiny).log() - Wbar).sum() + prior_term

    return true_lpj


def true_free_energy(model, data, states):
    pies = model.theta["pies"]
    pies_ = pies.clamp(1e-2, 1.0 - 1e-2)
    logjoints = (
        true_lpj(model, data, states)
        + to.sum(to.log(1 - pies_))
        - to.lgamma(data.type_as(pies)).to(data.device).sum(dim=1).unsqueeze(1)
    )
    return to.logsumexp(logjoints, dim=1).sum().item()


def test_lpj(simple_psc):
    N = 2
    D, H = simple_psc.shape
    states = fullem_for(simple_psc, N=N)
    K = states.K
    S = K.shape[1]
    assert (H, D) == (2, 1), "test assumes this shape for model but shape changed"
    assert K.shape == (N, S, H)
    data = to.tensor([[10], [4]], device=tvem.get_device(), dtype=simple_psc.config["data_dtype"])
    assert data.shape == (N, D)

    lpj = simple_psc.log_pseudo_joint(data, K)
    expected_lpj = true_lpj(simple_psc, data, states)

    assert expected_lpj.shape == lpj.shape
    assert to.allclose(lpj, expected_lpj)


def test_free_energy(simple_psc):
    N = 2
    D, H = simple_psc.shape
    assert (H, D) == (2, 1), "test assumes this shape for model but shape changed"
    states = fullem_for(simple_psc, N)
    K = states.K
    data = to.tensor([[10], [4]], device=tvem.get_device(), dtype=simple_psc.config["data_dtype"])
    assert data.shape == (N, D)

    states.lpj[:] = simple_psc.log_pseudo_joint(data, K)
    model_F = simple_psc.free_energy(to.arange(data.shape[0]), data, states)
    true_F = true_free_energy(simple_psc, data, states)
    assert math.isclose(model_F, true_F)


def test_generate_from_hidden(simple_psc):
    N = 1
    D, H = simple_psc.shape
    S = to.zeros(N, H, dtype=to.uint8, device=tvem.get_device())
    data = simple_psc.generate_data(N, S)
    assert data.shape == (N, D)


def test_generate_data(simple_psc):
    N = 3
    D, H = simple_psc.shape
    data, hidden_state = simple_psc.generate_data(N)
    assert data.shape == (N, D)
    assert hidden_state.shape == (N, H)
