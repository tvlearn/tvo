# -*- coding: utf-8 -*-
# Copyright (C) 2021 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import pytest
from typing import Dict, Tuple
from math import pi as MATH_PI
from munch import Munch

from tvo import get_device, get_run_policy
from tvo.variational import FullEM
from tvo.models import SSSC
from tvo.utils.parallel import pprint, init_processes


@pytest.fixture(scope="module", params=[pytest.param(get_device().type, marks=[pytest.mark.gpu])])
def add_gpu_mark():
    """No-op fixture, use it to add the 'gpu' mark to a test or fixture."""
    pass


@pytest.fixture(scope="module", params=[to.float32, to.float64], ids=["float32", "float64"])
def precision(request):
    return request.param


@pytest.fixture(scope="module")
def hyperparams():
    """Return an object containing hyperparameters N,D,H as data members."""
    return Munch(N=5, D=4, H=2)


@pytest.fixture(scope="function")
def model(hyperparams, precision, add_gpu_mark):
    return SSSC(H=hyperparams.H, D=hyperparams.D, precision=precision)


@pytest.fixture(scope="function")
def data(hyperparams, precision, add_gpu_mark):
    return to.rand((hyperparams.N, hyperparams.D), dtype=precision, device=get_device())


@pytest.fixture(scope="function")
def var_states(hyperparams, precision, add_gpu_mark):
    return FullEM(hyperparams.N, hyperparams.H, precision)


def lpj_looped(data: to.Tensor, states: to.Tensor, theta: Dict[str, to.tensor]) -> to.Tensor:
    W, sigma2, _pies, mu, Psi = (
        theta["W"],
        theta["sigma2"],
        theta["pies"],
        theta["mus"],
        theta["Psi"],
    )
    pies = _pies.clamp(1e-2, 1.0 - 1e-2)
    Kfloat = states.type_as(pies)
    N, D, S, H = data.shape + states.shape[1:]
    assert N == states.shape[0]
    assert W.shape == (D, H)
    assert sigma2.shape == (1,)
    assert pies.shape == (H,)
    assert mu.shape == (H,)
    assert Psi.shape == (H, H)

    lpj = to.zeros((N, S), dtype=W.dtype, device=W.device)

    s1 = to.log(pies / (1.0 - pies))  # (H,)

    for n in range(N):
        for s in range(S):
            Ws = W * Kfloat[n, s].unsqueeze(0)  # (W, H)
            data_normed = data[n] - Ws @ mu  # (D,)
            Cs = sigma2 * to.eye(D) + (Ws @ Psi) @ Ws.t()  # (D, D)
            try:
                CsInv = to.linalg.inv(Cs)  # (D, D)
            except Exception:
                CsInv = to.linalg.pinv(Cs)  # (D, D)
                pprint("sssc_lpj_looped: Took pseudo-inverse of Cs")

            lpj[n, s] += (Kfloat[n, s] * s1).sum()
            lpj[n, s] -= 0.5 * to.log(to.linalg.det(Cs))
            lpj[n, s] -= 0.5 * (data_normed * to.matmul(CsInv, data_normed)).sum()

    return lpj


def free_energy_looped(
    data: to.Tensor, states: to.Tensor, theta: Dict[str, to.tensor]
) -> Tuple[to.Tensor, float]:
    D = data.shape[1]
    lpj = lpj_looped(data, states, theta)  # (N, S)
    assert lpj.shape == states.shape[:2]
    const = to.log(1.0 - theta["pies"]).sum() - D / 2.0 * to.log(
        to.tensor([2.0 * MATH_PI], dtype=data.dtype, device=data.device)
    )
    logjoints = lpj + const
    return lpj, to.logsumexp(logjoints, dim=1).sum(dim=0).item()


def test_generate_from_hidden(hyperparams, model):
    N, D, H = hyperparams.N, hyperparams.D, hyperparams.H
    hidden_state = to.zeros(N, H, dtype=to.uint8, device=get_device())
    assert model.generate_data(hidden_state=hidden_state).shape == (N, D)


def test_generate_data(hyperparams, model):
    N, D, H = hyperparams.N, hyperparams.D, hyperparams.H
    data, hidden_state = model.generate_data(N=N)
    assert data.shape == (N, D)
    assert hidden_state.shape == (N, H)


def test_reformulated_lpj(model, data, var_states):
    K = var_states.K
    lpj_looped_ = lpj_looped(data, K, model.theta)
    lpj = model.log_pseudo_joint(data, K)
    tol = 1e-04
    assert lpj.device == K.device
    assert to.allclose(lpj, lpj_looped_, rtol=tol, atol=tol)


def test_straightforward_lpj(model, data, var_states):
    K = var_states.K
    lpj_looped_ = lpj_looped(data, K, model.theta)
    model.config["reformulated_lpj"] = False
    lpj = model.log_pseudo_joint(data, K)
    tol = 1e-04
    assert lpj.device == K.device
    assert to.allclose(lpj, lpj_looped_, rtol=tol, atol=tol)


def test_free_energy(model, data, var_states):
    lpj_looped, free_energy_looped_ = free_energy_looped(data, var_states.K, model.theta)
    var_states.lpj[:] = model.log_pseudo_joint(data, var_states.K)
    free_energy = model.free_energy(to.arange(data.shape[0]), data, var_states)
    tol = 1e-04
    assert to.allclose(lpj_looped, var_states.lpj, rtol=tol, atol=tol)
    assert to.isclose(
        to.tensor([free_energy_looped_]), to.tensor([free_energy]), rtol=tol, atol=tol
    )


@pytest.mark.gpu
def test_train(model, data, var_states):
    if get_run_policy() == "mpi":
        init_processes()

    N = data.shape[0]

    var_states.lpj[:] = model.log_joint(data, var_states.K)
    first_F = model.free_energy(idx=to.arange(N), batch=data, states=var_states)

    model.update_param_batch(idx=to.arange(N), batch=data, states=var_states)
    model.update_param_epoch()
    var_states.lpj[:] = model.log_joint(data, var_states.K)
    new_F = model.free_energy(idx=to.arange(N), batch=data, states=var_states)
    assert new_F > first_F
