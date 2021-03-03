# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from tvem.models import TVAE, BSC
from tvem.variational import FullEM
from tvem.utils import get
from tvem.utils.parallel import init_processes
import tvem
from math import pi as MATH_PI
import math
import pytest
from munch import Munch


@pytest.fixture(
    scope="module", params=[pytest.param(tvem.get_device().type, marks=[pytest.mark.gpu])]
)
def add_gpu_mark():
    """No-op fixture, use it to add the 'gpu' mark to a test or fixture."""
    pass


def fullem_for(tvae, N):
    D, H0 = tvae.shape
    return FullEM(N, H0, tvae.precision)


@pytest.fixture(scope="function")
def simple_tvae(add_gpu_mark):
    H0, H1, D = 2, 3, 1
    W = [to.ones((H0, H1)), to.ones((H1, D))]
    b = [to.zeros((H1)), to.zeros((D))]
    pi = to.full((H0,), 0.2)
    sigma2 = 0.01
    return TVAE(pi_init=pi, W_init=W, b_init=b, sigma2_init=sigma2)


def test_forward(simple_tvae):
    D, H1, H0 = simple_tvae.net_shape

    mlp_in = to.zeros((2, H0), device=tvem.get_device(), dtype=simple_tvae.precision)
    mlp_in[1, -1] = 1.0
    # assuming all W==1, all b==0 and ReLU activation
    expected_output = H1 * to.relu(mlp_in.sum(dim=1, keepdim=True))

    out = simple_tvae.forward(mlp_in)
    assert to.allclose(out, expected_output)

    mlp_in = to.rand(100, H0, device=tvem.get_device(), dtype=simple_tvae.precision)
    expected_output = H1 * mlp_in.sum(dim=1, keepdim=True)
    out = simple_tvae.forward(mlp_in)
    assert to.allclose(out, expected_output)


def true_lpj(tvae_model, data, states):
    # true lpj calculations make certain simplifying assumptions on tvae
    # parameters. check they are satisfied.
    d = tvem.get_device()
    pi, sigma2 = get(tvae_model.theta, "pies", "sigma2")
    assert all(math.isclose(pi[0], p) for p in pi)
    assert all(w.allclose(to.ones(1, device=d, dtype=tvae_model.precision)) for w in tvae_model.W)
    assert all(b.allclose(to.zeros(1, device=d, dtype=tvae_model.precision)) for b in tvae_model.b)

    D, H1, H0 = tvae_model.net_shape
    mlp_out = states.K.sum(dim=2, dtype=tvae_model.precision, keepdim=True).mul_(H1)
    assert mlp_out.allclose(tvae_model.forward(states.K))

    s1 = (data.unsqueeze(1) - mlp_out).pow_(2).sum(dim=2).div_(2 * sigma2)
    s2 = states.K.to(dtype=tvae_model.precision).matmul(to.log(pi / (1 - pi)))
    true_lpj = s2.sub_(s1)

    return true_lpj


def true_free_energy(tvae_model, data, states):
    D, H1, H0 = tvae_model.net_shape
    assert D == tvae_model.theta["b_1"].numel()
    assert D == tvae_model.theta["W_1"].shape[1]
    assert H0 == tvae_model.theta["pies"].numel()

    logjoints = (
        true_lpj(tvae_model, data, states)
        - D / 2 * to.log(2 * to.tensor(MATH_PI) * tvae_model.theta["sigma2"])
        + to.log(1 - tvae_model.theta["pies"][0]) * H0
    )
    return to.logsumexp(logjoints, dim=1).sum().item()


def test_lpj(simple_tvae):
    N = 2
    D, H1, H0 = simple_tvae.net_shape
    S = 2 ** H0
    states = fullem_for(simple_tvae, N=N)
    assert (H0, H1, D) == (2, 3, 1), "test assumes this shape for tvae but shape changed"
    assert states.K.shape == (N, S, H0)
    data = to.tensor([[0.0], [1.0]], device=tvem.get_device(), dtype=simple_tvae.precision)
    assert data.shape == (N, D)

    lpj = simple_tvae._log_pseudo_joint(data, states.K)
    expected_lpj = true_lpj(simple_tvae, data, states)

    assert expected_lpj.shape == lpj.shape
    assert to.allclose(lpj, expected_lpj)


def test_free_energy(simple_tvae):
    N = 2
    D, H1, H0 = simple_tvae.net_shape
    assert (H0, H1, D) == (2, 3, 1), "test assumes this shape for tvae but shape changed"
    states = fullem_for(simple_tvae, N)
    data = to.tensor([[0.0], [1.0]], device=tvem.get_device(), dtype=simple_tvae.precision)
    assert data.shape == (N, D)

    states.lpj[:] = simple_tvae.log_joint(data, states.K)
    tvae_F = simple_tvae.free_energy(to.arange(data.shape[0]), data, states)
    true_F = true_free_energy(simple_tvae, data, states)
    assert math.isclose(tvae_F, true_F)


@pytest.fixture(scope="function")
def tvae_and_corresponding_bsc(add_gpu_mark):
    precision = to.float64
    d = tvem.get_device()

    D, H0, H1 = 10, 2, 2
    assert H0 == H1
    W = [to.eye(H1, dtype=precision, device=d), to.rand(H1, D, dtype=precision, device=d)]
    b = [to.zeros((H1), dtype=precision, device=d), to.zeros((D), dtype=precision, device=d)]
    pi = to.full((H0,), 0.2, dtype=precision, device=d)
    sigma2 = 0.01
    tvae = TVAE(pi_init=pi, W_init=W, b_init=b, sigma2_init=sigma2, precision=precision)

    bsc_W = W[1].t()
    bsc_sigma2 = to.tensor([0.01], dtype=precision, device=d)
    bsc = BSC(H=H0, D=D, W_init=bsc_W, sigma2_init=bsc_sigma2, pies_init=pi)

    return tvae, bsc


def test_same_as_bsc(tvae_and_corresponding_bsc):
    tvae, bsc = tvae_and_corresponding_bsc

    data = to.tensor([[0.0] * 10, [1.0] * 10], dtype=tvae.precision, device=tvem.get_device())
    N = data.shape[0]

    states = fullem_for(tvae, N)

    states.lpj[:] = bsc.log_pseudo_joint(data, states.K)
    F_bsc = bsc.free_energy(to.arange(N), data, states)

    states.lpj[:] = tvae.log_joint(data, states.K)
    F_tvae = tvae.free_energy(to.arange(N), data, states)

    assert math.isclose(F_bsc, F_tvae, abs_tol=1e-5)


@pytest.fixture(scope="module")
def train_setup():
    N, D = 100, 25
    return Munch(N=N, D=D, shape=(D, 10, 10), data=to.rand(N, D, device=tvem.get_device()))


@pytest.fixture(scope="function", params=["analytical_pisigma", "gd_sigma", "gd_pisigma"])
def tvae(request, train_setup, add_gpu_mark):
    analytical_pi_updates = True
    analytical_sigma_updates = True
    if request.param == "gd_sigma":
        analytical_sigma_updates = False
    if request.param == "gd_pisigma":
        analytical_sigma_updates = False
        analytical_pi_updates = False

    return TVAE(
        train_setup.shape,
        precision=train_setup.data.dtype,
        analytical_pi_updates=analytical_pi_updates,
        analytical_sigma_updates=analytical_sigma_updates,
    )


@pytest.mark.gpu
def test_train(train_setup, tvae):
    if tvem.get_run_policy() == "mpi":
        init_processes()

    N = train_setup.N
    states = fullem_for(tvae, N)
    data = train_setup.data

    states.lpj[:] = tvae.log_joint(data, states.K)
    first_F = tvae.free_energy(idx=to.arange(N), batch=data, states=states)

    tvae.update_param_batch(idx=to.arange(N), batch=data, states=states)
    tvae.update_param_epoch()
    states.lpj[:] = tvae.log_joint(data, states.K)
    new_F = tvae.free_energy(idx=to.arange(N), batch=data, states=states)

    assert new_F > first_F


def copy_tvae(tvae):
    tvae_copy = TVAE(
        shape=tvae.net_shape,
        precision=tvae.precision,
        W_init=[w.detach().clone() for w in tvae.W],
        b_init=[b.detach().clone() for b in tvae.b],
        sigma2_init=float(tvae.theta["sigma2"].detach().clone().item()),
        pi_init=tvae.theta["pies"].detach().clone(),
        analytical_pi_updates=not tvae.theta["pies"].requires_grad,
        analytical_sigma_updates=not tvae.theta["sigma2"].requires_grad,
    )
    return tvae_copy


@pytest.mark.gpu
def test_gradients_independent_of_estep(train_setup, tvae):
    """Verify that weights are updated the same way independently of number of E-steps.

    This could not be the case if we screwed up gradient updates and they pick up on other
    calculations performed outside of the M-step.
    """
    if tvem.get_run_policy() == "mpi":
        init_processes()

    tvae_copy = copy_tvae(tvae)

    N = train_setup.N
    states = fullem_for(tvae, N)

    data = train_setup.data

    def epoch_with_esteps(tvae, n_esteps):
        for _ in range(n_esteps):
            states.update(idx=to.arange(N), batch=data, model=tvae)
        tvae.update_param_batch(idx=to.arange(N), batch=data, states=states)
        tvae.update_param_epoch()
        return tvae.theta

    theta_1 = epoch_with_esteps(tvae, 1)
    theta_10 = epoch_with_esteps(tvae_copy, 10)

    assert all(to.allclose(p1, p10) for p1, p10 in zip(theta_1.values(), theta_10.values()))


def test_generate_from_hidden(tvae):
    N = 1
    D, H0 = tvae.shape
    S = to.zeros(N, H0, dtype=to.uint8, device=tvem.get_device())
    data = tvae.generate_data(N, S)
    assert data.shape == (N, D)


def test_generate_data(tvae):
    N = 3
    D, H0 = tvae.shape
    data, hidden_state = tvae.generate_data(N)
    assert data.shape == (N, D)
    assert hidden_state.shape == (N, H0)
