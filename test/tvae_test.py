# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from tvem.models import GaussianTVAE, BernoulliTVAE, BSC
from tvem.variational import FullEM
from tvem.utils.parallel import init_processes
import tvem
from math import pi as MATH_PI
import math
import pytest
from munch import Munch
import copy


@pytest.fixture(
    scope="module", params=[pytest.param(tvem.get_device().type, marks=[pytest.mark.gpu])]
)
def add_gpu_mark():
    """No-op fixture, use it to add the 'gpu' mark to a test or fixture."""
    pass


def fullem_for(tvae, N):
    D, H0 = tvae.shape
    return FullEM(N, H0, tvae.precision)


@pytest.fixture(scope="function", params=("Gaussian", "Bernoulli"))
def simple_tvae(request, add_gpu_mark):
    H0, H1, D = 2, 3, 1
    W = [to.ones((H0, H1)), to.ones((H1, D))]
    b = [to.zeros((H1)), to.zeros((D))]
    pi = to.full((H0,), 0.2)
    if request.param == "Gaussian":
        sigma2 = 0.01
        return GaussianTVAE(pi_init=pi, W_init=W, b_init=b, sigma2_init=sigma2)
    elif request.param == "Bernoulli":
        return BernoulliTVAE(pi_init=pi, W_init=W, b_init=b)


def test_forward(simple_tvae):
    D, H1, H0 = simple_tvae.net_shape

    mlp_in = to.zeros((2, H0), device=tvem.get_device(), dtype=simple_tvae.precision)
    mlp_in[1, -1] = 1.0

    # assuming all W==1, all b==0 and ReLU activation
    e = H1 * to.relu(mlp_in.sum(dim=1, keepdim=True))
    expected_output = to.sigmoid(e) if isinstance(simple_tvae, BernoulliTVAE) else e

    out = simple_tvae.forward(mlp_in)
    assert to.allclose(out, expected_output)

    mlp_in = to.rand(100, H0, device=tvem.get_device(), dtype=simple_tvae.precision)
    e = H1 * mlp_in.sum(dim=1, keepdim=True)
    expected_output = to.sigmoid(e) if isinstance(simple_tvae, BernoulliTVAE) else e
    out = simple_tvae.forward(mlp_in)
    assert to.allclose(out, expected_output)


def true_lpj(tvae_model, data, states):
    # true lpj calculations make certain simplifying assumptions on tvae
    # parameters. check they are satisfied.
    d = tvem.get_device()
    pi = tvae_model.theta["pies"]
    assert all(math.isclose(pi[0], p) for p in pi)
    assert all(w.allclose(to.ones(1, device=d, dtype=tvae_model.precision)) for w in tvae_model.W)
    assert all(b.allclose(to.zeros(1, device=d, dtype=tvae_model.precision)) for b in tvae_model.b)

    D, H1, H0 = tvae_model.net_shape
    mlp_out = states.K.sum(dim=2, dtype=tvae_model.precision, keepdim=True).mul_(H1)
    mlp_out = to.sigmoid(mlp_out) if isinstance(tvae_model, BernoulliTVAE) else mlp_out
    assert mlp_out.allclose(tvae_model.forward(states.K))
    N, S = data.shape[0], states.K.shape[1]

    if isinstance(tvae_model, GaussianTVAE):
        s1 = (data.unsqueeze(1) - mlp_out).pow_(2).sum(dim=2).div_(2 * tvae_model.theta["sigma2"])
    elif isinstance(tvae_model, BernoulliTVAE):
        s1 = to.sum(
            to.nn.functional.binary_cross_entropy(
                mlp_out, data.unsqueeze(1).expand(N, S, D), reduction="none"
            ),
            dim=2,
        )
    s2 = states.K.to(dtype=tvae_model.precision) @ to.log(pi / (1 - pi))
    true_lpj = s2.sub_(s1)

    return true_lpj


def true_free_energy(tvae_model, data, states):
    D, H1, H0 = tvae_model.net_shape
    assert D == tvae_model.theta["b_1"].numel()
    assert D == tvae_model.theta["W_1"].shape[1]
    assert H0 == tvae_model.theta["pies"].numel()
    pi = tvae_model.theta["pies"]

    if isinstance(tvae_model, GaussianTVAE):
        logjoints = (
            true_lpj(tvae_model, data, states)
            - D / 2 * to.log(2 * to.tensor(MATH_PI) * tvae_model.theta["sigma2"])
            + to.log(1 - pi[0]) * H0
        )
    elif isinstance(tvae_model, BernoulliTVAE):
        logjoints = true_lpj(tvae_model, data, states) + to.log(1 - pi[0]) * H0

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
    tvae = GaussianTVAE(pi_init=pi, W_init=W, b_init=b, sigma2_init=sigma2, precision=precision)

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


@pytest.fixture(scope="function", params=["with_external", "without_external"])
def external_model(request):
    if request.param == "with_external":

        class ExternalModule(to.nn.Module):
            def __init__(self, request=None):
                super(ExternalModule, self).__init__()
                self.shape = (10, 10, 25)
                self.flatten = to.nn.Flatten()
                self.H0 = self.shape[0]
                self.conv1 = to.nn.ConvTranspose2d(
                    in_channels=1, out_channels=1, kernel_size=3, padding=1
                )  # pseudo-invert convolution
                self.linear_relu_stack = to.nn.Sequential(
                    to.nn.Linear(self.shape[0], self.shape[1]),
                    to.nn.ReLU(),
                    to.nn.Linear(self.shape[1], self.shape[2]),
                    to.nn.ReLU(),
                )

            def forward(self, x):
                h = self.linear_relu_stack(x)
                # dimentionalize for conv
                if len(h.shape) == 3:
                    h = h.unsqueeze(dim=1)  # add channel depth
                elif len(h.shape) == 2:
                    h = h.unsqueeze(dim=0).unsqueeze(dim=0)  # add channel and batch dims
                h = self.conv1(h)

                # de-dimentionalize
                h = h.squeeze()
                if len(h.shape) == 1:
                    h = h.unsqueeze(dim=0)  # add s dim

                return h

        return ExternalModule
    elif request.param == "without_external":
        return None


@pytest.fixture(
    scope="function",
    params=[
        "gaussian-analytical_pisigma",
        "gaussian-gd_sigma",
        "gaussian-gd_pisigma",
        "bernoulli-analytical_pi",
        "bernoulli-gd_pi",
    ],
)
def tvae(request, train_setup, external_model, add_gpu_mark):
    if request.param == "gaussian-analytical_pisigma":
        model = GaussianTVAE
        kwargs = {"analytical_pi_updates": True, "analytical_sigma_updates": True}
    elif request.param == "gaussian-gd_sigma":
        model = GaussianTVAE
        kwargs = {"analytical_pi_updates": True, "analytical_sigma_updates": False}
    elif request.param == "gaussian-gd_pisigma":
        model = GaussianTVAE
        kwargs = {"analytical_pi_updates": False, "analytical_sigma_updates": False}
    elif request.param == "bernoulli-analytical_pi":
        model = BernoulliTVAE
        kwargs = {"analytical_pi_updates": True}
    elif request.param == "bernoulli-gd_pi":
        model = BernoulliTVAE
        kwargs = {"analytical_pi_updates": False}

    if external_model is not None:
        if model == BernoulliTVAE:

            class external_bernoulli_model(external_model):
                def forward(self, x):
                    return to.sigmoid(super(external_bernoulli_model, self).forward(x))

            ext = external_bernoulli_model()
        elif model == GaussianTVAE:
            ext = external_model()
        kwargs["shape"] = None
    else:
        ext = None
        kwargs["shape"] = train_setup.shape

    return model(**kwargs, precision=train_setup.data.dtype, external_model=ext)


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
    if tvae._external_model is not None:
        kwargs = {
            "shape": None,
            "W_init": None,
            "b_init": None,
            "external_model": copy.deepcopy(tvae._external_model),
        }
    else:
        kwargs = {
            "shape": None,
            "W_init": [w.detach().clone() for w in tvae.W],
            "b_init": [b.detach().clone() for b in tvae.b],
            "external_model": None,
        }
    if isinstance(tvae, GaussianTVAE):
        tvae_copy = GaussianTVAE(
            precision=tvae.precision,
            sigma2_init=float(tvae.theta["sigma2"].detach().clone().item()),
            pi_init=tvae.theta["pies"].detach().clone(),
            analytical_pi_updates=not tvae.theta["pies"].requires_grad,
            analytical_sigma_updates=not tvae.theta["sigma2"].requires_grad,
            **kwargs,
        )
    elif isinstance(tvae, BernoulliTVAE):
        tvae_copy = BernoulliTVAE(
            precision=tvae.precision,
            pi_init=tvae.theta["pies"].detach().clone(),
            analytical_pi_updates=not tvae.theta["pies"].requires_grad,
            **kwargs,
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
