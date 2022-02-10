# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvo.exp import ExpConfig, EVOConfig, Training
from tvo.utils.model_protocols import Trainable
from tvo.variational import FullEM

import tvo
import numpy as np
import h5py
import torch as to
import pytest
import math


class BlackBoxBSC(Trainable):
    def __init__(self, H: int, D: int):
        self._theta = dict(pi=to.rand(H), logsigma=to.tensor(0.001), W=to.rand(H, D))
        self._k = to.log(to.tensor(2.0 * math.pi)) / 2

    def log_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        pi, logsigma, W = self._theta.values()
        pi_ = pi.clamp(1e-2, 1.0 - 1e-2)
        H, D = W.shape
        Kfloat = states.type_as(pi)
        logprior = Kfloat @ to.log(pi_ / (1 - pi_)) + to.sum(to.log(1 - pi_))
        logpygs = (
            -(data.unsqueeze(1) - Kfloat @ W).pow_(2).sum(dim=2).div_(2 * to.exp(2 * logsigma))
            - D * self._k
            - D * logsigma
        )
        return logprior + logpygs


@pytest.fixture(scope="module", params=[pytest.param(tvo.get_device().type, marks=pytest.mark.gpu)])
def setup(request):
    class Setup:
        _device = tvo.get_device()
        N, D, H = 2, 1, 2
        precision = to.float32
        pies_init = to.full((H,), 0.5, dtype=precision, device=_device)
        W_init = to.full((D, H), 1.0, dtype=precision, device=_device)
        sigma_init = to.tensor([1.0], dtype=precision, device=_device)

        conf = {"D": D, "H": H, "S": 2**H, "Snew": 0, "batch_size": N, "precision": precision}
        m = BlackBoxBSC(H, D)
        m._theta["pi"][:] = pies_init
        m._theta["W"][:] = W_init.T
        m._theta["logsigma"][...] = to.log(sigma_init)
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
                    2.0 * np.log(1.0) - (1.0 / 2) * 2.0**2,
                ],
                [-(1.0 / 2), np.log(1.0), np.log(1.0), 2.0 * np.log(1.0) - (1.0 / 2)],
            ],
            device=_device,
        )
        true_const = 2 * np.log(0.5) - 0.5 * np.log(2 * math.pi)
        # per datap.
        true_free_energy = to.log(to.exp(true_lpj + true_const).sum(dim=1)).sum() / N

    return Setup


def test_free_energy(setup) -> None:
    batch, states, m = setup.data, setup.all_s, setup.m
    states.lpj[:] = m.log_joint(batch, states.K[:])
    f = m.free_energy(idx=to.arange(setup.N), batch=batch, states=states) / setup.N
    assert math.isclose(f, setup.true_free_energy, rel_tol=1e-6)


def test_one_em_step(setup) -> None:
    m = setup.m
    N = setup.N
    batch, states = setup.data, setup.all_s
    states.lpj[:] = m.log_joint(batch, states.K[:])
    first_F = m.free_energy(idx=to.arange(N), batch=batch, states=states)
    m.update_param_batch(idx=to.arange(N), batch=batch, states=states)
    m.update_param_epoch()
    states.lpj[:] = m.log_joint(batch, states.K[:])
    new_F = m.free_energy(idx=to.arange(N), batch=batch, states=states)

    assert new_F > first_F


def test_training() -> None:
    with h5py.File("blackbox_test.h5", "w") as f:
        f.create_dataset("data", data=np.random.rand(100, 8))
    model = BlackBoxBSC(H=8, D=8)
    estep_conf = EVOConfig(n_states=8, n_parents=3, n_generations=2)
    exp = Training(ExpConfig(), estep_conf, model, train_data_file="blackbox_test.h5")
    for log in exp.run(10):
        log.print()
