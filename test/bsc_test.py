# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import numpy as np
import torch as to
import pytest
import math

import tvem
from tvem.models import BSC
from tvem.variational import FullEM

test_devices = [to.device('cpu')]
if 'TVEM_GPU' in os.environ:
    test_devices.append(to.device('cuda:0'))


@pytest.fixture(scope="module", params=test_devices)
def setup(request):
    class Setup:
        tvem._set_device(request.param)
        _device = tvem.get_device()
        N, D, H = 2, 1, 2
        dtype = to.float32
        pi_init = to.full((H,), .5, dtype=dtype, device=_device)
        W_init = to.full((D, H), 1., dtype=dtype, device=_device)
        sigma_init = to.tensor([1., ], dtype=dtype, device=_device)

        conf = {'N': N, 'D': D, 'H': H, 'S': 2**H,
                'Snew': 0, 'batch_size': N, 'dtype': dtype}
        m = BSC(conf, W_init, sigma_init, pi_init)
        conf = {'N': N, 'H': H, 'S': 2**H, 'dtype': dtype}
        all_s = FullEM(conf)
        all_s.lpj = to.zeros_like(all_s.lpj)
        data = to.tensor([[0], [1]], dtype=dtype, device=_device)
        # lpj = \sum_h s_h \log( \pi_h/(1-\pi_h) )
        #        - 1/(2\sigma^2) ( \vec{y}-W\vec{s})^T (\vec{y}-W\vec{s}) )
        # const = \sum_h \log(1-\pi_h) - (D/2) \log(2\pi\sigma^2)
        # free_energy_all_datapoints = to.log(to.exp(lpj + const).sum(dim=1)).sum()
        true_lpj = to.tensor([[0., np.log(1.)-(1./2), np.log(1.)-(1./2),
                               2.*np.log(1.)-(1./2)*2.**2],
                              [-(1./2), np.log(1.), np.log(1.), 2.*np.log(1.)-(1./2)]],
                             device=_device)
        true_const = 2*np.log(0.5) - 0.5 * np.log(2*math.pi)
        # per datap.
        true_free_energy = to.log(
            to.exp(true_lpj + true_const).sum(dim=1)).sum() / N
    return Setup


def test_lpj(setup):
    setup.m.init_epoch()
    lpj = setup.m.log_pseudo_joint(setup.data, setup.all_s.K)
    assert to.allclose(lpj, setup.true_lpj)


def test_free_energy(setup):
    setup.m.init_epoch()
    f = setup.m.free_energy(idx=to.arange(
        setup.N), batch=setup.data, states=setup.all_s) / setup.N
    assert math.isclose(f, setup.true_free_energy, rel_tol=1e-6)


def test_train(setup):
    m = setup.m
    N = setup.N
    data, states = setup.data, setup.all_s
    m.init_epoch()
    first_F = m.free_energy(idx=to.arange(N), batch=data, states=states)
    m.update_param_batch(idx=to.arange(N), batch=data, states=states)
    m.update_param_epoch()
    m.init_epoch()
    new_F = m.free_energy(idx=to.arange(N), batch=data, states=states)

    assert new_F > first_F


"""
TESTS TO DO:

- calls to generate_data and generate_from_hidden must return tensors with the correct shape
- a call to generate_from_hidden with all-zero latents must return an all-zero datapoint
"""
