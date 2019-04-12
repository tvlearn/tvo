# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import torch as to

from torch import Tensor
from typing import Dict, Tuple

import tvem
from tvem.util import get
from tvem.util.parallel import pprint, all_reduce
from tvem.variational import TVEMVariationalStates
from tvem.models.TVEMModel import TVEMModel


class BSC(TVEMModel):
    """Binary Sparse Coding (BSC)"""

    def __init__(self, conf, W_init: Tensor = None, sigma_init: Tensor = None,
                 pies_init: Tensor = None):
        device = tvem.get_device()

        required_keys = ('N', 'D', 'H', 'S', 'Snew', 'batch_size', 'dtype')
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.conf = conf
        self.required_keys = required_keys

        N, D, H, S, Snew, batch_size, dtype = get(conf, *required_keys)

        self.tmp = {
            "my_Wp": to.empty((D, H), dtype=dtype, device=device),
            "my_Wq": to.empty((H, H), dtype=dtype, device=device),
            "my_pies": to.empty(H, dtype=dtype, device=device),
            "my_sigma": to.empty(1, dtype=dtype, device=device),
            "pil_bar_cuda": to.empty(H, dtype=dtype, device=device),
            "pil_bar_cpu": to.empty(H, dtype=dtype, device=to.device('cpu')),
            "WT_cuda": to.empty((H, D), dtype=dtype, device=device),
            "WT_cpu": to.empty((H, D), dtype=dtype, device=to.device('cpu')),
            "pjc": to.empty((batch_size, S), dtype=dtype, device=device),
            "pjc_sum": to.empty(batch_size, dtype=dtype, device=device),
            "batch_Wbar": to.empty((batch_size, S+Snew, D), dtype=dtype, device=device),
            "batch_s_pjc": to.empty((batch_size, H), dtype=dtype, device=device),
            "batch_Wp": to.empty((batch_size, D, H), dtype=dtype, device=device),
            "batch_Wq": to.empty((batch_size, H, H), dtype=dtype, device=device),
            "batch_sigma": to.empty((batch_size,), dtype=dtype, device=device),
        }

        assert W_init is None or W_init.shape == (
            D, H) and W_init.device == device
        assert pies_init is None or pies_init.shape == (
            H,) and pies_init.device == device
        assert sigma_init is None or sigma_init.shape == (
            1,) and sigma_init.device == device

        theta = {
            'pies': pies_init if pies_init is not None else to.full((H,), 1./H,
                                                                    dtype=dtype, device=device),
            'W': W_init if W_init is not None else to.rand((D, H), dtype=dtype, device=device),
            'sigma': sigma_init if sigma_init is not None else to.tensor([1., ],
                                                                         dtype=dtype,
                                                                         device=device)}
        super().__init__(theta=theta)

    @property
    def sorted_by_lpj(self) -> Dict[str, Tensor]:

        tmp = self.tmp

        return {'batch_Wbar': tmp['batch_Wbar']}

    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Use hidden states to sample datapoints according to the noise model of BSC.

        :param hidden_state: a tensor with shape (N, H) where H is the number of hidden units.
        :returns: the datapoints, as a tensor with shape (N, D) where D is
                  the number of observables.
        """

        theta = self.theta

        dtype_f, device = theta['W'].dtype, theta['W'].device
        no_datapoints, D, H_gen = (hidden_state.shape[0],) + theta['W'].shape

        Wbar = to.zeros((no_datapoints, D), dtype=dtype_f, device=device)

        # Linear superposition
        for n in range(no_datapoints):
            for h in range(H_gen):
                if hidden_state[n, h]:
                    Wbar[n] += theta['W'][:, h]

        # Add noise according to the model parameters
        Y = Wbar + theta['sigma'] * \
            to.randn((no_datapoints, D), dtype=dtype_f, device=device)

        return Y

    def init_epoch(self):
        """Allocate and/or initialize tensors used during EM-step."""

        conf = self.conf
        theta = self.theta
        tmp = self.tmp

        D = conf['D']

        tmp["my_Wp"].fill_(0.)
        tmp["my_Wq"].fill_(0.)
        tmp["my_pies"].fill_(0.)
        tmp["my_sigma"].fill_(0.)

        tmp["pil_bar_cuda"][:] = to.log(
            theta["pies"]/(1.-theta["pies"]))
        tmp["pil_bar_cpu"][:] = tmp["pil_bar_cuda"].to(
            device=to.device('cpu'))

        tmp["WT_cuda"][:, :] = theta["W"].t()
        tmp["WT_cpu"][:, :] = tmp["WT_cuda"].to(
            device=to.device('cpu'))

        tmp["pre1"] = -1./2./theta["sigma"]/theta["sigma"]

        tmp["fenergy_const"] = to.log(1.-theta["pies"]).sum()\
            - D/2*to.log(2*math.pi*theta["sigma"]**2)

        tmp["infty"] = to.tensor(
            [float('inf')], dtype=theta["pies"].dtype, device=theta["pies"].device)

        tmp["indS_fill_upto"] = 0

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints for BSC."""

        theta = self.theta
        tmp = self.tmp
        sorted_by_lpj = self.sorted_by_lpj

        batch_size, S, _ = states.shape
        device_type = states.device.type

        pil_bar = tmp['pil_bar_%s' % device_type]
        WT = tmp['WT_%s' % device_type]
        pre1 = tmp['pre1']
        indS_fill_upto = tmp["indS_fill_upto"]

        # TODO Find solution to avoid byte->float casting
        statesfloat = states.to(dtype=theta['W'].dtype)

        # TODO Store batch_Wbar in storage allocated at beginning of EM-step, e.g.
        # to.matmul(tensor1=states, tensor2=tmp['WT'], out=tmp["batch_Wbar"])
        sorted_by_lpj['batch_Wbar'][:batch_size, indS_fill_upto:(
            indS_fill_upto+S), :] = to.matmul(statesfloat, WT)
        batch_Wbar = sorted_by_lpj['batch_Wbar'][:batch_size, indS_fill_upto:(
            indS_fill_upto+S), :]
        tmp['indS_fill_upto'] += S
        # is (batch_size,S)
        lpj = to.mul(to.sum(to.pow(batch_Wbar-data[:, None, :], 2), dim=2), pre1) +\
            to.einsum('ijk,k->ij', statesfloat, pil_bar)
        return lpj.to(device=states.device)

    def free_energy(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> float:

        conf = self.conf
        tmp = self.tmp

        N, D, H = get(conf, *('N', 'D', 'H'))
        fenergy_const = tmp['fenergy_const']

        lpj = self.log_pseudo_joint(batch, states.K[idx])
        B = 0. - to.max(lpj, dim=1, keepdim=True)[0]
        lpj_shifted_sum_chunk = (to.logsumexp(
            lpj + B.expand_as(lpj), dim=1) - B.flatten()).sum()
        all_reduce(lpj_shifted_sum_chunk)

        return (fenergy_const*N + lpj_shifted_sum_chunk).item()

    def update_param_batch(self, idx: Tensor, batch: Tensor,
                           states: TVEMVariationalStates) -> float:

        conf = self.conf
        tmp = self.tmp
        sorted_by_lpj = self.sorted_by_lpj

        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        # TODO Find solution to avoid byte->float casting
        Kfloat = K.to(dtype=lpj.dtype)

        N = conf['N']
        pjc, pjc_sum, batch_s_pjc, batch_Wp, batch_Wq, batch_sigma, my_pies, my_Wp, my_Wq,\
            my_sigma, indS_fill_upto,\
            fenergy_const = get(tmp, *('pjc', 'pjc_sum', 'batch_s_pjc', 'batch_Wp', 'batch_Wq',
                                       'batch_sigma', 'my_pies', 'my_Wp', 'my_Wq', 'my_sigma',
                                       'indS_fill_upto', 'fenergy_const'))
        batch_Wbar = sorted_by_lpj['batch_Wbar']

        B = 0. - to.max(lpj, dim=1, keepdim=True)[0]
        to.exp(lpj + B, out=pjc[:batch_size, :])

        pjc_sum[:batch_size] = to.sum(
            pjc[:batch_size, :], dim=1)  # is (batch_size,)

        batch_s_pjc[:batch_size, :] = to.einsum(
            'ij,ijl->il', (pjc[:batch_size, :], Kfloat))  # is (batch_size,H)
        batch_Wp[:batch_size, :, :] = to.einsum(
            'nd,nh->ndh', (batch, batch_s_pjc[:batch_size, :]))  # is (batch_size,D,H)
        batch_Wq[:batch_size, :, :] = to.einsum(
            'ij,ijkl->ikl', (pjc[:batch_size, :], to.einsum('ijk,ijl->ijkl', (Kfloat, Kfloat))))
        # is (batch_size,H,H)
        batch_sigma[:batch_size] = to.einsum('ij,ij->i', (pjc[:batch_size, :], to.sum(
            (batch[:, None, :]-batch_Wbar[:batch_size, :S, :])**2, dim=2)))
        # is (batch_size,)

        my_pies.add_(
            to.sum(batch_s_pjc[:batch_size, :] / pjc_sum[:batch_size, None], dim=0))
        my_Wp.add_(to.sum(batch_Wp[:batch_size, :, :] /
                          pjc_sum[:batch_size, None, None], dim=0))
        my_Wq.add_(to.sum(batch_Wq[:batch_size, :, :] /
                          pjc_sum[:batch_size, None, None], dim=0))
        my_sigma.add_(to.sum(batch_sigma[:batch_size] / pjc_sum[:batch_size]))

        # batch free energy
        lpj_shifted_sum_chunk = (to.logsumexp(
            pjc[:batch_size, :], dim=1) - B).sum()
        all_reduce(lpj_shifted_sum_chunk)

        return (fenergy_const*N + lpj_shifted_sum_chunk).item()

    def update_param_epoch(self) -> None:

        conf = self.conf
        theta = self.theta
        tmp = self.tmp

        N, H = get(conf, *('N', 'H'))
        my_pies, my_Wp, my_Wq, my_sigma = get(
            tmp, *('my_pies', 'my_Wp', 'my_Wq', 'my_sigma'))

        theta_new = {}

        all_reduce(my_pies)
        all_reduce(my_Wp)
        all_reduce(my_Wq)
        all_reduce(my_sigma)

        # Calculate updated W
        try:
            # to.gels assumes full-commrank matrices
            theta_new['W'] = to.gels(my_Wp.t(), my_Wq)[0].t()
            if to.isnan(theta_new['W']).any() or to.isinf(theta_new['W']).any():
                pprint("Infinite Wnew. Will not update W but add some noise instead.")
                W_old = theta['W']
                theta_new['W'] = W_old + to.randn(
                    W_old.shape, dtype=W_old.dtype, device=W_old.device)
        except RuntimeError:
            pprint("Infinite Wnew. Will not update W but add some noise instead.")
            W_old = theta['W']
            theta_new['W'] = W_old + \
                to.randn(W_old.shape, dtype=W_old.dtype, device=W_old.device)

        # Calculate updated pi
        theta_new['pies'] = my_pies / N
        if to.isnan(theta_new['pies']).any() or to.isinf(theta_new['pies']).any():
            pprint("Infinite pies. Will not update pies.")
            theta_new['pies'] = theta['pies']
        eps = 1.e-5
        theta_new['pies'][theta_new['pies'] <= eps] = eps

        # Calculate updated sigma
        theta_new['sigma'] = to.sqrt(my_sigma / N / ((H / 2)**2))
        if to.isnan(theta_new['sigma']).any() or to.isinf(theta_new['sigma']).any():
            pprint("Infinite sigma. Will not update sigma.")
            theta_new['sigma'] = theta['sigma']

        for key in theta:
            theta[key] = theta_new[key]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta['W'].shape
