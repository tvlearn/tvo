# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


from .TVEMModel import TVEMModel
from tvem.variational import TVEMVariationalStates  # type: ignore
from tvem.util.parallel import all_reduce
from torch import Tensor
import torch as to
from typing import Dict, Optional, Tuple
import tvem


class NoisyOR(TVEMModel):
    eps = 1e-7

    def __init__(self, H: int, D: int, W_init: Tensor = None, pi_init: Tensor = None):
        """Shallow NoisyOR model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes NoisyOR weights.
        :param pi_init: Tensor with shape (H,), initializes NoisyOR priors.
        """

        device = tvem.get_device()
        if W_init is not None:
            assert W_init.shape == (D, H)
        else:
            W_init = to.rand(D, H, device=device)
        if pi_init is not None:
            assert pi_init.shape == (H,)
            assert (pi_init <= 1.).all() and (pi_init >= 0).all()
        else:
            pi_init = to.full((H,), 1./H, device=device)

        super().__init__(theta={'pies': pi_init.to(device=device), 'W': W_init.to(device=device)})
        self.new_pi = to.zeros(H, device=device)
        self.Btilde = to.zeros(D, H, device=device)
        self.Ctilde = to.zeros(D, H, device=device)

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints for NoisyOR."""
        K = states
        Y = data
        assert K.dtype == to.uint8 and Y.dtype == to.uint8
        pi = self.theta['pies']
        W = self.theta['W']
        N, S, H = K.shape
        D = W.shape[0]
        dev = pi.device
        logPy = to.empty((N, S), device=dev)
        logPriors = to.matmul(K.type_as(pi), to.log(pi/(1-pi)))
        # the general routine for eem sometimes require evaluation of lpjs of all-zero states,
        # which is not supported for noisy-OR.
        # we instead manually set the lpjs of all-zero states to low values to decrease the
        # probability they will be selected as new variational states.
        # in case they are nevertheless selected by eem, they are discarded by noisy-OR.
        zeroStatesInd = to.nonzero((K == 0).all(dim=2))
        # https://discuss.pytorch.org/t/use-torch-nonzero-as-index/33218
        zeroStatesInd = (zeroStatesInd[:, 0], zeroStatesInd[:, 1])
        K[zeroStatesInd] = 1
        # prods_nkd = prod{h}{1-W_dh*K_nkh}
        prods = to.broadcast_tensors(to.empty(N, S, H, D), W.transpose(0, 1))[1].clone()
        prod_mask = (~K).unsqueeze(-1).expand(N, S, H, D)
        prods[prod_mask] = 0.
        prods = to.prod(1 - prods, dim=2)
        # logPy_nk = sum{d}{y_nd*log(1/prods_nkd - 1) + log(prods_nkd)}
        f1 = to.log(1./(prods + self.eps) - 1.)
        indeces = Y[:, None, :].expand(N, S, D)
        f1[~indeces] = 0.
        logPy[:, :] = to.sum(f1, dim=-1) + to.sum(to.log(prods), dim=2)
        K[zeroStatesInd] = 0
        # return lpj_nk
        lpj = logPriors + logPy
        lpj[zeroStatesInd] = -1e30  # arbitrary very low value
        assert not to.isnan(lpj).any(), 'some NoisyOR lpj values are nan!'
        return lpj.to(device=states.device)

    def update_param_batch(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates,
                           mstep_factors: Dict[str, Tensor] = None) -> Optional[float]:
        lpj = states.lpj[idx]
        K = states.K[idx]
        Kfloat = K.type_as(lpj)
        deltaY = (batch.any(dim=1) == 0).type_as(lpj)
        N = states.K.shape[0]

        # pi_h = sum{n}{<K_nkh>} / N
        self.new_pi += to.sum(self._mean_posterior(K.permute(2, 0, 1), lpj, deltaY), dim=1) / N
        assert not to.isnan(self.new_pi).any()

        # Ws_dnkh = 1 - (W_dh * K_nkh)
        Ws = 1 - self.theta['W'][:, None, None, :] * Kfloat[None, :, :, :]
        Ws_prod = to.prod(Ws, dim=-1, keepdim=True)
        B = Kfloat / ((Ws * (1 - Ws_prod)) + self.eps)
        self.Btilde.add_(to.einsum('ijk,ki->ij',
                                   self._mean_posterior(B.permute(0, 3, 1, 2), lpj, deltaY),
                                   batch.type_as(lpj) - 1))
        C = Ws_prod * B / Ws
        self.Ctilde.add_(to.sum(self._mean_posterior(C.permute(0, 3, 1, 2), lpj, deltaY), dim=2))
        assert not to.isnan(self.Ctilde).any()
        assert not to.isnan(self.Btilde).any()

        return None

    def update_param_epoch(self) -> None:
        all_reduce(self.new_pi)
        self.theta['pies'][:] = self.new_pi
        to.clamp(self.theta['pies'], self.eps, 1 - self.eps, out=self.theta['pies'])
        self.new_pi[:] = 0.

        all_reduce(self.Btilde)
        all_reduce(self.Ctilde)
        self.theta['W'][:] = 1 + self.Btilde / (self.Ctilde + self.eps)
        to.clamp(self.theta['W'], self.eps, 1 - self.eps, out=self.theta['W'])
        self.Btilde[:] = 0.
        self.Ctilde[:] = 0.

    def free_energy(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> float:
        pi = self.theta['pies']
        N = batch.shape[0]
        # deltaY_n is 1 if Y_nd == 0 for each d, 0 otherwise (shape=(N))
        deltaY = (batch.any(dim=1) == 0).type_as(states.lpj)
        F = N * to.sum(to.log(1 - pi))\
            + to.sum(to.log(to.sum(to.exp(states.lpj[idx]) + self.eps, dim=1) + deltaY))
        assert not (to.isnan(F) or to.isinf(F)), 'free energy is nan!'
        return F.item()

    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Use hidden states to sample datapoints according to the NoisyOR generative model.

        :param hidden_state: a tensor with shape (N, H) where H is the number of hidden units.
        :returns: the datapoints, as a tensor with shape (N, D) where D is
                  the number of observables.
        """
        N, H = hidden_state.shape
        expected_H = self.theta['pies'].numel()
        assert H == expected_H, f'input has wrong shape, expected {(N, expected_H)}, got {(N, H)}'
        W = self.theta['W']
        # py_nd = 1 - prod_h (1 - W_dh * s_nh)
        py = 1 - to.prod(1 - W[None, :, :] * hidden_state.type_as(W)[:, None, :], dim=2)
        return to.rand_like(py) < py

    @staticmethod
    def _mean_posterior(g: Tensor, lpj: Tensor, deltaY: Tensor):
        """Evaluate the mean of array g over the posterior probability distribution.

        The array is assumed to have shape (...,N,K) where N is the number of
        data-points and K is the number of configurations considered by TVEM
        (nConfs).

        The array returned has shape (...,N). Each component is the average of
        g[...,n,k], over k, weighted by lpj[k,n].
        """

        # Evaluate constants B_n by which we can translate lpj
        B = -to.max(lpj, dim=1, keepdim=True)[0]
        to.clamp(B, -80, 80, out=B)

        # sum{k}{g_ink*exp(lpj_nk + B)} / (sum{k}{exp(lpj_nk + B)}
        explpj = to.exp(lpj + B)
        denominator = to.sum(explpj, dim=1) + deltaY.type_as(B)*to.exp(B[:, 0])
        means = to.einsum('...ij,ij->...i', g.type_as(lpj), explpj) / (denominator + NoisyOR.eps)
        assert not (to.isnan(means).any() or to.isinf(means).any())
        return means

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta['W'].shape
