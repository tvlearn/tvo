# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


from tvo.utils.model_protocols import Optimized, Sampler
from tvo.variational import TVOVariationalStates  # type: ignore
from tvo.variational._utils import mean_posterior
from tvo.utils.parallel import all_reduce, broadcast
from torch import Tensor
import torch as to
from typing import Dict, Optional, Union, Tuple
import tvo


class NoisyOR(Optimized, Sampler):
    eps = 1e-7

    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        pi_init: Tensor = None,
        precision: to.dtype = to.float64,
    ):
        """Shallow NoisyOR model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes NoisyOR weights.
        :param pi_init: Tensor with shape (H,), initializes NoisyOR priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.
        """

        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        self._precision = precision

        device = tvo.get_device()

        if W_init is not None:
            assert W_init.shape == (D, H)
        else:
            W_init = to.rand(D, H, device=device)
            broadcast(W_init)

        if pi_init is not None:
            assert pi_init.shape == (H,)
            assert (pi_init <= 1.0).all() and (pi_init >= 0).all()
        else:
            pi_init = to.full((H,), 1.0 / H, device=device, dtype=self.precision)

        self._theta = {
            "pies": pi_init.to(device=device, dtype=precision),
            "W": W_init.to(device=device, dtype=precision),
        }

        self.new_pi = to.zeros(H, device=device, dtype=precision)
        self.Btilde = to.zeros(D, H, device=device, dtype=precision)
        self.Ctilde = to.zeros(D, H, device=device, dtype=precision)
        # number of datapoints processed in a training epoch
        self._train_datapoints = to.tensor([0], dtype=to.int, device=device)
        self._config = dict(H=H, D=D, precision=self.precision, device=device)
        self._shape = self.theta["W"].shape

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:  # type: ignore
        """Evaluate log-pseudo-joints for NoisyOR."""
        K = states
        Y = data
        assert K.dtype == to.uint8 and Y.dtype == to.uint8
        pi = self.theta["pies"]
        W = self.theta["W"]
        batch_size, S, H = K.shape
        D = W.shape[0]
        dev = pi.device

        logPriors = to.matmul(K.type_as(pi), to.log(pi / (1 - pi)))

        logPy = to.empty((batch_size, S), device=dev, dtype=self.precision)
        # We will manually set the lpjs of all-zero states to the appropriate value.
        # For now, transform all-zero states in all-one states, to avoid computation of log(0).
        zeroStatesInd = to.nonzero((K == 0).all(dim=2))
        # https://discuss.pytorch.org/t/use-torch-nonzero-as-index/33218
        zeroStatesInd = (zeroStatesInd[:, 0], zeroStatesInd[:, 1])
        K[zeroStatesInd] = 1
        # prods_nsd = prod{h}{1-W_dh*K_nkh}
        prods = (W * K.type_as(W).unsqueeze(2)).neg_().add_(1).prod(dim=-1)
        to.clamp(prods, self.eps, 1 - self.eps, out=prods)
        # logPy_nk = sum{d}{y_nd*log(1/prods_nkd - 1) + log(prods_nkd)}
        f1 = to.log(1.0 / prods - 1.0)
        indeces = 1 - Y[:, None, :].expand(batch_size, S, D)
        # convert to BoolTensor in pytorch>=1.2, leave it as ByteTensor in earlier versions
        indeces = indeces.type_as(to.empty(0) < 0)
        f1[indeces] = 0.0
        logPy[:, :] = to.sum(f1, dim=-1) + to.sum(to.log(prods), dim=2)
        K[zeroStatesInd] = 0

        lpj = logPriors + logPy
        # for all-zero states, set lpj to arbitrary very low value if y!=0, 0 otherwise
        # in the end we want exp(lpj(y,s=0)) = 1 if y=0, 0 otherwise
        lpj[zeroStatesInd] = -1e30 * data[zeroStatesInd[0]].any(dim=1).type_as(lpj)
        assert (
            not to.isnan(lpj).any() and not to.isinf(lpj).any()
        ), "some NoisyOR lpj values are invalid!"
        return lpj.to(device=states.device)  # (N, S)

    def update_param_batch(
        self,
        idx: Tensor,
        batch: Tensor,
        states: TVOVariationalStates,
        mstep_factors: Dict[str, Tensor] = None,
    ) -> Optional[float]:
        lpj = states.lpj[idx]
        K = states.K[idx]
        Kfloat = K.type_as(lpj)

        # pi_h = sum{n}{<K_hns>} / N
        # (division by N has to wait until after the mpi all_reduce)
        self.new_pi += mean_posterior(Kfloat, lpj).sum(dim=0)
        assert not to.isnan(self.new_pi).any()

        # Ws_nsdh = 1 - (W_dh * Kfloat_nsh)
        Ws = (self.theta["W"][None, None, :, :] * Kfloat[:, :, None, :]).neg_().add_(1)
        Ws_prod = to.prod(Ws, dim=3, keepdim=True)
        B = Kfloat.unsqueeze(2) / (Ws * Ws_prod.neg().add_(1)).add_(self.eps)  # (N,S,D,H)
        self.Btilde.add_(
            (mean_posterior(B, lpj) * (batch.type_as(lpj) - 1).unsqueeze(2)).sum(dim=0)
        )
        C = B.mul_(Ws_prod).div_(Ws)  # (N,S,D,H)
        self.Ctilde.add_(to.sum(mean_posterior(C, lpj), dim=0))
        assert not to.isnan(self.Ctilde).any()
        assert not to.isnan(self.Btilde).any()

        self._train_datapoints.add_(batch.shape[0])

        return None

    def update_param_epoch(self) -> None:
        all_reduce(self._train_datapoints)
        all_reduce(self.new_pi)
        N = self._train_datapoints.item()
        self.theta["pies"][:] = self.new_pi / N
        to.clamp(self.theta["pies"], self.eps, 1 - self.eps, out=self.theta["pies"])
        self.new_pi[:] = 0.0

        all_reduce(self.Btilde)
        all_reduce(self.Ctilde)
        self.theta["W"][:] = 1 + self.Btilde / (self.Ctilde + self.eps)
        to.clamp(self.theta["W"], self.eps, 1 - self.eps, out=self.theta["W"])
        self.Btilde[:] = 0.0
        self.Ctilde[:] = 0.0

        self._train_datapoints[:] = 0

    def log_joint(self, data, states, lpj=None):
        pi = self.theta["pies"]
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        # TODO: could pre-evaluate the constant factor once per epoch
        return to.sum(to.log(1 - pi)) + lpj

    def generate_data(
        self, N: int = None, hidden_state: Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        """Use hidden states to sample datapoints according to the NoisyOR generative model.

        :param hidden_state: a tensor with shape (N, H) where H is the number of hidden units.
        :returns: the datapoints, as a tensor with shape (N, D) where D is
                  the number of observables.
        """
        theta = self.theta
        W = theta["W"]
        D, H = W.shape

        if hidden_state is None:
            pies = theta["pies"]
            hidden_state = to.rand((N, H), dtype=pies.dtype, device=pies.device) < pies
            must_return_hidden_state = True
        else:
            if N is not None:
                shape = hidden_state.shape
                assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        # py_nd = 1 - prod_h (1 - W_dh * s_nh)
        py = 1 - to.prod(1 - W[None, :, :] * hidden_state.type_as(W)[:, None, :], dim=2)
        Y = (to.rand_like(py) < py).byte()

        return (Y, hidden_state) if must_return_hidden_state else Y
