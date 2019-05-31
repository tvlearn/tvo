# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.models import TVEMModel
from tvem.variational import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.parallel import all_reduce
from tvem.utils import get
import tvem
import torch as to
from typing import Tuple, List, Dict, Iterable, Optional, Sequence
from math import pi as MATH_PI


class TVAE(TVEMModel):
    def __init__(
        self,
        N: int,
        shape: Sequence[int],
        precision: to.dtype = to.float32,
        pi_init: to.Tensor = None,
        W_init: Iterable[to.Tensor] = None,
        b_init: Iterable[to.Tensor] = None,
        sigma2_init: float = None,
    ):
        """Create a TVAE model.

        :param shape: network shape, from observable to most hidden: (D,...,H1,H0)
        :param precision: one of to.float32 or to.float64, indicates the floating point precision
                          of model parameters.
        :param pi_init: optional tensor with initial prior values
        :param W_init: optional list of tensors with initial weight values
        :param b_init: optional list of tensors with initial
        :param sigma2_init: optional initial value for model variance
        """
        theta = {}
        self.precision = precision
        self._net_shape = tuple(reversed(shape))
        self.W = self._init_W(W_init)
        self.b = self._init_b(b_init)
        theta.update({f"W_{i}": W for i, W in enumerate(self.W)})
        theta.update({f"b_{i}": b for i, b in enumerate(self.b)})
        H0 = self._net_shape[0]
        theta["pies"] = self._init_pi(pi_init, H0)
        theta["sigma2"] = self._init_sigma2(sigma2_init)
        super().__init__(theta)

        self._new_pi = to.zeros(H0, dtype=precision, device=tvem.get_device())
        self._new_sigma2 = to.zeros(1, dtype=precision, device=tvem.get_device())
        self._N = N

    def _init_W(self, init: Optional[Iterable[to.Tensor]]) -> List[to.Tensor]:
        shape = self._net_shape
        if init is None:
            n_layers = len(shape) - 1
            W_shapes = ((shape[l], shape[l + 1]) for l in range(n_layers))
            W = (self._from_normal(s) for s in W_shapes)
        else:
            assert all(w.shape == (shape[l], shape[l + 1]) for l, w in enumerate(init))
            W = (w.clone() for w in init)
        return [
            w.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(True) for w in W
        ]

    def _init_b(self, init: Optional[Iterable[to.Tensor]]) -> List[to.Tensor]:
        if init is None:
            B = [self._from_normal((s,)) for s in self._net_shape[1:]]
        else:
            assert all(b.shape == (self._net_shape[l + 1],) for l, b in enumerate(init))
            B = [b.clone() for b in init]
        return [
            b.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(True) for b in B
        ]

    def _init_pi(self, init: Optional[to.Tensor], H0: int) -> to.Tensor:
        if init is None:
            pi = to.full((H0,), 1 / H0)
        else:
            assert init.shape == (H0,)
            pi = init.clone()
        return pi.to(device=tvem.get_device(), dtype=self.precision)

    def _init_sigma2(self, init: Optional[float]) -> to.Tensor:
        sigma2 = to.tensor([0.01] if init is None else init)
        return sigma2.to(device=tvem.get_device(), dtype=self.precision)

    @staticmethod
    def _from_normal(shape: Iterable[int]) -> to.Tensor:
        return to.distributions.Normal(0.0, 0.1).sample(shape)

    def log_pseudo_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        with to.no_grad():
            lpj, _ = self._lpj_and_mlpout(data, states)
        return lpj

    def _lpj_and_mlpout(self, data: to.Tensor, states: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        N, D = data.shape
        N_, S, H = states.shape
        assert N == N_, "Shape mismatch between data and states"
        pi, sigma2 = get(self.theta, "pies", "sigma2")
        states = states.to(dtype=self.precision)

        mlp_out = self._mlp_forward(states)  # (N, S, D)
        s1 = (data.unsqueeze(1) - mlp_out).pow_(2).sum(dim=2).div_(2 * sigma2)  # (N, S)
        s2 = states @ to.log(pi / (1.0 - pi))  # (N, S)
        lpj = s2 - s1
        assert lpj.shape == (N, S)
        return lpj, mlp_out

    def free_energy(self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates) -> float:
        with to.no_grad():
            return self._free_energy_from_logjoints(states.lpj[idx]).item()

    def _free_energy_from_logjoints(self, lpj: to.Tensor) -> to.Tensor:
        PI = to.tensor(MATH_PI)
        pi, sigma2 = get(self.theta, "pies", "sigma2")
        D = self._net_shape[-1]
        # TODO summands that do not depend on N can be brought outside of the logsumexp
        # logjoints has shape (N, S)
        logjoints = lpj - D / 2 * to.log(2 * PI * sigma2) + to.log(1 - pi).sum()
        Fn = to.logsumexp(logjoints, dim=1)
        assert Fn.shape == (lpj.shape[0],)
        return Fn.sum()

    def update_param_batch(
        self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates
    ) -> float:
        F, mlp_out = self._optimize_nn_params(idx, batch, states)
        with to.no_grad():
            self._accumulate_param_updates(idx, batch, states, mlp_out)
        return F

    def update_param_epoch(self) -> None:
        N, D = self._N, self._net_shape[-1]

        all_reduce(self._new_pi)
        self.theta["pies"][:] = self._new_pi / N
        # avoids infinites in lpj evaluation
        to.clamp(self.theta["pies"], 1e-5, 1 - 1e-5, out=self.theta["pies"])

        sigma2 = self.theta["sigma2"]
        all_reduce(self._new_sigma2)
        sigma2[:] = self._new_sigma2 / (N * D)
        # disallow arbitrary growth of sigma. at each iteration, it can grow by at most 1%
        to.clamp(
            sigma2,
            (sigma2 - sigma2.abs() * 0.01).item(),
            (sigma2 + sigma2.abs() * 0.01).item(),
            out=sigma2,
        )

        # TODO when using a proper optimizer, just call `zero_grad()` on that
        for W, b in zip(self.W, self.b):
            W.grad.zero_()
            b.grad.zero_()
        self._new_pi.zero_()
        self._new_sigma2.zero_()

    def generate_from_hidden(self, hidden_state: to.Tensor) -> Dict[str, to.Tensor]:
        with to.no_grad():
            mlp_out = self._mlp_forward(hidden_state)
        return to.distributions.Normal(loc=mlp_out, scale=self.theta["sigma2"]).sample()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of TVAE model as a bayes net: (D, H0)

        Neural network shape is ignored.
        """
        return tuple((self._net_shape[-1], self._net_shape[0]))

    @property
    def net_shape(self) -> Tuple[int, ...]:
        """Full TVAE network shape (D, Hn, Hn-1, ..., H0)."""
        return tuple(reversed(self._net_shape))

    def _optimize_nn_params(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates
    ) -> Tuple[float, to.Tensor]:
        """
        W, b are changed in-place. All other arguments are left untouched.
        :returns: F and mlp_output _before_ the weight update
        """
        learn_rate = 0.01  # TODO make this a configurable parameter

        lpj, mlp_out = self._lpj_and_mlpout(data, states.K[idx])
        F = self._free_energy_from_logjoints(lpj)
        loss = -F / data.shape[0]
        loss.backward()

        with to.no_grad():
            for w, b in zip(self.W, self.b):
                all_reduce(w.grad)
                all_reduce(b.grad)

            # TODO use a proper optimizer, e.g. adam
            for l in range(len(self.W)):
                self.W[l][:] = self.W[l] - learn_rate * self.W[l].grad
                self.b[l][:] = self.b[l] - learn_rate * self.b[l].grad

        return F.item(), mlp_out

    def _accumulate_param_updates(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates, mlp_out: to.Tensor
    ) -> None:
        """Evaluate partial updates to pi and sigma2."""
        # \pi_h = \frac{1}{N} \sum_n < K_nsh >_{q^n}
        K_batch = states.K[idx].type_as(states.lpj)
        self._new_pi.add_(mean_posterior(K_batch, states.lpj[idx]).sum(dim=0))

        # \sigma2 = \frac{1}{DN} \sum_{n,d} < (y^n_d - \vec{a}^L_d)^2 >_{q^n}
        y_minus_a_sqr = (data.unsqueeze(1) - mlp_out).pow_(2)  # (D, N, S)
        self._new_sigma2.add_(mean_posterior(y_minus_a_sqr, states.lpj[idx]).sum((0, 1)))

    def _mlp_forward(self, x: to.Tensor) -> to.Tensor:
        """Forward application of TVAE's MLP to the specified input."""
        assert x.shape[-1] == self._net_shape[0], "Incompatible shape in _mlp_forward input"

        # middle layers (relu)
        output = x.to(dtype=self.precision)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            output = to.relu(output @ W + b)

        # output layer (linear)
        output = output @ self.W[-1] + self.b[-1]

        return output
