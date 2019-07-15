# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.models import TVEMModel
from tvem.variational import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.parallel import all_reduce
from tvem.utils import get, CyclicLR
import torch.optim as opt
import tvem
import torch as to
import torch.distributed as dist
from typing import Tuple, List, Dict, Iterable, Optional, Sequence
from math import pi as MATH_PI


class TVAE(TVEMModel):
    def __init__(
        self,
        N: int = None,
        shape: Sequence[int] = None,
        precision: to.dtype = to.float64,
        min_lr: float = 0.001,
        max_lr: float = 0.01,
        cycliclr_step_size_up=400,
        pi_init: to.Tensor = None,
        W_init: Sequence[to.Tensor] = None,
        b_init: Sequence[to.Tensor] = None,
        sigma2_init: float = None,
        analytical_sigma_updates: bool = True,
        analytical_pi_updates: bool = True,
        clamp_sigma_updates: bool = False,
    ):
        """Create a TVAE model.

        :param N: Number of datapoints used for training. Only required if TVAE is to be trained
                  and one of the analytical_*_updates parameters is True. In MPI runs, must be the
                  _total_ number of datapoints.
        :param shape: Network shape, from observable to most hidden: (D,...,H1,H0).
                      Can be None if W_init is not None.
        :param precision: One of to.float32 or to.float64, indicates the floating point precision
                          of model parameters.
        :param pi_init: Optional tensor with initial prior values
        :param W_init: Optional list of tensors with initial weight values. Weight matrices
                       must be ordered from most hidden to observable layer. If this parameter
                       is not None, the shape parameter can be omitted.
        :param b_init: Optional list of tensors with initial.
        :param sigma2_init: Optional initial value for model variance.
        :param analytical_sigma_updates: Whether sigmas should be updated via the analytical
                                         max-likelihood solution rather than gradient descent.
        :param analytical_pi_updates: Whether priors should be updated via the analytical
                                      max-likelihood solution rather than gradient descent.
        :param clamp_sigma_updates: Whether to limit the rate at which sigma can be updated.
        """
        theta = {}
        self._clamp_sigma = clamp_sigma_updates
        self.precision = precision
        self._net_shape = self._get_net_shape(shape, W_init)
        self.W = self._init_W(W_init)
        self.b = self._init_b(b_init)
        self._N = N
        theta.update({f"W_{i}": W for i, W in enumerate(self.W)})
        theta.update({f"b_{i}": b for i, b in enumerate(self.b)})
        H0 = self._net_shape[0]
        theta["pies"] = self._init_pi(pi_init, H0, requires_grad=not analytical_pi_updates)
        theta["sigma2"] = self._init_sigma2(sigma2_init, requires_grad=not analytical_sigma_updates)
        super().__init__(theta)

        gd_parameters = self.W + self.b

        if analytical_sigma_updates:
            self._new_sigma2 = to.zeros(1, dtype=precision, device=tvem.get_device())
        else:
            gd_parameters.append(theta["sigma2"])

        if analytical_pi_updates:
            self._new_pi = to.zeros(H0, dtype=precision, device=tvem.get_device())
        else:
            gd_parameters.append(theta["pies"])

        self._optimizer = opt.Adam(gd_parameters, lr=min_lr)
        self._scheduler = CyclicLR(
            self._optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=cycliclr_step_size_up,
            cycle_momentum=False,
        )

    def _get_net_shape(self, shape: Sequence[int] = None, W_init: Sequence[to.Tensor] = None):
        if shape is not None:
            return tuple(reversed(shape))
        else:
            assert W_init is not None, "Must pass one of `shape` and `W_init` to TVAE.__init__"
            return tuple(w.shape[0] for w in W_init) + (W_init[-1].shape[1],)

    def _init_W(self, init: Optional[Iterable[to.Tensor]]) -> List[to.Tensor]:
        """Return weights initialized with Xavier or to specified init values.

        This method also makes sure that device and precision are the ones required by the model.
        """
        shape = self._net_shape
        if init is None:
            n_layers = len(shape) - 1
            W_shapes = ((shape[l], shape[l + 1]) for l in range(n_layers))
            W = map(to.nn.init.xavier_normal_, (to.empty(s) for s in W_shapes))
        else:
            assert all(
                w.shape == (shape[l], shape[l + 1]) for l, w in enumerate(init)
            ), "shapes: {}".format([w.shape for w in init])
            W = (w.clone() for w in init)
        return [
            w.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(True) for w in W
        ]

    def _init_b(self, init: Optional[Iterable[to.Tensor]]) -> List[to.Tensor]:
        """Return biases initialized to zeros or to specified init values.

        This method also makes sure that device and precision are the ones required by the model.
        """
        if init is None:
            B = [to.zeros(s) for s in self._net_shape[1:]]
        else:
            assert all(b.shape == (self._net_shape[l + 1],) for l, b in enumerate(init))
            B = [b.clone() for b in init]
        return [
            b.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(True) for b in B
        ]

    def _init_pi(self, init: Optional[to.Tensor], H0: int, requires_grad: bool) -> to.Tensor:
        if init is None:
            pi = to.full((H0,), 1 / H0)
        else:
            assert init.shape == (H0,)
            pi = init.clone()
        return pi.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(requires_grad)

    def _init_sigma2(self, init: Optional[float], requires_grad: bool) -> to.Tensor:
        sigma2 = to.tensor([0.01] if init is None else [init])
        return sigma2.to(device=tvem.get_device(), dtype=self.precision).requires_grad_(
            requires_grad
        )

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

        mlp_out = self.forward(states)  # (N, S, D)
        s1 = (data.unsqueeze(1) - mlp_out).pow_(2).sum(dim=2).div_(2 * sigma2)  # (N, S)
        s2 = states @ to.log(pi / (1.0 - pi))  # (N, S)
        lpj = s2 - s1
        assert lpj.shape == (N, S)
        assert not to.isnan(lpj).any() and not to.isinf(lpj).any()
        return lpj, mlp_out

    def free_energy(self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates) -> float:
        with to.no_grad():
            return super().free_energy(idx, batch, states)

    def log_joint(self, data, states, lpj=None):
        pi, sigma2 = get(self.theta, "pies", "sigma2")
        D = self._net_shape[-1]
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        # TODO: could pre-evaluate the constant factor once per epoch
        logjoints = lpj - D / 2 * to.log(2 * MATH_PI * sigma2) + to.log(1 - pi).sum()
        return logjoints

    def _free_energy_from_logjoints(self, logjoints: to.Tensor) -> to.Tensor:
        Fn = to.logsumexp(logjoints, dim=1)
        assert Fn.shape == (logjoints.shape[0],)
        assert not to.isnan(Fn).any() and not to.isinf(Fn).any()
        return Fn.sum()

    def update_param_batch(
        self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates
    ) -> float:
        F, mlp_out = self._optimize_nn_params(idx, batch, states)
        with to.no_grad():
            self._accumulate_param_updates(idx, batch, states, mlp_out)
        return F

    def update_param_epoch(self) -> None:
        pi = self.theta["pies"]
        sigma2 = self.theta["sigma2"]

        if pi.requires_grad and sigma2.requires_grad:
            return  # nothing to do
        else:
            # FIXME this is slightly incorrect in mpi runs:
            # the ShufflingSampler provides the same number of datapoints to each
            # worker by duplicating datapoints when needed. The total number of
            # datapoints processed in an epoch might be slightly higher than the
            # number of datapoints in the dataset.
            N, D = self._N, self._net_shape[-1]
            assert N is not None, "TVAE: N is None but model is being trained."

        if not pi.requires_grad:
            all_reduce(self._new_pi)
            pi[:] = self._new_pi / N
            # avoids infinites in lpj evaluation
            to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)
            self._new_pi.zero_()

        if not sigma2.requires_grad:
            all_reduce(self._new_sigma2)
            # disallow arbitrary growth of sigma. at each iteration, it can grow by at most 1%
            new_sigma_min = (sigma2 - sigma2.abs() * 0.01).item()
            new_sigma_max = (sigma2 + sigma2.abs() * 0.01).item()
            sigma2[:] = self._new_sigma2 / (N * D)
            if self._clamp_sigma:
                to.clamp(sigma2, new_sigma_min, new_sigma_max, out=sigma2)
            self._new_sigma2.zero_()

    def generate_from_hidden(self, hidden_state: to.Tensor) -> Dict[str, to.Tensor]:
        with to.no_grad():
            mlp_out = self.forward(hidden_state)
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
        lpj, mlp_out = self._lpj_and_mlpout(data, states.K[idx])
        F = self._free_energy_from_logjoints(self.log_joint(data, states, lpj))
        loss = -F / data.shape[0]
        loss.backward()

        self._mpi_average_grads()
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

        with to.no_grad():
            sigma2 = self.theta["sigma2"]
            if sigma2.requires_grad:
                to.clamp(sigma2, 1e-5, out=sigma2)
            pi = self.theta["pies"]
            if pi.requires_grad:
                to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)

        return F.item(), mlp_out

    def _accumulate_param_updates(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates, mlp_out: to.Tensor
    ) -> None:
        """Evaluate partial updates to pi and sigma2."""

        if not self.theta["pies"].requires_grad:
            # \pi_h = \frac{1}{N} \sum_n < K_nsh >_{q^n}
            K_batch = states.K[idx].type_as(states.lpj)
            self._new_pi.add_(mean_posterior(K_batch, states.lpj[idx]).sum(dim=0))

        if not self.theta["sigma2"].requires_grad:
            # \sigma2 = \frac{1}{DN} \sum_{n,d} < (y^n_d - \vec{a}^L_d)^2 >_{q^n}
            # TODO would it be better (faster or more numerically stable) to
            # sum over D _before_ taking the mean_posterior?
            y_minus_a_sqr = (data.unsqueeze(1) - mlp_out).pow_(2)  # (N, S, D)
            assert y_minus_a_sqr.shape == (idx.numel(), K_batch.shape[1], data.shape[1])
            self._new_sigma2.add_(mean_posterior(y_minus_a_sqr, states.lpj[idx]).sum((0, 1)))

    def forward(self, x: to.Tensor) -> to.Tensor:
        """Forward application of TVAE's MLP to the specified input."""
        assert x.shape[-1] == self._net_shape[0], "Incompatible shape in forward input"

        # middle layers (relu)
        output = x.to(dtype=self.precision)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            output = to.relu(output @ W + b)

        # output layer (linear)
        output = output @ self.W[-1] + self.b[-1]

        return output

    def _mpi_average_grads(self) -> None:
        if tvem.get_run_policy() != "mpi":
            return  # nothing to do

        # Average gradients across processes. See https://bit.ly/2FlJsxS
        with to.no_grad():
            n_procs = dist.get_world_size()
            for w, b in zip(self.W, self.b):
                all_reduce(w.grad)
                w.grad /= n_procs
                all_reduce(b.grad)
                b.grad /= n_procs
