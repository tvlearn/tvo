# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.utils.model_protocols import Trainable, Sampler, Reconstructor
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.parallel import all_reduce, broadcast, mpi_average_grads
from tvem.utils import get, CyclicLR
import torch.optim as opt
import tvem
import torch as to
from typing import Tuple, List, Dict, Iterable, Optional, Sequence, Union
from math import pi as MATH_PI
from abc import abstractmethod


def _get_net_shape(net_shape: Sequence[int] = None, W_init: Sequence[to.Tensor] = None):
    if net_shape is not None:
        return tuple(reversed(net_shape))
    else:
        assert (
            W_init is not None
        ), "Must pass one of `net_shape` and `W_init` to __init__ of the\
        TVAE model"
        return tuple(w.shape[0] for w in W_init) + (W_init[-1].shape[1],)


def _init_W(
    net_shape: Sequence[int], precision: to.dtype, init: Optional[Sequence[to.Tensor]]
) -> List[to.Tensor]:
    """Return weights initialized with Xavier or to specified init values.

    This method also makes sure that device and precision are the ones required by
    the model.
    """
    if init is None:
        n_layers = len(net_shape) - 1
        W_shapes = ((net_shape[ln], net_shape[ln + 1]) for ln in range(n_layers))
        W = list(map(to.nn.init.xavier_normal_, (to.empty(s) for s in W_shapes)))
    else:
        assert (
            len(init) == len(net_shape) - 1
        ), f"Shape is {net_shape} but {len(init)} weights passed"
        Wshapes = [w.shape for w in init]
        expected_Wshapes = [(net_shape[ln], net_shape[ln + 1]) for ln in range(len(init))]
        err_msg = f"Input W shapes: {Wshapes}\nExpected W shapes {expected_Wshapes}"
        assert all(ws == exp_s for ws, exp_s in zip(Wshapes, expected_Wshapes)), err_msg
        W = list(w.clone() for w in init)
    for w in W:
        broadcast(w)
    return [w.to(device=tvem.get_device(), dtype=precision).requires_grad_(True) for w in W]


def _init_b(
    net_shape: Sequence[int], precision: to.dtype, init: Optional[Iterable[to.Tensor]]
) -> List[to.Tensor]:
    """Return biases initialized to zeros or to specified init values.

    This method also makes sure that device and precision are the ones required by the model.
    """
    if init is None:
        B = [to.zeros(s) for s in net_shape[1:]]
    else:
        assert all(b.shape == (net_shape[ln + 1],) for ln, b in enumerate(init))
        B = [b.clone() for b in init]
    return [b.to(device=tvem.get_device(), dtype=precision).requires_grad_(True) for b in B]


def _init_pi(
    precision: to.dtype, init: Optional[to.Tensor], H0: int, requires_grad: bool
) -> to.Tensor:
    if init is None:
        pi = to.full((H0,), 1 / H0)
    else:
        assert init.shape == (H0,)
        pi = init.clone()
    return pi.to(device=tvem.get_device(), dtype=precision).requires_grad_(requires_grad)


def _init_sigma2(precision: to.dtype, init: Optional[float], requires_grad: bool) -> to.Tensor:
    sigma2 = to.tensor([0.01] if init is None else [init])
    return sigma2.to(device=tvem.get_device(), dtype=precision).requires_grad_(requires_grad)


class _TVAE(Trainable, Sampler, Reconstructor):
    _theta: Dict[str, to.Tensor]
    _precision: to.dtype
    _net_shape: Sequence[int]
    _scheduler: opt.lr_scheduler._LRScheduler
    _optimizer: opt.Optimizer
    _external_model: Optional[to.nn.Module] = None

    @abstractmethod
    def log_joint(self, data: to.Tensor, states: to.Tensor, lpj: to.Tensor = None) -> to.Tensor:
        ...

    def _log_pseudo_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        with to.no_grad():
            lpj, _ = self._lpj_and_mlpout(data, states)
        return lpj

    @abstractmethod
    def _lpj_and_mlpout(self, data: to.Tensor, states: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        ...

    def free_energy(self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates) -> float:
        with to.no_grad():
            return super().free_energy(idx, batch, states)

    def _free_energy_from_logjoints(self, logjoints: to.Tensor) -> to.Tensor:
        Fn = to.logsumexp(logjoints, dim=1)
        assert Fn.shape == (logjoints.shape[0],)
        assert not to.isnan(Fn).any() and not to.isinf(Fn).any()
        return Fn.sum()

    def update_param_batch(
        self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates
    ) -> float:
        if to.isnan(batch).any():
            raise RuntimeError("There are NaNs in this batch")
        F, mlp_out = self._optimize_nn_params(idx, batch, states)
        with to.no_grad():
            self._accumulate_param_updates(idx, batch, states, mlp_out)
        return F

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
        Gradient-based optimized parameters are changed in-place. All other arguments are left
        untouched.

        :returns: F and mlp_output _before_ the weight update
        """
        assert self._optimizer is not None  # to make mypy happy

        lpj, mlp_out = self._lpj_and_mlpout(data, states.K[idx])
        F = self._free_energy_from_logjoints(self.log_joint(data, states.K[idx], lpj))
        loss = -F / data.shape[0]
        loss.backward()

        mpi_average_grads(self.theta)
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

        return F.item(), mlp_out

    def _accumulate_param_updates(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates, mlp_out: to.Tensor
    ) -> None:
        pass

    def data_estimator(
        self, idx: to.Tensor, batch: to.Tensor, states: TVEMVariationalStates
    ) -> to.Tensor:  # type: ignore
        r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """

        lpj = states.lpj[idx]
        K = states.K[idx]

        with to.no_grad():
            means = self.forward(K)  # N,S,D

        return mean_posterior(means, lpj)  # N, D

    @abstractmethod
    def forward(self, x: to.Tensor) -> to.Tensor:
        """Forward application of TVAE's MLP to the specified input."""
        ...


class GaussianTVAE(_TVAE):
    def __init__(
        self,
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
        external_model: Optional[to.nn.Module] = None,
        optimizer: Optional[opt.Optimizer] = None,
    ):
        """Create a TVAE model with Gaussian observables.

        :param shape: Network shape, from observable to most hidden: (D,...,H1,H0). One of shape,
                      (W_init, b_init), external_model must be specified exclusively.
        :param precision: One of to.float32 or to.float64, indicates the floating point precision
                          of model parameters.
        :param min_lr: See docs of tvem.utils.CyclicLR
        :param max_lr: See docs of tvem.utils.CyclicLR
        :param cycliclr_step_size_up: See docs of tvem.utils.CyclicLR
        :param pi_init: Optional tensor with initial prior values
        :param W_init: Optional list of tensors with initial weight values. Weight matrices
                       must be ordered from most hidden to observable layer. One of shape,
                       (W_init, b_init), external_model must be specified exclusively.
        :param b_init: Optional list of tensors with initial bias. One of shape,
                       (W_init, b_init), external_model must be specified exclusively.
        :param sigma2_init: Optional initial value for model variance.
        :param analytical_sigma_updates: Whether sigmas should be updated via the analytical
                                         max-likelihood solution rather than gradient descent.
        :param analytical_pi_updates: Whether priors should be updated via the analytical
                                      max-likelihood solution rather than gradient descent.
        :param clamp_sigma_updates: Whether to limit the rate at which sigma can be updated.
        :param external_model: Optional decoder neural network. One of shape, (W_init, b_init),
                               external_model must be specified exclusively.
        :param optimizer: Gradient optimizer (defaults to Adam if not specified)
        """
        self._theta: Dict[str, to.Tensor] = {}
        self._clamp_sigma = clamp_sigma_updates
        self._precision = precision
        self._external_model = external_model
        assert (
            (shape is not None and W_init is None and b_init is None and external_model is None)
            or (
                shape is None
                and W_init is not None
                and b_init is not None
                and external_model is None
            )
            or (shape is None and W_init is None and b_init is None and external_model is not None)
        ), "Must exclusively specify one one `shape`, (`W_init`, `b_init`), `external_model`"

        if external_model is not None:
            assert hasattr(
                external_model, "H0"
            ), "for externally defined models, H0 has to be provided manually"
            assert hasattr(
                external_model, "shape"
            ), "for externally defined models, shape has to be provided manually"
            H0 = external_model.H0
            self._net_shape = external_model.shape
            self.W = self.b = None
            gd_parameters = list(external_model.parameters())
        else:
            self._net_shape = _get_net_shape(shape, W_init)
            H0 = self._net_shape[0]
            self.W = _init_W(self._net_shape, precision, W_init)
            self.b = _init_b(self._net_shape, precision, b_init)
            self._theta.update({f"W_{i}": W for i, W in enumerate(self.W)})
            self._theta.update({f"b_{i}": b for i, b in enumerate(self.b)})
            gd_parameters = self.W + self.b

        self._theta["pies"] = _init_pi(
            precision, pi_init, H0, requires_grad=not analytical_pi_updates
        )
        self._theta["sigma2"] = _init_sigma2(
            precision, sigma2_init, requires_grad=not analytical_sigma_updates
        )

        self._min_lr, self._max_lr, self._step_size_up = min_lr, max_lr, cycliclr_step_size_up

        if analytical_sigma_updates:
            self._new_sigma2 = to.zeros(1, dtype=precision, device=tvem.get_device())
            self._analytical_sigma_updates = True
        else:
            gd_parameters.append(self._theta["sigma2"])
            self._analytical_sigma_updates = False

        if analytical_pi_updates:
            self._new_pi = to.zeros(H0, dtype=precision, device=tvem.get_device())
            self._analytical_pi_updates = True
        else:
            gd_parameters.append(self._theta["pies"])
            self._analytical_pi_updates = False

        if optimizer is None:
            self._optimizer = opt.Adam(gd_parameters, lr=min_lr)
        else:
            self._optimizer = optimizer(gd_parameters)

        self._scheduler = CyclicLR(
            self._optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=cycliclr_step_size_up,
            cycle_momentum=False,
        )
        # number of datapoints processed in a training epoch
        self._train_datapoints = to.tensor([0], dtype=to.int, device=tvem.get_device())
        self._config = dict(
            net_shape=self._net_shape,
            precision=self.precision,
            min_lr=self._min_lr,
            max_lr=self._max_lr,
            step_size_up=self._step_size_up,
            analytical_sigma_updates=self._analytical_sigma_updates,
            analytical_pi_updates=self._analytical_pi_updates,
            clamp_sigma_updates=self._clamp_sigma,
            device=tvem.get_device(),
        )

    def _lpj_and_mlpout(self, data: to.Tensor, states: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        N = data.shape[0]
        N_, S, H = states.shape
        assert N == N_, "Shape mismatch between data and states"
        pi, sigma2 = get(self.theta, "pies", "sigma2")
        states = states.to(dtype=self.precision)

        mlp_out = self.forward(states)  # (N, S, D)

        # nansum used to automatically ignore missing data
        s1 = to.nansum((data.unsqueeze(1) - mlp_out).pow_(2), dim=2).div_(2 * sigma2)  # (N, S)
        s2 = states @ to.log(pi / (1.0 - pi))  # (N, S)
        lpj = s2 - s1
        assert lpj.shape == (N, S)
        assert not to.isnan(lpj).any() and not to.isinf(lpj).any()
        return lpj, mlp_out

    def log_joint(self, data, states, lpj=None):
        pi, sigma2 = get(self.theta, "pies", "sigma2")
        D = data.shape[1] - to.isnan(data).sum(dim=1)  # (N,): ignores missing data
        D = D.unsqueeze(1)  # (N, 1)
        if lpj is None:
            lpj = self._log_pseudo_joint(data, states)
        # TODO: could pre-evaluate the constant factor once per epoch
        logjoints = lpj - D / 2 * to.log(2 * MATH_PI * sigma2) + to.log(1 - pi).sum()
        return logjoints

    def update_param_epoch(self) -> None:
        pi = self.theta["pies"]
        sigma2 = self.theta["sigma2"]

        if tvem.get_run_policy() == "mpi":
            with to.no_grad():
                for p in self.theta.values():
                    if p.requires_grad:
                        broadcast(p)

        if pi.requires_grad and sigma2.requires_grad:
            return  # nothing to do
        else:
            D = self._net_shape[-1]
            all_reduce(self._train_datapoints)
            N = self._train_datapoints.item()

        if not pi.requires_grad:
            all_reduce(self._new_pi)
            pi[:] = self._new_pi / N
            # avoids infinites in lpj evaluation
            to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)
            self._new_pi.zero_()

        # FIXME in case of missing data there is a correction that should be applied here
        if not sigma2.requires_grad:
            all_reduce(self._new_sigma2)
            # disallow arbitrary growth of sigma. at each iteration, it can grow by at most 1%
            new_sigma_min = (sigma2 - sigma2.abs() * 0.01).item()
            new_sigma_max = (sigma2 + sigma2.abs() * 0.01).item()
            sigma2[:] = self._new_sigma2 / (N * D)
            if self._clamp_sigma:
                to.clamp(sigma2, new_sigma_min, new_sigma_max, out=sigma2)
            self._new_sigma2.zero_()

        self._train_datapoints[:] = 0

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        H = self.shape[-1]
        if hidden_state is None:
            pies = self.theta["pies"]
            hidden_state = to.rand((N, H), dtype=pies.dtype, device=pies.device) < pies
            must_return_hidden_state = True
        else:
            if N is not None:
                shape = hidden_state.shape
                assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        with to.no_grad():
            mlp_out = self.forward(hidden_state)
        Y = to.distributions.Normal(loc=mlp_out, scale=to.sqrt(self.theta["sigma2"])).sample()

        return (Y, hidden_state) if must_return_hidden_state else Y

    def _optimize_nn_params(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates
    ) -> Tuple[float, to.Tensor]:
        F, mlp_out = super()._optimize_nn_params(idx, data, states)

        with to.no_grad():
            sigma2 = self.theta["sigma2"]
            if sigma2.requires_grad:
                to.clamp(sigma2, 1e-5, out=sigma2)
            pi = self.theta["pies"]
            if pi.requires_grad:
                to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)

        return F, mlp_out

    def _accumulate_param_updates(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates, mlp_out: to.Tensor
    ) -> None:
        """Evaluate partial updates to pi and sigma2."""
        K_batch = states.K[idx].type_as(states.lpj)

        if not self.theta["pies"].requires_grad:
            # \pi_h = \frac{1}{N} \sum_n < K_nsh >_{q^n}
            self._new_pi.add_(mean_posterior(K_batch, states.lpj[idx]).sum(dim=0))

        if not self.theta["sigma2"].requires_grad:
            # \sigma2 = \frac{1}{DN} \sum_{n,d} < (y^n_d - \vec{a}^L_d)^2 >_{q^n}
            # TODO would it be better (faster or more numerically stable) to
            # sum over D _before_ taking the mean_posterior?
            y_minus_a_sqr = (data.unsqueeze(1) - mlp_out).pow_(2)  # (N, S, D)
            assert y_minus_a_sqr.shape == (idx.numel(), K_batch.shape[1], data.shape[1])
            self._new_sigma2.add_(mean_posterior(y_minus_a_sqr, states.lpj[idx]).sum((0, 1)))

        self._train_datapoints.add_(data.shape[0])

    def forward(self, x: to.Tensor) -> to.Tensor:
        """Forward application of TVAE's MLP to the specified input."""
        assert x.shape[-1] == self._net_shape[0], "Incompatible shape in forward input"

        output = x.to(dtype=self.precision)
        if self._external_model is not None:
            output = self._external_model.forward(output)
        else:
            assert isinstance(self.W, Sequence) and isinstance(
                self.b, Sequence
            )  # to make mypy happy

            # middle layers (relu)
            for W, b in zip(self.W[:-1], self.b[:-1]):
                output = to.relu(output @ W + b)

            # output layer (linear)
            output = output @ self.W[-1] + self.b[-1]

        return output


class BernoulliTVAE(_TVAE):
    def __init__(
        self,
        shape: Sequence[int] = None,
        precision: to.dtype = to.float64,
        min_lr: float = 0.001,
        max_lr: float = 0.01,
        cycliclr_step_size_up=400,
        pi_init: to.Tensor = None,
        W_init: Sequence[to.Tensor] = None,
        b_init: Sequence[to.Tensor] = None,
        analytical_pi_updates: bool = True,
        external_model: Optional[to.nn.Module] = None,
        optimizer: Optional[opt.Optimizer] = None,
    ):
        """Create a TVAE model with Bernoulli observables.

        :param shape: Network shape, from observable to most hidden: (D,...,H1,H0).
                      Can be None if W_init is not None.
        :param precision: One of to.float32 or to.float64, indicates the floating point precision
                          of model parameters.
        :param min_lr: See docs of tvem.utils.CyclicLR
        :param max_lr: See docs of tvem.utils.CyclicLR
        :param cycliclr_step_size_up: See docs of tvem.utils.CyclicLR
        :param pi_init: Optional tensor with initial prior values
        :param W_init: Optional list of tensors with initial weight values. Weight matrices
                       must be ordered from most hidden to observable layer. If this parameter
                       is not None, the shape parameter can be omitted.
        :param b_init: Optional list of tensors with initial.
        :param analytical_pi_updates: Whether priors should be updated via the analytical
                                      max-likelihood solution rather than gradient descent.
        """
        self._theta: Dict[str, to.Tensor] = {}
        self._precision = precision
        self._external_model = external_model
        assert (
            (shape is not None and W_init is None and b_init is None and external_model is None)
            or (
                shape is None
                and W_init is not None
                and b_init is not None
                and external_model is None
            )
            or (shape is None and W_init is None and b_init is None and external_model is not None)
        ), "Must exclusively specify one one `shape`, (`W_init`, `b_init`), `external_model`"

        if external_model is not None:
            assert hasattr(
                external_model, "H0"
            ), "for externally defined models, H0 has to be provided manually"
            assert hasattr(
                external_model, "shape"
            ), "for externally defined models, shape has to be provided manually"
            H0 = external_model.H0
            self._net_shape = external_model.shape
            self.W = self.b = None
            gd_parameters = list(external_model.parameters())
        else:
            self._net_shape = _get_net_shape(shape, W_init)
            H0 = self._net_shape[0]
            self.W = _init_W(self._net_shape, precision, W_init)
            self.b = _init_b(self._net_shape, precision, b_init)
            self._theta.update({f"W_{i}": W for i, W in enumerate(self.W)})
            self._theta.update({f"b_{i}": b for i, b in enumerate(self.b)})
            gd_parameters = self.W + self.b

        self._theta["pies"] = _init_pi(
            precision, pi_init, H0, requires_grad=not analytical_pi_updates
        )

        self._min_lr, self._max_lr, self._step_size_up = min_lr, max_lr, cycliclr_step_size_up

        if analytical_pi_updates:
            self._new_pi = to.zeros(H0, dtype=precision, device=tvem.get_device())
            self._analytical_pi_updates = True
        else:
            gd_parameters.append(self._theta["pies"])
            self._analytical_pi_updates = False

        if optimizer is None:
            self._optimizer = opt.Adam(gd_parameters, lr=min_lr)
        else:
            self._optimizer = optimizer(gd_parameters)

        self._scheduler = CyclicLR(
            self._optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=cycliclr_step_size_up,
            cycle_momentum=False,
        )
        # number of datapoints processed in a training epoch
        self._train_datapoints = to.tensor([0], dtype=to.int, device=tvem.get_device())
        self._config = dict(
            net_shape=self._net_shape,
            precision=self.precision,
            min_lr=self._min_lr,
            max_lr=self._max_lr,
            step_size_up=self._step_size_up,
            analytical_pi_updates=self._analytical_pi_updates,
            device=tvem.get_device(),
        )

    def _lpj_and_mlpout(self, data: to.Tensor, states: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        N, D = data.shape
        N_, S, H = states.shape
        assert N == N_, "Shape mismatch between data and states"
        pi = self.theta["pies"]
        states = states.to(dtype=self.precision)

        mlp_out = self.forward(states)  # (N, S, D)

        # nansum used to automatically ignore missing data
        s1 = to.nansum(
            to.nn.functional.binary_cross_entropy(
                mlp_out, data.unsqueeze(1).expand(N, S, D), reduction="none"
            ),
            dim=2,
        )  # (N, S)
        s2 = states @ to.log(pi / (1.0 - pi))  # (N, S)
        lpj = s2 - s1
        assert lpj.shape == (N, S)
        assert not to.isnan(lpj).any() and not to.isinf(lpj).any()
        return lpj, mlp_out

    def log_joint(self, data, states, lpj=None):
        D = data.shape[1] - to.isnan(data).sum(dim=1)  # (N,): ignores missing data
        D = D.unsqueeze(1)  # (N, 1)
        if lpj is None:
            lpj = self._log_pseudo_joint(data, states)
        # TODO: could pre-evaluate the constant factor once per epoch
        logjoints = lpj + to.log(1 - self.theta["pies"]).sum()
        return logjoints

    def update_param_epoch(self) -> None:
        pi = self.theta["pies"]

        if tvem.get_run_policy() == "mpi":
            with to.no_grad():
                for p in self.theta.values():
                    if p.requires_grad:
                        broadcast(p)

        if pi.requires_grad:
            return  # nothing to do
        else:
            all_reduce(self._train_datapoints)
            N = self._train_datapoints.item()
            all_reduce(self._new_pi)
            pi[:] = self._new_pi / N
            # avoids infinites in lpj evaluation
            to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)
            self._new_pi.zero_()

        self._train_datapoints[:] = 0

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        H = self.shape[-1]
        if hidden_state is None:
            pies = self.theta["pies"]
            hidden_state = to.rand((N, H), dtype=pies.dtype, device=pies.device) < pies
            must_return_hidden_state = True
        else:
            if N is not None:
                shape = hidden_state.shape
                assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        with to.no_grad():
            mlp_out = self.forward(hidden_state)
        Y = to.distributions.Bernoulli(mlp_out).sample()

        return (Y, hidden_state) if must_return_hidden_state else Y

    def _optimize_nn_params(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates
    ) -> Tuple[float, to.Tensor]:
        F, mlp_out = super()._optimize_nn_params(idx, data, states)

        with to.no_grad():
            pi = self.theta["pies"]
            if pi.requires_grad:
                to.clamp(pi, 1e-5, 1 - 1e-5, out=pi)

        return F, mlp_out

    def _accumulate_param_updates(
        self, idx: to.Tensor, data: to.Tensor, states: TVEMVariationalStates, mlp_out: to.Tensor
    ) -> None:
        """Evaluate partial updates to pi."""
        K_batch = states.K[idx].type_as(states.lpj)

        if not self.theta["pies"].requires_grad:
            # \pi_h = \frac{1}{N} \sum_n < K_nsh >_{q^n}
            self._new_pi.add_(mean_posterior(K_batch, states.lpj[idx]).sum(dim=0))

        self._train_datapoints.add_(data.shape[0])

    def forward(self, x: to.Tensor) -> to.Tensor:
        """Forward application of TVAE's MLP to the specified input."""
        assert x.shape[-1] == self._net_shape[0], "Incompatible shape in forward input"

        output = x.to(dtype=self.precision)
        if self._external_model is not None:
            output = self._external_model.forward(output)
        else:
            assert isinstance(self.W, Sequence) and isinstance(
                self.b, Sequence
            )  # to make mypy happy

            # middle layers (relu)
            output = x.to(dtype=self.precision)
            for W, b in zip(self.W[:-1], self.b[:-1]):
                output = to.relu(output @ W + b)

            # output layer (sigmoid)
            output = to.sigmoid(output @ self.W[-1] + self.b[-1])

        return output
