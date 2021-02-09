# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor
from typing import Union, Tuple

import tvem
from tvem.utils.model_protocols import Sampler, Optimized, Reconstructor
from tvem.utils import CyclicLR
from tvem.utils.parallel import mpi_average_grads
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.parallel import all_reduce, broadcast


class PSC(Sampler, Optimized, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
        data_dtype: to.dtype = to.int64,
        analytical_pi_updates: bool = True,
        analytical_W_updates: bool = False,
        min_lr: float = 0.001,
        max_lr: float = 0.01,
        cycliclr_step_size_up=400,
    ):
        """BSC-like sparse coding model with Poisson instead of Gaussian noise.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes BSC weights.
        :param pies_init: Tensor with shape (H,), initializes BSC priors.
        :param precision: Floating point precision required.
        :param data_dtype: Integer precision of data.
        :param analytical_pi_updates: Whether priors should be updated via the analytical
                                      max-likelihood solution rather than gradient descent.
        :param analytical_W_updates: Whether weights should be updated via the analytical
                                      max-likelihood solution rather than gradient descent.
        :param min_lr: Hyperparameter of learning rate scheduler
        :param max_lr: Hyperparameter of learning rate scheduler
        :param cycliclr_step_size_up: Hyperparameter of learning rate scheduler
        """
        assert not analytical_W_updates, "analytical W updates not implemented"  # FIXME
        assert precision in (to.float32, to.float64), "precision must be one of to.float{32,64}"
        assert data_dtype in (
            to.int8,
            to.int16,
            to.int32,
            to.int64,
        ), "data_dtype must be one of to.int{8,16,32,64}"
        self._precision = precision
        device = tvem.get_device()

        if pies_init is None:
            pies_init = to.rand(H, device=device, dtype=precision)
        if W_init is None:
            W_init = to.rand(H, D, device=device, dtype=precision)
        self._theta = dict(
            pies=pies_init.requires_grad_((not analytical_pi_updates)),
            W=W_init.requires_grad_(not analytical_W_updates),
        )

        gd_parameters = []
        if analytical_pi_updates:
            self._new_pies = to.zeros(H, dtype=precision, device=device)
            self._analytical_pi_updates = True
        else:
            gd_parameters.append(self._theta["pies"])
            self._analytical_pi_updates = False

        if analytical_W_updates:
            self._analytical_W_updates = True
        else:
            gd_parameters.append(self._theta["W"])
            self._analytical_W_updates = False
        self.my_N = to.tensor([0], dtype=to.int, device=device)

        if len(gd_parameters) == 0:
            self._optimizer = None
            self._scheduler = None
        else:
            self._optimizer = to.optim.Adam(gd_parameters, lr=min_lr)
            self._scheduler = CyclicLR(
                self._optimizer,
                base_lr=min_lr,
                max_lr=max_lr,
                step_size_up=cycliclr_step_size_up,
                cycle_momentum=False,
            )

        self._config = dict(
            H=H,
            D=D,
            precision=precision,
            data_dtype=data_dtype,
            analytical_pi_updates=self._analytical_pi_updates,
            min_lr=min_lr,
            max_lr=max_lr,
            cycliclr_step_size_up=cycliclr_step_size_up,
            device=device,
        )
        self._shape = self.theta["W"].shape[::-1]
        self._tiny = to.finfo(precision).tiny

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:  # type: ignore
        """Evaluate log-pseudo-joints for PSC."""
        assert states.dtype == to.uint8  # and data.dtype == self.config["data_dtype"]
        N, S = states.shape[:2]
        pies, W = self.theta["pies"], self.theta["W"]
        pies_ = pies.clamp(1e-2, 1.0 - 1e-2)
        Kfloat = states.type_as(pies)
        priorterm = Kfloat @ to.log(pies_ / (1 - pies_))

        Wbar = (Kfloat @ W).clamp(min=self._tiny)
        lpj = (data.unsqueeze(1).type_as(pies) * Wbar.log() - Wbar).sum(dim=2) + priorterm
        assert lpj.shape == (N, S)
        assert not to.isnan(lpj).any() and not to.isinf(lpj).any()
        return lpj

    def log_joint(self, data: Tensor, states: Tensor, lpj: Tensor = None) -> Tensor:
        """Evaluate log-joints for PSC."""
        assert states.dtype == to.uint8  # and data.dtype == self.config["data_dtype"]
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        pies = self.theta["pies"]
        pies_ = pies.clamp(1e-2, 1.0 - 1e-2)
        logjoints = (
            lpj
            + to.sum(to.log(1 - pies_))
            - to.lgamma(data.type_as(pies) + 1.0 + self._tiny)
            .to(data.device)
            .sum(dim=1)
            .unsqueeze(1)
        )  # TODO: Evaluate prior term only once per epoch and factorial term only once per run
        assert logjoints.shape == lpj.shape
        assert not to.isnan(logjoints).any() and not to.isinf(logjoints).any()
        return logjoints

    @property
    def shape(self) -> Tuple[int, int]:
        return self.theta["W"].shape[::-1]

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        precision, device = self.precision, tvem.get_device()
        D, H = self.shape

        assert not (N is None and hidden_state is None), "Must provide either N or hidden_state"
        if hidden_state is None:
            hidden_state = to.rand((N, H), dtype=precision, device=device) < self.theta["pies"]
            must_return_hidden_state = True
        else:
            shape = hidden_state.shape
            if N is None:
                N = shape[0]
            assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        # Linearly superimpose generative fields
        Wbar = to.zeros((N, D), dtype=precision, device=device)
        assert N is not None  # to make mypy happy
        for n in range(N):
            for h in range(H):
                if hidden_state[n, h]:
                    Wbar[n] += self.theta["W"][h]

        # Add noise
        Y = to.poisson(Wbar).type(self.config["data_dtype"]).to(device)

        return (Y, hidden_state) if must_return_hidden_state else Y

    def _gd_step(self, loss: to.Tensor):
        assert self._optimizer is not None  # to make mypy happy
        assert self._scheduler is not None  # to make mypy happy
        loss.backward()
        mpi_average_grads(self.theta)
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad()

        with to.no_grad():
            if self.theta["pies"].requires_grad:
                to.clamp(self.theta["pies"], 1e-5, 1 - 1e-5, out=self.theta["pies"])

    def _analytical_msteps(self, idx: to.Tensor, states: "TVEMVariationalStates"):
        """Evaluate analytical partial M-step updates"""
        if self._analytical_pi_updates:
            assert not self.theta["pies"].requires_grad
            K_batch = states.K[idx].type_as(states.lpj)
            # \pi_h = \frac{1}{N} \sum_n < K_nsh >_{q^n}
            self._new_pies.add_(mean_posterior(K_batch, states.lpj[idx]).sum(dim=0))

    def update_param_batch(
        self, idx: to.Tensor, batch: to.Tensor, states: "TVEMVariationalStates"
    ) -> float:
        """Execute batch-wise M-step or batch-wise section of an M-step computation.

        :param idx: indexes of the datapoints that compose the batch within the dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all variational states for this dataset

        If the model allows it, as an optimization this method can return this batch's free energy
        evaluated _before_ the model parameter update. If the batch's free energy is returned here,
        Trainers will skip a direct per-batch call to the free_energy method.
        """
        log_joints = self.log_joint(batch, states.K[idx])
        F = to.logsumexp(log_joints, dim=1).sum(dim=0)
        if self._optimizer is not None:
            loss = -F / batch.shape[0]
            self._gd_step(loss)

        with to.no_grad():
            self._analytical_msteps(idx, states)

        self.my_N.add_(len(idx))

        return F

    def update_param_epoch(self) -> None:
        pies, W = self.theta["pies"], self.theta["W"]

        if tvem.get_run_policy() == "mpi":
            with to.no_grad():
                for p in self.theta.values():
                    if p.requires_grad:
                        broadcast(p)

        if pies.requires_grad and W.requires_grad:
            return  # nothing to do
        else:
            all_reduce(self.my_N)
            N = self.my_N.item()

            if not pies.requires_grad:
                all_reduce(self._new_pies)
                pies[:] = self._new_pies / N
                # avoids infinites in lpj evaluation
                to.clamp(pies, 1e-5, 1 - 1e-5, out=pies)
                self._new_pies.zero_()

        self.my_N[:] = 0.0

    def data_estimator(self, idx: Tensor, states: TVEMVariationalStates) -> Tensor:
        """Estimator used for data reconstruction. Data reconstruction can only be supported
            by a model if it implements this method. The estimator to be implemented is defined
            as follows:""" r"""
            :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
            """
        K = states.K[idx]
        # TODO Find solution to avoid byte->float casting of `K`
        # TODO Pre-allocate tensor and use `out` argument of to.matmul
        return mean_posterior(K.to(dtype=self.precision) @ self.theta["W"], states.lpj[idx])
