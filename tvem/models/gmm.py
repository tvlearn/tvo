# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


import torch as to
import math
from torch.distributions.one_hot_categorical import OneHotCategorical

from torch import Tensor
from typing import Union, Tuple, Dict, Any

import tvem
from tvem.utils.parallel import pprint, all_reduce, broadcast
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.model_protocols import Optimized, Sampler, Reconstructor
from tvem.utils.sanity import fix_theta

# pytorch 1.2 deprecates to.gels in favour of to.lstsq
lstsq = to.lstsq if int(to.__version__.split(".")[1]) >= 2 else to.gels


class GMM(Optimized, Sampler, Reconstructor):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        sigma2_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
    ):
        """Gaussian Mixture model (GMM).

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes GM weights.
        :param pies_init: Tensor with shape (H,), initializes GM priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.

        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        self._precision = precision

        device = tvem.get_device()

        if W_init is not None:
            assert W_init.shape == (D, H)
            W_init = W_init.to(dtype=precision, device=device)
        else:
            W_init = to.rand((D, H), dtype=precision, device=device)
            broadcast(W_init)

        if pies_init is not None:
            assert pies_init.shape == (H,)
            pies_init = pies_init.to(dtype=precision, device=device)
        else:
            pies_init = to.full((H,), 1.0 / H, dtype=precision, device=device)

        if sigma2_init is not None:
            assert sigma2_init.shape == (1,)
            sigma2_init = sigma2_init.to(dtype=precision, device=device)
        else:
            sigma2_init = to.tensor([1.0], dtype=precision, device=device)

        self._theta = {"pies": pies_init, "W": W_init, "sigma2": sigma2_init}
        eps, inf = 1.0e-5, math.inf
        self.policy = {
            "W": [None, to.full_like(self._theta["W"], -inf), to.full_like(self._theta["W"], inf)],
            "pies": [
                None,
                to.full_like(self._theta["pies"], eps),
                to.full_like(self._theta["pies"], 1.0 - eps),
            ],
            "sigma2": [
                None,
                to.full_like(self._theta["sigma2"], eps),
                to.full_like(self._theta["sigma2"], inf),
            ],
        }

        self.my_Wp = to.zeros((D, H), dtype=precision, device=device)
        self.my_Wq = to.zeros((H), dtype=precision, device=device)
        self.my_pies = to.zeros(H, dtype=precision, device=device)
        self.my_sigma2 = to.zeros(1, dtype=precision, device=device)
        self.my_N = to.tensor([0], dtype=to.int, device=device)
        self._config = dict(H=H, D=D, precision=precision, device=device)
        self._shape = self.theta["W"].shape

    def log_pseudo_joint(self, data: Tensor, states: Tensor, **kwargs: Dict[str, Any]) -> Tensor:
        """Evaluate log-pseudo-joints for GMM.

        TODO: Make use of notnan to neglect missing data
        """
        Kfloat = states.to(
            dtype=self.theta["W"].dtype
        )  # N,C,C # TODO Find solution to avoid byte->float casting
        Wbar = to.matmul(
            Kfloat, self.theta["W"].t()
        )  # N,C,D  # TODO Pre-allocate tensor and use `out` argument of to.matmul
        lpj = to.mul(
            to.sum(to.pow(Wbar - data[:, None, :], 2), dim=2), -1 / 2 / self.theta["sigma2"]
        ) + to.matmul(Kfloat, to.log(self.theta["pies"]))
        return lpj.to(device=states.device)

    def log_joint(
        self,
        data: Tensor,
        states: Tensor,
        lpj: Tensor = None,
        **kwargs: Dict[str, Any],
    ) -> Tensor:
        """Evaluate log-joints for GMM."""
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states, **kwargs)
        D = self.shape[0]
        return lpj - D / 2 * to.log(2 * math.pi * self.theta["sigma2"])

    def update_param_batch(
        self,
        idx: Tensor,
        batch: Tensor,
        states: Tensor,
        **kwargs: Dict[str, Any],
    ) -> None:
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        Kfloat = K.to(dtype=lpj.dtype)  # TODO Find solution to avoid byte->float casting
        Wbar = to.matmul(
            Kfloat, self.theta["W"].t()
        )  # N,S,D # TODO Find solution to re-use evaluations from E-step

        batch_s_pjc = mean_posterior(Kfloat, lpj)  # is (batch_size,H) mean_posterior(Kfloat, lpj)
        batch_Wp = batch.unsqueeze(2) * batch_s_pjc.unsqueeze(1)  # is (batch_size,D,H)
        batch_sigma2 = mean_posterior(to.sum((batch[:, None, :] - Wbar) ** 2, dim=2), lpj)

        self.my_pies.add_(to.sum(batch_s_pjc, dim=0))
        self.my_Wp.add_(to.sum(batch_Wp, dim=0))
        self.my_Wq.add_(to.sum(batch_s_pjc, dim=0))
        self.my_sigma2.add_(to.sum(batch_sigma2))
        self.my_N.add_(batch_size)

        return None

    def update_param_epoch(self) -> None:
        theta = self.theta
        policy = self.policy

        all_reduce(self.my_Wp)
        all_reduce(self.my_Wq)
        all_reduce(self.my_pies)
        all_reduce(self.my_sigma2)
        all_reduce(self.my_N)

        N = self.my_N.item()
        D = self.shape[0]

        # Calculate updated W
        Wold_noisy = theta["W"] + 0.1 * to.randn_like(theta["W"])
        broadcast(Wold_noisy)
        theta_new = {}
        try:
            theta_new["W"] = self.my_Wp / self.my_Wq[None, :]
        except RuntimeError:
            pprint("Inversion error. Will not update W but add some noise instead.")
            theta_new["W"] = Wold_noisy

        # Calculate updated pi
        theta_new["pies"] = self.my_pies / N

        # Calculate updated sigma^2
        theta_new["sigma2"] = self.my_sigma2 / N / D

        policy["W"][0] = Wold_noisy
        policy["pies"][0] = theta["pies"]
        policy["sigma2"][0] = theta["sigma2"]
        fix_theta(theta_new, policy)
        for key in theta:
            theta[key] = theta_new[key]

        self.my_Wp[:] = 0.0
        self.my_Wq[:] = 0.0
        self.my_pies[:] = 0.0
        self.my_sigma2[:] = 0.0
        self.my_N[:] = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta["W"].shape

    def generate_data(
        self, N: int = None, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        precision, device = self.precision, tvem.get_device()
        D, H = self.shape

        if hidden_state is None:
            assert N is not None
            pies = self.theta["pies"]
            hidden_state = OneHotCategorical(probs=pies).sample([N]) == 1
            must_return_hidden_state = True
        else:
            shape = hidden_state.shape
            if N is None:
                N = shape[0]
            assert shape == (N, H), f"hidden_state has shape {shape}, expected ({N},{H})"
            must_return_hidden_state = False

        Wbar = to.zeros((N, D), dtype=precision, device=device)

        for n in range(N):
            for h in range(H):
                if hidden_state[n, h]:
                    Wbar[n] += self.theta["W"][:, h]

        # Add noise according to the model parameters
        Y = Wbar + to.sqrt(self.theta["sigma2"]) * to.randn((N, D), dtype=precision, device=device)

        return (Y, hidden_state) if must_return_hidden_state else Y

    def data_estimator(
        self,
        idx: Tensor,
        batch: Tensor,
        states: TVEMVariationalStates,
        **kwargs: Dict[str, Any],
    ) -> Tensor:

        # Not yet implemented

        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:""" r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        # Not
        K = states.K[idx]
        # TODO Find solution to avoid byte->float casting of `K`
        # TODO Pre-allocate tensor and use `out` argument of to.matmul
        return mean_posterior(
            to.matmul(K.to(dtype=self.precision), self.theta["W"].t()), states.lpj[idx]
        )
