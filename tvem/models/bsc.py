# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import torch as to

from torch import Tensor
from typing import Dict, Tuple, Any

import tvem
from tvem.utils.parallel import pprint, all_reduce, broadcast
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.models.TVEMModel import TVEMModel
from tvem.utils.sanity import fix_theta

# pytorch 1.2 deprecates to.gels in favour of to.lstsq
lstsq = to.lstsq if int(to.__version__.split(".")[1]) >= 2 else to.gels


class BSC(TVEMModel):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        sigma_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
    ):
        """Shallow Binary Sparse Coding (BSC) model.

        :param H: Number of hidden units.
        :param D: Number of observables.
        :param W_init: Tensor with shape (D,H), initializes BSC weights.
        :param pies_init: Tensor with shape (H,), initializes BSC priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.

        """
        assert precision in (to.float32, to.float64), "precision must be one of torch.float{32,64}"
        self.precision = precision

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

        if sigma_init is not None:
            assert sigma_init.shape == (1,)
            sigma_init = sigma_init.to(dtype=precision, device=device)
        else:
            sigma_init = to.tensor([1.0], dtype=precision, device=device)

        theta = {"pies": pies_init, "W": W_init, "sigma": sigma_init}
        eps, inf = 1.0e-5, math.inf
        self.policy = {
            "W": [None, to.full_like(theta["W"], -inf), to.full_like(theta["W"], inf)],
            "pies": [
                None,
                to.full_like(theta["pies"], eps),
                to.full_like(theta["pies"], 1.0 - eps),
            ],
            "sigma": [None, to.full_like(theta["sigma"], eps), to.full_like(theta["sigma"], inf)],
        }

        super().__init__(theta=theta)

    def init_storage(self, S: int, Snew: int, batch_size: int):
        """Allocate tensors used in E- and M-step."""
        device = tvem.get_device()
        precision = self.precision
        D, H = self.theta["W"].shape
        self.storage = {
            "my_Wp": to.empty((D, H), dtype=precision, device=device),
            "my_Wq": to.empty((H, H), dtype=precision, device=device),
            "my_pies": to.empty(H, dtype=precision, device=device),
            "my_sigma": to.empty(1, dtype=precision, device=device),
            "pil_bar": to.empty(H, dtype=precision, device=device),
            "WT": to.empty((H, D), dtype=precision, device=device),
            "batch_Wbar": to.empty((batch_size, S + Snew, D), dtype=precision, device=device),
            "batch_s_pjc": to.empty((batch_size, H), dtype=precision, device=device),
            "batch_Wp": to.empty((batch_size, D, H), dtype=precision, device=device),
            "batch_Wq": to.empty((H, H), dtype=precision, device=device),
            "batch_sigma": to.empty((batch_size,), dtype=precision, device=device),
            "indS_filled": 0,
            "my_N": to.tensor([0], dtype=to.int, device=device),
        }

    @property
    def sorted_by_lpj(self) -> Dict[str, Tensor]:
        return {"batch_Wbar": self.storage["batch_Wbar"]}

    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Use hidden states to sample datapoints according to the noise model of BSC.

        :param hidden_state: a tensor with shape (N, H) where H is the number of hidden units.
        :returns: the datapoints, as a tensor with shape (N, D) where D is
                  the number of observables.
        """

        theta = self.theta

        precision, device = theta["W"].dtype, theta["W"].device
        no_datapoints, D, H_gen = (hidden_state.shape[0],) + theta["W"].shape

        Wbar = to.zeros((no_datapoints, D), dtype=precision, device=device)

        # Linear superposition
        for n in range(no_datapoints):
            for h in range(H_gen):
                if hidden_state[n, h]:
                    Wbar[n] += theta["W"][:, h]

        # Add noise according to the model parameters
        Y = Wbar + theta["sigma"] * to.randn((no_datapoints, D), dtype=precision, device=device)

        return Y

    def init_epoch(self):
        """Initialize tensors used in E- and M-step."""
        theta = self.theta
        storage = self.storage
        D = theta["W"].shape[0]

        for k in ("my_Wp", "my_Wq", "my_pies", "my_sigma"):
            storage[k].fill_(0.0)
        storage["pil_bar"][:] = to.log(theta["pies"] / (1.0 - theta["pies"]))
        storage["WT"][:, :] = theta["W"].t()
        storage["pre1"] = -1.0 / 2.0 / theta["sigma"] / theta["sigma"]
        storage["fenergy_const"] = to.log(1.0 - theta["pies"]).sum() - D / 2 * to.log(
            2 * math.pi * theta["sigma"] ** 2
        )

    def init_batch(self):
        """Reset counter for how many states tensors in sorted_by_lpj have been evaluated."""
        self.storage["indS_filled"] = 0

    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints for BSC."""
        batch_size, S, _ = states.shape
        indS_filled = self.storage["indS_filled"]

        # Type casting for compatibility with float tensors
        # TODO Find solution to avoid byte->float casting
        Kfloat = states.to(dtype=self.theta["W"].dtype)

        # Pre-compute Ws
        # TODO Use `out` argument of to.matmul, e.g.
        # to.matmul(tensor1=Kfloat, tensor2=self.storage['WT'], out=Wbar)
        Wbar = self.sorted_by_lpj["batch_Wbar"][:batch_size, indS_filled : (indS_filled + S), :]
        Wbar[:, :, :] = to.matmul(Kfloat, self.storage["WT"])
        self.storage["indS_filled"] += S

        # Compute lpj, is (batch_size, S)
        lpj = to.mul(
            to.sum(to.pow(Wbar - data[:, None, :], 2), dim=2), self.storage["pre1"]
        ) + to.matmul(Kfloat, self.storage["pil_bar"])
        return lpj.to(device=states.device)

    def log_joint(self, data: Tensor, states: Tensor, lpj: Tensor = None) -> Tensor:
        """Evaluate log-joints for BSC."""
        if lpj is None:
            lpj = self.log_pseudo_joint(data, states)
        return lpj + self.storage["fenergy_const"]

    def update_param_batch(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> None:
        storage = self.storage
        sorted_by_lpj = self.sorted_by_lpj
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        # Type casting for compatibility with float tensors
        # TODO Find solution to avoid byte->float casting
        Kfloat = K.to(dtype=lpj.dtype)

        storage["batch_s_pjc"][:batch_size, :] = mean_posterior(Kfloat, lpj)  # is (batch_size,H)
        storage["batch_Wp"][:batch_size, :, :] = batch.unsqueeze(2) * storage["batch_s_pjc"][
            :batch_size
        ].unsqueeze(
            1
        )  # is (batch_size,D,H)
        Kq = Kfloat.mul(tvem.variational._utils._lpj2pjc(lpj)[:, :, None])
        storage["batch_Wq"][:, :] = to.einsum("ijk,ijl->kl", Kq, Kfloat)  # is (batch_size,H,H)
        storage["batch_sigma"][:batch_size] = mean_posterior(
            to.sum(
                (batch[:, None, :] - sorted_by_lpj["batch_Wbar"][:batch_size, :S, :]) ** 2, dim=2
            ),
            lpj,
        )  # is (batch_size,)

        storage["my_pies"].add_(to.sum(storage["batch_s_pjc"][:batch_size, :], dim=0))
        storage["my_Wp"].add_(to.sum(storage["batch_Wp"][:batch_size, :, :], dim=0))
        storage["my_Wq"].add_(storage["batch_Wq"])
        storage["my_sigma"].add_(to.sum(storage["batch_sigma"][:batch_size]))
        storage["my_N"].add_(batch_size)

        return None

    def update_param_epoch(self) -> None:
        theta = self.theta
        storage = self.storage
        policy = self.policy

        for k in ("my_pies", "my_Wp", "my_Wq", "my_sigma", "my_N"):
            all_reduce(storage[k])
        N = storage["my_N"].item()
        storage["my_N"][:] = 0

        theta_new = {}
        # Calculate updated W
        Wold_noisy = theta["W"] + 0.1 * to.randn_like(theta["W"])
        broadcast(Wold_noisy)
        try:
            theta_new["W"] = lstsq(storage["my_Wp"].t(), storage["my_Wq"])[0].t()
        except RuntimeError:
            pprint("Inversion error. Will not update W but add some noise instead.")
            theta_new["W"] = Wold_noisy

        # Calculate updated pi
        theta_new["pies"] = storage["my_pies"] / N

        # Calculate updated sigma
        theta_new["sigma"] = to.sqrt(storage["my_sigma"] / N / theta["W"].shape[0])

        policy["W"][0] = Wold_noisy
        policy["pies"][0] = theta["pies"]
        policy["sigma"][0] = theta["sigma"]
        fix_theta(theta_new, policy)
        for key in theta:
            theta[key] = theta_new[key]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.theta["W"].shape

    def data_estimator(self, idx: Tensor, states: Tensor) -> Tensor:
        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        return mean_posterior(self.sorted_by_lpj["batch_Wbar"][:batch_size, :S, :], lpj)

    @property
    def config(self) -> Dict[str, Any]:
        D, H = self.theta["W"].shape
        return dict(H=H, D=D, precision=self.precision)
